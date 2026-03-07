import threading
import time
import os
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from dataset import EMGDataset
from util import record_emg
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import torchmetrics
import tkinter as tk
from tkinter import font, ttk
from model import HDClassifier
import scipy.signal as signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ================= CONSTANTS =================
NUM_SAMPLES = 5000
NUM_CHANNELS = 8
SAMPLES_PER_POINT = 50
BATCH_SIZE = 1
STATE_DICT = {0: "Relax", 1: "Clench", 2: "Triton", 3: "L(ove)", 4: "Surfer"}
SAMPLING_RATE = 500
DIMENSIONS = 10000
GENERAL_MODEL_PATH = "general_model.pt"
CALIB_DIR = "calibration_data"

os.makedirs(CALIB_DIR, exist_ok=True)
for cls_name in STATE_DICT.values():
    os.makedirs(os.path.join(CALIB_DIR, cls_name), exist_ok=True)


class VaderGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Vader Hand Control")
        self.root.configure(bg='black')

        torch.manual_seed(42)
        np.random.seed(42)

        self.stop_event = threading.Event()
        self.continue_event = threading.Event()
        self.hardware_initialized = False

        self.board_shim = None

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
        self.general_model = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
        self._load_general_model()

        self.title_font = font.Font(size=28, weight='bold')
        self.button_font = font.Font(size=14)

        self.pred_history = []

        # ==== DISPLAY LABEL ====
        self.display = tk.Label(
            root,
            text="Welcome to Vader!",
            fg="red",
            bg="black",
            font=self.title_font,
            justify='center'
        )
        self.display.pack(pady=10)

        # ==== PROGRESS BAR ====
        style = ttk.Style()
        style.theme_use('classic')
        style.configure('Red.Horizontal.TProgressbar',
                        troughcolor='black',
                        background='red')

        self.progress = ttk.Progressbar(
            root,
            style='Red.Horizontal.TProgressbar',
            orient='horizontal',
            mode='determinate'
        )
        self.progress.pack(fill='x', padx=50, pady=10)

        # ==== FIX: Bandpass upper cutoff must be < Nyquist (250 Hz) ====
        nyq = SAMPLING_RATE / 2
        self.bp_b, self.bp_a = signal.butter(4, [20 / nyq, 200 / nyq], btype='band')
        self.notch_b, self.notch_a = signal.iirnotch(60 / nyq, Q=30)

        self.zi_bp = [signal.lfilter_zi(self.bp_b, self.bp_a)
                      for _ in range(NUM_CHANNELS)]
        self.zi_notch = [signal.lfilter_zi(self.notch_b, self.notch_a)
                         for _ in range(NUM_CHANNELS)]

        # ==== Waveform display parameters ====
        self.offset_step = 300
        self.rms_window = 100
        self.var_threshold = 5.0
        self.display_buffer = np.zeros((NUM_CHANNELS, 500))
        self.channel_enabled = [tk.BooleanVar(value=True)
                                 for _ in range(NUM_CHANNELS)]

        self._init_waveform_panel()

        # ==== Channel toggle UI ====
        toggle_frame = tk.Frame(root, bg='black')
        toggle_frame.pack()
        for ch in range(NUM_CHANNELS):
            cb = tk.Checkbutton(
                toggle_frame,
                text=f"Ch {ch + 1}",
                variable=self.channel_enabled[ch],
                fg='white',
                bg='black',
                selectcolor='black'
            )
            cb.pack(side='left')

        # ==== BUTTONS ====
        btn_frame = tk.Frame(root, bg='black')
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Start Calibration",
                  command=self.start_process,
                  bg='red').pack(side='left', padx=5)

        tk.Button(btn_frame, text="Use General Model",
                  command=self.start_general_inference,   # FIX: method now exists
                  bg='red').pack(side='left', padx=5)

        tk.Button(btn_frame, text="Aggregate",
                  command=self.start_aggregate,           # FIX: method now exists
                  bg='red').pack(side='left', padx=5)

        tk.Button(btn_frame, text="Infer Aggregate",
                  command=self.start_aggregate_inference,
                  bg='red').pack(side='left', padx=5)

        self.stop_btn = tk.Button(btn_frame,
                                  text="Stop",
                                  command=self.stop_process,
                                  bg='red')
        self.stop_btn.pack(side='left', padx=5)

    # ====================================================
    # ==== Thread-safe UI update =========================
    # ====================================================

    def update_display(self, text):
        self.root.after(0, lambda: self.display.config(text=text))

    def update_progress(self, val):
        self.root.after(0, lambda: self.progress.config(value=val))

    # ====================================================
    # ==== Waveform panel initialization =================
    # ====================================================

    def _init_waveform_panel(self):
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(-self.offset_step,
                         NUM_CHANNELS * self.offset_step)
        self.ax.set_title("Filtered EMG + RMS", color='white')
        self.ax.tick_params(colors='white')

        self.lines = []
        self.rms_lines = []

        for ch in range(NUM_CHANNELS):
            line, = self.ax.plot([], [], linewidth=0.8, color='cyan')
            rms_line, = self.ax.plot([], [], linewidth=2, color='yellow')
            self.lines.append(line)
            self.rms_lines.append(rms_line)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    # ====================================================
    # ==== Live plot loop ================================
    # ====================================================

    def start_live_plot(self):
        threading.Thread(target=self._live_plot_loop,
                         daemon=True).start()

    def _live_plot_loop(self):
        while not self.stop_event.is_set():
            if not self.hardware_initialized:
                time.sleep(0.1)
                continue

            # FIX: board_shim may be None (mocked hardware); skip the data
            # count check and use simulated data directly.
            if self.board_shim is not None:
                if self.board_shim.get_board_data_count() < SAMPLES_PER_POINT:
                    time.sleep(0.01)
                    continue
                data = self.board_shim.get_board_data(SAMPLES_PER_POINT)
                emg = np.array(data[:NUM_CHANNELS])
            else:
                # Simulated data when hardware is mocked
                emg = np.random.randn(NUM_CHANNELS, SAMPLES_PER_POINT) * 50

            for ch in range(NUM_CHANNELS):
                emg[ch], self.zi_bp[ch] = signal.lfilter(
                    self.bp_b, self.bp_a, emg[ch], zi=self.zi_bp[ch])
                emg[ch], self.zi_notch[ch] = signal.lfilter(
                    self.notch_b, self.notch_a, emg[ch], zi=self.zi_notch[ch])

            self.display_buffer = np.roll(
                self.display_buffer, -SAMPLES_PER_POINT, axis=1)
            self.display_buffer[:, -SAMPLES_PER_POINT:] = emg

            self.root.after(0, self._update_plot)
            time.sleep(0.02)

    # ====================================================
    # ==== Vertical offset + RMS + bad-channel detection =
    # ====================================================

    def _update_plot(self):
        for ch in range(NUM_CHANNELS):

            if not self.channel_enabled[ch].get():
                self.lines[ch].set_data([], [])
                self.rms_lines[ch].set_data([], [])
                continue

            signal_data = self.display_buffer[ch]
            offset = ch * self.offset_step
            shifted = signal_data + offset

            self.lines[ch].set_data(
                np.arange(len(signal_data)),
                shifted
            )

            rms = np.sqrt(
                np.convolve(signal_data ** 2,
                            np.ones(self.rms_window) / self.rms_window,
                            mode='same')
            )

            self.rms_lines[ch].set_data(
                np.arange(len(signal_data)),
                rms + offset
            )

            # Bad-channel detection (turns line red if near-flat signal)
            if np.var(signal_data) < self.var_threshold:
                self.lines[ch].set_color('red')
            else:
                self.lines[ch].set_color('cyan')

        self.canvas.draw_idle()

    # ====================================================
    # ==== HARDWARE ======================================
    # ====================================================

    def init_hardware(self):
        if self.hardware_initialized:
            return
        # ---- Use CSV replay instead of random noise ----
        from csv_board import CSVBoard
        self.board_shim = CSVBoard("EMG_mindrove60hz_5_10.csv")
    # ------------------------------------------------
        #board_id = BoardIds.MINDROVE_WIFI_BOARD
        #params = MindRoveInputParams()
        # Uncomment below when real hardware is connected:
        # self.board_shim = BoardShim(board_id, params)
        # self.board_shim.prepare_session()
        # self.board_shim.start_stream(450000)

        self.hardware_initialized = True
        self.start_live_plot()

    # ====================================================
    # ==== Model persistence =============================
    # ====================================================

    def _save_general_model(self):
        torch.save(self.general_model.state_dict(), GENERAL_MODEL_PATH)

    def _load_general_model(self):
        if os.path.exists(GENERAL_MODEL_PATH):
            self.general_model.load_state_dict(
                torch.load(GENERAL_MODEL_PATH, map_location=self.device)
            )

    # ====================================================
    # ==== Inference helpers =============================
    # ====================================================

    def _inference_step(self, model):
        """Run one inference step and update display with smoothed prediction."""
        start_time = time.time()

        rot, win = record_emg(
            self.board_shim,
            SAMPLES_PER_POINT,
            NUM_CHANNELS,
            SAMPLING_RATE
        )

        xb = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(self.device)

        out = model(xb)
        cls = out.argmax().item()

        self.pred_history.append(cls)
        if len(self.pred_history) > 5:
            self.pred_history.pop(0)

        final_cls = max(set(self.pred_history), key=self.pred_history.count)
        latency = (time.time() - start_time) * 1000

        self.update_display(
            f"Pose: {STATE_DICT[final_cls]}\n"
            f"Latency: {latency:.1f} ms"
        )

    def _run_inference_loop(self, model):
        """Continuously run inference until stop_event is set."""
        while not self.stop_event.is_set():
            try:
                self._inference_step(model)
            except Exception as e:
                self.update_display(f"Inference error:\n{e}")
                break

    # ====================================================
    # ==== FIX: start_general_inference (was missing) ====
    # ====================================================

    def start_general_inference(self):
        """Start inference using the pre-trained general model."""
        self.stop_event.clear()
        self.pred_history.clear()
        self.update_display("Running General Model...")
        threading.Thread(
            target=self._ensure_hardware_then,
            args=(lambda: self._run_inference_loop(self.general_model),),
            daemon=True
        ).start()

    # ====================================================
    # ==== FIX: start_aggregate (was missing) ============
    # ====================================================

    def start_aggregate(self):
        """
        Aggregate calibration data from CALIB_DIR to build/update a model
        that combines the general model weights with user-specific data.
        This trains self.aggregate_model and saves it.
        """
        self.stop_event.clear()
        self.update_display("Aggregating calibration data...")
        threading.Thread(target=self._aggregate_worker, daemon=True).start()

    def _aggregate_worker(self):
        try:
            # Collect all saved calibration samples
            all_data, all_labels = [], []
            for label_idx, cls_name in STATE_DICT.items():
                cls_dir = os.path.join(CALIB_DIR, cls_name)
                for fname in os.listdir(cls_dir):
                    if fname.endswith('.npy'):
                        arr = np.load(os.path.join(cls_dir, fname))
                        all_data.append(arr)
                        all_labels.append(label_idx)

            if not all_data:
                self.update_display("No calibration data found.\nRun calibration first.")
                return

            dataset = EMGDataset(all_data, all_labels)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Start from general model weights
            self.aggregate_model = HDClassifier(
                DIMENSIONS, len(STATE_DICT), NUM_CHANNELS
            ).to(self.device)
            self.aggregate_model.load_state_dict(self.general_model.state_dict())

            # Simple HD retraining pass
            self.aggregate_model.train()
            total = len(loader)
            for i, (xb, yb) in enumerate(loader):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.aggregate_model.train_step(xb, yb)
                self.update_progress(int((i + 1) / total * 100))

            self.update_display("Aggregate model ready!")
            self.update_progress(100)

        except Exception as e:
            self.update_display(f"Aggregate error:\n{e}")

    # ====================================================
    # ==== start_aggregate_inference =====================
    # ====================================================

    def start_aggregate_inference(self):
        """Start inference using the aggregated user model."""
        if not hasattr(self, 'aggregate_model'):
            self.update_display("No aggregate model found.\nRun Aggregate first.")
            return
        self.stop_event.clear()
        self.pred_history.clear()
        self.update_display("Running Aggregate Model...")
        threading.Thread(
            target=self._ensure_hardware_then,
            args=(lambda: self._run_inference_loop(self.aggregate_model),),
            daemon=True
        ).start()

    # ====================================================
    # ==== CONTROL =======================================
    # ====================================================

    def _ensure_hardware_then(self, fn):
        """Initialize hardware if needed, then call fn()."""
        self.init_hardware()
        fn()

    def stop_process(self):
        """Signal all background threads to stop."""
        self.stop_event.set()
        self.update_display("Stopped.")

    def start_process(self):
        """Start calibration (initializes hardware)."""
        # FIX: reset stop_event so threads can run again after a previous stop
        self.stop_event.clear()
        threading.Thread(target=self.init_hardware, daemon=True).start()


if __name__ == '__main__':
    root = tk.Tk()
    app = VaderGUI(root)
    root.mainloop()