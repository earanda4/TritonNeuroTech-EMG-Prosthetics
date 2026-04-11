import threading
import time
import os
import numpy as np
import scipy
import torch
from util import record_emg
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import tkinter as tk
from tkinter import font, ttk
from mindrove_bilstm import FinalPushLSTM, BILSTM_MODEL_PATH, BILSTM_NUM_CLASSES
import scipy.signal as signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ================= CONSTANTS =================
NUM_CHANNELS = 4
SAMPLES_PER_POINT = 50
SAMPLING_RATE = 500
CALIB_DIR = "calibration_data"

# BiLSTM uses 4 channels; labels match the training data in mindrove_bilstm.py
BILSTM_STATE_DICT = {0: "Relax", 1: "Fist", 2: "Rock", 3: "Peace", 4: "Shaka"}

os.makedirs(CALIB_DIR, exist_ok=True)

THEMES = {
    'Dark': {
        'bg': 'black',       'fg': 'red',        'text_fg': 'white',
        'btn_bg': 'red',     'bilstm_bg': 'dark red', 'bilstm_fg': 'white',
        'check_fg': 'white', 'plot_bg': 'black',  'plot_fg': 'white',
        'emg_color': 'cyan', 'rms_color': 'yellow',
    },
    'Light': {
        'bg': '#f0f0f0',     'fg': '#8b0000',    'text_fg': '#111111',
        'btn_bg': '#c0392b', 'bilstm_bg': '#7b241c', 'bilstm_fg': 'white',
        'check_fg': '#111111', 'plot_bg': '#ffffff', 'plot_fg': '#111111',
        'emg_color': '#0055cc', 'rms_color': '#cc6600',
    },
    'High Contrast': {
        'bg': 'black',       'fg': 'yellow',     'text_fg': 'yellow',
        'btn_bg': '#555500', 'bilstm_bg': '#333300', 'bilstm_fg': 'yellow',
        'check_fg': 'yellow','plot_bg': 'black',  'plot_fg': 'yellow',
        'emg_color': 'white','rms_color': 'yellow',
    },
}


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

        self.bilstm_model = FinalPushLSTM(input_size=NUM_CHANNELS, num_classes=BILSTM_NUM_CLASSES).to(self.device)
        self._load_bilstm_model()

        self.current_theme = 'Dark'
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
        self.pb_style = ttk.Style()
        self.pb_style.theme_use('classic')
        self.pb_style.configure('Themed.Horizontal.TProgressbar',
                                troughcolor='black', background='red')

        self.progress = ttk.Progressbar(
            root,
            style='Themed.Horizontal.TProgressbar',
            orient='horizontal',
            mode='determinate'
        )
        self.progress.pack(fill='x', padx=50, pady=10)

        # ==== Bandpass / notch filters ====
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
        self.toggle_frame = tk.Frame(root, bg='black')
        self.toggle_frame.pack()
        self.channel_checkboxes = []
        for ch in range(NUM_CHANNELS):
            cb = tk.Checkbutton(
                self.toggle_frame,
                text=f"Ch {ch + 1}",
                variable=self.channel_enabled[ch],
                fg='white',
                bg='black',
                selectcolor='black'
            )
            cb.pack(side='left')
            self.channel_checkboxes.append(cb)

        # ==== BUTTONS ====
        self.btn_frame = tk.Frame(root, bg='black')
        self.btn_frame.pack(pady=10)

        self.start_cal_btn = tk.Button(self.btn_frame, text="Start Calibration",
                                       command=self.start_process, bg='red')
        self.start_cal_btn.pack(side='left', padx=5)

        self.bilstm_btn = tk.Button(self.btn_frame, text="BiLSTM Model",
                                    command=self.start_bilstm_inference,
                                    bg='dark red', fg='white')
        self.bilstm_btn.pack(side='left', padx=5)

        self.test_btn = tk.Button(self.btn_frame, text="Test Connection",
                                  command=self.test_connection, bg='green')
        self.test_btn.pack(side='left', padx=5)

        self.stop_btn = tk.Button(self.btn_frame, text="Stop",
                                  command=self.stop_process, bg='red')
        self.stop_btn.pack(side='left', padx=5)

        tk.Button(self.btn_frame, text="Accessibility",
                  command=self._open_accessibility,
                  bg='gray', fg='white').pack(side='left', padx=5)

    # ====================================================
    # ==== Test ==========================================
    # ====================================================
    def test_connection(self):
        """Force-start hardware and stream a few samples to verify data flow."""
        self.update_display("Testing connection...")
        self.stop_event.clear()

        def _test():
            try:
                self.init_hardware()
                time.sleep(1)  # let the buffer fill

                if self.board_shim is None:
                    self.update_display("No board connected.")
                    return

                count = self.board_shim.get_board_data_count()
                if count == 0:
                    self.update_display("Board found but\nno data flowing!")
                    return

                data = self.board_shim.get_board_data(min(count, 50))
                emg = np.array(data[:NUM_CHANNELS])
                rms_per_channel = np.sqrt(np.mean(emg**2, axis=1))

                lines = ["Connection OK!", f"Samples received: {count}"]
                for ch, rms in enumerate(rms_per_channel):
                    lines.append(f"Ch{ch+1} RMS: {rms:.1f}")

                self.update_display("\n".join(lines))

            except Exception as e:
                self.update_display(f"Connection failed:\n{e}")

        threading.Thread(target=_test, daemon=True).start()

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

            show_rms = getattr(self, 'show_rms', None)
            if show_rms is None or show_rms.get():
                self.rms_lines[ch].set_data(np.arange(len(signal_data)), rms + offset)
            else:
                self.rms_lines[ch].set_data([], [])

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
        # from csv_board import CSVBoard
        # self.board_shim = CSVBoard("EMG_mindrove60hz_5_10.csv")
    # ------------------------------------------------
        #board_id = BoardIds.MINDROVE_WIFI_BOARD
        #params = MindRoveInputParams()
        # Uncomment below when real hardware is connected:

        board_id = BoardIds.MINDROVE_WIFI_BOARD
        params = MindRoveInputParams()
        self.board_shim = BoardShim(board_id, params)
        self.board_shim.prepare_session()
        self.board_shim.start_stream(450000)

        self.hardware_initialized = True
        self.start_live_plot()

    # ====================================================
    # ==== Model persistence =============================
    # ====================================================

    def _load_bilstm_model(self):
        if os.path.exists(BILSTM_MODEL_PATH):
            self.bilstm_model.load_state_dict(
                torch.load(BILSTM_MODEL_PATH, map_location=self.device)
            )
            self.bilstm_model.eval()
            print(f"BiLSTM model loaded from {BILSTM_MODEL_PATH}")
        else:
            print(f"No BiLSTM model found at {BILSTM_MODEL_PATH}. Train it by running mindrove_bilstm.py.")

    # ====================================================
    # ==== BiLSTM inference ==============================
    # ====================================================

    def start_bilstm_inference(self):
        """Start inference using the trained BiLSTM model."""
        if not os.path.exists(BILSTM_MODEL_PATH):
            self.update_display("No BiLSTM model found.\nRun mindrove_bilstm.py first.")
            return
        self.stop_event.clear()
        self.pred_history.clear()
        self.update_display("Running BiLSTM Model...")
        threading.Thread(
            target=self._ensure_hardware_then,
            args=(lambda: self._bilstm_inference_loop(),),
            daemon=True
        ).start()

    def _bilstm_inference_loop(self):
        while not self.stop_event.is_set():
            try:
                self._bilstm_inference_step()
            except Exception as e:
                self.update_display(f"BiLSTM error:\n{e}")
                break

    def _bilstm_inference_step(self):
        start_time = time.time()

        _, win = record_emg(
            self.board_shim,
            SAMPLES_PER_POINT,
            NUM_CHANNELS,
            SAMPLING_RATE
        )

        # win shape: (NUM_CHANNELS, SAMPLES_PER_POINT) — take first 4 channels
        emg = np.array(win[:NUM_CHANNELS], dtype=np.float32).T  # (SAMPLES_PER_POINT, 4)

        # Per-window z-score normalization to match training StandardScaler behaviour
        mean = emg.mean(axis=0, keepdims=True)
        std = emg.std(axis=0, keepdims=True) + 1e-8
        emg = (emg - mean) / std

        xb = torch.tensor(emg, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, seq, 4)

        with torch.no_grad():
            out = self.bilstm_model(xb)
        cls = out.argmax().item()

        self.pred_history.append(cls)
        if len(self.pred_history) > 5:
            self.pred_history.pop(0)

        final_cls = max(set(self.pred_history), key=self.pred_history.count)
        latency = (time.time() - start_time) * 1000

        self.update_display(
            f"Pose: {BILSTM_STATE_DICT[final_cls]}\n"
            f"Latency: {latency:.1f} ms"
        )

    # ====================================================
    # ==== Accessibility ================================
    # ====================================================

    def _open_accessibility(self):
        win = tk.Toplevel(self.root)
        win.title("Accessibility")
        t = THEMES[self.current_theme]
        win.configure(bg=t['bg'])

        def section(label):
            tk.Label(win, text=label, bg=t['bg'], fg=t['text_fg'],
                     font=('Arial', 11, 'bold')).pack(anchor='w', padx=14, pady=(12, 2))

        # ---- Font size ----
        section("Display font size")
        font_var = tk.IntVar(value=self.title_font.cget('size'))
        size_frame = tk.Frame(win, bg=t['bg'])
        size_frame.pack(fill='x', padx=14)
        tk.Label(size_frame, text="A", bg=t['bg'], fg=t['text_fg'],
                 font=('Arial', 9)).pack(side='left')
        tk.Scale(size_frame, from_=14, to=48, orient='horizontal',
                 variable=font_var, bg=t['bg'], fg=t['text_fg'],
                 troughcolor=t['plot_bg'], highlightthickness=0,
                 command=lambda v: self._apply_font_size(int(v))
                 ).pack(side='left', fill='x', expand=True)
        tk.Label(size_frame, text="A", bg=t['bg'], fg=t['text_fg'],
                 font=('Arial', 18)).pack(side='left')

        # ---- Theme ----
        section("Color theme")
        theme_var = tk.StringVar(value=self.current_theme)
        for name in THEMES:
            tk.Radiobutton(win, text=name, variable=theme_var, value=name,
                           bg=t['bg'], fg=t['text_fg'], selectcolor=t['bg'],
                           activebackground=t['bg'], activeforeground=t['text_fg'],
                           command=lambda n=name: self._apply_theme(n)
                           ).pack(anchor='w', padx=28)

        # ---- Waveform scale ----
        section("Waveform channel spacing")
        scale_var = tk.IntVar(value=self.offset_step)
        tk.Scale(win, from_=50, to=800, orient='horizontal',
                 variable=scale_var, bg=t['bg'], fg=t['text_fg'],
                 troughcolor=t['plot_bg'], highlightthickness=0,
                 command=lambda v: self._apply_waveform_scale(int(v))
                 ).pack(fill='x', padx=14)

        # ---- RMS overlay toggle ----
        section("RMS overlay")
        self.show_rms = getattr(self, 'show_rms', tk.BooleanVar(value=True))
        tk.Checkbutton(win, text="Show RMS envelope", variable=self.show_rms,
                       bg=t['bg'], fg=t['text_fg'], selectcolor=t['bg'],
                       activebackground=t['bg'], activeforeground=t['text_fg']
                       ).pack(anchor='w', padx=28)

        tk.Button(win, text="Close", command=win.destroy,
                  bg=t['btn_bg'], fg='white').pack(pady=14)

    def _apply_font_size(self, size):
        self.title_font.configure(size=size)

    def _apply_waveform_scale(self, step):
        self.offset_step = step
        self.ax.set_ylim(-step, NUM_CHANNELS * step)

    def _apply_theme(self, name):
        self.current_theme = name
        t = THEMES[name]

        # Root + frames
        self.root.configure(bg=t['bg'])
        self.toggle_frame.configure(bg=t['bg'])
        self.btn_frame.configure(bg=t['bg'])

        # Display label
        self.display.configure(bg=t['bg'], fg=t['fg'])

        # Progress bar
        self.pb_style.configure('Themed.Horizontal.TProgressbar',
                                troughcolor=t['plot_bg'], background=t['btn_bg'])

        # Channel checkboxes
        for cb in self.channel_checkboxes:
            cb.configure(bg=t['bg'], fg=t['check_fg'], selectcolor=t['bg'],
                         activebackground=t['bg'], activeforeground=t['check_fg'])

        # Action buttons
        self.start_cal_btn.configure(bg=t['btn_bg'], fg='white')
        self.bilstm_btn.configure(bg=t['bilstm_bg'], fg=t['bilstm_fg'])
        self.test_btn.configure(bg='green', fg='white')
        self.stop_btn.configure(bg=t['btn_bg'], fg='white')

        # Matplotlib plot
        self.ax.set_facecolor(t['plot_bg'])
        self.fig.patch.set_facecolor(t['plot_bg'])
        self.ax.set_title("Filtered EMG + RMS", color=t['plot_fg'])
        self.ax.tick_params(colors=t['plot_fg'])
        for line in self.lines:
            line.set_color(t['emg_color'])
        for rline in self.rms_lines:
            rline.set_color(t['rms_color'])
        self.canvas.draw_idle()

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