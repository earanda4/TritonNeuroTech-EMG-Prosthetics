import threading
import time
import os
import numpy as np
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

# ─────────────────────────── CONSTANTS ────────────────────────────────────────
NUM_SAMPLES        = 5000
NUM_CHANNELS       = 8
SAMPLES_PER_POINT  = 50
BATCH_SIZE         = 1
STATE_DICT         = {0: "Relax", 1: "Clench", 2: "Triton", 3: "L(ove)", 4: "Surfer"}
SAMPLING_RATE      = 500
DIMENSIONS         = 10000
GENERAL_MODEL_PATH = "general_model.pt"
CALIB_DIR          = "calibration_data"

os.makedirs(CALIB_DIR, exist_ok=True)
for cls_name in STATE_DICT.values():
    os.makedirs(os.path.join(CALIB_DIR, cls_name), exist_ok=True)


# ─────────────────────────── HELPERS ──────────────────────────────────────────
def load_calibration_dataset(calib_dir=CALIB_DIR):
    data_list, labels = [], []
    for cls, name in STATE_DICT.items():
        cls_dir = os.path.join(calib_dir, name)
        if not os.path.exists(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if fn.endswith('.npy'):
                arr = np.load(os.path.join(cls_dir, fn))
                data_list.append(arr)
                labels.append(cls)
    if data_list:
        X = np.stack(data_list, axis=0)
        Y = np.array(labels, dtype=int)
        return EMGDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(Y, dtype=torch.long))
    empty_X = np.zeros((0, SAMPLES_PER_POINT, NUM_CHANNELS), dtype=np.float32)
    empty_Y = np.zeros((0,), dtype=int)
    return EMGDataset(torch.tensor(empty_X, dtype=torch.float32),
                      torch.tensor(empty_Y, dtype=torch.long))


# ─────────────────────────── GUI CLASS ────────────────────────────────────────
class VaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vader Hand Control")
        try:
            self.root.state('zoomed')
            self.root.attributes('-zoomed', True)
        except Exception:
            pass
        self.root.configure(bg='black')

        torch.manual_seed(42)
        np.random.seed(42)

        # stop_event is SET only by stop_process / errors.
        # It is CLEARED only at the top of each start_* method.
        # reset_ui() intentionally does NOT touch it.
        self.stop_event           = threading.Event()
        self.continue_event       = threading.Event()
        self.hardware_initialized = False
        self.board_shim           = None
        self.pred_history         = []
        self.current_gesture      = "—"
        self.current_rotation     = "None"

        self.device = torch.device(
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()           else "cpu"
        )

        self.model         = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
        self.general_model = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
        self._load_general_model()

        # Fonts
        self.title_font   = font.Font(family="Helvetica", size=28, weight='bold')
        self.gesture_font = font.Font(family="Helvetica", size=52, weight='bold')
        self.label_font   = font.Font(family="Helvetica", size=13)
        self.button_font  = font.Font(family="Helvetica", size=14)

        # Signal-processing filters
        nyq = SAMPLING_RATE / 2
        self.bp_b,    self.bp_a    = signal.butter(4, [20 / nyq, 200 / nyq], btype='band')
        self.notch_b, self.notch_a = signal.iirnotch(60 / nyq, Q=30)
        self.zi_bp    = [signal.lfilter_zi(self.bp_b,    self.bp_a)    for _ in range(NUM_CHANNELS)]
        self.zi_notch = [signal.lfilter_zi(self.notch_b, self.notch_a) for _ in range(NUM_CHANNELS)]

        self.offset_step     = 300
        self.rms_window      = 100
        self.var_threshold   = 5.0
        self.display_buffer  = np.zeros((NUM_CHANNELS, 500))
        self.channel_enabled = [tk.BooleanVar(value=True) for _ in range(NUM_CHANNELS)]

        self._build_layout()

    # ══════════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ══════════════════════════════════════════════════════════════════════════
    def _build_layout(self):
        # Header
        header = tk.Frame(self.root, bg='black')
        header.pack(fill='x', padx=20, pady=(14, 4))

        tk.Label(header, text="VADER HAND CONTROL",
                 fg='red', bg='black', font=self.title_font).pack(side='left')

        style = ttk.Style()
        style.theme_use('classic')
        style.configure('Red.Horizontal.TProgressbar', troughcolor='#111', background='red')
        self.progress = ttk.Progressbar(
            header, style='Red.Horizontal.TProgressbar',
            orient='horizontal', mode='determinate', length=320
        )
        self.progress.pack(side='right', padx=10, ipady=3)
        tk.Label(header, text="Progress:", fg='#888', bg='black',
                 font=self.label_font).pack(side='right')

        # Split pane
        pane = tk.PanedWindow(self.root, orient='horizontal',
                              bg='#1a1a1a', sashwidth=5, sashrelief='flat')
        pane.pack(fill='both', expand=True, padx=8, pady=4)

        left  = tk.Frame(pane, bg='black')
        right = tk.Frame(pane, bg='#0d0d0d')
        pane.add(left,  stretch='always')
        pane.add(right, stretch='always')

        self._build_signal_panel(left)
        self._build_params_panel(right)

        # Footer buttons
        footer = tk.Frame(self.root, bg='black')
        footer.pack(fill='x', padx=10, pady=(4, 12))
        self._build_buttons(footer)

        self.root.update_idletasks()
        pane.sash_place(0, self.root.winfo_width() // 2, 0)

    def _build_signal_panel(self, parent):
        gesture_bar = tk.Frame(parent, bg='#0a0a0a', pady=6)
        gesture_bar.pack(fill='x')

        tk.Label(gesture_bar, text="DETECTED GESTURE",
                 fg='#555', bg='#0a0a0a',
                 font=font.Font(family='Helvetica', size=10, weight='bold')).pack()

        self.gesture_label = tk.Label(
            gesture_bar, text="—",
            fg='#ff3333', bg='#0a0a0a', font=self.gesture_font
        )
        self.gesture_label.pack()

        self.rotation_label = tk.Label(
            gesture_bar, text="Rotation: None",
            fg='#aaaaaa', bg='#0a0a0a',
            font=font.Font(family='Helvetica', size=13)
        )
        self.rotation_label.pack()

        # Matplotlib waveform
        self.fig = Figure(figsize=(7, 5), dpi=95)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(-self.offset_step, NUM_CHANNELS * self.offset_step)
        self.ax.set_title("Filtered EMG · RMS Envelope", color='#888', fontsize=10)
        self.ax.tick_params(colors='#555')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#333')

        self.lines     = []
        self.rms_lines = []
        for _ in range(NUM_CHANNELS):
            ln,    = self.ax.plot([], [], linewidth=0.9, color='cyan')
            rmsln, = self.ax.plot([], [], linewidth=1.8, color='#ffdd00', alpha=0.7)
            self.lines.append(ln)
            self.rms_lines.append(rmsln)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toggle_frame = tk.Frame(parent, bg='black')
        toggle_frame.pack(pady=4)
        for ch in range(NUM_CHANNELS):
            tk.Checkbutton(
                toggle_frame, text=f"Ch {ch + 1}",
                variable=self.channel_enabled[ch],
                fg='white', bg='black', selectcolor='black',
                font=font.Font(size=10)
            ).pack(side='left', padx=2)

    def _build_params_panel(self, parent):
        tk.Label(parent, text="PARAMETERS & STATUS",
                 fg='#555', bg='#0d0d0d',
                 font=font.Font(family='Helvetica', size=10, weight='bold')
                 ).pack(pady=(10, 2))

        self.display = tk.Label(
            parent,
            text="Welcome to Vader!",
            fg='red', bg='#0d0d0d',
            font=font.Font(family='Helvetica', size=22, weight='bold'),
            justify='center', wraplength=500
        )
        self.display.pack(expand=True, fill='both', pady=10, padx=10)

        param_grid = tk.Frame(parent, bg='#0d0d0d')
        param_grid.pack(fill='x', padx=16, pady=6)

        params = [
            ("Sampling Rate",   f"{SAMPLING_RATE} Hz"),
            ("Channels",        str(NUM_CHANNELS)),
            ("HD Dimensions",   f"{DIMENSIONS:,}"),
            ("Window Size",     f"{SAMPLES_PER_POINT} samples"),
            ("Bandpass Filter", "20–200 Hz"),
            ("Notch Filter",    "60 Hz"),
            ("Classes",         ", ".join(STATE_DICT.values())),
            ("Device",          str(self.device)),
        ]
        for i, (name, val) in enumerate(params):
            tk.Label(param_grid, text=name + ":",
                     fg='#666', bg='#0d0d0d',
                     font=font.Font(family='Helvetica', size=11),
                     anchor='e', width=18
                     ).grid(row=i, column=0, sticky='e', pady=2, padx=(0, 6))
            tk.Label(param_grid, text=val,
                     fg='#cccccc', bg='#0d0d0d',
                     font=font.Font(family='Helvetica', size=11, weight='bold'),
                     anchor='w'
                     ).grid(row=i, column=1, sticky='w', pady=2)

        self.continue_btn = tk.Button(
            parent, text="Continue →",
            fg='black', bg='red', font=self.button_font,
            width=16, state='disabled', command=self._on_continue
        )
        self.continue_btn.pack(pady=8)

    def _build_buttons(self, parent):
        defs = [
            ("Start Calibration",  self.start_process,             'start_btn'),
            ("Use General Model",  self.start_general_inference,   'general_btn'),
            ("Aggregate",          self.start_aggregate,           'aggregate_btn'),
            ("Infer Aggregate",    self.start_aggregate_inference, 'aggregate_infer_btn'),
            ("Stop",               self.stop_process,              'stop_btn'),
        ]
        for label, cmd, attr in defs:
            btn = tk.Button(parent, text=label, command=cmd,
                            fg='black', bg='red', font=self.button_font, width=16,
                            state='disabled' if attr == 'stop_btn' else 'normal')
            btn.pack(side='left', expand=True, padx=4)
            setattr(self, attr, btn)

    # ══════════════════════════════════════════════════════════════════════════
    # THREAD-SAFE UI HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def update_display(self, text):
        self.root.after(0, lambda: self.display.config(text=text))

    def update_progress(self, val):
        self.root.after(0, lambda: self.progress.config(value=val))

    def _on_continue(self):
        self.continue_btn.pack_forget()
        self.continue_event.set()

    def _set_gesture(self, gesture: str, rotation: str = "None"):
        self.root.after(0, lambda: (
            self.gesture_label.config(text=gesture),
            self.rotation_label.config(text=f"Rotation: {rotation}")
        ))

    def _set_buttons_running(self):
        self.root.after(0, lambda: (
            self.start_btn.config(state='disabled'),
            self.general_btn.config(state='disabled'),
            self.aggregate_btn.config(state='disabled'),
            self.aggregate_infer_btn.config(state='disabled'),
            self.stop_btn.config(state='normal'),
        ))

    def reset_ui(self):
        """Restore buttons & status. Does NOT touch stop_event."""
        self.root.after(0, lambda: (
            self.start_btn.config(state='normal'),
            self.general_btn.config(state='normal'),
            self.aggregate_btn.config(state='normal'),
            self.aggregate_infer_btn.config(state='normal'),
            self.stop_btn.config(state='disabled'),
            self.display.config(text="Stopped." if self.stop_event.is_set()
                                else "Welcome to Vader!"),
            self.progress.config(value=0),
        ))
        self._set_gesture("—")

    # ══════════════════════════════════════════════════════════════════════════
    # HARDWARE  ── Real MindRove ARB (Wi-Fi)
    # ══════════════════════════════════════════════════════════════════════════
    def init_hardware(self):
        if self.hardware_initialized:
            return

        self.update_display("Connecting to MindRove…")

        board_id = BoardIds.MINDROVE_WIFI_BOARD   # ARB uses Wi-Fi transport
        params   = MindRoveInputParams()

        self.board_shim = BoardShim(board_id, params)
        self.board_shim.prepare_session()
        self.board_shim.start_stream(450000)

        # Wait up to 10 s for first samples, interruptible by stop
        self.update_display("Waiting for data stream…")
        deadline = time.time() + 10
        while time.time() < deadline and not self.stop_event.is_set():
            if self.board_shim.get_board_data_count() > 0:
                # Flush stale startup samples
                self.board_shim.get_board_data(self.board_shim.get_board_data_count())
                break
            time.sleep(0.1)

        if self.stop_event.is_set():
            self._release_hardware()
            return

        self.hardware_initialized = True
        self.update_display("MindRove connected ✓")
        self.start_live_plot()

    def _release_hardware(self):
        if self.board_shim:
            try:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
            except Exception:
                pass
            self.board_shim = None
        self.hardware_initialized = False

    def _ensure_hardware_then(self, fn):
        self.init_hardware()
        if not self.stop_event.is_set():
            fn()
        # Thread is finishing — clean up and restore UI
        self._release_hardware()
        self.reset_ui()

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE WAVEFORM
    # ══════════════════════════════════════════════════════════════════════════
    def start_live_plot(self):
        threading.Thread(target=self._live_plot_loop, daemon=True).start()

    def _live_plot_loop(self):
        while not self.stop_event.is_set():
            if not self.hardware_initialized or self.board_shim is None:
                time.sleep(0.1)
                continue

            if self.board_shim.get_board_data_count() < SAMPLES_PER_POINT:
                time.sleep(0.01)
                continue

            data = self.board_shim.get_board_data(SAMPLES_PER_POINT)
            emg  = np.array(data[:NUM_CHANNELS], dtype=float)

            for ch in range(NUM_CHANNELS):
                emg[ch], self.zi_bp[ch]    = signal.lfilter(
                    self.bp_b,    self.bp_a,    emg[ch], zi=self.zi_bp[ch])
                emg[ch], self.zi_notch[ch] = signal.lfilter(
                    self.notch_b, self.notch_a, emg[ch], zi=self.zi_notch[ch])

            self.display_buffer = np.roll(self.display_buffer, -SAMPLES_PER_POINT, axis=1)
            self.display_buffer[:, -SAMPLES_PER_POINT:] = emg

            self.root.after(0, self._refresh_plot)
            time.sleep(0.02)

    def _refresh_plot(self):
        for ch in range(NUM_CHANNELS):
            sig        = self.display_buffer[ch]
            enabled    = self.channel_enabled[ch].get()
            has_signal = np.var(sig) >= self.var_threshold

            if not enabled or not has_signal:
                self.lines[ch].set_data([], [])
                self.rms_lines[ch].set_data([], [])
                continue

            offset  = ch * self.offset_step
            shifted = sig + offset
            self.lines[ch].set_data(np.arange(len(sig)), shifted)
            self.lines[ch].set_color('cyan')

            rms = np.sqrt(np.convolve(sig ** 2,
                                      np.ones(self.rms_window) / self.rms_window,
                                      mode='same'))
            self.rms_lines[ch].set_data(np.arange(len(sig)), rms + offset)

        self.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # STOP
    # ══════════════════════════════════════════════════════════════════════════
    def stop_process(self):
        """
        Signal all background threads to stop.
        Only SETS the event — never clears it.
        Each worker thread's finally block calls reset_ui() once it exits,
        which avoids the race condition where clearing the event here would
        let threads keep running.
        """
        self.stop_event.set()
        self.continue_event.set()   # unblock any continue_event.wait()
        self.update_display("Stopping…")
        self._set_gesture("—")
        # Release hardware immediately so record_emg's poll loop unblocks
        self._release_hardware()

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════
    def _load_general_model(self):
        if os.path.exists(GENERAL_MODEL_PATH):
            try:
                self.general_model.load_state_dict(
                    torch.load(GENERAL_MODEL_PATH, map_location=self.device))
            except Exception:
                pass

    def _save_general_model(self):
        torch.save(self.general_model.state_dict(), GENERAL_MODEL_PATH)

    def _save_window(self, cls_index, window):
        cls_dir  = os.path.join(CALIB_DIR, STATE_DICT[cls_index])
        existing = len([f for f in os.listdir(cls_dir) if f.endswith('.npy')])
        np.save(os.path.join(cls_dir, f"window_{existing}.npy"), window)

    # ══════════════════════════════════════════════════════════════════════════
    # INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    def _inference_step(self, model):
        num_classes = len(STATE_DICT)
        votes       = np.zeros(num_classes, dtype=int)
        rotation    = None

        for _ in range(10):
            if self.stop_event.is_set():
                return
            rot, win = record_emg(self.board_shim, SAMPLES_PER_POINT,
                                  NUM_CHANNELS, SAMPLING_RATE)
            if rotation is None and rot != "None":
                rotation = rot
            xb  = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = model(xb)
            votes[out.argmax().item()] += 1

        if self.stop_event.is_set():
            return

        cls    = int(votes.argmax())
        action = STATE_DICT[cls]
        self._set_gesture(action, rotation or "None")
        self.update_display(f"Pose: {action}\nRotation: {rotation or 'None'}")

    def _run_inference_loop(self, model):
        while not self.stop_event.is_set():
            try:
                self._inference_step(model)
            except Exception as e:
                if not self.stop_event.is_set():
                    self.update_display(f"Inference error:\n{e}")
                break

    # ══════════════════════════════════════════════════════════════════════════
    # BUTTON ACTIONS  ── each clears stop_event before launching its thread
    # ══════════════════════════════════════════════════════════════════════════
    def start_process(self):
        self.stop_event.clear()
        self._set_buttons_running()
        threading.Thread(target=self.run_sequence, daemon=True).start()

    def start_general_inference(self):
        self.stop_event.clear()
        self.pred_history.clear()
        self._set_buttons_running()
        self.update_display("Connecting to MindRove…")
        threading.Thread(
            target=self._ensure_hardware_then,
            args=(lambda: self._run_inference_loop(self.general_model),),
            daemon=True
        ).start()

    def start_aggregate(self):
        self.stop_event.clear()
        self._set_buttons_running()
        threading.Thread(target=self.run_aggregate, daemon=True).start()

    def start_aggregate_inference(self):
        if not hasattr(self, 'aggregate_model'):
            self.update_display("No aggregate model.\nRun Aggregate first.")
            return
        self.stop_event.clear()
        self._set_buttons_running()
        self.update_display("Running Aggregate Model…")
        threading.Thread(
            target=self._ensure_hardware_then,
            args=(lambda: self._run_inference_loop(self.aggregate_model),),
            daemon=True
        ).start()

    # ══════════════════════════════════════════════════════════════════════════
    # CALIBRATION SEQUENCE
    # ══════════════════════════════════════════════════════════════════════════
    def run_sequence(self):
        try:
            self.init_hardware()
            if self.stop_event.is_set():
                return

            num_classes = len(STATE_DICT)
            X, Y = [], []

            for cls_idx, cls_name in STATE_DICT.items():
                if self.stop_event.is_set():
                    break
                self.update_display(f"Get ready for:\n{cls_name}")
                # Interruptible 2-second countdown
                for _ in range(20):
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.1)

                samples_collected = 0
                target = NUM_SAMPLES // num_classes
                while samples_collected < target and not self.stop_event.is_set():
                    rot, win = record_emg(self.board_shim, SAMPLES_PER_POINT,
                                          NUM_CHANNELS, SAMPLING_RATE)
                    self._save_window(cls_idx, win)
                    X.append(win)
                    Y.append(cls_idx)
                    samples_collected += 1
                    self.update_progress(int(samples_collected / target * 100))
                    self.update_display(
                        f"Recording: {cls_name}\n{samples_collected}/{target}")

            if not X or self.stop_event.is_set():
                return

            # Build models
            full_ds   = EMGDataset(X, Y)
            train_len = int(len(full_ds) * 0.8)
            train_ds, test_ds = torch.utils.data.random_split(
                full_ds, [train_len, len(full_ds) - train_len])
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

            self.model         = HDClassifier(DIMENSIONS, num_classes, NUM_CHANNELS).to(self.device)
            self.general_model = HDClassifier(DIMENSIONS, num_classes, NUM_CHANNELS).to(self.device)
            self._load_general_model()

            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    self.model.build(xb, yb)
                    self.general_model.build(xb, yb, lr=0.1)

            self.model.normalize()
            self._save_general_model()

            # Evaluate
            results = {}
            for name, m in [("Custom", self.model), ("General", self.general_model)]:
                acc  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
                prec = torchmetrics.Precision(task="multiclass", average="none",
                                              num_classes=num_classes).to(self.device)
                rec  = torchmetrics.Recall(task="multiclass", average="none",
                                           num_classes=num_classes).to(self.device)
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        out = m(xb)
                        acc.update(out, yb)
                        prec.update(out, yb)
                        rec.update(out, yb)
                p = prec.compute().cpu().numpy() * 100
                r = rec.compute().cpu().numpy()  * 100
                results[name] = (acc.compute().item() * 100, p, r)

            lines = []
            for name, (a, p, r) in results.items():
                lines.append(f"{name}  Acc: {a:.1f}%")
                for i in range(num_classes):
                    lines.append(f"  {STATE_DICT[i]}  P:{p[i]:.1f}%  R:{r[i]:.1f}%")
                lines.append("")
            self.update_display("\n".join(lines))

            # Wait for Continue (stop_process also fires continue_event)
            self.root.after(0, lambda: (
                self.continue_btn.pack(pady=8),
                self.continue_btn.config(state='normal')
            ))
            self.continue_event.wait()
            self.continue_event.clear()

            if not self.stop_event.is_set():
                time.sleep(1)
                self.run_custom_inference()

        except Exception as e:
            if not self.stop_event.is_set():
                self.update_display(f"Error: {e}")
        finally:
            self._release_hardware()
            self.reset_ui()

    # ══════════════════════════════════════════════════════════════════════════
    # INFERENCE RUNNERS
    # ══════════════════════════════════════════════════════════════════════════
    def run_custom_inference(self, loop=True):
        try:
            if not self.hardware_initialized:
                self.init_hardware()
            if self.stop_event.is_set():
                return
            self.update_display("Custom inference running…")
            self.progress['value'] = 0
            while loop and not self.stop_event.is_set():
                self._inference_step(self.model)
        except Exception as e:
            if not self.stop_event.is_set():
                self.update_display(f"Error: {e}")
        finally:
            self._release_hardware()
            self.reset_ui()

    def run_aggregate(self):
        try:
            self.update_display("Aggregating calibration data…")
            ds     = load_calibration_dataset()
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            agg    = HDClassifier(DIMENSIONS, len(STATE_DICT), NUM_CHANNELS).to(self.device)
            with torch.no_grad():
                for X, Y in loader:
                    if self.stop_event.is_set():
                        break
                    agg.build(X.to(self.device), Y.to(self.device))
            if not self.stop_event.is_set():
                agg.normalize()
                self.aggregate_model = agg
                self.update_display(
                    f"Aggregate model ready\n({len(loader.dataset)} samples)")
                time.sleep(2)
        except Exception as e:
            if not self.stop_event.is_set():
                self.update_display(f"Error: {e}")
        finally:
            self.reset_ui()


# ─────────────────────────── ENTRY POINT ──────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    app  = VaderGUI(root)
    root.mainloop()
