"""psychobox_style_ui.py

A Tkinter-based user interface inspired by the PsychoBox layout. This is
an independent, user-friendly UI with a sidebar, top header, metrics
selection, file open, run analysis (threaded placeholder), results
pane, progress bar and status region.

Place this file in the `SQAT4PY` folder and run it with your project's
Python environment. It's intentionally lightweight and does not include
recording features.
"""
import os
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# Project metric functions
from metrics_loudness import Loudness_ISO532_1, EPNL_FAR_Part36
from metrics_sharpness import Sharpness_DIN45692
from metrics_roughness import Roughness_Daniel1997
from metrics_fluctuation import FluctuationStrength_Osses2016
from metrics_tonality import Tonality_Aures1985
from metrics_annoyance import (
    PsychoacousticAnnoyance_Di2016,
    PsychoacousticAnnoyance_Zwicker1999,
    PsychoacousticAnnoyance_More2010,
)
from utilities import see

METRICS = [
    "Loudness (ISO 532-1)",
    "Sharpness (DIN 45692)",
    "Roughness (Daniel 1997)",
    "Fluctuation Strength (Osses 2016)",
    "Tonality (Aures 1985)",
    "Annoyance (Di 2016)",
    "EPNL (FAR Part 36)",
]


class PsychoBoxStyleUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Psychobox-like UI — SQAT")
        self.geometry("1100x720")

        # state
        self.file_path = None
        self.insig = None
        self.fs = None
        self.last_out = None

        # threading
        self._worker = None
        self._result_q = queue.Queue()

        self._build_ui()

    def _build_ui(self):
        # top header
        header = ttk.Frame(self, padding=10)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="SQAT — Psychobox-style Interface", font=(None, 14, "bold")).pack(side=tk.LEFT)

        # main area with left sidebar and content
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # sidebar
        sidebar = ttk.Frame(main, width=200, padding=(8, 8))
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="Navigation", font=(None, 10, "bold")).pack(anchor="w", pady=(0, 6))
        for name in ("Home", "Metrics", "Graphs", "Settings"):
            b = ttk.Button(sidebar, text=name)
            b.pack(fill=tk.X, pady=4)

        # content area
        content = ttk.Frame(main, padding=10)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # File + metric row
        row = ttk.Frame(content)
        row.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        self.btn_open = ttk.Button(row, text="Open WAV…", command=self.on_open)
        self.btn_open.pack(side=tk.LEFT)

        self.lbl_file = ttk.Label(row, text="No file selected", width=60)
        self.lbl_file.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(row, text="Metric:").pack(side=tk.LEFT, padx=(16, 4))
        self.cmb_metric = ttk.Combobox(row, values=METRICS, state="readonly", width=36)
        self.cmb_metric.set(METRICS[0])
        self.cmb_metric.pack(side=tk.LEFT)

        # parameters and actions
        params_frame = ttk.LabelFrame(content, text="Parameters", padding=8)
        params_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 10))

        ttk.Label(params_frame, text="time_skip [s]:").grid(row=0, column=0, sticky="e")
        self.var_time_skip = tk.DoubleVar(value=0.5)
        ttk.Entry(params_frame, textvariable=self.var_time_skip, width=10).grid(row=0, column=1, sticky="w", padx=(6, 16))

        ttk.Label(params_frame, text="Field:").grid(row=0, column=2, sticky="e")
        self.var_field = tk.IntVar(value=0)
        ttk.Combobox(params_frame, values=["0 = free field", "1 = diffuse field"], state="readonly", width=22, textvariable=self.var_field).grid(row=0, column=3, sticky="w", padx=(6, 0))

        actions = ttk.Frame(content)
        actions.pack(side=tk.TOP, fill=tk.X)
        self.var_show = tk.BooleanVar(value=False)
        ttk.Checkbutton(actions, text="Show plots after run", variable=self.var_show).pack(side=tk.LEFT)

        self.btn_run = ttk.Button(actions, text="Run analysis", command=self.on_run)
        self.btn_run.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_view_wave = ttk.Button(actions, text="View waveform", command=self.on_view_waveform, state="disabled")
        self.btn_view_wave.pack(side=tk.LEFT, padx=8)

        # results area
        self.txt = tk.Text(content, height=18, wrap="word")
        self.txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.txt.insert("end", "Welcome — open a WAV file, select a metric and run the analysis.\n")

        # footer status
        footer = ttk.Frame(self, padding=(8, 6))
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.status = ttk.Label(footer, text="Ready", width=20, anchor="e")
        self.status.pack(side=tk.RIGHT)

    # Handlers
    def on_open(self):
        path = filedialog.askopenfilename(title="Choose a WAV file", filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        self.file_path = path
        self.lbl_file.config(text=os.path.basename(path))
        self.btn_view_wave.config(state="normal")
        self.status.config(text=f"Loaded: {os.path.basename(path)}")

    def on_view_waveform(self):
        if not self.file_path:
            return
        # Lightweight placeholder — user can replace with real plot function
        messagebox.showinfo("Waveform", f"Would show waveform for:\n{self.file_path}")

    def _set_running(self, running: bool):
        for w in (self.btn_run, self.btn_open, self.cmb_metric, self.btn_view_wave):
            w.config(state=("disabled" if running else "normal"))
        if running:
            self.progress.start(10)
            self.status.config(text="Analyzing…")
            self.config(cursor="watch")
        else:
            self.progress.stop()
            self.status.config(text="Ready")
            self.config(cursor="")

    def on_run(self):
        if not self.file_path:
            messagebox.showinfo("Select file", "Please choose a WAV file first.")
            return

        metric = self.cmb_metric.get()
        params = {"time_skip": float(self.var_time_skip.get()), "field": int(self.var_field.get())}
        want_plots = bool(self.var_show.get())

        self._result_q = queue.Queue()
        self._set_running(True)
        self.txt.delete("1.0", "end")
        self.txt.insert("end", f"Metric: {metric}\nFile: {self.file_path}\n\nWorking…\n")

        self._worker = threading.Thread(target=self._worker_func, args=(metric, params, want_plots), daemon=True)
        self._worker.start()
        self.after(100, self._poll_worker)

    def _worker_func(self, metric, params, want_plots):
        try:
            # Call the real metric functions from the SQAT package.
            out = self._run_metric(metric, params, want_plots)
            # Provide a short summary field if missing
            if isinstance(out, dict) and "summary" not in out:
                out = dict(out)
                out["summary"] = f"Completed {metric}"
            self._result_q.put(("ok", out))
        except Exception as e:
            self._result_q.put(("err", str(e)))

    def _run_metric(self, metric, p, show_plots: bool):
        """Map the selected metric to the appropriate function call.

        The metric functions accept either a filename or (insig, fs).
        We pass the selected file path and minimal parameters. This
        mirrors the behaviour in `main_UI_updated.py`.
        """
        if not self.file_path:
            raise RuntimeError("No input file selected")

        # Common params
        time_skip = float(p.get("time_skip", 0.0))
        field = int(p.get("field", 0))

        if metric == "Loudness (ISO 532-1)":
            return Loudness_ISO532_1(self.file_path, fs=None, field=field, method=2, time_skip=time_skip, show=show_plots)

        if metric == "Sharpness (DIN 45692)":
            return Sharpness_DIN45692(self.file_path, fs=None, weight_type="DIN45692", LoudnessField=field, LoudnessMethod=2, time_skip=time_skip, show_sharpness=show_plots, show_loudness=False)

        if metric == "Roughness (Daniel 1997)":
            return Roughness_Daniel1997(self.file_path, fs=None, time_skip=time_skip, show=show_plots)

        if metric == "Fluctuation Strength (Osses 2016)":
            # default to time-varying method (1)
            return FluctuationStrength_Osses2016(self.file_path, fs=None, method=1, time_skip=time_skip, show=show_plots)

        if metric == "Tonality (Aures 1985)":
            return Tonality_Aures1985(self.file_path, fs=None, field=field, time_skip=time_skip, show=show_plots)

        if metric == "Annoyance (Di 2016)":
            return PsychoacousticAnnoyance_Di2016(self.file_path, fs=None, field=field, time_skip=time_skip, show=show_plots)

        if metric == "Annoyance (Zwicker 1999)":
            return PsychoacousticAnnoyance_Zwicker1999(self.file_path, fs=None, field=field, time_skip=time_skip, show=show_plots)

        if metric == "Annoyance (More 2010)":
            return PsychoacousticAnnoyance_More2010(self.file_path, fs=None, field=field, time_skip=time_skip, show=show_plots)

        if metric == "EPNL (FAR Part 36)":
            return EPNL_FAR_Part36(self.file_path, fs=None, method_epnl=1, threshold_epnl=None)

        # Unknown metric
        raise ValueError(f"Unknown metric: {metric}")

    def _poll_worker(self):
        try:
            msg = self._result_q.get_nowait()
        except queue.Empty:
            if self._worker and self._worker.is_alive():
                self.after(100, self._poll_worker)
                return
            else:
                # no message but thread finished — treat as error
                self._set_running(False)
                return

        kind, payload = msg
        self._set_running(False)
        if kind == "ok":
            out = payload
            self.last_out = out
            self.txt.insert("end", "\nDone.\n\n")
            self.txt.insert("end", out.get("summary", "No summary"))
            self.btn_view_wave.config(state="normal")
        else:
            self.txt.insert("end", f"\nError: {payload}\n")


def run_app():
    app = PsychoBoxStyleUI()
    app.mainloop()


if __name__ == "__main__":
    run_app()
