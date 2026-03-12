# main.py
# SQAT – GUI for psychoacoustic metrics (with progress bar & threaded runs)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, Optional
import threading, queue

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Project modules ---
from utilities import wav2sig, see
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

METRICS = [
    "Loudness (ISO 532-1)",
    "Sharpness (DIN 45692)",
    "Roughness (Daniel 1997)",
    "Fluctuation Strength (Osses 2016)",
    "Tonality (Aures 1985)",
    "Annoyance (Di 2016)",
    "Annoyance (Zwicker 1999)",
    "Annoyance (More 2010)",
    "EPNL (FAR Part 36)",
]

DEFAULTS = {
    "Loudness (ISO 532-1)": dict(field=0, method=2, time_skip=0.5),
    "Sharpness (DIN 45692)": dict(field=0, method=2, weight_type="DIN45692", time_skip=0.5),
    "Roughness (Daniel 1997)": dict(time_skip=0.0),
    "Fluctuation Strength (Osses 2016)": dict(method_fs=1, time_skip=0.0),
    "Tonality (Aures 1985)": dict(field=0, time_skip=0.0),
    "Annoyance (Di 2016)": dict(field=0, time_skip=0.2),
    "Annoyance (Zwicker 1999)": dict(field=0, time_skip=0.2),
    "Annoyance (More 2010)": dict(field=0, time_skip=0.2),
    "EPNL (FAR Part 36)": dict(method_epnl=1, threshold_epnl=None),
}

class SQATApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SQAT – Psychoacoustic Metrics GUI")
        self.geometry("1000x740")

        # state
        self.file_path: Optional[str] = None
        self.insig = None
        self.fs: Optional[int] = None
        self.last_metric: Optional[str] = None
        self.last_out: Optional[Dict[str, Any]] = None

        # threading state
        self._worker: Optional[threading.Thread] = None
        self._result_q: "queue.Queue" = queue.Queue()

        self._build_ui()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_pick = ttk.Button(top, text="Open WAV…", command=self.on_pick_file)
        self.btn_pick.grid(row=0, column=0, padx=(0, 8))

        self.lbl_file = ttk.Label(top, text="No file selected", width=60)
        self.lbl_file.grid(row=0, column=1, sticky="w")

        ttk.Label(top, text="Metric:").grid(row=1, column=0, pady=(10, 0), sticky="e")
        self.cmb_metric = ttk.Combobox(top, values=METRICS, state="readonly", width=38)
        self.cmb_metric.set(METRICS[0])
        self.cmb_metric.grid(row=1, column=1, sticky="w", pady=(10, 0))
        self.cmb_metric.bind("<<ComboboxSelected>>", lambda e: self._refresh_params())

        self.var_show = tk.BooleanVar(value=False)
        self.chk_show = ttk.Checkbutton(top, text="Show plots after run", variable=self.var_show)
        self.chk_show.grid(row=1, column=2, padx=(20, 0), sticky="w")

        # Dynamic params
        self.params_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        self.params_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self._build_param_widgets()
        self._refresh_params()

        # Action buttons
        btns = ttk.Frame(self, padding=(10, 0))
        btns.pack(side=tk.TOP, fill=tk.X)

        self.btn_analyze = ttk.Button(btns, text="Run analysis", command=self.on_analyze)
        self.btn_analyze.pack(side=tk.LEFT)

        self.btn_plot = ttk.Button(btns, text="Open graphs window", command=self.open_graph_window, state="disabled")
        self.btn_plot.pack(side=tk.LEFT, padx=8)

        self.btn_wave = ttk.Button(btns, text="View waveform & spectrogram", command=self.on_view_waveform, state="disabled")
        self.btn_wave.pack(side=tk.LEFT, padx=8)

        # Results
        self.txt = tk.Text(self, height=20, wrap="word")
        self.txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.txt.insert("end", "Pick a WAV file, select a metric, set parameters, then click “Run analysis”.\n")

        # Footer: progress + status
        footer = ttk.Frame(self, padding=(10, 6))
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.status = ttk.Label(footer, text="Ready", anchor="e", width=18)
        self.status.pack(side=tk.RIGHT)

    def _build_param_widgets(self):
        # generic params
        self.var_time_skip = tk.DoubleVar(value=0.0)
        self.entry_time_skip = self._make_param_row("time_skip [s]:", self.var_time_skip, row=0)

        self.var_field = tk.IntVar(value=0)
        self.row_field = self._make_combo_row("Loudness field:", ["0 = free field", "1 = diffuse field"], self.var_field, row=1)

        self.var_method = tk.IntVar(value=2)
        self.row_method = self._make_combo_row("Loudness method:", ["1 = stationary", "2 = time-varying"], self.var_method, row=2)

        self.var_weight = tk.StringVar(value="DIN45692")
        self.row_weight = self._make_combo_row("Sharpness weight:", ["DIN45692", "aures", "bismarck"], self.var_weight, row=3)

        self.var_method_fs = tk.IntVar(value=1)
        self.row_method_fs = self._make_combo_row("FS method:", ["0 = stationary", "1 = time-varying"], self.var_method_fs, row=4)

        # EPNL-specific
        self.var_method_epnl = tk.IntVar(value=1)
        self.row_method_epnl = self._make_combo_row("EPNL method:", ["1 = waveform (GUI)"], self.var_method_epnl, row=5)

        self.var_threshold_epnl = tk.StringVar(value="")
        ttk.Label(self.params_frame, text="EPNL tone-threshold [PNdB] (optional):").grid(row=6, column=0, sticky="e", pady=2, padx=(0, 6))
        self.entry_threshold_epnl = ttk.Entry(self.params_frame, textvariable=self.var_threshold_epnl, width=10)
        self.entry_threshold_epnl.grid(row=6, column=1, sticky="w")

    def _make_param_row(self, label, var, row):
        ttk.Label(self.params_frame, text=label).grid(row=row, column=0, sticky="e", pady=2, padx=(0, 6))
        ent = ttk.Entry(self.params_frame, textvariable=var, width=10)
        ent.grid(row=row, column=1, sticky="w")
        return ent

    def _make_combo_row(self, label, items, var, row):
        ttk.Label(self.params_frame, text=label).grid(row=row, column=0, sticky="e", pady=2, padx=(0, 6))
        combo = ttk.Combobox(self.params_frame, values=items, state="readonly", width=28)
        combo.grid(row=row, column=1, sticky="w")
        def on_sel(_):
            idx = combo.current()
            if isinstance(var.get(), int):
                var.set(idx)
            else:
                var.set(combo.get())
        combo.current(0)
        combo.bind("<<ComboboxSelected>>", on_sel)
        return combo

    def _refresh_params(self):
        metric = self.cmb_metric.get()
        d = DEFAULTS[metric]

        for w in (self.row_field, self.row_method, self.row_weight, self.row_method_fs, self.row_method_epnl, self.entry_threshold_epnl):
            try: w.grid_remove()
            except Exception: pass

        if "time_skip" in d:
            self.var_time_skip.set(float(d["time_skip"]))
            self.entry_time_skip.grid()
        else:
            self.entry_time_skip.grid_remove()

        if metric in ("Loudness (ISO 532-1)", "Sharpness (DIN 45692)", "Tonality (Aures 1985)",
                      "Annoyance (Di 2016)", "Annoyance (Zwicker 1999)", "Annoyance (More 2010)"):
            self.var_field.set(int(d.get("field", 0))); self.row_field.grid()
        if metric in ("Loudness (ISO 532-1)", "Sharpness (DIN 45692)"):
            self.var_method.set(int(d.get("method", 2))); self.row_method.grid()
        if metric == "Sharpness (DIN 45692)":
            self.var_weight.set(d.get("weight_type", "DIN45692")); self.row_weight.grid()
        if metric == "Fluctuation Strength (Osses 2016)":
            self.var_method_fs.set(int(d.get("method_fs", 1))); self.row_method_fs.grid()
        if metric == "EPNL (FAR Part 36)":
            self.var_method_epnl.set(int(d.get("method_epnl", 1))); self.row_method_epnl.grid()
            self.entry_threshold_epnl.grid()

    # -------------------------
    # Handlers
    # -------------------------
    def on_pick_file(self):
        path = filedialog.askopenfilename(
            title="Choose a WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            insig, fs = wav2sig(path)  # 0 dBFS = 94 dB SPL
        except Exception as e:
            messagebox.showerror("Error", f"Could not read WAV:\n{e}")
            return

        self.file_path = path
        self.insig = insig
        self.fs = fs

        self.lbl_file.config(text=path)
        self.btn_wave.config(state="normal")
        self.status.config(text=f"Loaded @ {fs} Hz")

    def on_view_waveform(self):
        if not self.file_path:
            return
        try:
            see(self.file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show waveform:\n{e}")

    def _set_running(self, running: bool):
        # enable/disable controls and manage progress animation
        for w in (self.btn_analyze, self.btn_pick, self.cmb_metric, self.btn_wave):
            w.config(state=("disabled" if running else "normal"))
        # Plot button only enabled if we have results and not currently running
        self.btn_plot.config(state=("disabled" if running else ("normal" if self.last_out else "disabled")))

        if running:
            self.progress.start(8)  # ms per move; smaller = faster
            self.config(cursor="watch")
            self.status.config(text="Analyzing…")
        else:
            self.progress.stop()
            self.config(cursor="")
            self.status.config(text="Ready")

    def on_analyze(self):
        if self.insig is None or self.fs is None:
            messagebox.showinfo("Select file", "Please choose a WAV file first.")
            return

        metric = self.cmb_metric.get()
        params = {"time_skip": float(self.var_time_skip.get())}

        if metric in ("Loudness (ISO 532-1)", "Sharpness (DIN 45692)", "Tonality (Aures 1985)",
                      "Annoyance (Di 2016)", "Annoyance (Zwicker 1999)", "Annoyance (More 2010)"):
            params["field"] = int(self.var_field.get())
        if metric in ("Loudness (ISO 532-1)", "Sharpness (DIN 45692)"):
            params["method"] = int(self.var_method.get())
        if metric == "Sharpness (DIN 45692)":
            params["weight_type"] = str(self.var_weight.get())
        if metric == "Fluctuation Strength (Osses 2016)":
            params["method_fs"] = int(self.var_method_fs.get())
        if metric == "EPNL (FAR Part 36)":
            params["method_epnl"] = 1
            thr_txt = self.var_threshold_epnl.get().strip()
            params["threshold_epnl"] = float(thr_txt) if thr_txt else None

        want_plots_after = bool(self.var_show.get())

        # start worker
        self._result_q = queue.Queue()
        self._set_running(True)
        self.txt.delete("1.0", "end")
        self.txt.insert("end", f"Metric: {metric}\nFile: {self.lbl_file.cget('text')}\n\nWorking…\n")

        self._worker = threading.Thread(
            target=self._analyze_worker,
            args=(metric, params, want_plots_after),
            daemon=True,
        )
        self._worker.start()
        self.after(100, self._poll_worker)

    # -------------------------
    # Threading helpers
    # -------------------------
    def _analyze_worker(self, metric: str, params: Dict[str, Any], want_plots_after: bool):
        try:
            # Disable internal plotting while threaded (Matplotlib/Tk aren’t thread-safe).
            out = self._run_metric(metric, params, show_plots=False)
            self._result_q.put(("ok", metric, out, want_plots_after))
        except Exception as e:
            self._result_q.put(("err", str(e)))

    def _poll_worker(self):
        try:
            msg = self._result_q.get_nowait()
        except queue.Empty:
            # still running
            if self._worker and self._worker.is_alive():
                self.after(120, self._poll_worker)
                return
            # thread ended but nothing in queue (shouldn’t happen)
            self._set_running(False)
            return

        kind = msg[0]
        if kind == "ok":
            _, metric, out, want_plots_after = msg
            self.last_metric = metric
            self.last_out = out
            summary = self._summarize(metric, out)
            self.txt.delete("1.0", "end")
            self.txt.insert("end", f"Metric: {metric}\nFile: {self.lbl_file.cget('text')}\n\n{summary}\n")
            self.btn_plot.config(state="normal")
            self._set_running(False)
            self.status.config(text="Done")
            if want_plots_after:
                # open the embedded graph window post-run
                self.open_graph_window()
        else:
            _, err = msg
            self._set_running(False)
            messagebox.showerror("Error", f"Analysis failed:\n{err}")

    # -------------------------
    # Metric execution
    # -------------------------
    def _run_metric(self, metric: str, p: Dict[str, Any], show_plots: bool):
        insig, fs = self.insig, self.fs

        if metric == "Loudness (ISO 532-1)":
            return Loudness_ISO532_1(
                insig=insig, fs=fs,
                field=p["field"], method=p["method"],
                time_skip=p["time_skip"], show=show_plots
            )
        if metric == "Sharpness (DIN 45692)":
            return Sharpness_DIN45692(
                insig=insig, fs=fs,
                weight_type=p["weight_type"],
                LoudnessField=p["field"],
                LoudnessMethod=p["method"],
                time_skip=p["time_skip"],
                show_sharpness=show_plots,
                show_loudness=False
            )
        if metric == "Roughness (Daniel 1997)":
            return Roughness_Daniel1997(
                insig=insig, fs=fs,
                time_skip=p["time_skip"], show=show_plots
            )
        if metric == "Fluctuation Strength (Osses 2016)":
            return FluctuationStrength_Osses2016(
                insig=insig, fs=fs,
                method=p["method_fs"],
                time_skip=p["time_skip"], show=show_plots
            )
        if metric == "Tonality (Aures 1985)":
            return Tonality_Aures1985(
                insig=insig, fs=fs,
                LoudnessField=p["field"],
                time_skip=p["time_skip"], show=show_plots
            )
        if metric == "Annoyance (Di 2016)":
            return PsychoacousticAnnoyance_Di2016(
                insig=insig, fs=fs,
                LoudnessField=p["field"],
                time_skip=p["time_skip"], show=show_plots, showPA=show_plots
            )
        if metric == "Annoyance (Zwicker 1999)":
            return PsychoacousticAnnoyance_Zwicker1999(
                insig=insig, fs=fs,
                LoudnessField=p["field"],
                time_skip=p["time_skip"], show=show_plots, showPA=show_plots
            )
        if metric == "Annoyance (More 2010)":
            return PsychoacousticAnnoyance_More2010(
                insig=insig, fs=fs,
                LoudnessField=p["field"],
                time_skip=p["time_skip"], show=show_plots, showPA=show_plots
            )
        if metric == "EPNL (FAR Part 36)":
            return EPNL_FAR_Part36(
                insig=insig, fs=fs,
                method=1,
                threshold=p.get("threshold_epnl", None),
                show=show_plots
            )
        raise ValueError(f"Unknown metric: {metric}")

    # -------------------------
    # Result summarizer
    # -------------------------
    def _get_first_scalar(self, val):
        if isinstance(val, (list, tuple)):
            return float(val[0])
        if isinstance(val, np.ndarray):
            return float(val.ravel()[0])
        try:
            return float(val)
        except Exception:
            return val

    def _summarize(self, metric: str, out: Dict[str, Any]) -> str:
        lines = []
        if metric == "Loudness (ISO 532-1)" and "Loudness" in out:
            lines.append(f"Loudness: {self._get_first_scalar(out['Loudness']):.3f} sone")
        if metric == "Sharpness (DIN 45692)" and "Sharpness" in out:
            lines.append(f"Sharpness: {self._get_first_scalar(out['Sharpness']):.3f} acum")

        units = {"N":"sone","S":"acum","R":"asper","FS":"vacil","K":"t.u.","PA":"annoy"}
        family = {
            "Loudness (ISO 532-1)":"N","Sharpness (DIN 45692)":"S","Roughness (Daniel 1997)":"R",
            "Fluctuation Strength (Osses 2016)":"FS","Tonality (Aures 1985)":"K",
            "Annoyance (Di 2016)":"PA","Annoyance (Zwicker 1999)":"PA","Annoyance (More 2010)":"PA",
        }.get(metric, None)
        if family:
            for keylab, pretty in (("mean","mean"),("5","5th percentile"),("95","95th percentile"),("max","max"),("min","min")):
                k = f"{family}{keylab}"
                if k in out:
                    lines.append(f"{pretty.title() if keylab=='mean' else pretty}: {self._get_first_scalar(out[k]):.3f} {units[family]}")
        if "ScalarPA" in out:
            lines.append(f"Scalar PA: {self._get_first_scalar(out['ScalarPA']):.3f} annoy")
        if "PA5" in out:
            lines.append(f"PA5 (5th percentile): {self._get_first_scalar(out['PA5']):.3f} annoy")

        if metric == "EPNL (FAR Part 36)":
            if "EPNL" in out:   lines.append(f"EPNL: {self._get_first_scalar(out['EPNL']):.3f} EPNdB")
            if "PNLM" in out:   lines.append(f"PNLM (max PNL): {self._get_first_scalar(out['PNLM']):.3f} PNdB")
            if "PNLTM" in out:  lines.append(f"PNLTM (max tone-corrected): {self._get_first_scalar(out['PNLTM']):.3f} PNdB")

        if not lines:
            for tkey, ykey, label in (
                ("time","InstantaneousLoudness","Loudness"),
                ("time","InstantaneousSharpness","Sharpness"),
                ("time","InstantaneousRoughness","Roughness"),
                ("time","InstantaneousFluctuationStrength","Fluctuation strength"),
                ("time","InstantaneousTonality","Tonality"),
                ("time","InstantaneousPA","Annoyance"),
                ("time","PNL","PNL"),
                ("time","PNLT","PNLT"),
            ):
                if tkey in out and ykey in out:
                    y = np.asarray(out[ykey]).ravel()
                    lines.append(f"{label} – mean: {np.mean(y):.3f}, max: {np.max(y):.3f}")
                    break
        return "\n".join(lines) if lines else "No summary values found."

    # -------------------------
    # Embedded graphs
    # -------------------------
    def open_graph_window(self):
        if not self.last_out:
            return
        out = self.last_out
        metric = self.last_metric or ""

        if metric == "EPNL (FAR Part 36)" and "time" in out and ("PNL" in out or "PNLT" in out):
            return self._plot_epnl(out)

        y_key = None
        if "InstantaneousPA" in out: y_key = "InstantaneousPA"
        elif metric == "Loudness (ISO 532-1)" and "InstantaneousLoudness" in out: y_key = "InstantaneousLoudness"
        elif metric == "Sharpness (DIN 45692)" and "InstantaneousSharpness" in out: y_key = "InstantaneousSharpness"
        elif metric == "Roughness (Daniel 1997)" and "InstantaneousRoughness" in out: y_key = "InstantaneousRoughness"
        elif metric == "Fluctuation Strength (Osses 2016)" and "InstantaneousFluctuationStrength" in out: y_key = "InstantaneousFluctuationStrength"
        elif metric == "Tonality (Aures 1985)" and "InstantaneousTonality" in out: y_key = "InstantaneousTonality"

        if not y_key or "time" not in out:
            messagebox.showinfo("No plot", "This metric didn’t return a time-varying series to plot.")
            return

        t = np.asarray(out["time"]).ravel()
        y = np.asarray(out[y_key]).ravel()

        win = tk.Toplevel(self)
        win.title(f"Graphs – {metric}")
        win.geometry("900x480")

        fig = Figure(figsize=(8.6, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(t, y)
        ax.set_xlabel("Time [s]")

        ylabels = {
            "InstantaneousLoudness": "Loudness, N [sone]",
            "InstantaneousSharpness": "Sharpness, S [acum]",
            "InstantaneousRoughness": "Roughness, R [asper]",
            "InstantaneousFluctuationStrength": "Fluctuation strength, FS [vacil]",
            "InstantaneousTonality": "Tonality, K [t.u.]",
            "InstantaneousPA": "Annoyance, PA [annoy]",
        }
        ax.set_ylabel(ylabels.get(y_key, y_key))

        pref_map = {
            "Loudness (ISO 532-1)": "N5",
            "Sharpness (DIN 45692)": "S5",
            "Roughness (Daniel 1997)": "R5",
            "Fluctuation Strength (Osses 2016)": "FS5",
            "Tonality (Aures 1985)": "K5",
            "Annoyance (Di 2016)": "PA5",
            "Annoyance (Zwicker 1999)": "PA5",
            "Annoyance (More 2010)": "PA5",
        }
        p5key = pref_map.get(metric, None)
        if p5key and p5key in out and len(t) >= 2:
            p5 = self._get_first_scalar(out[p5key])
            ax.plot([t[0], t[-1]], [p5, p5], linestyle="--")

        ax.grid(True, linestyle=":", alpha=0.6)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_epnl(self, out: Dict[str, Any]):
        t = np.asarray(out["time"]).ravel()
        y_pnl = np.asarray(out.get("PNL", [])).ravel() if "PNL" in out else None
        y_pnlt = np.asarray(out.get("PNLT", [])).ravel() if "PNLT" in out else None

        win = tk.Toplevel(self)
        win.title("Graphs – EPNL (FAR Part 36)")
        win.geometry("900x480")

        fig = Figure(figsize=(8.6, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        if y_pnl is not None and y_pnl.size == t.size:
            ax.plot(t, y_pnl, label="PNL (PNdB)")
        if y_pnlt is not None and y_pnlt.size == t.size:
            ax.plot(t, y_pnlt, label="PNLT (PNdB)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Level [PNdB]")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = SQATApp()
    app.mainloop()
