# main.py
# SQAT – GUI for psychoacoustic metrics (with progress bar & threaded runs)
# Updated: Multi-file support with figure saving + full plots

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, Optional, List
import threading, queue
import os
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

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
from plotting_metrics import create_metric_plot, create_metric_plot_split


METRICS = [
    "Loudness (ISO 532-1)",
    "Loudness (ECMA 418-2)",
    "Sharpness (DIN 45692)",
    "Roughness (Daniel 1997)",
    "Roughness (ECMA 418-2)",
    "Fluctuation Strength (Osses 2016)",
    "Tonality (Aures 1985)",
    "Tonality (ECMA 418-2)",
    "Annoyance (Di 2016)",
    "Annoyance (Zwicker 1999)",
    "Annoyance (More 2010)",
    "EPNL (FAR Part 36)",
]

DEFAULTS = {
    "Loudness (ISO 532-1)": dict(field=0, method=2, time_skip=0.5),
    "Loudness (ECMA 418-2)": dict(field=0, method=2, time_skip=0.5),
    "Sharpness (DIN 45692)": dict(field=0, method=2, weight_type="DIN45692", time_skip=0.5),
    "Roughness (Daniel 1997)": dict(time_skip=0.0),
    "Roughness (ECMA 418-2)": dict(time_skip=0.5),
    "Fluctuation Strength (Osses 2016)": dict(method_fs=1, time_skip=0.0),
    "Tonality (Aures 1985)": dict(field=0, time_skip=0.0),
    "Tonality (ECMA 418-2)": dict(field=0, time_skip=0.5),
    "Annoyance (Di 2016)": dict(field=0, time_skip=0.2),
    "Annoyance (Zwicker 1999)": dict(field=0, time_skip=0.2),
    "Annoyance (More 2010)": dict(field=0, time_skip=0.2),
    "EPNL (FAR Part 36)": dict(method_epnl=1, threshold_epnl=None),
}



PARAM_SPECS = {
    "time_skip": {
        "label": "time_skip [s]:",
        "kind": "entry",
        "width": 12,
    },
    "field": {
        "label": "Sound field:",
        "kind": "combo",
        "options": [
            ("0 = free field", 0),
            ("1 = diffuse field", 1),
        ],
        "width": 28,
    },
    "method": {
        "label": "Method:",
        "kind": "combo",
        "options": [
            ("1 = stationary", 1),
            ("2 = time-varying", 2),
        ],
        "width": 28,
    },
    "weight_type": {
        "label": "Sharpness weight:",
        "kind": "combo",
        "options": [
            ("DIN45692", "DIN45692"),
            ("aures", "aures"),
            ("bismarck", "bismarck"),
        ],
        "width": 28,
    },
    "method_fs": {
        "label": "FS method:",
        "kind": "combo",
        "options": [
            ("0 = stationary", 0),
            ("1 = time-varying", 1),
        ],
        "width": 28,
    },
    "threshold_epnl": {
        "label": "EPNL tone-threshold [PNdB] (optional):",
        "kind": "entry",
        "width": 12,
    },
}

METRIC_PARAM_LAYOUT = {
    "Loudness (ISO 532-1)": ["field", "method", "time_skip"],
    "Loudness (ECMA 418-2)": ["field", "method", "time_skip"],
    "Sharpness (DIN 45692)": ["field", "method", "weight_type", "time_skip"],
    "Roughness (Daniel 1997)": ["time_skip"],
    "Roughness (ECMA 418-2)": ["field", "method", "time_skip"],
    "Fluctuation Strength (Osses 2016)": ["method_fs", "time_skip"],
    "Tonality (Aures 1985)": ["field", "time_skip"],
    "Tonality (ECMA 418-2)": ["field", "method", "time_skip"],
    "Annoyance (Di 2016)": ["field", "time_skip"],
    "Annoyance (Zwicker 1999)": ["field", "time_skip"],
    "Annoyance (More 2010)": ["field", "time_skip"],
    "EPNL (FAR Part 36)": ["threshold_epnl"],
}

class SQATApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SQAT – Psychoacoustic Metrics GUI")
        self.geometry("1180x820")

        # state
        self.file_paths: List[str] = []
        self.file_label_map: Dict[str, str] = {}
        self.label_to_file: Dict[str, str] = {}
        self.current_file_index: int = 0
        self.insig = None
        self.fs: Optional[int] = None
        self.last_metric: Optional[str] = None
        self.last_out: Optional[Dict[str, Any]] = None
        self.save_dir: Optional[str] = None
        self.excel_dir: Optional[str] = None

        self.metric_params: Dict[str, Dict[str, Any]] = {
            metric: dict(DEFAULTS.get(metric, {})) for metric in METRICS
        }
        self.current_config_metric: str = METRICS[0]
        self.metric_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # threading state
        self._worker: Optional[threading.Thread] = None
        self._result_q: "queue.Queue" = queue.Queue()
        self._analysis_finished = False
        self._total_steps = 1
        self._completed_steps = 0

        self._build_ui()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_pick = ttk.Button(top, text="Open WAV files…", command=self.on_pick_files)
        self.btn_pick.grid(row=0, column=0, padx=(0, 8), sticky="w")

        self.lbl_file = ttk.Label(top, text="No files selected", width=42)
        self.lbl_file.grid(row=0, column=1, sticky="w")

        ttk.Label(top, text="Active file:").grid(row=0, column=2, padx=(16, 6), sticky="e")
        self.cmb_active_file = ttk.Combobox(top, state="disabled", width=42)
        self.cmb_active_file.grid(row=0, column=3, sticky="w")
        self.cmb_active_file.bind("<<ComboboxSelected>>", self._on_active_file_changed)

        ttk.Label(top, text="Plot metric:").grid(row=1, column=0, pady=(10, 0), sticky="e")
        self.cmb_plot_metric = ttk.Combobox(top, state="disabled", width=38)
        self.cmb_plot_metric.grid(row=1, column=1, sticky="w", pady=(10, 0))
        self.cmb_plot_metric.bind("<<ComboboxSelected>>", lambda e: self._refresh_plot_target())

        self.var_show = tk.BooleanVar(value=False)
        self.chk_show = ttk.Checkbutton(top, text="Show plots after run", variable=self.var_show)
        self.chk_show.grid(row=1, column=2, padx=(16, 0), sticky="w", pady=(10, 0))

        self.var_save = tk.BooleanVar(value=False)
        self.chk_save = ttk.Checkbutton(top, text="Save figures", variable=self.var_save, command=self.on_save_toggle)
        self.chk_save.grid(row=1, column=3, padx=(16, 0), sticky="w", pady=(10, 0))

        self.var_excel = tk.BooleanVar(value=False)
        self.chk_excel = ttk.Checkbutton(top, text="Export to Excel", variable=self.var_excel, command=self.on_excel_toggle)
        self.chk_excel.grid(row=1, column=4, padx=(16, 0), sticky="w", pady=(10, 0))

        self.var_split = tk.BooleanVar(value=False)
        self.chk_split = ttk.Checkbutton(top, text="Split figures", variable=self.var_split)
        self.chk_split.grid(row=1, column=5, padx=(16, 0), sticky="w", pady=(10, 0))

        select_frame = ttk.LabelFrame(self, text="Metrics to analyze", padding=10)
        select_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        self.lst_metrics = tk.Listbox(select_frame, selectmode=tk.EXTENDED, exportselection=False, height=7)
        self.lst_metrics.grid(row=0, column=0, rowspan=2, sticky="nsew")
        scroll = ttk.Scrollbar(select_frame, orient="vertical", command=self.lst_metrics.yview)
        scroll.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(6, 10))
        self.lst_metrics.configure(yscrollcommand=scroll.set)
        for metric in METRICS:
            self.lst_metrics.insert(tk.END, metric)
        self.lst_metrics.selection_set(0)
        self.lst_metrics.bind("<<ListboxSelect>>", self._on_metrics_selection_changed)

        ttk.Label(
            select_frame,
            text="Select one or more metrics on the left, then edit the parameters of the metric chosen below.",
        ).grid(row=0, column=2, sticky="w")

        editor = ttk.Frame(select_frame)
        editor.grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Label(editor, text="Edit parameters for:").pack(side=tk.LEFT)
        self.cmb_config_metric = ttk.Combobox(editor, state="readonly", width=38)
        self.cmb_config_metric.pack(side=tk.LEFT, padx=(8, 0))
        self.cmb_config_metric.bind("<<ComboboxSelected>>", self._on_config_metric_changed)

        select_frame.columnconfigure(0, weight=1)
        select_frame.columnconfigure(2, weight=1)

        self.params_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        self.params_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))
        self._build_param_widgets()
        self._sync_config_metric_choices(preferred=self.current_config_metric)
        self._refresh_params()

        btns = ttk.Frame(self, padding=(10, 0))
        btns.pack(side=tk.TOP, fill=tk.X)

        self.btn_analyze = ttk.Button(btns, text="Run analysis", command=self.on_analyze)
        self.btn_analyze.pack(side=tk.LEFT)

        self.btn_plot = ttk.Button(btns, text="Open graphs window", command=self.open_graph_window, state="disabled")
        self.btn_plot.pack(side=tk.LEFT, padx=8)

        self.btn_wave = ttk.Button(btns, text="View waveform & spectrogram", command=self.on_view_waveform, state="disabled")
        self.btn_wave.pack(side=tk.LEFT, padx=8)

        self.txt = tk.Text(self, height=20, wrap="word")
        self.txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.txt.insert(
            "end",
            "Select WAV files, choose one or more metrics, tune each metric's parameters, then click 'Run analysis'.\n",
        )

        footer = ttk.Frame(self, padding=(10, 6))
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=1, value=0)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.status = ttk.Label(footer, text="Ready", anchor="e", width=24)
        self.status.pack(side=tk.RIGHT)

    def _build_param_widgets(self):
        self.param_controls: Dict[str, Dict[str, Any]] = {}
        for key, spec in PARAM_SPECS.items():
            label = ttk.Label(self.params_frame, text=spec["label"])
            label.grid(row=0, column=0, sticky="e", pady=2, padx=(0, 6))

            if spec["kind"] == "entry":
                var = tk.StringVar(value="")
                widget = ttk.Entry(self.params_frame, textvariable=var, width=spec.get("width", 12))
            else:
                var = tk.StringVar(value=spec["options"][0][0])
                widget = ttk.Combobox(
                    self.params_frame,
                    values=[display for display, _ in spec["options"]],
                    textvariable=var,
                    state="readonly",
                    width=spec.get("width", 28),
                )

            widget.grid(row=0, column=1, sticky="w")
            self.param_controls[key] = {
                "label": label,
                "widget": widget,
                "var": var,
                "spec": spec,
            }

    def _get_selected_metrics(self) -> List[str]:
        indices = list(self.lst_metrics.curselection())
        return [self.lst_metrics.get(i) for i in indices]

    def _sync_config_metric_choices(self, preferred: Optional[str] = None):
        selected = self._get_selected_metrics()
        editable_metrics = selected or [self.current_config_metric or METRICS[0]]
        self.cmb_config_metric["values"] = editable_metrics

        target = preferred if preferred in editable_metrics else editable_metrics[0]
        self.current_config_metric = target
        self.cmb_config_metric.set(target)
        self._refresh_params()

    def _on_metrics_selection_changed(self, _event=None):
        self._store_current_param_state()
        selected = self._get_selected_metrics()
        preferred = self.current_config_metric if self.current_config_metric in selected else (selected[0] if selected else self.current_config_metric)
        self._sync_config_metric_choices(preferred=preferred)

    def _on_config_metric_changed(self, _event=None):
        previous_metric = getattr(self, "current_config_metric", None)
        self._store_current_param_state(previous_metric)
        picked = self.cmb_config_metric.get().strip()
        if picked:
            self.current_config_metric = picked
        self._refresh_params()

    def _control_value_to_storage(self, key: str):
        ctrl = self.param_controls[key]
        raw = ctrl["var"].get()
        if ctrl["spec"]["kind"] == "entry":
            return raw.strip()

        for display, value in ctrl["spec"]["options"]:
            if raw == display:
                return value
        return raw

    def _set_control_value(self, key: str, value: Any):
        ctrl = self.param_controls[key]
        spec = ctrl["spec"]

        if spec["kind"] == "entry":
            ctrl["var"].set("" if value is None else str(value))
            return

        reverse = {stored: display for display, stored in spec["options"]}
        display = reverse.get(value)
        if display is None and spec["options"]:
            display = spec["options"][0][0]
        ctrl["var"].set(display)

    def _store_current_param_state(self, metric: Optional[str] = None):
        metric = metric or getattr(self, "current_config_metric", None)
        if not metric:
            return

        params = dict(self.metric_params.get(metric, {}))
        for key in METRIC_PARAM_LAYOUT.get(metric, []):
            params[key] = self._control_value_to_storage(key)
        self.metric_params[metric] = params

    def _refresh_params(self):
        metric = getattr(self, "current_config_metric", METRICS[0])
        defaults = dict(DEFAULTS.get(metric, {}))
        stored = dict(self.metric_params.get(metric, {}))
        merged = {**defaults, **stored}
        self.metric_params[metric] = merged

        for ctrl in self.param_controls.values():
            ctrl["label"].grid_remove()
            ctrl["widget"].grid_remove()

        row = 0
        for key in METRIC_PARAM_LAYOUT.get(metric, []):
            ctrl = self.param_controls[key]
            ctrl["label"].grid(row=row, column=0, sticky="e", pady=2, padx=(0, 6))
            ctrl["widget"].grid(row=row, column=1, sticky="w")
            self._set_control_value(key, merged.get(key, ""))
            row += 1

        self.params_frame.configure(text=f"Parameters – {metric}")

    def _build_file_label_maps(self):
        basenames = [Path(path).name for path in self.file_paths]
        duplicates = {name for name in basenames if basenames.count(name) > 1}
        self.file_label_map = {}
        self.label_to_file = {}

        for path in self.file_paths:
            label = str(path) if Path(path).name in duplicates else Path(path).name
            self.file_label_map[path] = label
            self.label_to_file[label] = path

    def _selected_active_file_path(self) -> Optional[str]:
        label = self.cmb_active_file.get().strip()
        if label and label in self.label_to_file:
            return self.label_to_file[label]
        if self.file_paths:
            return self.file_paths[min(self.current_file_index, len(self.file_paths) - 1)]
        return None

    def _on_active_file_changed(self, _event=None):
        file_path = self._selected_active_file_path()
        if file_path and file_path in self.file_paths:
            self.current_file_index = self.file_paths.index(file_path)

    def _append_terminal(self, text: str):
        self.txt.insert("end", text.rstrip() + "\n")
        self.txt.see("end")

    # -------------------------
    # Handlers
    # -------------------------
    def on_pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Choose WAV files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not paths:
            return

        self.file_paths = list(paths)
        self.current_file_index = 0
        self._build_file_label_maps()

        self.lbl_file.config(text=f"{len(self.file_paths)} file(s) selected")
        file_labels = [self.file_label_map[path] for path in self.file_paths]
        self.cmb_active_file["values"] = file_labels
        self.cmb_active_file.set(file_labels[0])
        self.cmb_active_file.config(state="readonly")
        self.btn_wave.config(state="normal")
        self.status.config(text=f"Loaded {len(self.file_paths)} file(s)")

    def on_save_toggle(self):
        if self.var_save.get():
            save_path = filedialog.askdirectory(title="Choose directory to save figures")
            if not save_path:
                self.var_save.set(False)
                return
            self.save_dir = save_path

    def on_excel_toggle(self):
        if self.var_excel.get():
            excel_path = filedialog.askdirectory(title="Choose directory to save Excel file")
            if not excel_path:
                self.var_excel.set(False)
                return
            self.excel_dir = excel_path

    def on_view_waveform(self):
        if not self.file_paths:
            return
        try:
            target_file = self._selected_active_file_path() or self.file_paths[0]
            see(target_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show waveform:\n{e}")

    def _set_running(self, running: bool):
        button_state = "disabled" if running else "normal"
        for w in (self.btn_analyze, self.btn_pick):
            try:
                w.config(state=button_state)
            except Exception:
                pass

        self.lst_metrics.config(state=("disabled" if running else "normal"))
        self.chk_show.config(state=button_state)
        self.chk_save.config(state=button_state)
        self.chk_excel.config(state=button_state)
        self.chk_split.config(state=button_state)

        active_file_state = "disabled" if running or not self.file_paths else "readonly"
        config_metric_state = "disabled" if running else "readonly"
        plot_metric_state = "disabled" if running or not self.metric_results else "readonly"
        wave_state = "disabled" if running or not self.file_paths else "normal"
        plot_button_state = "disabled" if running or not self.metric_results else "normal"

        self.cmb_active_file.config(state=active_file_state)
        self.cmb_config_metric.config(state=config_metric_state)
        self.cmb_plot_metric.config(state=plot_metric_state)
        self.btn_wave.config(state=wave_state)
        self.btn_plot.config(state=plot_button_state)

        if running:
            self.progress.configure(mode="determinate", maximum=max(self._total_steps, 1), value=0)
            self.config(cursor="watch")
            self.status.config(text=f"0/{self._total_steps}")
        else:
            self.config(cursor="")
            if self._analysis_finished:
                self.status.config(text="Done")
            elif self._completed_steps:
                self.status.config(text=f"{self._completed_steps}/{self._total_steps}")
            else:
                self.status.config(text="Ready")

    def _resolve_metric_params(self, metric: str) -> Dict[str, Any]:
        raw = {**DEFAULTS.get(metric, {}), **self.metric_params.get(metric, {})}
        params: Dict[str, Any] = {}

        def parse_float(name: str, default: float = 0.0) -> float:
            value = raw.get(name, default)
            if value in (None, ""):
                return float(default)
            return float(value)

        if "time_skip" in METRIC_PARAM_LAYOUT.get(metric, []):
            params["time_skip"] = parse_float("time_skip", DEFAULTS.get(metric, {}).get("time_skip", 0.0))
        if "field" in METRIC_PARAM_LAYOUT.get(metric, []):
            params["field"] = int(raw.get("field", DEFAULTS.get(metric, {}).get("field", 0)))
        if "method" in METRIC_PARAM_LAYOUT.get(metric, []):
            params["method"] = int(raw.get("method", DEFAULTS.get(metric, {}).get("method", 1)))
        if "weight_type" in METRIC_PARAM_LAYOUT.get(metric, []):
            params["weight_type"] = str(raw.get("weight_type", DEFAULTS.get(metric, {}).get("weight_type", "DIN45692")))
        if "method_fs" in METRIC_PARAM_LAYOUT.get(metric, []):
            params["method_fs"] = int(raw.get("method_fs", DEFAULTS.get(metric, {}).get("method_fs", 1)))
        if metric == "EPNL (FAR Part 36)":
            params["method_epnl"] = 1
            threshold_raw = raw.get("threshold_epnl", DEFAULTS.get(metric, {}).get("threshold_epnl", None))
            params["threshold_epnl"] = None if threshold_raw in (None, "") else float(threshold_raw)

        return params

    def on_analyze(self):
        if not self.file_paths:
            messagebox.showinfo("Select files", "Please choose WAV files first.")
            return

        self._store_current_param_state()
        selected_metrics = self._get_selected_metrics()
        if not selected_metrics:
            messagebox.showinfo("Select metrics", "Please choose at least one metric to analyze.")
            return

        try:
            params_by_metric = {metric: self._resolve_metric_params(metric) for metric in selected_metrics}
        except Exception as e:
            messagebox.showerror("Invalid parameters", f"Please review the metric parameters:\n{e}")
            return

        want_plots_after = bool(self.var_show.get())
        want_save = bool(self.var_save.get())
        want_excel = bool(self.var_excel.get())
        want_split = bool(self.var_split.get())

        self.metric_results = {}
        self.last_metric = None
        self.last_out = None
        self._result_q = queue.Queue()
        self._analysis_finished = False
        self._completed_steps = 0
        self._total_steps = max(len(self.file_paths) * len(selected_metrics), 1)

        self._set_running(True)
        self.txt.delete("1.0", "end")
        self._append_terminal(f"Files: {len(self.file_paths)}")
        self._append_terminal(f"Metrics: {', '.join(selected_metrics)}")
        self._append_terminal("")
        self._append_terminal("Starting analysis...")

        self._worker = threading.Thread(
            target=self._analyze_worker,
            args=(selected_metrics, params_by_metric, want_plots_after, want_save, want_excel, want_split),
            daemon=True,
        )
        self._worker.start()
        self.after(100, self._poll_worker)

    # -------------------------
    # Threading helpers
    # -------------------------
    def _analyze_worker(
        self,
        selected_metrics: List[str],
        params_by_metric: Dict[str, Dict[str, Any]],
        want_plots_after: bool,
        want_save: bool,
        want_excel: bool,
        want_split: bool,
    ):
        try:
            results_by_metric = {metric: [] for metric in selected_metrics}
            total_steps = len(self.file_paths) * len(selected_metrics)
            completed_steps = 0

            for file_idx, file_path in enumerate(self.file_paths, start=1):
                filename = Path(file_path).name
                self._result_q.put(("log", f"[{file_idx}/{len(self.file_paths)}] Loading {filename}"))

                insig = None
                fs = None
                load_error = None
                try:
                    insig, fs = wav2sig(file_path)
                except Exception as e:
                    load_error = str(e)
                    self._result_q.put(("log", f"    Failed to read file: {load_error}"))

                for metric_idx, metric in enumerate(selected_metrics, start=1):
                    self._result_q.put(("log", f"    ({metric_idx}/{len(selected_metrics)}) {metric}: running"))
                    try:
                        if load_error is not None:
                            raise RuntimeError(load_error)
                        out = self._run_metric_worker(metric, params_by_metric[metric], insig, fs, show_plots=False)
                        results_by_metric[metric].append((file_path, out, None))
                        summary = self._summarize(metric, out).splitlines()[0]
                        self._result_q.put(("log", f"        done – {summary}"))
                    except Exception as e:
                        results_by_metric[metric].append((file_path, None, str(e)))
                        self._result_q.put(("log", f"        error – {e}"))

                    completed_steps += 1
                    self._result_q.put(("progress", completed_steps, total_steps, metric, filename))

            self._result_q.put(
                (
                    "ok",
                    results_by_metric,
                    want_plots_after,
                    want_save,
                    want_excel,
                    want_split,
                )
            )
        except Exception as e:
            self._result_q.put(("err", str(e)))

    def _poll_worker(self):
        got_terminal_update = False
        finished = False

        while True:
            try:
                msg = self._result_q.get_nowait()
            except queue.Empty:
                break

            kind = msg[0]
            if kind == "log":
                self._append_terminal(msg[1])
                got_terminal_update = True
            elif kind == "progress":
                _, step, total, metric, filename = msg
                self._completed_steps = step
                self._total_steps = max(total, 1)
                self.progress.configure(maximum=self._total_steps, value=step)
                self.status.config(text=f"{step}/{self._total_steps}")
            elif kind == "ok":
                finished = True
                (
                    _,
                    results_by_metric,
                    want_plots_after,
                    want_save,
                    want_excel,
                    want_split,
                ) = msg
                self._handle_success(results_by_metric, want_plots_after, want_save, want_excel, want_split)
            elif kind == "err":
                finished = True
                _, err = msg
                self._analysis_finished = False
                self._set_running(False)
                messagebox.showerror("Error", f"Analysis failed:\n{err}")

        if finished:
            return

        if self._worker and self._worker.is_alive():
            self.after(120 if got_terminal_update else 180, self._poll_worker)
        else:
            if not self._analysis_finished:
                self._analysis_finished = False
                self._set_running(False)

    def _handle_success(
        self,
        results_by_metric: Dict[str, List[tuple]],
        want_plots_after: bool,
        want_save: bool,
        want_excel: bool,
        want_split: bool,
    ):
        self.metric_results = {}
        self._append_terminal("\n=== Summary ===")

        last_success_metric = None
        last_success_out = None

        for metric, results in results_by_metric.items():
            self._append_terminal(f"\n[{metric}]")
            self.metric_results[metric] = {}

            for file_path, out, err in results:
                filename = self.file_label_map.get(file_path, Path(file_path).name)
                if err:
                    self._append_terminal(f"  - {filename}: ERROR – {err}")
                    continue

                self.metric_results[metric][file_path] = out
                summary = self._summarize(metric, out).replace("\n", " | ")
                self._append_terminal(f"  - {filename}: {summary}")

                if want_save and self.save_dir:
                    self._save_figure(metric, out, file_path, want_split)

                last_success_metric = metric
                last_success_out = out

            if not self.metric_results[metric]:
                del self.metric_results[metric]

        if want_excel and self.excel_dir:
            self._export_results_excel(results_by_metric)

        self.last_metric = last_success_metric
        self.last_out = last_success_out
        self._update_plot_metric_choices()

        self._analysis_finished = True
        self._set_running(False)

        if want_plots_after and self.metric_results:
            self.open_graph_window()

        if want_save and self.save_dir:
            self._append_terminal(f"\nFigures saved to: {self.save_dir}")

    def _update_plot_metric_choices(self):
        available_metrics = list(self.metric_results.keys())
        if not available_metrics:
            self.cmb_plot_metric.set("")
            self.cmb_plot_metric["values"] = []
            self.cmb_plot_metric.config(state="disabled")
            self.btn_plot.config(state="disabled")
            return

        current = self.cmb_plot_metric.get().strip()
        target = current if current in available_metrics else available_metrics[0]
        self.cmb_plot_metric["values"] = available_metrics
        self.cmb_plot_metric.set(target)
        self.cmb_plot_metric.config(state="readonly")
        self.btn_plot.config(state="normal")
        self._refresh_plot_target()

    def _refresh_plot_target(self):
        metric = self.cmb_plot_metric.get().strip()
        if not metric or metric not in self.metric_results:
            return

        available_files = list(self.metric_results[metric].keys())
        if not available_files:
            return

        current_path = self._selected_active_file_path()
        target_path = current_path if current_path in available_files else available_files[0]
        target_label = self.file_label_map.get(target_path, Path(target_path).name)
        self.cmb_active_file.set(target_label)
        self._on_active_file_changed()

    # -------------------------
    # Metric execution
    # -------------------------
    def _run_metric_worker(self, metric: str, p: Dict[str, Any], insig, fs, show_plots: bool):
        if metric == "Loudness (ISO 532-1)":
            return Loudness_ISO532_1(
                insig=insig,
                fs=fs,
                field=p["field"],
                method=p["method"],
                time_skip=p["time_skip"],
                show=show_plots,
            )
        if metric == "Sharpness (DIN 45692)":
            return Sharpness_DIN45692(
                insig=insig,
                fs=fs,
                weight_type=p["weight_type"],
                LoudnessField=p["field"],
                LoudnessMethod=p["method"],
                time_skip=p["time_skip"],
                show_sharpness=show_plots,
                show_loudness=False,
            )
        if metric == "Roughness (Daniel 1997)":
            return Roughness_Daniel1997(insig=insig, fs=fs, time_skip=p["time_skip"], show=show_plots)
        if metric == "Fluctuation Strength (Osses 2016)":
            return FluctuationStrength_Osses2016(
                insig=insig,
                fs=fs,
                method=p["method_fs"],
                time_skip=p["time_skip"],
                show=show_plots,
            )
        if metric == "Tonality (Aures 1985)":
            return Tonality_Aures1985(
                insig=insig,
                fs=fs,
                LoudnessField=p["field"],
                time_skip=p["time_skip"],
                show=show_plots,
            )
        if metric == "Annoyance (Di 2016)":
            return PsychoacousticAnnoyance_Di2016(
                insig=insig,
                fs=fs,
                LoudnessField=p["field"],
                time_skip=p["time_skip"],
                show=show_plots,
                showPA=show_plots,
            )
        if metric == "Annoyance (Zwicker 1999)":
            return PsychoacousticAnnoyance_Zwicker1999(
                insig=insig,
                fs=fs,
                LoudnessField=p["field"],
                time_skip=p["time_skip"],
                show=show_plots,
                showPA=show_plots,
            )
        if metric == "Annoyance (More 2010)":
            return PsychoacousticAnnoyance_More2010(
                insig=insig,
                fs=fs,
                LoudnessField=p["field"],
                time_skip=p["time_skip"],
                show=show_plots,
                showPA=show_plots,
            )
        if metric == "EPNL (FAR Part 36)":
            return EPNL_FAR_Part36(
                insig=insig,
                fs=fs,
                method=1,
                threshold=p.get("threshold_epnl", None),
                show=show_plots,
            )
        if metric == "Loudness (ECMA 418-2)":
            try:
                from metrics_ecma import Loudness_ECMA418_2
            except Exception as e:
                raise RuntimeError(f"ECMA loudness unavailable: {e}")
            return Loudness_ECMA418_2(
                insig=insig,
                fs=fs,
                field=p.get("field", 0),
                method=p.get("method", 1),
                time_skip=p.get("time_skip", 0),
                show=show_plots,
            )
        if metric == "Roughness (ECMA 418-2)":
            try:
                from metrics_ecma import Roughness_ECMA418_2
            except Exception as e:
                raise RuntimeError(f"ECMA roughness unavailable: {e}")
            return Roughness_ECMA418_2(
                insig=insig,
                fs=fs,
                field=p.get("field", 0),
                method=p.get("method", 1),
                time_skip=p.get("time_skip", 0),
                show=show_plots,
            )
        if metric == "Tonality (ECMA 418-2)":
            try:
                from metrics_ecma import Tonality_ECMA418_2
            except Exception as e:
                raise RuntimeError(f"ECMA tonality unavailable: {e}")
            out = Tonality_ECMA418_2(
                insig=insig,
                fs=fs,
                field=p.get("field", 0),
                method=p.get("method", 1),
                time_skip=p.get("time_skip", 0),
                show=show_plots,
            )
            if isinstance(out, dict):
                if "tonalityTDep" in out and "InstantaneousTonality" not in out:
                    out["InstantaneousTonality"] = out["tonalityTDep"]
                if "tonalityAvg" in out and "Kmean" not in out:
                    out["Kmean"] = out["tonalityAvg"]
                if "Kmean" in out and "Tmean" not in out:
                    out["Tmean"] = out["Kmean"]
            return out
        raise ValueError(f"Unknown metric: {metric}")

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
        if metric == "Loudness (ECMA 418-2)" and "Loudness" in out:
            lines.append(f"Loudness (ECMA): {self._get_first_scalar(out['Loudness']):.3f} sone")
        if metric == "Sharpness (DIN 45692)" and "Sharpness" in out:
            lines.append(f"Sharpness: {self._get_first_scalar(out['Sharpness']):.3f} acum")

        units = {"N": "sone", "S": "acum", "R": "asper", "FS": "vacil", "K": "t.u.", "T": "t.u.", "PA": "annoy"}
        family = {
            "Loudness (ISO 532-1)": "N",
            "Loudness (ECMA 418-2)": "N",
            "Sharpness (DIN 45692)": "S",
            "Roughness (Daniel 1997)": "R",
            "Roughness (ECMA 418-2)": "R",
            "Fluctuation Strength (Osses 2016)": "FS",
            "Tonality (Aures 1985)": "K",
            "Tonality (ECMA 418-2)": "T",
            "Annoyance (Di 2016)": "PA",
            "Annoyance (Zwicker 1999)": "PA",
            "Annoyance (More 2010)": "PA",
        }.get(metric, None)
        if family:
            for keylab, pretty in (("mean", "mean"), ("5", "5th percentile"), ("95", "95th percentile"), ("max", "max"), ("min", "min")):
                k = f"{family}{keylab}"
                if k in out:
                    lines.append(f"{pretty.title() if keylab == 'mean' else pretty}: {self._get_first_scalar(out[k]):.3f} {units[family]}")
        if "ScalarPA" in out:
            lines.append(f"Scalar PA: {self._get_first_scalar(out['ScalarPA']):.3f} annoy")
        if "PA5" in out:
            lines.append(f"PA5 (5th percentile): {self._get_first_scalar(out['PA5']):.3f} annoy")

        if metric == "EPNL (FAR Part 36)":
            if "EPNL" in out:
                lines.append(f"EPNL: {self._get_first_scalar(out['EPNL']):.3f} EPNdB")
            if "PNLM" in out:
                lines.append(f"PNLM (max PNL): {self._get_first_scalar(out['PNLM']):.3f} PNdB")
            if "PNLTM" in out:
                lines.append(f"PNLTM (max tone-corrected): {self._get_first_scalar(out['PNLTM']):.3f} PNdB")

        if not lines:
            for tkey, ykey, label in (
                ("time", "InstantaneousLoudness", "Loudness"),
                ("time", "InstantaneousSharpness", "Sharpness"),
                ("time", "InstantaneousRoughness", "Roughness"),
                ("time", "InstantaneousFluctuationStrength", "Fluctuation strength"),
                ("time", "InstantaneousTonality", "Tonality"),
                ("time", "InstantaneousPA", "Annoyance"),
                ("time", "PNL", "PNL"),
                ("time", "PNLT", "PNLT"),
            ):
                if tkey in out and ykey in out:
                    y = np.asarray(out[ykey]).ravel()
                    lines.append(f"{label} – mean: {np.mean(y):.3f}, max: {np.max(y):.3f}")
                    break
        return "\n".join(lines) if lines else "No summary values found."

    # -------------------------
    # Embedded graphs - using plotting_metrics module
    # -------------------------
    def open_graph_window(self):
        metric = self.cmb_plot_metric.get().strip() or self.last_metric
        file_path = self._selected_active_file_path()
        out = None
        if metric and file_path:
            out = self.metric_results.get(metric, {}).get(file_path)

        if out is None:
            out = self.last_out
            metric = metric or self.last_metric

        if not out or not metric:
            messagebox.showinfo("No data", "No analysis results to display")
            return

        fig = create_metric_plot(metric, out)
        if fig is None:
            messagebox.showinfo("No plot", "This metric didn't return plottable data.")
            return

        file_label = self.file_label_map.get(file_path, Path(file_path).name) if file_path else "latest result"

        win = tk.Toplevel(self)
        win.title(f"Graphs – {metric} – {file_label}")
        win.geometry("1200x700")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_full_metric_plot(self, metric: str, out: Dict[str, Any]) -> Optional[Figure]:
        return create_metric_plot(metric, out)

    # -------------------------
    # Figure saving
    # -------------------------
    def _save_figure(self, metric: str, out: Dict[str, Any], file_path: str, split: bool = False):
        filename = Path(file_path).stem

        if split:
            figures = create_metric_plot_split(metric, out)
            if isinstance(figures, dict):
                for component_name, fig in figures.items():
                    figure_name = f"{metric.replace(' ', '_').replace('(', '').replace(')', '')}_{component_name}_{filename}.png"
                    figure_path = os.path.join(self.save_dir, figure_name)
                    fig.savefig(figure_path, dpi=100, bbox_inches="tight")
            else:
                fig = create_metric_plot(metric, out)
                if fig:
                    figure_name = f"{metric.replace(' ', '_').replace('(', '').replace(')', '')}_{filename}.png"
                    figure_path = os.path.join(self.save_dir, figure_name)
                    fig.savefig(figure_path, dpi=100, bbox_inches="tight")
        else:
            fig = create_metric_plot(metric, out)
            if fig:
                figure_name = f"{metric.replace(' ', '_').replace('(', '').replace(')', '')}_{filename}.png"
                figure_path = os.path.join(self.save_dir, figure_name)
                fig.savefig(figure_path, dpi=100, bbox_inches="tight")

    # -------------------------
    # Excel export
    # -------------------------
    def _export_results_excel(self, results_by_metric: Dict[str, List[tuple]]):
        """Export all metric results into one workbook with one summary sheet per metric."""
        wb = Workbook()
        # Remove default sheet; sheets are created per metric.
        wb.remove(wb.active)

        exported_metrics = 0
        total_rows = 0
        total_files = set()

        for metric, results in results_by_metric.items():
            if not any(out is not None for _, out, err in results if not err):
                continue
            ws = wb.create_sheet(self._safe_sheet_name(metric, wb.sheetnames))
            row_count, files = self._write_metric_summary_sheet(ws, metric, results)
            if row_count > 0:
                exported_metrics += 1
                total_rows += row_count
                total_files.update(files)
            else:
                wb.remove(ws)

        if exported_metrics == 0:
            self._append_terminal("\nNo successful results were available for Excel export.")
            messagebox.showwarning("Excel Export", "No data to export")
            return

        excel_path = os.path.join(self.excel_dir, "metrics_comparison.xlsx")
        wb.save(excel_path)
        messagebox.showinfo(
            "Excel Export",
            f"Results saved to:\n{excel_path}\n\nWorkbook contains {exported_metrics} metric sheet(s), "
            f"{total_rows} parameters, from {len(total_files)} file(s)."
        )

    def _write_metric_summary_sheet(self, ws, metric: str, results: List[tuple]):
        """Write one metric summary sheet with one column per file."""
        # Title
        ws['A1'] = f"Metric: {metric}"
        header_cell = ws['A1']
        header_cell.font = Font(bold=True, size=12, color="FFFFFF")
        header_cell.fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")

        # Process files sequentially - one column per file
        collected_rows = {}  # {row_label: {filename: value}}
        all_files = []

        for file_path, out, err in results:
            filename = Path(file_path).stem

            if err or not out:
                continue

            # Extract statistics from this file
            stats = self._extract_metric_stats(metric, out)

            for stat_name, stat_value in stats.items():
                if stat_name not in collected_rows:
                    collected_rows[stat_name] = {}
                collected_rows[stat_name][filename] = stat_value

            if filename not in all_files:
                all_files.append(filename)

        if not collected_rows:
            return 0, []

        # Now write all collected data to the summary sheet
        # Sort rows by priority
        def sort_key(item):
            label = item[0]
            if "[PRIMARY]" in label:
                return (0, label)
            elif "[SECONDARY]" in label:
                return (1, label)
            elif "[TERTIARY]" in label:
                return (2, label)
            elif "[METADATA]" in label:
                return (3, label)
            else:
                return (4, label)
        
        sorted_rows = dict(sorted(collected_rows.items(), key=sort_key))

        # Write column headers (filenames)
        ws['A2'] = "Parameter"
        for col_offset, filename in enumerate(all_files, start=1):
            col_letter = get_column_letter(col_offset + 1)  # B, C, D, ...
            ws[f'{col_letter}2'] = filename

        # Write data rows
        for row_idx, (row_label, values_dict) in enumerate(sorted_rows.items(), start=3):
            # Clean label for display
            display_label = row_label.replace("[PRIMARY] ", "").replace("[SECONDARY] ", "").replace("[TERTIARY] ", "").replace("[METADATA] ", "").replace("[OTHER] ", "")

            ws.cell(row=row_idx, column=1, value=display_label)
            for col_offset, filename in enumerate(all_files, start=1):
                value = values_dict.get(filename, "")
                ws.cell(row=row_idx, column=col_offset + 1, value=value)

        # Format the sheet
        self._format_excel_sheet(ws, len(all_files))
        return len(sorted_rows), all_files

    def _safe_sheet_name(self, metric: str, existing_names: List[str]) -> str:
        """Create a valid unique Excel sheet name (max 31 chars)."""
        invalid_chars = '\\/*?:[]'
        name = "".join(ch for ch in metric if ch not in invalid_chars).strip() or "Metric"
        name = name[:31]

        if name not in existing_names:
            return name

        counter = 2
        while True:
            suffix = f" ({counter})"
            base = name[:31 - len(suffix)]
            candidate = f"{base}{suffix}"
            if candidate not in existing_names:
                return candidate
            counter += 1

    def _extract_metric_stats(self, metric: str, out: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ALL relevant statistics from metric output dictionary, organized by importance"""
        stats = {}
        
        # Map of metric families to their keys
        family_keys = {
            "Loudness": ("N", "sone"),
            "Sharpness": ("S", "acum"),
            "Roughness": ("R", "asper"),
            "Fluctuation": ("FS", "vacil"),
            "Tonality": ("K", "t.u."),
            "Annoyance": ("PA", "annoy"),
        }
        
        # Find which family this metric belongs to
        metric_family = None
        metric_unit = None
        for family, (prefix, unit) in family_keys.items():
            if family in metric:
                metric_family = prefix
                metric_unit = unit
                break
        
        # ===== PRIMARY STATISTICS (Most Important) =====
        # Percentile-based statistics (from get_statistics)
        if metric_family:
            priority_keys = [
                ("5", "5th percentile"),
                ("mean", "Mean"),
                ("95", "95th percentile"),
                ("max", "Max"),
                ("min", "Min"),
                ("std", "Std Dev"),
            ]
            for key_suffix, label in priority_keys:
                key = f"{metric_family}{key_suffix}"
                if key in out:
                    val = self._get_first_scalar(out[key])
                    stats[f"[PRIMARY] {label} ({metric_unit})"] = val
        
        # Overall loudness/sharpness values
        if "Loudness" in metric:
            if "Loudness" in out:
                stats["[PRIMARY] Overall Loudness (sone)"] = self._get_first_scalar(out["Loudness"])
            if "LoudnessLevel" in out:
                stats["[PRIMARY] Loudness Level (phon)"] = self._get_first_scalar(out["LoudnessLevel"])
        
        if "Sharpness" in metric and "Sharpness" in out:
            stats["[PRIMARY] Overall Sharpness (acum)"] = self._get_first_scalar(out["Sharpness"])
        
        # Annoyance-specific primary statistics
        if "Annoyance" in metric:
            if "ScalarPA" in out:
                stats["[PRIMARY] Scalar PA (annoy)"] = self._get_first_scalar(out["ScalarPA"])
            
            # Component weights
            if "ws" in out:
                ws_arr = np.asarray(out["ws"]).ravel()
                if ws_arr.size > 0:
                    stats["[PRIMARY] Weight Sharpness - Mean"] = float(np.mean(ws_arr))
                    stats["[PRIMARY] Weight Sharpness - Max"] = float(np.max(ws_arr))
            
            if "wfr" in out:
                wfr_arr = np.asarray(out["wfr"]).ravel()
                if wfr_arr.size > 0:
                    stats["[PRIMARY] Weight Roughness/Fluctuation - Mean"] = float(np.mean(wfr_arr))
                    stats["[PRIMARY] Weight Roughness/Fluctuation - Max"] = float(np.max(wfr_arr))
            
            if "wt" in out:
                wt_arr = np.asarray(out["wt"]).ravel()
                if wt_arr.size > 0:
                    stats["[PRIMARY] Weight Tonality - Mean"] = float(np.mean(wt_arr))
                    stats["[PRIMARY] Weight Tonality - Max"] = float(np.max(wt_arr))
            
            # Add component percentiles (5th percentile)
            if "L" in out and isinstance(out["L"], dict):
                if "N5" in out["L"]:
                    stats["[SECONDARY] Component N5 - Loudness (sone)"] = self._get_first_scalar(out["L"]["N5"])
                if "Nmean" in out["L"]:
                    stats["[SECONDARY] Component Nmean - Loudness (sone)"] = self._get_first_scalar(out["L"]["Nmean"])
            
            if "S" in out and isinstance(out["S"], dict):
                if "S5" in out["S"]:
                    stats["[SECONDARY] Component S5 - Sharpness (acum)"] = self._get_first_scalar(out["S"]["S5"])
                if "Smean" in out["S"]:
                    stats["[SECONDARY] Component Smean - Sharpness (acum)"] = self._get_first_scalar(out["S"]["Smean"])
            
            if "R" in out and isinstance(out["R"], dict):
                if "R5" in out["R"]:
                    stats["[SECONDARY] Component R5 - Roughness (asper)"] = self._get_first_scalar(out["R"]["R5"])
                if "Rmean" in out["R"]:
                    stats["[SECONDARY] Component Rmean - Roughness (asper)"] = self._get_first_scalar(out["R"]["Rmean"])
            
            if "FS" in out and isinstance(out["FS"], dict):
                if "FS5" in out["FS"]:
                    stats["[SECONDARY] Component FS5 - Fluctuation (vacil)"] = self._get_first_scalar(out["FS"]["FS5"])
                if "FSmean" in out["FS"]:
                    stats["[SECONDARY] Component FSmean - Fluctuation (vacil)"] = self._get_first_scalar(out["FS"]["FSmean"])
            
            if "K" in out and isinstance(out["K"], dict):
                if "K5" in out["K"]:
                    stats["[SECONDARY] Component K5 - Tonality (t.u.)"] = self._get_first_scalar(out["K"]["K5"])
                if "Kmean" in out["K"]:
                    stats["[SECONDARY] Component Kmean - Tonality (t.u.)"] = self._get_first_scalar(out["K"]["Kmean"])
        
        # EPNL-specific primary
        if "EPNL" in metric:
            if "EPNL" in out:
                stats["[PRIMARY] EPNL (EPNdB)"] = self._get_first_scalar(out["EPNL"])
            if "PNLM" in out:
                stats["[PRIMARY] PNLM - Max PNL (PNdB)"] = self._get_first_scalar(out["PNLM"])
            if "PNLTM" in out:
                stats["[PRIMARY] PNLTM - Max Tone-Corrected (PNdB)"] = self._get_first_scalar(out["PNLTM"])
        
        # ===== SECONDARY: INSTANTANEOUS DATA SUMMARIES =====
        # For time-varying signals, extract summary stats from instantaneous arrays
        instantaneous_keys = {
            "InstantaneousLoudness": "Loudness",
            "InstantaneousLoudnessLevel": "Loudness Level",
            "InstantaneousSharpness": "Sharpness",
            "InstantaneousRoughness": "Roughness",
            "InstantaneousFluctuationStrength": "Fluctuation Strength",
            "InstantaneousTonality": "Tonality",
            "InstantaneousPA": "Annoyance",
            "InstantaneousSPL": "SPL",
        }
        
        for key, label in instantaneous_keys.items():
            if key in out:
                arr = np.asarray(out[key]).ravel()
                if arr.size > 1:  # Only if array
                    stats[f"[SECONDARY] {label} - Mean"] = float(np.mean(arr))
                    stats[f"[SECONDARY] {label} - Max"] = float(np.max(arr))
                    stats[f"[SECONDARY] {label} - Min"] = float(np.min(arr))
                    stats[f"[SECONDARY] {label} - Std"] = float(np.std(arr))
        
        # ===== TERTIARY: SPECIFIC LOUDNESS DATA =====
        if "SpecificLoudness" in out:
            arr = np.asarray(out["SpecificLoudness"]).ravel()
            if arr.size > 0:
                stats[f"[TERTIARY] Specific Loudness - Mean"] = float(np.mean(arr))
                stats[f"[TERTIARY] Specific Loudness - Max"] = float(np.max(arr))
                stats[f"[TERTIARY] Specific Loudness - Min"] = float(np.min(arr))
        
        if "TimeAveragedSPL" in out:
            arr = np.asarray(out["TimeAveragedSPL"]).ravel()
            if arr.size > 0:
                stats[f"[TERTIARY] Time-Averaged SPL - Mean"] = float(np.mean(arr))
                stats[f"[TERTIARY] Time-Averaged SPL - Max"] = float(np.max(arr))
        
        # ===== QUATERNARY: ARRAY DIMENSIONS & METADATA =====
        # Record dimensions of major arrays for reference
        for key in ["barkAxis", "time", "time_insig", "InstantaneousLoudness", "InstantaneousSpecificLoudness"]:
            if key in out:
                val = out[key]
                arr = np.asarray(val).ravel()
                if arr.size > 0:
                    stats[f"[METADATA] {key} - size"] = int(arr.size)
        
        # ===== FALLBACK: Extract any remaining scalar keys =====
        for key in out.keys():
            if key not in stats:
                val = out[key]
                # Only add if it's a scalar or small array
                if isinstance(val, (int, float)):
                    stats[f"[OTHER] {key}"] = val
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.asarray(val).ravel()
                    if 1 <= arr.size <= 5:  # Only small arrays
                        stats[f"[OTHER] {key}"] = float(arr[0]) if arr.size == 1 else f"array({arr.size})"
        
        return stats

    def _add_matrix_sheet_single(self, wb, out: Dict, filename: str, metric: str):
        """Add a single matrix data sheet for one file (sequential processing)"""
        # Define which arrays to export for each metric family
        # These match the complete output dictionaries from each metric function
        matrix_keys = {
            "Loudness": [
                "InstantaneousLoudness", "InstantaneousLoudnessLevel", "InstantaneousSPL",
                "SpecificLoudness", "InstantaneousSpecificLoudness", "TimeAveragedSPL",
                "time", "time_insig", "barkAxis", "ThirdOctaveLevel"
            ],
            "Sharpness": [
                "InstantaneousSharpness", "time", "loudness"
            ],
            "Roughness": [
                "InstantaneousRoughness", "InstantaneousSpecificRoughness", 
                "TimeAveragedSpecificRoughness", "time", "barkAxis", "dz"
            ],
            "Fluctuation": [
                "InstantaneousFluctuationStrength", "time"
            ],
            "Tonality": [
                "InstantaneousTonality", "time", "freqAxis"
            ],
            "Annoyance": [
                "InstantaneousPA", "time",
                # Component metric matrices from sub-calculations
                "ws", "wfr", "wt",
                # Nested component metrics (loudness, sharpness, roughness, fluctuation, tonality)
                "L", "S", "R", "FS", "K"
            ],
            "EPNL": [
                "PNL", "PNLT", "time", "t_f"
            ],
        }
        
        # Determine which metric family we're dealing with
        selected_keys = []
        for family in matrix_keys.keys():
            if family in metric:
                selected_keys = matrix_keys[family]
                break
        
        if not selected_keys:
            return  # No matching metric
        
        # Create sheet for this file
        sheet_name = filename[:31]  # Excel sheet name limit is 31 chars
        ws = wb.create_sheet(sheet_name)
        
        # Title
        ws['A1'] = f"Data Matrices - {filename}"
        ws['A1'].font = Font(bold=True, size=12, color="FFFFFF")
        ws['A1'].fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
        ws.merge_cells('A1:Z1')
        
        current_row = 2
        
        # Export each matrix/array
        for key in selected_keys:
            if key not in out:
                continue
            
            data = out[key]
            
            # Special handling for nested dicts (Annoyance components)
            if isinstance(data, dict):
                # This is a sub-metric dict (L, S, R, FS, K)
                ws.cell(row=current_row, column=1, value=f"=== {key} (Component Metric) ===")
                header_cell = ws.cell(row=current_row, column=1)
                header_cell.font = Font(bold=True, color="FFFFFF", size=10)
                header_cell.fill = PatternFill(start_color="8B7355", end_color="8B7355", fill_type="solid")
                current_row += 1
                
                # Export key arrays from this sub-metric
                component_arrays = {
                    "L": ["InstantaneousLoudness", "InstantaneousLoudnessLevel", "SpecificLoudness", "barkAxis"],
                    "S": ["InstantaneousSharpness"],
                    "R": ["InstantaneousRoughness", "TimeAveragedSpecificRoughness", "barkAxis"],
                    "FS": ["InstantaneousFluctuationStrength"],
                    "K": ["InstantaneousTonality", "freqAxis"],
                }
                
                if key in component_arrays:
                    for subkey in component_arrays[key]:
                        if subkey in data:
                            subdata = data[subkey]
                            subarr = np.asarray(subdata).ravel()
                            
                            if subarr.size == 0:
                                continue
                            
                            # Add section header for sub-array
                            ws.cell(row=current_row, column=1, value=f"  {subkey}")
                            subheader = ws.cell(row=current_row, column=1)
                            subheader.font = Font(bold=True, color="FFFFFF", size=9)
                            subheader.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                            current_row += 1
                            
                            # Add dimension info
                            ws.cell(row=current_row, column=1, value="Dim")
                            ws.cell(row=current_row, column=2, value=subarr.size)
                            dim_cell = ws.cell(row=current_row, column=1)
                            dim_cell.font = Font(italic=True, size=8)
                            current_row += 1
                            
                            # Add data
                            for idx, val in enumerate(subarr, start=1):
                                row = current_row + (idx - 1) // 10
                                col = 1 + ((idx - 1) % 10)
                                cell = ws.cell(row=row, column=col, value=float(val))
                                cell.number_format = '0.0000'
                                cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            current_row += (subarr.size + 9) // 10 + 1
                continue
            
            arr = np.asarray(data).ravel()
            
            if arr.size == 0:
                continue
            
            # Add section header
            ws.cell(row=current_row, column=1, value=key)
            header_cell = ws.cell(row=current_row, column=1)
            header_cell.font = Font(bold=True, color="FFFFFF", size=11)
            header_cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            ws.merge_cells(f'A{current_row}:D{current_row}')
            current_row += 1
            
            # Add array dimensions info
            ws.cell(row=current_row, column=1, value="Dimension")
            ws.cell(row=current_row, column=2, value=arr.size)
            ws.cell(row=current_row, column=3, value="values")
            dim_cell = ws.cell(row=current_row, column=1)
            dim_cell.font = Font(italic=True, size=9, color="666666")
            current_row += 1
            
            # Add data values
            for idx, val in enumerate(arr, start=1):
                row = current_row + (idx - 1) // 10  # 10 values per row
                col = 1 + ((idx - 1) % 10)
                cell = ws.cell(row=row, column=col, value=float(val))
                cell.number_format = '0.0000'
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Update row for next section
            current_row += (arr.size + 9) // 10 + 2  # Account for wrapped rows
        
        # Set column widths
        for col in range(1, 11):
            ws.column_dimensions[chr(64 + col)].width = 15

    def _format_excel_sheet(self, ws, num_files=1):
        """Apply formatting to Excel worksheet"""
        # Define styles
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)
        
        # Priority-based row colors
        primary_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
        primary_font = Font(bold=True, size=10)
        
        secondary_fill = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")
        secondary_font = Font(size=9)
        
        tertiary_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
        tertiary_font = Font(size=9, italic=True)
        
        metadata_font = Font(size=8, color="666666")
        
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Format header row (Row 2)
        for cell in ws[2]:
            if cell.value:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = border
        
        # Format data rows with category-based styling
        for row_idx, row in enumerate(ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=1, max_col=ws.max_column), start=3):
            row_label = ws.cell(row=row_idx, column=1).value or ""
            
            # Determine styling based on parameter type
            if "[PRIMARY]" in row_label or any(key in row_label for key in ["5th percentile", "Mean", "Overall", "Max", "Min", "Std"]):
                row_fill = primary_fill
                label_font = primary_font
            elif "[SECONDARY]" in row_label or "Instantaneous" in row_label:
                row_fill = secondary_fill
                label_font = secondary_font
            elif "[TERTIARY]" in row_label or "Specific" in row_label:
                row_fill = tertiary_fill
                label_font = tertiary_font
            else:
                row_fill = None
                label_font = metadata_font
            
            for col_idx, cell in enumerate(row, start=1):
                if col_idx == 1:  # Parameter name column
                    cell.font = label_font
                    if row_fill:
                        cell.fill = row_fill
                    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                else:  # Data columns
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00'
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    if row_fill:
                        cell.fill = row_fill
                cell.border = border
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 50
        for col in range(2, ws.max_column + 1):
            ws.column_dimensions[chr(64 + col)].width = 18


if __name__ == "__main__":
    app = SQATApp()
    app.mainloop()

