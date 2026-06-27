"""
FINAL_VAL_MALR.py
=================
Alternate validation script.  Reports per (metric, case):

  * Relative error on **mean** and **std** scalars  (as in FINAL_VAL.py)
  * **Normalised Mean Absolute Error (NMAE %)** on the full instantaneous
    time series, replacing the percentile comparisons (5th, 10th, 50th,
    95th, …).

NMAE is defined as:

    NMAE = 100 * mean( |py_t - ref_t| ) / mean( |ref_t| )   [%]

It is the time-average of the absolute frame-by-frame error, expressed as a
percentage of the mean reference level — i.e. the regular time-series average
relative error.  Unlike the signed mean error (which reduces to the
mean-scalar relative error), it does not let positive and negative deviations
cancel, so it captures how far apart the two curves run on average.  It
reuses each metric's scalar PASS/FAIL band.

Outputs
-------
  validation/python_results/results_python_malr.json
  validation/report/figures/<metric>_overview.png      (same figures)
  validation/report/FINAL_VAL_MALR.tex
  validation/comparison/FINAL_VAL_MALR.csv

Usage
-----
  cd SQAT4PY
  python validation/FINAL_VAL_MALR.py

  # Optionally compile the LaTeX report:
  cd validation/report
  pdflatex FINAL_VAL_MALR.tex
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent
SF_ROOT  = ROOT / "sound_files"
REF_DIR  = SF_ROOT / "reference"
EX_DIR   = SF_ROOT / "examples"
PY_JSON  = HERE / "python_results" / "results_python_malr.json"
MAT_JSON = HERE / "matlab_results" / "results_matlab.json"
FIG_DIR  = HERE / "report" / "figures"
TEX_FILE = HERE / "report" / "FINAL_VAL_MALR.tex"
CSV_FILE = HERE / "comparison" / "FINAL_VAL_MALR.csv"

for d in (PY_JSON.parent, FIG_DIR, HERE / "report", HERE / "comparison"):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ── scalar tolerances (relative %, for mean/std and NMAE rows) ────────────────
#   The per-metric bands in mdef["tol"] drive the verdicts; these globals are
#   only the fallback default.  NMAE (the time-series average relative error)
#   reuses the same band as the mean/std scalars.
TOL_PASS = 1.0
TOL_FAIL = 2.0


def _parse_tolerances(argv: list[str]) -> None:
    global TOL_PASS, TOL_FAIL
    import argparse
    ap = argparse.ArgumentParser(add_help=True,
                                 description="SQAT4PY NMAE validation")
    ap.add_argument("--pass-tol", type=float, default=TOL_PASS)
    ap.add_argument("--fail-tol", type=float, default=TOL_FAIL)
    args, _ = ap.parse_known_args(argv)
    TOL_PASS = float(args.pass_tol)
    TOL_FAIL = float(args.fail_tol)


# ── metric imports ─────────────────────────────────────────────────────────────
from utilities import wav2sig
from metrics_loudness    import Loudness_ISO532_1
from metrics_roughness   import Roughness_Daniel1997
from metrics_sharpness   import Sharpness_DIN45692
from metrics_fluctuation import FluctuationStrength_Osses2016
from metrics_tonality    import Tonality_Aures1985
from metrics_epnl        import EPNL_FAR_Part36
from metrics_annoyance   import (PsychoacousticAnnoyance_Di2016,
                                 PsychoacousticAnnoyance_Zwicker1999,
                                 PsychoacousticAnnoyance_More2010)

try:
    from metrics_ecma import (Loudness_ECMA418_2,
                               Roughness_ECMA418_2,
                               Tonality_ECMA418_2)
    _ECMA_OK = True
except Exception:
    _ECMA_OK = False

# ── metric registry ────────────────────────────────────────────────────────────
#   scalars : only mean + std are compared per-scalar (with relative % error).
#             Percentile statistics are dropped in favour of NMAE.
#   EPNL    : all three scalars kept (they are event-level quantities,
#             not percentiles of a distribution).
# ──────────────────────────────────────────────────────────────────────────────
AUDIO_TESTER = "tester.wav"
AUDIO_A320   = "5sec_A320.wav"

METRICS: list[dict] = [
    # ── Classic ────────────────────────────────────────────────────────────────
    {
        "id":          "Loudness_ISO532_1",
        "label":       "Loudness  ISO 532-1",
        "unit":        "sone",
        "group":       "classic",
        "tol":         (1.0, 2.0),
        "scalars":     ["Nmean", "Nstd"],
        "main_scalar": "Nmean",
        "audio_cases": {
            "reference": ("RefSignal_Loudness_ISO532_1.wav", 94.0),
            "tester":    (AUDIO_TESTER, 94.0),
            "a320":      (AUDIO_A320,   94.0),
        },
        "params":     dict(field=0, method=2, time_skip=0.5),
        "available":  True,
        "theory_ref": {"Nmean": 1.0},
    },
    {
        "id":          "Roughness_Daniel1997",
        "label":       "Roughness  Daniel 1997",
        "unit":        "asper",
        "group":       "classic",
        "tol":         (2.0, 5.0),
        "scalars":     ["Rmean", "Rstd"],
        "main_scalar": "Rmean",
        "audio_cases": {
            "reference": ("RefSignal_Roughness_Daniel1997.wav", 94.0),
            "tester":    (AUDIO_TESTER, 94.0),
            "a320":      (AUDIO_A320,   94.0),
        },
        "params":     dict(time_skip=0.2),
        "available":  True,
        "theory_ref": {"Rmean": 1.0},
    },
    {
        "id":          "Sharpness_DIN45692",
        "label":       "Sharpness  DIN 45692",
        "unit":        "acum",
        "group":       "classic",
        "tol":         (1.0, 2.0),
        "scalars":     ["Smean", "Sstd"],
        "main_scalar": "Smean",
        "audio_cases": {
            "reference": ("RefSignal_Sharpness_DIN45692.wav", None),
            "tester":    (AUDIO_TESTER, 94.0),
            "a320":      (AUDIO_A320,   94.0),
        },
        "params":     dict(weight_type="DIN45692", LoudnessField=0,
                           LoudnessMethod=2, time_skip=0.5),
        "available":  True,
        "theory_ref": {"Smean": 1.0},
    },
    {
        "id":          "FluctuationStrength_Osses2016",
        "label":       "Fluct. Strength  Osses 2016",
        "unit":        "vacil",
        "group":       "classic",
        "tol":         (2.0, 5.0),
        "scalars":     ["FSmean", "FSstd"],
        "main_scalar": "FSmean",
        "audio_cases": {
            "reference": ("RefSignal_FluctuationStrength_Osses2016.wav", 94.0),
            "tester":    (AUDIO_TESTER, 94.0),
            "a320":      (AUDIO_A320,   94.0),
        },
        "params":     dict(method=1, time_skip=0.2),
        "available":  True,
        "theory_ref": {"FSmean": 1.0},
    },
    {
        "id":          "Tonality_Aures1985",
        "label":       "Tonality  Aures 1985",
        "unit":        "t.u.",
        "group":       "classic",
        "tol":         (2.0, 5.0),
        "scalars":     ["Kmean", "Kstd"],
        "main_scalar": "Kmean",
        "audio_cases": {
            "reference": ("RefSignal_Tonality_Aures1985.wav", 94.0),
            "tester":    (AUDIO_TESTER, 94.0),
            "a320":      (AUDIO_A320,   94.0),
        },
        "params":     dict(LoudnessField=0, time_skip=0.2),
        "available":  True,
        "theory_ref": {"Kmean": 1.0},
    },
    {
        "id":          "EPNL_FAR_Part36",
        "label":       "EPNL  FAR Part 36",
        "unit":        "EPNdB",
        "group":       "classic",
        "tol":         (1.0, 2.0),
        "scalars":     ["EPNL", "PNLTM", "PNLM"],   # event-level: no percentiles
        "main_scalar": "EPNL",
        "audio_cases": {
            "tester": (AUDIO_TESTER, 94.0),
            "a320":   (AUDIO_A320,   94.0),
        },
        "params":     dict(dt=0.5, threshold=10.0),
        "available":  True,
        "theory_ref": {},
    },
    # ── Psychoacoustic Annoyance ───────────────────────────────────────────────
    {
        "id":          "PsychoacousticAnnoyance_Di2016",
        "label":       "PA  Di 2016",
        "unit":        "-",
        "group":       "annoyance",
        "tol":         (5.0, 10.0),
        "scalars":     ["PAmean", "PAstd"],
        "main_scalar": "PAmean",
        "audio_cases": {
            "tester": (AUDIO_TESTER, 94.0),
            "a320":   (AUDIO_A320,   94.0),
        },
        "params":     dict(LoudnessField=0, time_skip=0.5),
        "available":  True,
        "theory_ref": {},
    },
    {
        "id":          "PsychoacousticAnnoyance_Zwicker1999",
        "label":       "PA  Zwicker 1999",
        "unit":        "-",
        "group":       "annoyance",
        "tol":         (5.0, 10.0),
        "scalars":     ["PAmean", "PAstd"],
        "main_scalar": "PAmean",
        "audio_cases": {
            "tester": (AUDIO_TESTER, 94.0),
            "a320":   (AUDIO_A320,   94.0),
        },
        "params":     dict(LoudnessField=0, time_skip=0.5),
        "available":  True,
        "theory_ref": {},
    },
    {
        "id":          "PsychoacousticAnnoyance_More2010",
        "label":       "PA  More 2010",
        "unit":        "-",
        "group":       "annoyance",
        "tol":         (5.0, 10.0),
        "scalars":     ["PAmean", "PAstd"],
        "main_scalar": "PAmean",
        "audio_cases": {
            "tester": (AUDIO_TESTER, 94.0),
            "a320":   (AUDIO_A320,   94.0),
        },
        "params":     dict(LoudnessField=0, time_skip=0.5),
        "available":  True,
        "theory_ref": {},
    },
    # ── ECMA-418-2 / Sottek Hearing Model ─────────────────────────────────────
    {
        "id":          "Loudness_ECMA418_2",
        "label":       "Loudness  ECMA-418-2",
        "unit":        "sone",
        "group":       "ecma",
        "tol":         (2.0, 5.0),
        "scalars":     ["Nmean", "Nstd"],
        "main_scalar": "Nmean",
        "audio_cases": {
            "reference": ("RefSignal_Loudness_ECMA418_2.wav", 94.0),
            "tester":    (AUDIO_TESTER, 94.0),
            "a320":      (AUDIO_A320,   94.0),
        },
        "params":     dict(field=0, method=1, time_skip=0.5),
        "available":  _ECMA_OK,
        "theory_ref": {"Nmean": 1.0},
    },
    {
        "id":          "Roughness_ECMA418_2",
        "label":       "Roughness  ECMA-418-2",
        "unit":        "asper",
        "group":       "ecma",
        "tol":         (2.0, 5.0),
        "scalars":     ["Rmean", "Rstd"],
        "main_scalar": "Rmean",
        "audio_cases": {
            "reference": ("RefSignal_Roughness_ECMA418_2.wav", 94.0),
            "tester":    (AUDIO_TESTER, 94.0),
            "a320":      (AUDIO_A320,   94.0),
        },
        "params":     dict(field=0, method=1, time_skip=0.5),
        "available":  _ECMA_OK,
        "theory_ref": {"Rmean": 1.0},
    },
    {
        "id":          "Tonality_ECMA418_2",
        "label":       "Tonality  ECMA-418-2",
        "unit":        "t.u.",
        "group":       "ecma",
        "tol":         (2.0, 5.0),
        "scalars":     ["Tmean", "Tstd"],
        "main_scalar": "Tmean",
        "audio_cases": {
            "reference": ("RefSignal_Tonality_ECMA418_2.wav", 94.0),
            "tester":    (AUDIO_TESTER, 94.0),
            "a320":      (AUDIO_A320,   94.0),
        },
        "params":     dict(field=0, method=1, time_skip=0.5),
        "available":  _ECMA_OK,
        "theory_ref": {"Tmean": 1.0},
    },
]

CASE_LABEL = {
    "reference": "Reference signal",
    "tester":    "tester.wav",
    "a320":      "5sec_A320.wav",
}

_TARGET_SPL = {"RefSignal_Sharpness_DIN45692.wav": 60.0}

# ── figure config (unchanged from FINAL_VAL.py) ────────────────────────────────
METRIC_CFG: dict[str, dict] = {
    "Loudness_ISO532_1": dict(
        time_key="InstantaneousLoudness", time_label="Loudness (sone)",
        spec_key="SpecificLoudness_avg",  spec_axis="barkAxis",
        spec_label="Spec. Loudness (sone/Bark)", spec_xlabel="Bark", spec_xlog=False,
        extra_time_keys=[], extra_labels=[],
    ),
    "Roughness_Daniel1997": dict(
        time_key="InstantaneousRoughness", time_label="Roughness (asper)",
        spec_key="TimeAveragedSpecificRoughness", spec_axis="barkAxis",
        spec_label="Spec. Roughness (asper/Bark)", spec_xlabel="Bark", spec_xlog=False,
        extra_time_keys=[], extra_labels=[],
    ),
    "Sharpness_DIN45692": dict(
        time_key="InstantaneousSharpness", time_label="Sharpness (acum)",
        spec_key=None, spec_axis=None, spec_label=None, spec_xlabel=None, spec_xlog=False,
        extra_time_keys=[], extra_labels=[],
    ),
    "FluctuationStrength_Osses2016": dict(
        time_key="InstantaneousFluctuationStrength", time_label="Fluct. Strength (vacil)",
        spec_key="TimeAveragedSpecificFluctuationStrength", spec_axis="barkAxis",
        spec_label="Spec. FS (vacil/Bark)", spec_xlabel="Bark", spec_xlog=False,
        extra_time_keys=[], extra_labels=[],
    ),
    "Tonality_Aures1985": dict(
        time_key="InstantaneousTonality", time_label="Tonality (t.u.)",
        spec_key=None, spec_axis=None, spec_label=None, spec_xlabel=None, spec_xlog=False,
        extra_time_keys=[], extra_labels=[],
    ),
    "EPNL_FAR_Part36": dict(
        time_key="PNLT", time_label="PNLT (TPNdB)",
        spec_key="SPL_TOB_avg", spec_axis="TOB_freq",
        spec_label="SPL (dB)", spec_xlabel="Freq (Hz)", spec_xlog=True,
        extra_time_keys=["PNL"], extra_labels=["PNL"],
    ),
    "PsychoacousticAnnoyance_Di2016": dict(
        time_key="InstantaneousPA", time_label="PA Di 2016 (-)",
        spec_key=None, spec_axis=None, spec_label=None, spec_xlabel=None, spec_xlog=False,
        extra_time_keys=["wt", "wfr", "ws"], extra_labels=["wt", "wfr", "ws"],
    ),
    "PsychoacousticAnnoyance_Zwicker1999": dict(
        time_key="InstantaneousPA", time_label="PA Zwicker 1999 (-)",
        spec_key=None, spec_axis=None, spec_label=None, spec_xlabel=None, spec_xlog=False,
        extra_time_keys=["wfr", "ws"], extra_labels=["wfr", "ws"],
    ),
    "PsychoacousticAnnoyance_More2010": dict(
        time_key="InstantaneousPA", time_label="PA More 2010 (-)",
        spec_key=None, spec_axis=None, spec_label=None, spec_xlabel=None, spec_xlog=False,
        extra_time_keys=["wt", "wfr", "ws"], extra_labels=["wt", "wfr", "ws"],
    ),
    "Loudness_ECMA418_2": dict(
        time_key="loudnessTDep", time_label="Loudness ECMA (sone_HMS)",
        spec_key="specLoudnessPowAvg", spec_axis="bandCentreFreqs",
        spec_label="Spec. Loudness (sone_HMS)", spec_xlabel="Freq (Hz)", spec_xlog=True,
        extra_time_keys=[], extra_labels=[],
    ),
    "Roughness_ECMA418_2": dict(
        time_key="roughnessTDep", time_label="Roughness ECMA (asper)",
        spec_key="specRoughnessAvg", spec_axis="bandCentreFreqs",
        spec_label="Spec. Roughness (asper)", spec_xlabel="Freq (Hz)", spec_xlog=True,
        extra_time_keys=[], extra_labels=[],
    ),
    "Tonality_ECMA418_2": dict(
        time_key="tonalityTDep", time_label="Tonality ECMA (tu_HMS)",
        spec_key="specTonalityAvg", spec_axis="bandCentreFreqs",
        spec_label="Spec. Tonality (tu_HMS)", spec_xlabel="Freq (Hz)", spec_xlog=True,
        extra_time_keys=[], extra_labels=[],
    ),
}

# ── signal loading ─────────────────────────────────────────────────────────────

def _resolve_sf(fname: str) -> Path:
    for d in (REF_DIR, EX_DIR):
        cand = d / fname
        if cand.exists():
            return cand
    return REF_DIR / fname


def _dbfs_for_spl(wav_path: Path, target_spl: float) -> float:
    _, d = wavfile.read(str(wav_path))
    if d.dtype.kind in "iu":
        d = d.astype(np.float64) / np.iinfo(d.dtype).max
    else:
        d = d.astype(np.float64)
    if d.ndim > 1:
        d = d.mean(axis=1)
    rms_raw = float(np.sqrt(np.mean(d ** 2)))
    p_target = 2e-5 * 10.0 ** (target_spl / 20.0)
    return 94.0 + 20.0 * np.log10(p_target / rms_raw)


def load_signal(fname: str, dBFS_in: float | None) -> tuple[np.ndarray, int]:
    wav_path = _resolve_sf(fname)
    if dBFS_in is None:
        dBFS_in = _dbfs_for_spl(wav_path, _TARGET_SPL[fname])
    return wav2sig(str(wav_path), dBFS=dBFS_in)


# ── small helpers ──────────────────────────────────────────────────────────────

def _sc(v) -> float:
    arr = np.asarray(v).ravel()
    return float(arr[0]) if arr.size > 0 else float("nan")

def _vec(v) -> list:
    return np.asarray(v).ravel().tolist()

def _pull(out: dict, keys: list[str]) -> dict[str, float]:
    return {k: _sc(out[k]) for k in keys if k in out}

def _spec_avg(arr) -> np.ndarray:
    a = np.asarray(arr)
    return np.mean(a, axis=0) if a.ndim == 2 else a.ravel()


# ── time-series error helpers ───────────────────────────────────────────────────

def _align_series(py_ts, ref_ts, py_t=None, ref_t=None):
    """Align two instantaneous series for frame-by-frame comparison.

    Returns ``(py_a, ref_a)`` sampled at matching instants.  When both time
    vectors are supplied (and consistent with their series), the Python series
    is resampled onto the MATLAB time base over the overlapping interval, so
    frames are compared at *matching instants* rather than by raw index — this
    guards against differing frame rates or time_skip cropping between the two
    implementations.  When time vectors are unavailable, falls back to index
    alignment on the common length.  Returns ``(None, None)`` when alignment
    is impossible (empty inputs or no temporal overlap).
    """
    py  = np.asarray(py_ts,  dtype=float).ravel()
    ref = np.asarray(ref_ts, dtype=float).ravel()
    if py.size == 0 or ref.size == 0:
        return None, None

    pt = np.asarray(py_t,  dtype=float).ravel() if py_t  is not None else None
    rt = np.asarray(ref_t, dtype=float).ravel() if ref_t is not None else None

    if (pt is not None and rt is not None
            and pt.size == py.size and rt.size == ref.size
            and pt.size >= 2 and rt.size >= 2
            and np.all(np.diff(pt) > 0) and np.all(np.diff(rt) > 0)):
        # Time-aligned comparison on the overlapping interval.
        t0, t1 = max(pt[0], rt[0]), min(pt[-1], rt[-1])
        sel = (rt >= t0) & (rt <= t1)
        if sel.sum() == 0:
            return None, None
        return np.interp(rt[sel], pt, py), ref[sel]

    # Fall back to index alignment on the common length.
    n = min(py.size, ref.size)
    return py[:n], ref[:n]


def compute_nmae_pct(py_ts, ref_ts, py_t=None, ref_t=None) -> float:
    """
    Normalised Mean Absolute Error (%) between two instantaneous time series —
    the regular time-series average relative error.

        NMAE = 100 * mean( |py_t - ref_t| ) / mean( |ref_t| )

    It averages the absolute frame-by-frame error and expresses it as a
    percentage of the mean reference level.  Series are aligned by
    :func:`_align_series`.  All overlapping frames are used.  Returns NaN when
    alignment fails or the reference level is zero.
    """
    py_a, ref_a = _align_series(py_ts, ref_ts, py_t, ref_t)
    if py_a is None:
        return float("nan")

    denom = float(np.mean(np.abs(ref_a)))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(100.0 * np.mean(np.abs(py_a - ref_a)) / denom)


# ── per-metric Python runners (unchanged) ─────────────────────────────────────

def _run_one(mdef: dict, fname: str, dBFS_val: float | None) -> dict:
    mid = mdef["id"]
    p   = mdef["params"]
    sig, fs = load_signal(fname, dBFS_val)
    scalars: dict = {}
    vectors: dict = {}

    if mid == "Loudness_ISO532_1":
        out = Loudness_ISO532_1(sig, fs, field=p["field"], method=p["method"],
                                time_skip=p["time_skip"], show=False)
        scalars = _pull(out, ["Nmean", "Nstd", "Nmax", "Nmin",
                               "N5", "N10", "N50", "N95"])
        vectors = {
            "time":                  _vec(out.get("time", [])),
            "InstantaneousLoudness": _vec(out.get("InstantaneousLoudness", [])),
            "barkAxis":              _vec(out.get("barkAxis", [])),
            "SpecificLoudness_avg":  _spec_avg(out.get("SpecificLoudness", [])).tolist(),
        }

    elif mid == "Roughness_Daniel1997":
        out = Roughness_Daniel1997(sig, fs, time_skip=p["time_skip"], show=False)
        scalars = _pull(out, ["Rmean", "Rstd", "Rmax", "Rmin",
                               "R5", "R10", "R50", "R95"])
        vectors = {
            "time":                         _vec(out.get("time", [])),
            "InstantaneousRoughness":        _vec(out.get("InstantaneousRoughness", [])),
            "barkAxis":                      _vec(out.get("barkAxis", [])),
            "TimeAveragedSpecificRoughness": _vec(out.get("TimeAveragedSpecificRoughness", [])),
        }

    elif mid == "Sharpness_DIN45692":
        out = Sharpness_DIN45692(insig=sig, fs=fs,
                                 weight_type=p["weight_type"],
                                 LoudnessField=p["LoudnessField"],
                                 LoudnessMethod=p["LoudnessMethod"],
                                 time_skip=p["time_skip"],
                                 show_sharpness=False, show_loudness=False)
        scalars = _pull(out, ["Smean", "Sstd", "Smax", "Smin",
                               "S5", "S10", "S50", "S95"])
        vectors = {
            "time":                   _vec(out.get("time", [])),
            "InstantaneousSharpness": _vec(out.get("InstantaneousSharpness", [])),
        }

    elif mid == "FluctuationStrength_Osses2016":
        out = FluctuationStrength_Osses2016(sig, fs, method=p["method"],
                                             time_skip=p["time_skip"], show=False)
        scalars = _pull(out, ["FSmean", "FSstd", "FSmax", "FSmin",
                               "FS5", "FS10", "FS50", "FS95"])
        vectors = {
            "time":                                    _vec(out.get("time", [])),
            "InstantaneousFluctuationStrength":         _vec(out.get("InstantaneousFluctuationStrength", [])),
            "barkAxis":                                _vec(out.get("barkAxis", [])),
            "TimeAveragedSpecificFluctuationStrength": _vec(out.get("TimeAveragedSpecificFluctuationStrength", [])),
        }

    elif mid == "Tonality_Aures1985":
        out = Tonality_Aures1985(sig, fs, LoudnessField=p["LoudnessField"],
                                 time_skip=p["time_skip"], show=False)
        scalars = _pull(out, ["Kmean", "Kstd", "Kmax", "Kmin",
                               "K5", "K10", "K50", "K95"])
        vectors = {
            "time":                  _vec(out.get("time", [])),
            "InstantaneousTonality": _vec(out.get("InstantaneousTonality", [])),
        }

    elif mid == "EPNL_FAR_Part36":
        out = EPNL_FAR_Part36(sig, fs, dt=p["dt"], threshold=p["threshold"], show=False)
        scalars = {k: float(out[k]) for k in ["EPNL", "PNLTM", "PNLM"] if k in out}
        vectors = {
            "time":        _vec(out.get("time", [])),
            "PNL":         _vec(out.get("PNL", [])),
            "PNLT":        _vec(out.get("PNLT", [])),
            "TOB_freq":    _vec(out.get("TOB_freq", [])),
            "SPL_TOB_avg": _vec(out.get("SPL_TOB_avg", [])),
        }

    elif mid == "PsychoacousticAnnoyance_Di2016":
        out = PsychoacousticAnnoyance_Di2016(insig=sig, fs=fs,
                                              LoudnessField=p["LoudnessField"],
                                              time_skip=p["time_skip"],
                                              show=False, showPA=False)
        scalars = _pull(out, ["PAmean", "PAstd", "PAmax", "PAmin",
                               "PA5", "PA10", "PA50", "PA95"])
        vectors = {
            "time":            _vec(out.get("time", [])),
            "InstantaneousPA": _vec(out.get("InstantaneousPA", [])),
        }

    elif mid == "PsychoacousticAnnoyance_Zwicker1999":
        out = PsychoacousticAnnoyance_Zwicker1999(insig=sig, fs=fs,
                                                   LoudnessField=p["LoudnessField"],
                                                   time_skip=p["time_skip"],
                                                   show=False, showPA=False)
        scalars = _pull(out, ["PAmean", "PAstd", "PAmax", "PAmin",
                               "PA5", "PA10", "PA50", "PA95"])
        vectors = {
            "time":            _vec(out.get("time", [])),
            "InstantaneousPA": _vec(out.get("InstantaneousPA", [])),
        }

    elif mid == "PsychoacousticAnnoyance_More2010":
        out = PsychoacousticAnnoyance_More2010(insig=sig, fs=fs,
                                                LoudnessField=p["LoudnessField"],
                                                time_skip=p["time_skip"],
                                                show=False, showPA=False)
        scalars = _pull(out, ["PAmean", "PAstd", "PAmax", "PAmin",
                               "PA5", "PA10", "PA50", "PA95"])
        vectors = {
            "time":            _vec(out.get("time", [])),
            "InstantaneousPA": _vec(out.get("InstantaneousPA", [])),
        }

    elif mid == "Loudness_ECMA418_2":
        out = Loudness_ECMA418_2(sig, fs, field=p["field"], method=p["method"],
                                 time_skip=p["time_skip"], show=False)
        scalars = _pull(out, ["Nmean", "Nstd", "Nmax", "Nmin",
                               "N5", "N10", "N50", "N95"])
        vectors = {
            "time":               _vec(out.get("time", [])),
            "loudnessTDep":       _vec(out.get("InstantaneousLoudness", [])),
            "bandCentreFreqs":    _vec(out.get("barkAxis", [])),
            "specLoudnessPowAvg": _vec(out.get("SpecificLoudness_powavg", [])),
        }

    elif mid == "Roughness_ECMA418_2":
        out = Roughness_ECMA418_2(sig, fs, field=p["field"], method=p["method"],
                                  time_skip=p["time_skip"], show=False)
        scalars = _pull(out, ["Rmean", "Rstd", "Rmax", "Rmin",
                               "R5", "R10", "R50", "R90"])
        vectors = {
            "time":             _vec(out.get("time", [])),
            "roughnessTDep":    _vec(out.get("InstantaneousRoughness", [])),
            "bandCentreFreqs":  _vec(out.get("barkAxis", [])),
            "specRoughnessAvg": _vec(out.get("SpecificRoughness", [])),
        }

    elif mid == "Tonality_ECMA418_2":
        out = Tonality_ECMA418_2(sig, fs, field=p["field"], method=p["method"],
                                 time_skip=p["time_skip"], show=False)
        scalars = _pull(out, ["Tmean", "Tstd", "Tmax", "Tmin",
                               "T5", "T10", "T50", "T95"])
        vectors = {
            "time":            _vec(out.get("timeOut", out.get("time", []))),
            "tonalityTDep":    _vec(out.get("tonalityTDep", [])),
            "bandCentreFreqs": _vec(out.get("bandCentreFreqs", [])),
            "specTonalityAvg": _vec(out.get("specTonalityAvg", [])),
        }

    else:
        raise ValueError(f"Unknown metric: {mid}")

    return {"status": "OK", "scalars": scalars, "vectors": vectors}


# ── run all Python metrics ─────────────────────────────────────────────────────

def run_all_python() -> dict[str, dict]:
    print(f"\n{'='*65}")
    print("  SQAT4PY -- FINAL_VAL_MALR: running all Python metrics")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    if not _ECMA_OK:
        print("  [!] sottek_hearing_model not found -- ECMA-418-2 metrics skipped")
    print(f"{'='*65}")

    python: dict[str, dict] = {}

    for mdef in METRICS:
        mid = mdef["id"]
        if not mdef["available"]:
            print(f"\n  [{mid}]  SKIPPED")
            python[mid] = {}
            continue

        print(f"\n  [{mid}]")
        python[mid] = {}

        for case_name, (fname, dBFS_val) in mdef["audio_cases"].items():
            print(f"    {case_name:<12s} ({fname}) ... ", end="", flush=True)
            try:
                result = _run_one(mdef, fname, dBFS_val)
                python[mid][case_name] = result
                val = result["scalars"].get(mdef["main_scalar"], float("nan"))
                print(f"OK  {mdef['main_scalar']}={val:.4f} {mdef['unit']}")
            except Exception:
                err = traceback.format_exc()
                python[mid][case_name] = {"status": "ERROR", "error": err,
                                          "scalars": {}, "vectors": {}}
                print(f"FAILED: {err.splitlines()[-1]}")

    payload = {
        "implementation": "python",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "dBFS_default": 94.0,
        "results": python,
    }
    PY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Python JSON -> {PY_JSON}")

    return python


# ── figure generation (unchanged) ─────────────────────────────────────────────

_EXTRA_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#bcbd22"]
_CASE_LABEL   = {"reference": "reference", "tester": "tester", "a320": "A320"}


def generate_figure(metric: str, cfg: dict,
                    p_data: dict, m_data: dict,
                    cases: list[str], out_path: Path) -> None:
    has_spec = cfg["spec_key"] is not None
    n_rows   = 2 if has_spec else 1
    n_cols   = len(cases)
    fig_w    = 4.8 * n_cols
    fig_h    = 3.2 * n_rows + 0.4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for ci, case in enumerate(cases):
        m_entry = m_data.get(case, {})
        m_ok    = m_entry.get("status") == "OK"
        m_vecs  = m_entry.get("vectors", {}) if m_ok else {}
        m_t     = np.asarray(m_vecs.get("time", []))
        m_y     = np.asarray(m_vecs.get(cfg["time_key"], []))

        p_entry = p_data.get(case, {})
        p_ok    = p_entry.get("status") == "OK"
        p_vecs  = p_entry.get("vectors", {}) if p_ok else {}
        p_t     = np.asarray(p_vecs.get("time", []))
        p_y     = np.asarray(p_vecs.get(cfg["time_key"], []))

        ax = axes[0, ci]
        ax.set_title(_CASE_LABEL.get(case, case), fontsize=9, pad=3)

        if m_t.size > 0 and m_y.size > 0:
            ax.plot(m_t, m_y, color="steelblue", lw=1.6, label="MATLAB", zorder=3)
        if p_t.size > 0 and p_y.size > 0:
            ax.plot(p_t, p_y, color="tomato", lw=1.4, ls="--", label="Python", zorder=4)

        for ek, el, ec in zip(cfg["extra_time_keys"], cfg["extra_labels"], _EXTRA_COLORS):
            ey = np.asarray(m_vecs.get(ek, []))
            if ey.size > 0 and m_t.size > 0:
                ax.plot(m_t[:len(ey)], ey, color=ec, lw=1.0, ls=":", label=el)
            py_ey = np.asarray(p_vecs.get(ek, []))
            if py_ey.size > 0 and p_t.size > 0:
                ax.plot(p_t[:len(py_ey)], py_ey, color=ec, lw=1.0, ls="-.", label=f"{el} (Py)")

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(cfg["time_label"], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.4, alpha=0.5)
        if ci == 0:
            ax.legend(fontsize=6.5, loc="best")

        if has_spec:
            ax2 = axes[1, ci]
            m_xa = np.asarray(m_vecs.get(cfg["spec_axis"], []))
            m_sp = np.asarray(m_vecs.get(cfg["spec_key"],  []))
            if m_xa.size > 0 and m_sp.size > 0:
                ax2.plot(m_xa, m_sp, color="steelblue", lw=1.6, label="MATLAB")
            p_xa = np.asarray(p_vecs.get(cfg["spec_axis"], []))
            p_sp = np.asarray(p_vecs.get(cfg["spec_key"],  []))
            if p_xa.size > 0 and p_sp.size > 0:
                ax2.plot(p_xa, p_sp, color="tomato", lw=1.4, ls="--", label="Python")
            if cfg["spec_xlog"] and m_xa.size > 0 and m_xa.min() > 0:
                ax2.set_xscale("log")
            ax2.set_xlabel(cfg.get("spec_xlabel", ""), fontsize=8)
            ax2.set_ylabel(cfg["spec_label"],           fontsize=8)
            ax2.tick_params(labelsize=7)
            ax2.grid(True, lw=0.4, alpha=0.5)
            if ci == 0:
                ax2.legend(fontsize=6.5, loc="best")

    plt.suptitle(metric.replace("_", " "), fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── build comparison rows ──────────────────────────────────────────────────────

def build_rows(python: dict[str, dict], matlab: dict | None) -> list[dict]:
    """
    For each (metric, case):
      - One row per scalar in mdef["scalars"]  (mean, std) with relative % error.
      - One NMAE row (time-series average relative error, %) computed from the
        full instantaneous time series (requires MATLAB vectors; skipped when
        matlab is None).
    """
    rows: list[dict] = []

    for mdef in METRICS:
        mid      = mdef["id"]
        # Per-metric scalar tolerances (the global TOL_* are only a fallback
        # default). These drive both the mean/std verdicts and the NMAE verdict.
        tol_p, tol_f = mdef.get("tol", (TOL_PASS, TOL_FAIL))
        py_mid   = python.get(mid, {})
        mat_mid  = (matlab or {}).get(mid, {})

        for case_name in mdef["audio_cases"]:
            py_entry  = py_mid.get(case_name, {})
            py_ok     = py_entry.get("status") == "OK"
            py_sc     = py_entry.get("scalars", {}) if py_ok else {}

            mat_sc: dict = {}
            mat_entry = mat_mid.get(case_name, {})
            if mat_entry.get("status") == "OK":
                mat_sc = mat_entry.get("scalars", {})

            # ── scalar rows (mean + std) ───────────────────────────────────────
            for scalar in mdef["scalars"]:
                py_val = float("nan")
                if scalar in py_sc:
                    raw = py_sc[scalar]
                    py_val = float(raw[0]) if isinstance(raw, list) else float(raw)

                mat_val: float | None = None
                if scalar in mat_sc:
                    raw = mat_sc[scalar]
                    mat_val = float(raw[0]) if isinstance(raw, list) else float(raw)

                theory_val = (mdef["theory_ref"].get(scalar)
                              if case_name == "reference" else None)

                if mat_val is not None:
                    ref_val, ref_src = mat_val, "MATLAB"
                elif theory_val is not None:
                    ref_val, ref_src = theory_val, "Theory"
                else:
                    ref_val, ref_src = None, "n/a"

                abs_err = float("nan")
                rel_pct = float("nan")
                verdict = "N/A"
                if ref_val is not None and not np.isnan(py_val):
                    abs_err = py_val - ref_val
                    if ref_val != 0 and abs(ref_val) >= 5e-5:
                        rel_pct = 100.0 * abs_err / ref_val
                    elif abs(abs_err) < 5e-5:
                        rel_pct = 0.0
                    if not np.isnan(rel_pct):
                        a = abs(rel_pct)
                        verdict = "PASS" if a < tol_p else ("WARN" if a < tol_f else "FAIL")

                rows.append({
                    "metric":       mid,
                    "metric_label": mdef["label"],
                    "group":        mdef["group"],
                    "case":         case_name,
                    "case_label":   CASE_LABEL.get(case_name, case_name),
                    "scalar":       scalar,
                    "unit":         mdef["unit"],
                    "py_val":       py_val,
                    "mat_val":      mat_val,
                    "ref_val":      ref_val,
                    "ref_src":      ref_src,
                    "abs_err":      abs_err,
                    "rel_pct":      rel_pct,
                    "tol_p":        tol_p,
                    "tol_f":        tol_f,
                    "verdict":      verdict,
                    "is_primary":   scalar == mdef["main_scalar"],
                    "py_ok":        py_ok,
                })

            # ── NMAE row (time-series average relative error, %) ───────────────
            # An additive error on the full instantaneous curve, read on the
            # same scale as the mean/std relative errors, so it reuses the
            # metric's scalar PASS/FAIL band (tol_p, tol_f).
            ts_key = METRIC_CFG.get(mid, {}).get("time_key")
            if ts_key is None:
                continue

            py_vecs  = py_entry.get("vectors",  {}) if py_ok else {}
            mat_vecs = mat_entry.get("vectors", {}) if mat_entry.get("status") == "OK" else {}

            py_ts  = py_vecs.get(ts_key,  [])
            ref_ts = mat_vecs.get(ts_key, [])

            # Only compute NMAE when the MATLAB time series is available.
            if not ref_ts:
                continue

            # Pass the time vectors so the two series are compared at matching
            # instants (falls back to index alignment when times are absent).
            nmae = compute_nmae_pct(py_ts, ref_ts,
                                    py_vecs.get("time", []),
                                    mat_vecs.get("time", []))
            nmae_verdict = "N/A"
            if np.isfinite(nmae):
                a = abs(nmae)
                nmae_verdict = ("PASS" if a < tol_p
                                else "WARN" if a < tol_f else "FAIL")

            rows.append({
                "metric":        mid,
                "metric_label":  mdef["label"],
                "group":         mdef["group"],
                "case":          case_name,
                "case_label":    CASE_LABEL.get(case_name, case_name),
                "scalar":        "NMAE",
                "unit":          "%",
                "py_val":        nmae,
                "mat_val":       None,
                "ref_val":       None,
                "ref_src":       "time-avg |err| / |ref|",
                "abs_err":       float("nan"),
                "rel_pct":       nmae,
                "tol_p":         tol_p,
                "tol_f":         tol_f,
                "verdict":       nmae_verdict,
                "is_primary":    False,
                "is_nmae":       True,
                "py_ok":         py_ok,
            })

    return rows


# ── CSV writer ─────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], csv_path: Path) -> None:
    import csv

    def _fmt(v, nd=6):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.{nd}f}"

    fields = ["metric", "case", "scalar", "unit",
              "python_value", "matlab_value", "ref_source",
              "abs_error", "rel_error_pct", "tol_pass", "tol_fail", "status"]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if r.get("is_nmae"):
                w.writerow({
                    "metric":        r["metric"],
                    "case":          r["case"],
                    "scalar":        "NMAE",
                    "unit":          "%",
                    "python_value":  _fmt(r["py_val"], nd=4),
                    "matlab_value":  "n/a (time series)",
                    "ref_source":    r["ref_src"],
                    "abs_error":     "N/A",
                    "rel_error_pct": _fmt(r["rel_pct"], nd=4),
                    "tol_pass":      f"<{r['tol_p']}%",
                    "tol_fail":      f">={r['tol_f']}%",
                    "status":        r["verdict"],
                })
            else:
                w.writerow({
                    "metric":        r["metric"],
                    "case":          r["case"],
                    "scalar":        r["scalar"],
                    "unit":          r["unit"],
                    "python_value":  _fmt(r["py_val"]),
                    "matlab_value":  _fmt(r["mat_val"]),
                    "ref_source":    r["ref_src"],
                    "abs_error":     _fmt(r["abs_err"]),
                    "rel_error_pct": _fmt(r["rel_pct"], nd=4),
                    "tol_pass":      f"<{r['tol_p']}%",
                    "tol_fail":      f">={r['tol_f']}%",
                    "status":        r["verdict"],
                })


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def _tex(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
             .replace("#", r"\#").replace("{", r"\{").replace("}", r"\}"))

def _tf(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{nd}f}"

def _tpct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\text{--}"
    sign = "+" if v >= 0 else ""
    return rf"{sign}{v:.3f}\%"

def _tverd(v: str) -> str:
    c = {"PASS": "passgreen", "WARN": "warnyellow", "FAIL": "failred"}.get(v)
    return rf"\cellcolor{{{c}}}\textbf{{{v}}}" if c else "--"

_PREAMBLE = r"""\documentclass[a4paper,10pt]{article}
\usepackage[top=2.2cm,bottom=2.2cm,left=2cm,right=2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,graphicx,float,xcolor,colortbl,caption,array,hyperref,microtype}
\definecolor{passgreen} {HTML}{C6EFCE}
\definecolor{warnyellow}{HTML}{FFEB9C}
\definecolor{failred}   {HTML}{FFC7CE}
\definecolor{headblue}  {HTML}{2563EB}
\definecolor{rowalt}    {HTML}{F8FAFC}
\hypersetup{colorlinks=true, linkcolor=headblue}
\captionsetup{font=small, labelfont=bf, skip=4pt}
\setlength{\parindent}{0pt}
\pagestyle{plain}
"""


def build_latex_table(mdef: dict, rows: list[dict], has_matlab: bool) -> str:
    """Per-metric results table — values, errors and verdicts only.

    Acceptance tolerances are reported separately by build_tol_table(), so
    they no longer appear in this table's caption or as a column.
    """
    mid         = mdef["id"]
    metric_rows = [r for r in rows if r["metric"] == mid]

    # 7 columns: Case | Stat | Python | <ref> | Abs err | Rel err | Status
    col_spec = r"l l r r r r c"
    ref_col  = "MATLAB" if has_matlab else "Theory"
    header   = (rf"\textbf{{Case}} & \textbf{{Stat}} & \textbf{{Python}} & \textbf{{{ref_col}}} "
                r"& \textbf{Abs err} & \textbf{Rel err (\%)} & \textbf{Status} \\")

    L: list[str] = []
    A = L.append

    A(r"\begin{longtable}{" + col_spec + "}")
    A(r"\caption{" + _tex(mdef["label"]) + rf"  [{_tex(mdef['unit'])}] "
      rf"— Python vs {ref_col} per audio case. "
      r"Acceptance tolerances are listed in the following table.} \\")
    A(r"\toprule")
    A(header)
    A(r"\midrule\endfirsthead")
    A(r"\toprule " + header + r"\midrule\endhead")
    A(r"\midrule\multicolumn{7}{r}{\small\itshape continued\ldots}\\\endfoot")
    A(r"\bottomrule\endlastfoot")

    prev_case = None
    for i, r in enumerate(metric_rows):
        first_in_case = r["case"] != prev_case
        if first_in_case:
            if prev_case is not None:
                A(r"\midrule")
            prev_case = r["case"]

        bg = r"\rowcolor{rowalt}" if i % 2 == 0 else ""
        cl = _tex(r["case_label"]) if first_in_case else ""

        if r.get("is_nmae"):
            nmae_s = _tf(r["py_val"], nd=3)
            A(rf"{bg}{cl} & \textit{{NMAE}} "
              rf"& \multicolumn{{3}}{{c}}{{\textit{{time-avg $|$err$|$ / $|$ref$|$}}}} "
              rf"& ${nmae_s}\%$ & {_tverd(r['verdict'])} \\")
        else:
            sk = (rf"\textbf{{{_tex(r['scalar'])}}}" if r["is_primary"]
                  else _tex(r["scalar"]))
            if has_matlab or r["ref_src"] == "Theory":
                A(rf"{bg}{cl} & {sk} & {_tf(r['py_val'])} & {_tf(r['mat_val'])} "
                  rf"& {_tf(r['abs_err'])} & ${_tpct(r['rel_pct'])}$ "
                  rf"& {_tverd(r['verdict'])} \\")
            else:
                A(rf"{bg}{cl} & {sk} & {_tf(r['py_val'])} & -- "
                  rf"& -- & -- & -- \\")

    A(r"\end{longtable}")
    return "\n".join(L)


def build_tol_table(mdef: dict) -> str:
    """Per-metric acceptance-tolerance table (separate from the results).

    The scalar mean/std rows and the NMAE time-series error share a single
    per-metric relative-error band.
    """
    tol_p, tol_f = mdef.get("tol", (TOL_PASS, TOL_FAIL))
    scalars_str  = ", ".join(mdef["scalars"])

    L: list[str] = []
    A = L.append

    A(r"\begin{longtable}{l l l}")
    A(r"\caption{Acceptance tolerances — " + _tex(mdef["label"]) +
      r". The mean/std scalars and the NMAE time-series error share a fixed "
      r"per-metric relative-error band (see Methodology).} \\")
    hdr = (r"\textbf{Quantity} & \textbf{PASS} ($\tau_\mathrm{pass}$) "
           r"& \textbf{FAIL} ($\tau_\mathrm{fail}$) \\")
    A(r"\toprule")
    A(hdr)
    A(r"\midrule\endfirsthead")
    A(r"\toprule " + hdr + r"\midrule\endhead")
    A(r"\bottomrule\endlastfoot")

    A(rf"Scalars ({_tex(scalars_str)}) & $<{tol_p:g}\%$ & $\geq{tol_f:g}\%$ \\")
    A(rf"NMAE (time series) & $<{tol_p:g}\%$ & $\geq{tol_f:g}\%$ \\")

    A(r"\end{longtable}")
    return "\n".join(L)


def build_methodology_section() -> str:
    """Explanatory page describing the scalar (mean/std) and NMAE errors."""
    L: list[str] = []
    A = L.append

    A(r"\section{Methodology and acceptance criteria}")

    A(r"\subsection{Scalar errors (mean and standard deviation)}")
    A(r"Each metric produces an instantaneous time series whose \emph{mean} "
      r"and \emph{standard deviation} are compared against the reference "
      r"implementation (MATLAB, or the theoretical value for calibrated "
      r"reference signals). The agreement is quantified by the relative error")
    A(r"\begin{equation*}"
      r"\varepsilon_\mathrm{rel}=100\,\frac{x_\mathrm{py}-x_\mathrm{ref}}"
      r"{x_\mathrm{ref}}\quad[\%],"
      r"\end{equation*}")
    A(r"evaluated separately for the mean and the standard deviation. When "
      r"$|x_\mathrm{ref}|<5\times10^{-5}$ the relative error is ill-defined "
      r"and is reported as $0$ if the absolute error is also negligible. The "
      r"verdict uses two per-metric thresholds "
      r"$(\tau_\mathrm{pass},\tau_\mathrm{fail})$:")
    A(r"\begin{equation*}"
      r"\textbf{PASS}\ \text{if}\ |\varepsilon_\mathrm{rel}|<\tau_\mathrm{pass},"
      r"\qquad"
      r"\textbf{WARN}\ \text{if}\ \tau_\mathrm{pass}\le|\varepsilon_\mathrm{rel}|"
      r"<\tau_\mathrm{fail},\qquad"
      r"\textbf{FAIL}\ \text{otherwise.}"
      r"\end{equation*}")
    A(r"These thresholds are not taken from any standard; they are engineering "
      r"acceptance bands chosen to reflect how reproducibly each metric can be "
      r"expected to match the reference implementation. The decisive factor is "
      r"the \emph{numerical conditioning} of the computation, not the "
      r"perceptual importance of the metric, which sorts the metrics into three "
      r"tiers:")
    A(r"\begin{itemize}")
    A(r"\item \textbf{Tight — PASS $<1\%$, FAIL $\geq2\%$ "
      r"(loudness, sharpness, EPNL).} These are deterministic, well-conditioned "
      r"computations with no envelope or modulation processing — sharpness, for "
      r"instance, is essentially a weighted centroid of the specific-loudness "
      r"pattern. Given the same input the two implementations agree to well "
      r"under $1\%$, so a larger discrepancy genuinely indicates a problem.")
    A(r"\item \textbf{Medium — PASS $<2\%$, FAIL $\geq5\%$ "
      r"(roughness, fluctuation strength, tonality, ECMA-418-2).} These rely on "
      r"temporal-envelope and modulation analysis (band-pass filtering of the "
      r"envelope, modulation-frequency weighting, frame-based spectral "
      r"estimation), which is sensitive to frame length, filter design and FFT "
      r"details. Cross-implementation differences of a few percent are expected "
      r"and benign, so a $1\%$ band would flag harmless numerical noise as a "
      r"failure.")
    A(r"\item \textbf{Loose — PASS $<5\%$, FAIL $\geq10\%$ "
      r"(psychoacoustic annoyance: Di, Zwicker, More).} These models combine "
      r"loudness, sharpness, roughness and fluctuation strength (and tonality), "
      r"so the errors of the underlying components accumulate.")
    A(r"\end{itemize}")
    A(r"The exact PASS/FAIL values for each metric are repeated in its "
      r"tolerance table.")

    A(r"\subsection{Time-series error (NMAE)}")
    A(r"To compare the full instantaneous curves rather than only their "
      r"summary statistics, the \emph{normalised mean absolute error} averages "
      r"the frame-by-frame absolute difference between the two curves and "
      r"expresses it as a percentage of the mean reference level:")
    A(r"\begin{equation*}"
      r"\mathrm{NMAE}=100\,"
      r"\frac{\frac{1}{|\mathcal{F}|}\sum_{t\in\mathcal{F}}"
      r"\bigl|x_\mathrm{py}(t)-x_\mathrm{ref}(t)\bigr|}"
      r"{\frac{1}{|\mathcal{F}|}\sum_{t\in\mathcal{F}}"
      r"\bigl|x_\mathrm{ref}(t)\bigr|}\quad[\%],"
      r"\end{equation*}")
    A(r"where $\mathcal{F}$ is the set of overlapping frames. The Python "
      r"series is resampled onto the MATLAB time base over the overlapping "
      r"interval, so frames are compared at matching instants. Note that the "
      r"\emph{signed} mean error normalised by the reference level reduces "
      r"algebraically to the mean-scalar relative error already reported; "
      r"taking the \emph{absolute} value per frame prevents positive and "
      r"negative deviations from cancelling, so NMAE measures how far apart "
      r"the two curves run on average. It is read on the same scale as the "
      r"scalar relative errors and uses the same per-metric PASS/FAIL band.")
    return "\n".join(L)


def build_config_table(metrics_list: list[dict]) -> str:
    L: list[str] = []
    A = L.append
    A(r"\begin{longtable}{p{3.2cm} p{6.0cm} p{6.0cm}}")
    A(r"\caption{Run configuration — audio cases (with dBFS) and function "
      r"parameters. Acceptance tolerances are reported per metric in the "
      r"respective sections.} \\")
    hdr = (r"\textbf{Metric} & \textbf{Audio cases (dBFS)} & "
           r"\textbf{Parameters} \\")
    A(r"\toprule")
    A(hdr)
    A(r"\midrule\endfirsthead")
    A(r"\toprule " + hdr + r"\midrule\endhead")
    A(r"\bottomrule\endlastfoot")

    for mdef in metrics_list:
        cases = "; ".join(
            f"{c}: {fn} ({'auto' if d is None else f'{d:g}'})"
            for c, (fn, d) in mdef["audio_cases"].items()
        )
        params = ", ".join(f"{k}={v}" for k, v in mdef["params"].items()) or "—"
        A(rf"{_tex(mdef['label'])} & {_tex(cases)} & {_tex(params)} \\")
    A(r"\end{longtable}")
    return "\n".join(L)


def write_latex(rows: list[dict], fig_files: dict[str, Path],
                has_matlab: bool, tex_path: Path) -> None:
    ts  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ref = "MATLAB" if has_matlab else "Theoretical references"

    L: list[str] = [_PREAMBLE, r"\begin{document}"]
    A = L.append

    A(r"\begin{titlepage}\centering\vspace*{2.5cm}")
    A(r"{\color{headblue}\rule{\linewidth}{2pt}}\vspace{0.4cm}")
    A(r"{\Huge\bfseries SQAT4PY\\[0.3em]"
      r"\Large Full Metrics Validation Report}")
    A(r"\vspace{0.4cm}{\color{headblue}\rule{\linewidth}{2pt}}\vspace{1.5cm}")
    A(r"{\large\begin{tabular}{ll}")
    A(rf"\textbf{{Generated:}} & {_tex(ts)} \\[4pt]")
    A(rf"\textbf{{Reference:}} & {_tex(ref)} \\[4pt]")
    A(r"\textbf{Error measures:} & mean/std relative error and NMAE "
      r"(time-series average relative error) \\[4pt]")
    A(r"\textbf{Acceptance band:} & per-metric relative-error band "
      r"(shared by scalars and NMAE) \\[4pt]")
    A(rf"\textbf{{ECMA metrics:}} & "
      rf"{'Available' if _ECMA_OK else 'SKIPPED (sottek\\_hearing\\_model not installed)'} \\")
    A(r"\end{tabular}}\vfill"
      r"{\small\color{gray}Auto-generated by \texttt{validation/FINAL\_VAL\_MALR.py}}")
    A(r"\end{titlepage}\newpage\tableofcontents\newpage")

    A(build_methodology_section())
    A(r"\clearpage")

    A(r"\section{Run Configuration}")
    A(build_config_table([m for m in METRICS if m["available"]]))
    A(r"\clearpage")

    groups = [("classic",   "Classic Metrics"),
              ("annoyance", "Psychoacoustic Annoyance"),
              ("ecma",      "ECMA-418-2 / Sottek Hearing Model")]

    for group_id, group_title in groups:
        group_mdefs = [m for m in METRICS if m["group"] == group_id and m["available"]]
        if not group_mdefs:
            continue
        A(rf"\section{{{group_title}}}")

        for mdef in group_mdefs:
            mid = mdef["id"]
            A(rf"\subsection{{{_tex(mdef['label'])}}}")

            fp = fig_files.get(mid)
            if fp and fp.exists():
                A(r"\begin{figure}[H]\centering")
                A(rf"\includegraphics[width=\textwidth]{{figures/{fp.name}}}")
                A(rf"\caption{{{_tex(mid).replace(r'\_', ' ')} "
                  r"— MATLAB (solid blue) vs Python (dashed red)}}")
                A(r"\end{figure}")

            A(build_latex_table(mdef, rows, has_matlab))
            A(build_tol_table(mdef))
            A(r"\clearpage")

    A(r"\end{document}")
    tex_path.write_text("\n".join(L), encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # The console summary contains non-ASCII glyphs (em dashes). On a default
    # Windows console (cp1252) printing them raises UnicodeEncodeError, so
    # force UTF-8 output when the stream supports reconfiguration.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    _parse_tolerances(sys.argv[1:])
    print(f"  Tolerances : per-metric relative-error band "
          f"(default PASS < {TOL_PASS:g}%   FAIL >= {TOL_FAIL:g}%);")
    print(f"               shared by mean/std scalars and NMAE.")

    python = run_all_python()

    matlab: dict | None = None
    if MAT_JSON.exists():
        with open(MAT_JSON, encoding="utf-8") as fh:
            raw = json.load(fh)
        matlab = raw.get("results", raw)
        print(f"\n  MATLAB results loaded <- {MAT_JSON}")
    else:
        print(f"\n  MATLAB results NOT found -- using theoretical references where available.")
        print(f"  (NMAE rows require MATLAB vectors and will be omitted)")

    rows = build_rows(python, matlab)

    print(f"\n  Generating figures -> {FIG_DIR}")
    fig_files: dict[str, Path] = {}

    for mdef in METRICS:
        mid = mdef["id"]
        if not mdef["available"] or not python.get(mid):
            continue
        cfg = METRIC_CFG.get(mid)
        if cfg is None:
            continue

        p_data    = python[mid]
        m_data    = (matlab or {}).get(mid, {})
        all_cases = list(dict.fromkeys(
            list(mdef["audio_cases"].keys()) +
            [c for c in m_data if m_data[c].get("status") == "OK"]
        ))

        print(f"    [{mid}]  cases={all_cases} ... ", end="", flush=True)
        out_path = FIG_DIR / f"{mid}_overview.png"
        try:
            generate_figure(mid, cfg, p_data, m_data, all_cases, out_path)
            fig_files[mid] = out_path
            print("OK")
        except Exception as exc:
            print(f"FAILED: {exc}")

    write_csv(rows, CSV_FILE)
    print(f"\n  CSV   -> {CSV_FILE}")

    write_latex(rows, fig_files, has_matlab=matlab is not None, tex_path=TEX_FILE)
    print(f"  LaTeX -> {TEX_FILE}")

    # ── console summary ────────────────────────────────────────────────────────
    scalar_rows = [r for r in rows if not r.get("is_nmae")]
    nmae_rows   = [r for r in rows if r.get("is_nmae")]

    print(f"\n{'='*70}")
    print("  SCALAR SUMMARY  (mean / std, reference signals only)")
    print(f"  {'Metric':<38s}  {'Stat':<6s}  {'Python':>8s}  {'Ref':>8s}  {'Err%':>8s}  Status")
    print(f"  {'-'*65}")
    for r in scalar_rows:
        if r["case"] != "reference":
            continue
        py_s  = f"{r['py_val']:.4f}" if not np.isnan(r["py_val"]) else "N/A"
        ref_s = f"{r['ref_val']:.4f}" if r["ref_val"] is not None else "N/A"
        rel_s = f"{r['rel_pct']:+.2f}%" if not np.isnan(r["rel_pct"]) else "N/A"
        print(f"  {r['metric_label']:<38s}  {r['scalar']:<6s}  "
              f"{py_s:>8s}  {ref_s:>8s}  {rel_s:>8s}  {r['verdict']}")

    if nmae_rows:
        print(f"\n{'='*80}")
        print("  NMAE SUMMARY  (all audio cases — time-avg |error| / |ref|, units: %)")
        print(f"  {'Metric':<38s}  {'Case':<16s}  {'NMAE%':>8s}  "
              f"{'tol_p':>7s}  {'tol_f':>7s}  Status")
        print(f"  {'-'*75}")
        for r in nmae_rows:
            nmae_s = f"{r['py_val']:.3f}" if np.isfinite(r["py_val"]) else "N/A"
            tp_s   = f"{r['tol_p']:g}%"
            tf_s   = f"{r['tol_f']:g}%"
            print(f"  {r['metric_label']:<38s}  {r['case_label']:<16s}  "
                  f"{nmae_s:>8s}  {tp_s:>7s}  {tf_s:>7s}  {r['verdict']}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
