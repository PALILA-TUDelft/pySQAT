"""
metrics_epnl.py
===============
Python implementation of EPNL (Effective Perceived Noise Level) per
FAR Part 36 / ICAO Annex 16.

Direct translation of the MATLAB implementation in
  original_matlab/psychoacoustic_metrics/EPNL_FAR_Part36/

References
----------
[1] FAR 14 CFR Parts 36 and 91, Docket No. FAA-2003-16526; Amendment 36-26
[2] ICAO Annex 16, Environmental Protection, Volume I, 8th Ed.
[3] ICAO Doc 9501, Environmental Technical Manual, Volume I, 2nd Ed.

Usage
-----
  from metrics_epnl import EPNL_FAR_Part36
  out = EPNL_FAR_Part36(insig, fs, dt=0.5, threshold=10, show=False)
  out = EPNL_FAR_Part36(spl_tob_matrix, method=0, dt=0.5)
"""

from __future__ import annotations
import warnings
import numpy as np
from numpy.typing import NDArray

__all__ = ["EPNL_FAR_Part36"]

# ── Noys formulating table (Table A36-3 / A2-3) ─────────────────────────────
# Columns: band, freq, SPLa, SPLb, SPLc, SPLd, SPLe, Mb, Mc, Md, Me
_NOY_TABLE = np.array([
    [ 1,   50, 91.0, 64.0, 52.0, 49.0, 55.0, 0.043478, 0.030103, 0.07952,  0.058098],
    [ 2,   63, 85.9, 60.0, 51.0, 44.0, 51.0, 0.040570, 0.030103, 0.06816,  0.058098],
    [ 3,   80, 87.3, 56.0, 49.0, 39.0, 46.0, 0.036831, 0.030103, 0.06816,  0.052288],
    [ 4,  100, 79.9, 53.0, 47.0, 34.0, 42.0, 0.036831, 0.030103, 0.05964,  0.047534],
    [ 5,  125, 79.8, 51.0, 46.0, 30.0, 39.0, 0.035336, 0.030103, 0.053013, 0.043573],
    [ 6,  160, 76.0, 48.0, 45.0, 27.0, 36.0, 0.033333, 0.030103, 0.053013, 0.043573],
    [ 7,  200, 74.0, 46.0, 43.0, 24.0, 33.0, 0.033333, 0.030103, 0.053013, 0.040221],
    [ 8,  250, 74.9, 44.0, 42.0, 21.0, 30.0, 0.032051, 0.030103, 0.053013, 0.037349],
    [ 9,  315, 94.6, 42.0, 41.0, 18.0, 27.0, 0.030675, 0.030103, 0.053013, 0.034859],
    [10,  400,  np.inf, 40.0, 40.0, 16.0, 25.0, 0.030103, 0.0,  0.053013, 0.034859],
    [11,  500,  np.inf, 40.0, 40.0, 16.0, 25.0, 0.030103, 0.0,  0.053013, 0.034859],
    [12,  630,  np.inf, 40.0, 40.0, 16.0, 25.0, 0.030103, 0.0,  0.053013, 0.034859],
    [13,  800,  np.inf, 40.0, 40.0, 16.0, 25.0, 0.030103, 0.0,  0.053013, 0.034859],
    [14, 1000,  np.inf, 40.0, 40.0, 16.0, 25.0, 0.030103, 0.0,  0.053013, 0.034859],
    [15, 1250,  np.inf, 38.0, 38.0, 15.0, 23.0, 0.030103, 0.0,  0.05964,  0.034859],
    [16, 1600,  np.inf, 34.0, 34.0, 12.0, 21.0, 0.02996,  0.0,  0.053013, 0.040221],
    [17, 2000,  np.inf, 32.0, 32.0,  9.0, 18.0, 0.02996,  0.0,  0.053013, 0.037349],
    [18, 2500,  np.inf, 30.0, 30.0,  5.0, 15.0, 0.02996,  0.0,  0.047712, 0.034859],
    [19, 3150,  np.inf, 29.0, 29.0,  4.0, 14.0, 0.02996,  0.0,  0.047712, 0.034859],
    [20, 4000,  np.inf, 29.0, 29.0,  5.0, 14.0, 0.02996,  0.0,  0.053013, 0.034859],
    [21, 5000,  np.inf, 30.0, 30.0,  6.0, 15.0, 0.02996,  0.0,  0.053013, 0.034859],
    [22, 6300,  np.inf, 31.0, 31.0, 10.0, 17.0, 0.02996,  0.0,  0.06816,  0.037349],
    [23, 8000, 44.3,   37.0, 34.0, 17.0, 23.0, 0.042285, 0.02996, 0.07952, 0.037349],
    [24,10000, 50.7,   41.0, 37.0, 21.0, 29.0, 0.042285, 0.02996, 0.05964, 0.043573],
])

TOB_FREQS = _NOY_TABLE[:, 1].astype(float)   # 24 centre frequencies


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spl_to_noys(SPL: NDArray) -> NDArray:
    """
    Convert [nTime, 24] SPL matrix to perceived noisiness (Noys).
    Implements Table A36-3 mathematical formulation (FAR Part 36 §A36.4.7).
    """
    SPLa = _NOY_TABLE[:, 2]
    SPLb = _NOY_TABLE[:, 3]
    SPLc = _NOY_TABLE[:, 4]
    SPLd = _NOY_TABLE[:, 5]
    SPLe = _NOY_TABLE[:, 6]
    Mb   = _NOY_TABLE[:, 7]
    Mc   = _NOY_TABLE[:, 8]
    Md   = _NOY_TABLE[:, 9]
    Me   = _NOY_TABLE[:, 10]

    nT, nF = SPL.shape
    nn = np.zeros((nT, nF))

    for i in range(nF):
        s = SPL[:, i]
        # Region selection per band
        nn[:, i] = np.where(
            s >= SPLa[i],
            10.0 ** (Mc[i] * (s - SPLc[i])),
            np.where(
                s >= SPLb[i],
                10.0 ** (Mb[i] * (s - SPLb[i])),
                np.where(
                    s >= SPLe[i],
                    0.3 * 10.0 ** (Me[i] * (s - SPLe[i])),
                    np.where(
                        s >= SPLd[i],
                        0.1 * 10.0 ** (Md[i] * (s - SPLd[i])),
                        0.0
                    )
                )
            )
        )
    return nn


def _get_pnl(SPL: NDArray):
    """SPL [nT,24] → PN [nT], PNL [nT], PNLM, PNLM_idx."""
    nn  = _spl_to_noys(SPL)
    nmax = nn.max(axis=1)
    PN  = 0.85 * nmax + 0.15 * nn.sum(axis=1)
    PN  = np.maximum(PN, 1e-30)
    PNL = 40.0 + (10.0 / np.log10(2.0)) * np.log10(PN)
    idx = int(np.argmax(PNL))
    return PN, PNL, float(PNL[idx]), idx


def _get_pnlt(SPL: NDArray, freq_bands: NDArray, PNL: NDArray):
    """
    Compute tone-corrected PNL (PNLT) following FAR Part 36 §A36.4.8-A36.4.10.
    Steps 1-10 of the 10-step procedure.
    """
    nT, nF = SPL.shape
    # index_80 = first band index where freq >= 80 Hz (0-based)
    i80 = int(np.argmax(freq_bands >= 80.0))

    # Step 1: slopes S
    S = np.zeros((nF, nT))
    for k in range(nT):
        for i in range(i80 + 1, nF):
            S[i, k] = SPL[k, i] - SPL[k, i - 1]

    # Steps 2–3: encircle SPL where |ΔS| > 5 dB
    delS = np.zeros_like(S)
    SPLs = np.zeros_like(S)
    for k in range(nT):
        for i in range(i80, nF):
            d = abs(S[i, k] - S[i - 1, k])
            if d > 5.0:
                delS[i, k] = S[i, k]
                if S[i, k] > 0 and S[i, k] > S[i - 1, k]:
                    SPLs[i, k] = SPL[k, i]
                elif S[i, k] <= 0 and S[i - 1, k] > 0:
                    SPLs[i - 1, k] = SPL[k, i - 1]

    # Step 4: new SPL', SPLP
    SPLP = np.zeros((nF, nT))
    for k in range(nT):
        for i in range(nF):
            if SPLs[i, k] == 0.0:
                SPLP[i, k] = SPL[k, i]
            elif SPLs[i, k] > 0 and i < nF - 1:
                SPLP[i, k] = 0.5 * (SPL[k, i - 1] + SPL[k, i + 1])
            elif SPLs[i, k] > 0 and i == nF - 1:
                SPLP[i, k] = SPL[k, i - 1] + S[i - 1, k]
            elif SPLs[i, k] <= 0 and i == nF - 1:
                SPLP[i, k] = SPL[k, i]

    # Step 5: recompute slope SP (+ imaginary 25th band)
    SP = np.zeros((nF + 1, nT))
    for k in range(nT):
        for i in range(i80 + 1, nF - 1):
            SP[i, k] = SPLP[i, k] - SPLP[i - 1, k]
        SP[i80, k]   = SP[i80 + 1, k]
        SP[nF - 1, k] = SPLP[nF - 1, k] - SPLP[nF - 2, k]
        SP[nF, k]    = SP[nF - 1, k]

    # Step 6: average-of-three-adjacent slopes, SB
    SB = np.zeros((nF + 1, nT))
    for k in range(nT):
        for i in range(i80, nF - 1):
            SB[i, k] = (SP[i, k] + SP[i + 1, k] + SP[i + 2, k]) / 3.0

    # Step 7: final adjusted SPL, SPLPP
    SPLPP = np.zeros((nF, nT))
    for k in range(nT):
        SPLPP[i80, k] = SPL[k, i80]
        for i in range(i80 + 1, nF - 1):
            SPLPP[i, k] = SPLPP[i - 1, k] + SB[i - 1, k]
        SPLPP[nF - 1, k] = SPLPP[nF - 2, k] + SB[nF - 2, k]

    # Step 8: difference F = SPL - SPLPP, zero if F <= 1.5
    F = np.zeros((nF, nT))
    for k in range(nT):
        for i in range(i80, nF):
            diff = SPL[k, i] - SPLPP[i, k]
            F[i, k] = diff if diff > 1.5 else 0.0

    # Steps 9–10: tone correction C, then Cmax
    C    = np.zeros((nF, nT))
    Cmax = np.zeros(nT)
    for k in range(nT):
        for i in range(i80, nF):
            f = freq_bands[i]
            fi = F[i, k]
            if fi <= 1.5:
                c = 0.0
            elif 50 <= f < 500:
                c = fi / 3.0 - 0.5  if fi < 3.0 else (fi / 6.0 if fi < 20.0 else 10.0 / 3.0)
            elif 500 <= f <= 5000:
                c = 2*fi/3.0 - 1.0 if fi < 3.0 else (fi / 3.0 if fi < 20.0 else 20.0 / 3.0)
            else:  # > 5000
                c = fi / 3.0 - 0.5  if fi < 3.0 else (fi / 6.0 if fi < 20.0 else 10.0 / 3.0)
            C[i, k] = c
        Cmax[k] = C[:, k].max()

    PNLT   = PNL + Cmax
    idx    = int(np.argmax(PNLT))
    PNLTM  = float(PNLT[idx])

    # Bandsharing adjustment (≥5 points available)
    if nT >= 5 and 2 <= idx <= nT - 3:
        Cavg = Cmax[idx - 2:idx + 3].mean()
        DeltaB = Cavg * Cmax[idx] if Cavg > Cmax[idx] else 0.0
        PNLTM += DeltaB

    return PNLT, PNLTM, idx


def _duration_correction(PNLT: NDArray, PNLTM: float, idx: int,
                          dt: float, threshold: float):
    """Duration correction D and integration bounds (t1, t2 indices)."""
    decay = PNLTM - threshold
    # t1: first index before peak where PNLT > decay
    i1_arr = np.where(PNLT[:idx + 1] > decay)[0]
    if len(i1_arr) == 0:
        i1 = 0
    else:
        i1 = int(i1_arr[0])

    # t2: first index after peak where PNLT < decay
    i2_arr = np.where(PNLT[idx:] < decay)[0]
    if len(i2_arr) == 0:
        i2 = len(PNLT) - 1
        warnings.warn(
            "The signal does not decay by more than the threshold within the "
            "available duration. An indicative EPNL value is calculated from "
            "the available duration, but should not be used for aircraft noise "
            "certification."
        )
    else:
        i2 = int(i2_arr[0]) + idx

    segment = PNLT[i1:i2 + 1]
    D = 10.0 * np.log10(np.sum(10.0 ** (segment / 10.0))) - PNLTM + 10.0 * np.log10(dt / 10.0)
    return float(D), i1, i2


def _compute_spl_dt(insig: NDArray, fs: int, dt: float = 0.5):
    """
    Compute 1/3-octave band SPL averaged over dt-second intervals.
    ob13_iso532_1 returns pressure time series (Pa), which are squared,
    averaged over dt windows, then converted to SPL re 20 µPa.

    Returns
    -------
    SPL_dt : ndarray shape (nT, 24)
    time   : ndarray shape (nT,)   – centre times of each dt interval
    freqs  : ndarray shape (24,)   – TOB centre frequencies
    """
    from sound_metrics import ob13_iso532_1

    # ob13 returns pressure time series p [N, 24] (Pa, same units as insig)
    p_tob, freqs = ob13_iso532_1(insig, fs, fmin=50.0, fmax=10000.0)
    p_tob = np.array(p_tob, dtype=float)

    I_REF = 4e-10        # (20 µPa)^2, reference pressure squared
    TINY  = 1e-12

    win = int(round(fs * dt))
    n_samples = p_tob.shape[0]
    nT  = int(np.ceil(n_samples / win))   # MATLAB: num_times = ceil(len_insig/Nbins)
    if nT < 1:
        raise ValueError(f"Signal too short for dt={dt}s averaging.")

    SPL_dt = np.zeros((nT, 24))
    for k in range(nT):
        seg = p_tob[k * win:(k + 1) * win, :]   # pressure block (last may be partial)
        # MATLAB: mean(buffer(p^2, Nbins), 1). buffer zero-pads the final block
        # to length Nbins, so the average divides the sum of squares by the FULL
        # Nbins (not by the number of available samples).
        p2_mean = np.sum(seg ** 2, axis=0) / win
        SPL_dt[k, :] = 10.0 * np.log10((p2_mean + TINY) / I_REF)

    time = (np.arange(nT) + 0.5) * dt
    return SPL_dt, time, freqs


# ── Main entry point ──────────────────────────────────────────────────────────

def EPNL_FAR_Part36(insig, fs: int = None, method: int | None = None,
                    dt: float = 0.5, threshold: float | None = 10.0,
                    show: bool = False, dBFS: float = 94.0) -> dict:
    """
    Effective Perceived Noise Level per FAR Part 36 / ICAO Annex 16.

    Parameters
    ----------
    insig : str or ndarray
        Calibrated audio signal (Pa), path to a .wav file, or a
        third-octave SPL matrix with shape ``(n_time, 24)`` when
        ``method=0``.
    fs : int, optional
        Sampling frequency (required for waveform input).
    method : {0, 1}, optional
        ``0`` uses ``insig`` directly as a third-octave SPL matrix.
        ``1`` treats ``insig`` as waveform input. If omitted, the mode is
        inferred and defaults to waveform mode.
    dt : float
        Time averaging interval in seconds (default 0.5 s per FAR Part 36).
    threshold : float
        PNLT decay for duration correction in TPNdB (default 10).
    show : bool
        If True, print summary.
    dBFS : float
        dBFS calibration for .wav files (default 94).

    Returns
    -------
    dict with keys
        EPNL    : float   Effective Perceived Noise Level (EPNdB)
        PNLTM   : float   Maximum PNLT (TPNdB)
        PNLM    : float   Maximum PNL (PNdB)
        time    : ndarray dt-averaged time vector (s)
        InstantaneousSPL : ndarray overall SPL vs time (dB)
        PNL     : ndarray Perceived Noise Level vs time (PNdB)
        PNLT    : ndarray Tone-corrected PNL vs time (TPNdB)
        TOB_freq    : ndarray 24 third-octave centre frequencies (Hz)
        SPL_TOB_avg : ndarray time-averaged TOB spectrum [24] (dB)
    """
    if threshold is None:
        threshold = 10.0

    if method is None:
        method = 1
        if not isinstance(insig, str):
            arr = np.asarray(insig)
            if arr.ndim == 2 and arr.shape[1] == 24 and fs is None:
                method = 0

    if method == 0:
        SPL_dt = np.asarray(insig, dtype=float)
        if SPL_dt.ndim != 2 or SPL_dt.shape[1] != 24:
            raise ValueError("For method=0, insig must be an (n_time, 24) third-octave SPL matrix.")
        time = (np.arange(SPL_dt.shape[0]) + 0.5) * dt
        freqs = TOB_FREQS.copy()
        instantaneous_spl = 10.0 * np.log10(
            np.sum(10.0 ** (SPL_dt / 10.0), axis=1).clip(min=1e-30)
        )
    else:
        if isinstance(insig, str):
            from utilities import wav2sig
            insig, fs = wav2sig(insig, fs, dBFS)

        if fs is None:
            raise ValueError("fs must be provided when insig is a waveform array.")

        insig = np.asarray(insig, dtype=float)
        if insig.ndim > 1:
            insig = insig.mean(axis=1)  # stereo → mono

        # MATLAB resamples the signal to 48 kHz before the 1/3-octave filtering,
        # because the ISO 532-1 filter bank is validated only at fs = 48 kHz.
        if int(fs) != 48000:
            from scipy.signal import resample_poly
            g = int(np.gcd(48000, int(fs)))
            insig = resample_poly(insig, 48000 // g, int(fs) // g)
            fs = 48000

        # 1. 1/3-octave SPL averaged over dt intervals
        SPL_dt, time, freqs = _compute_spl_dt(insig, int(fs), dt)
        instantaneous_spl = 10.0 * np.log10(
            np.sum(10.0 ** (SPL_dt / 10.0), axis=1).clip(min=1e-30)
        )

    # 2. PNL
    PN, PNL, PNLM, PNLM_idx = _get_pnl(SPL_dt)

    # 3. PNLT
    PNLT, PNLTM, PNLTM_idx = _get_pnlt(SPL_dt, freqs, PNL)

    # 4. Duration correction → EPNL
    D, i1, i2 = _duration_correction(PNLT, PNLTM, PNLTM_idx, dt, threshold)
    EPNL = PNLTM + D

    if show:
        print(f"EPNL = {EPNL:.2f} EPNdB   (PNLTM={PNLTM:.2f} TPNdB, D={D:.2f} dB)")

    # Time-averaged TOB spectrum
    spl_avg = 10.0 * np.log10(
        np.mean(10.0 ** (SPL_dt / 10.0), axis=0).clip(min=1e-30)
    )

    return {
        "EPNL":        float(EPNL),
        "PNLTM":       float(PNLTM),
        "PNLM":        float(PNLM),
        "time":        time,
        "InstantaneousSPL": instantaneous_spl,
        "PNL":         PNL,
        "PNLT":        PNLT,
        "TOB_freq":    freqs,
        "SPL_TOB_avg": spl_avg,
        "SPL_TOB_spectra": SPL_dt,
    }
