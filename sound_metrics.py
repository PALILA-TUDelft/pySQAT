
##############################
#### SOUND METRICS MODULE ####
###############################

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, resample_poly, bilinear_zpk, zpk2tf, freqz
from typing import Tuple, Sequence
import soundfile as sf
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------
#### MAIN FUNCTIONS ####
# ----------------------

def ob13_iso532_1(
        insig: np.ndarray,
        fs: int | float,
        fmin: int | float = 25,
        fmax: int | float = 12_500
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    This code employs a hard-coded implementation of a one-third octave band
    filterbank for a sampling frequency fs=48000 Hz. For this reason, if
    the input fs adopts a value different from 48 kHz, the signal is
    appropriately resampled. 

    This code was extracted from the loudness implementation from the AARAE
    toolbox refactored by the SQAT toolbox team. Therefore, it complies with
    ISO 532-1:2017 (and IEC 61260-1:2014). However, the freq range computed
    is only between 25 Hz - 12.5 kHz

    ORIGINAL MATLAB CODE:
    Do_OB13_ISO532_1.m   (Manor, 2015) (Felix Greco, 2023) [Last Update: 20/10/2023]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 2023-10-01

    Parameters
    ----------
    insig : (N,) ndarray
        Mono audio signal.
    fs : int or float
        Input sampling rate (Hz).
    fmin, fmax : float, optional
        Lowest / highest one-third-octave centre frequency to keep (Hz).
        Valid range 25 Hz – 12 500 Hz.

    Returns
    -------
    outsig : (N, nBands) ndarray
        Signal filtered into one-third-octave bands.
    fc : (nBands,) ndarray
        Centre frequencies (Hz) of the returned bands.
    """

    # 1) Argument Checks
    if fmin < 25:
        print("fmin < 25 Hz → clamped to 25 Hz")
        fmin = 25
    if fmax > 12_500:
        print("fmax > 12 500 Hz → clamped to 12 500 Hz")
        fmax = 12_500
    if insig.ndim != 1:
        raise ValueError("insig must be mono (1-D)")

    # 2) Resampling to 48 kHz (if needed)
    if fs != 48_000:
        gcd = np.gcd(int(fs), 48_000)
        up   = 48_000 // gcd
        down = int(fs) // gcd
        insig = resample_poly(insig, up, down)
        fs = 48_000
        print("Resampled to 48 kHz to match hard-coded filter coefficients")

    N = len(insig)

    # 3) Hard-coded IIR Coefficients

    # base numerators / denominators (three identical biquads)
    br = np.array([[1, 2, 1],
                   [1, 0,-1],
                   [1,-2, 1]], dtype=float)
    ar = np.array([[1,-2, 1],
                   [1,-2, 1],
                   [1,-2, 1]], dtype=float)

    # ad  : (3, 3, 28)  coefficient offsets
    # filtgain : (28,)  overall gains
    ad = np.array([  # shape (3,3,28)
        [[ 0.000000e+00, -6.70260e-04,  6.59453e-04],
         [ 0.000000e+00, -3.75071e-04,  3.61926e-04],
         [ 0.000000e+00, -3.06523e-04,  2.97634e-04]],     # 25 Hz
        [[ 0.000000e+00, -8.47258e-04,  8.30131e-04],
         [ 0.000000e+00, -4.76448e-04,  4.55616e-04],
         [ 0.000000e+00, -3.88773e-04,  3.74685e-04]],     # 31.5 Hz
        [[ 0.000000e+00, -1.07210e-03,  1.04496e-03],
         [ 0.000000e+00, -6.06567e-04,  5.73553e-04],
         [ 0.000000e+00, -4.94004e-04,  4.71677e-04]],     # 40 Hz
        [[ 0.000000e+00, -1.35836e-03,  1.31535e-03],
         [ 0.000000e+00, -7.74327e-04,  7.22007e-04],
         [ 0.000000e+00, -6.29154e-04,  5.93771e-04]],     # 50 Hz
        [[ 0.000000e+00, -1.72380e-03,  1.65564e-03],
         [ 0.000000e+00, -9.91780e-04,  9.08866e-04],
         [ 0.000000e+00, -8.03529e-04,  7.47455e-04]],     # 63 Hz
        [[ 0.000000e+00, -2.19188e-03,  2.08388e-03],
         [ 0.000000e+00, -1.27545e-03,  1.14406e-03],
         [ 0.000000e+00, -1.02976e-03,  9.40900e-04]],     # 80 Hz
        [[ 0.000000e+00, -2.79386e-03,  2.62274e-03],
         [ 0.000000e+00, -1.64828e-03,  1.44006e-03],
         [ 0.000000e+00, -1.32520e-03,  1.18438e-03]],     # 100 Hz
        [[ 0.000000e+00, -3.57182e-03,  3.30071e-03],
         [ 0.000000e+00, -2.14252e-03,  1.81258e-03],
         [ 0.000000e+00, -1.71397e-03,  1.49082e-03]],     # 125 Hz
        [[ 0.000000e+00, -4.58305e-03,  4.15355e-03],
         [ 0.000000e+00, -2.80413e-03,  2.28135e-03],
         [ 0.000000e+00, -2.23006e-03,  1.87646e-03]],     # …
        [[ 0.000000e+00, -5.90655e-03,  5.22622e-03],
         [ 0.000000e+00, -3.69947e-03,  2.87118e-03],
         [ 0.000000e+00, -2.92205e-03,  2.36178e-03]],
        [[ 0.000000e+00, -7.65243e-03,  6.57493e-03],
         [ 0.000000e+00, -4.92540e-03,  3.61318e-03],
         [ 0.000000e+00, -3.86007e-03,  2.97240e-03]],
        [[ 0.000000e+00, -1.00023e-02,  8.29610e-03],
         [ 0.000000e+00, -6.63788e-03,  4.55999e-03],
         [ 0.000000e+00, -5.15982e-03,  3.75306e-03]],
        [[ 0.000000e+00, -1.31230e-02,  1.04220e-02],
         [ 0.000000e+00, -9.02274e-03,  5.73132e-03],
         [ 0.000000e+00, -6.94543e-03,  4.71734e-03]],
        [[ 0.000000e+00, -1.73693e-02,  1.30947e-02],
         [ 0.000000e+00, -1.24176e-02,  7.20526e-03],
         [ 0.000000e+00, -9.46002e-03,  5.93145e-03]],
        [[ 0.000000e+00, -2.31934e-02,  1.64308e-02],
         [ 0.000000e+00, -1.73009e-02,  9.04761e-03],
         [ 0.000000e+00, -1.30358e-02,  7.44926e-03]],
        [[ 0.000000e+00, -3.13292e-02,  2.06370e-02],
         [ 0.000000e+00, -2.44342e-02,  1.13731e-02],
         [ 0.000000e+00, -1.82108e-02,  9.36778e-03]],
        [[ 0.000000e+00, -4.28261e-02,  2.59325e-02],
         [ 0.000000e+00, -3.49619e-02,  1.43046e-02],
         [ 0.000000e+00, -2.57855e-02,  1.17912e-02]],
        [[ 0.000000e+00, -5.91733e-02,  3.25054e-02],
         [ 0.000000e+00, -5.06072e-02,  1.79513e-02],
         [ 0.000000e+00, -3.69401e-02,  1.48094e-02]],
        [[ 0.000000e+00, -8.26348e-02,  4.05894e-02],
         [ 0.000000e+00, -7.40348e-02,  2.24476e-02],
         [ 0.000000e+00, -5.34977e-02,  1.85371e-02]],
        [[ 0.000000e+00, -1.17018e-01,  5.08116e-02],
         [ 0.000000e+00, -1.09516e-01,  2.81387e-02],
         [ 0.000000e+00, -7.85097e-02,  2.32872e-02]],
        [[ 0.000000e+00, -1.67714e-01,  6.37872e-02],
         [ 0.000000e+00, -1.63378e-01,  3.53729e-02],
         [ 0.000000e+00, -1.16419e-01,  2.93723e-02]],
        [[ 0.000000e+00, -2.42528e-01,  7.98576e-02],
         [ 0.000000e+00, -2.45161e-01,  4.43370e-02],
         [ 0.000000e+00, -1.73972e-01,  3.70015e-02]],
        [[ 0.000000e+00, -3.53142e-01,  9.96330e-02],
         [ 0.000000e+00, -3.69163e-01,  5.53535e-02],
         [ 0.000000e+00, -2.61399e-01,  4.65428e-02]],
        [[ 0.000000e+00, -5.16316e-01,  1.24177e-01],
         [ 0.000000e+00, -5.55473e-01,  6.89403e-02],
         [ 0.000000e+00, -3.93998e-01,  5.86715e-02]],
        [[ 0.000000e+00, -7.56635e-01,  1.55023e-01],
         [ 0.000000e+00, -8.34281e-01,  8.58123e-02],
         [ 0.000000e+00, -5.94547e-01,  7.43960e-02]],
        [[ 0.000000e+00, -1.10165e+00,  1.91713e-01],
         [ 0.000000e+00, -1.23939e+00,  1.05243e-01],
         [ 0.000000e+00, -8.91666e-01,  9.40354e-02]],
        [[ 0.000000e+00, -1.58477e+00,  2.39049e-01],
         [ 0.000000e+00, -1.80505e+00,  1.28794e-01],
         [ 0.000000e+00, -1.32500e+00,  1.21333e-01]],
        [[ 0.000000e+00, -2.50630e+00,  1.42308e-01],
         [ 0.000000e+00, -2.19464e+00,  2.76470e-01],
         [ 0.000000e+00, -1.90231e+00,  1.47304e-01]], # 12.5 kHz
    ]).reshape(28, 3, 3).transpose(1, 2, 0)

    filtgain = np.array([
        4.30764e-11, 8.59340e-11, 1.71424e-10, 3.41944e-10, 6.82035e-10,
        1.36026e-09, 2.71261e-09, 5.40870e-09, 1.07826e-08, 2.14910e-08,
        4.28228e-08, 8.54316e-08, 1.70009e-07, 3.38215e-07, 6.71990e-07,
        1.33531e-06, 2.65172e-06, 5.25477e-06, 1.03780e-05, 2.04870e-05,
        4.05198e-05, 7.97914e-05, 1.56511e-04, 3.04954e-04, 5.99157e-04,
        1.16544e-03, 2.27488e-03, 3.91006e-03
    ], dtype=float)

    # 4) Nominal Centre Frequencies (28 ISO bands)
    fc_all = 10.0 ** ((np.arange(28) - 16) / 10.0) * 1_000.0  # Hz

    # 5) Band Selection
    keep = np.logical_and(fc_all >= fmin, fc_all <= fmax)
    fc = fc_all[keep]
    ad = ad[:, :, keep]
    filtgain = filtgain[keep]
    nBands = len(fc)

    # 6) Cascade Filtering
    outsig = np.zeros((N, nBands), dtype=float)

    for n in range(nBands):
        # cascade the 3 biquads
        stage_1 = lfilter(br[0], ar[0] - ad[0, :, n], insig)
        stage_2 = lfilter(br[1], ar[1] - ad[1, :, n], stage_1)
        stage_3 = lfilter(br[2], ar[2] - ad[2, :, n], stage_2)
        outsig[:, n] = filtgain[n] * stage_3

    return outsig, fc

def gen_weighting_filters(
        fs: int | float,
        weighting: str = "A",
        plot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This code employs generates the weighting filters. SOurced from
    IEC 61672-1:2013, IEC 61672-1:2013, and Osses2010 (thesis), section 2.2.6.

    ORIGINAL MATLAB CODE:
    Gen_weighting_filters.m   (Osses, Lotinga, 2016) [Last Update:03/04/2025]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 2023-10-01

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    weighting : {'A','B','C','D','Z'}, case-insensitive.
    plot : bool
        If True, show a response graph like MATLAB’s internal demo.

    Returns
    -------
    b, a : ndarray
        IIR transfer-function coefficients ready for scipy.signal.lfilter.
    """

    # 1) IEC 61672-1:2013 Weighting Filter Design (rad s-¹)
    w1 = 129.4273156550629
    w2 = 676.4015402329549
    w3 = 4636.125126885012
    w4 = 76618.52601685845
    w5 = 995.88 # Source: Osses2010

    # 2) Select Analogue Prototype
    wt = weighting.upper()
    if wt == "A":
        K = 7.3901e9
        z = [0, 0, 0, 0]                                 # four zeros at DC
        p = [-w1, -w1, -w2, -w3, -w4, -w4]               # six poles

    elif wt == "B":
        K = 5.9862e9
        z = [0, 0, 0]
        p = [-w1, -w1, -w5, -w4, -w4]

    elif wt == "C":
        K = 5.9124e9
        z = [0, 0]
        p = [-w1, -w1, -w4, -w4]

    elif wt == "D":
        K = 91103.49
        # one DC zero plus quadratic (complex-conjugate) zeros
        z = [0] + np.roots([1, 6532, 4.0975e7]).tolist()
        # two real poles + quadratic conjugate pole pair
        p = [-1773.6, -7288.5] + np.roots([1, 21514, 3.8836e8]).tolist()

    elif wt == "R":
        b = [1, -2, 1]
        a = [1, -1.99004745483398, 0.99007225036621]
        return b, a

    elif wt == "Z":          # Step 2b – Trivial flat filter
        return np.array([1.0]), np.array([1.0])

    else:                     # Step 2c – Bad argument
        raise ValueError("weighting must be 'A', 'B', 'C', 'D' or 'Z'.")

    # 3) Bilinear Transform (Tustin) 
    zd, pd, kd = bilinear_zpk(np.array(z), np.array(p), K, fs)

    # 4) Convert to Digital Filter Coefficients
    b, a = zpk2tf(zd, pd, kd)

    # 5) Plotting (if requested)
    if plot:
            # Make the *upper* half of the FFT grid: N/2 points between 0-fs/2
            N = 16384
            worN = N // 2
            w, h = freqz(b, a, worN=worN, fs=fs)

            plt.figure()
            plt.semilogx(w, 20 * np.log10(np.abs(h)))

            # match the MATLAB axes and tick marks
            plt.title(f"{wt}-weighting frequency response (fs = {fs:g} Hz)")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude [dB]")
            plt.grid(True, which="both")
            plt.xlim(20, fs / 2)
            plt.ylim(-60, 12)
            plt.xticks([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])

            plt.show()

    return b, a

def do_slm(
    insig: np.ndarray,
    fs: int | float,
    weight_freq: str = "A",
    weight_time: str = "f",
    dBFS: float = 100.0,
    plot: bool = False,
) -> tuple[np.ndarray, float]:
    """
    This code computes the frequency and time weighting SPL.

    ORIGINAL MATLAB CODE:
    Do_SLM.m   (Osses, 2016) [Last Update: 22/03/2023]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 2023-10-01

    Parameters
    ----------
    insig : 1-D array_like
        Audio waveform in full-scale units (e.g. ±1 from WAV).
    fs : int | float
        Sampling rate [Hz].
    weight_freq : {'A','C','Z'}, default 'A'
        IEC-61672 frequency weighting.
    weight_time : {'f','s','i'}, default 'f'
        Time weighting: fast (125 ms), slow (1 s) or impulse (35 ms).
    dBFS : float, default 100
        Level of a full-scale *sine* in dB SPL (94 is common).
    plot : bool, default False
        If True, draws a quick-look plot like the original MATLAB version.

    Returns
    -------
    outsig_dB : ndarray
        Instantaneous weighted level [dB(…)].
    dBFS : float
        Echo of the `dBFS` argument (handy for callers).
    """
    # 1) Ensure Mono Vector
    insig = np.asarray(insig, dtype=float).squeeze()
    if insig.ndim != 1:
        raise ValueError("Input signal must be a 1-D (mono) array")

    # 2) Get Frequency-weighting IIR
    b_w, a_w = gen_weighting_filters(fs, weight_freq)
    sig_w = lfilter(b_w, a_w, insig)

    # 3) Calibrate to Pascals
    dBoffset = 0.93
    cal_coeff = 10 ** ((dBFS + dBoffset - 94) / 20.0)
    sig_cal = cal_coeff * sig_w         # Pa

    # 4) Apply IEC Time Integrator
    sig_int = _integrator(sig_cal, fs, weight_time)

    # 5) Convert to dB SPL & Clamp
    outsig_dB = 20.0 * np.log10(np.abs(sig_int) / 2e-5)
    outsig_dB = np.maximum(outsig_dB, 0.0)   # set < 0 dB to 0 dB

    # 6) Plotting (if requested)
    if plot:
        leq = _leq(outsig_dB)
        t = np.arange(len(outsig_dB)) / fs
        plt.figure()
        plt.plot(t, outsig_dB)
        plt.grid(True)
        unit = "dB" if weight_freq.upper() == "Z" else f"dB({weight_freq.upper()})"
        plt.xlabel("Time [s]")
        plt.ylabel(f"Amplitude [{unit}]")
        plt.title(f"Level • Leq = {leq:0.1f} {unit}")
        plt.show()

    return outsig_dB, dBFS

def get_leq(
    levels: Sequence[float] | np.ndarray,
    fs: float | None = None,
    dt: float | None = None,
    framelen_s: float | None = None
) -> np.ndarray:
    """
    This code calculates the equivalent continuous sound level (Leq) from a dB‐SPL trace.

    ORIGINAL MATLAB CODE:
    Get_Leq.m   (Osses, 2014-2017) [Last Update: 13/07/2016]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 2023-10-01

    Parameters
    ----------
    levels : array-like
        A-weighted (or other) SPL values in dB.  NaNs are ignored.
    fs : float, optional
        Sampling rate of `levels` in hertz.  If omitted, the whole vector is
        treated as one frame and a single Leq is returned.
    dt : float, optional
        Hop size between consecutive Leq values, in seconds.  Required when
        `fs` is given.
    framelen_s : float, optional
        Frame length in seconds.  Defaults to `dt` when omitted.

    Returns
    -------
    leq : np.ndarray
        Vector of Leq values in dB.  Length is 1 for whole-signal mode,
        otherwise equal to the number of frames.
    """
    # 1) Ensure 1-D Shape
    levels = np.asarray(levels).ravel()

    # 2a) Whole-signal Mode
    if fs is None:
        valid = np.isfinite(levels)
        if not valid.any():
            return np.array([np.nan])
        power = 10.0 ** (levels[valid] / 10.0)
        return np.array([10.0 * np.log10(power.mean())])

    # 2b) Running-Leq Mode
    if dt is None:
        raise ValueError("`dt` (hop size in seconds) must be supplied with `fs`.")
    if framelen_s is None:
        framelen_s = dt

    H = int(round(dt * fs))          # hop in samples
    N = int(round(framelen_s * fs))  # window length in samples
    if H <= 0 or N <= 0:
        raise ValueError("`dt*fs` and `framelen_s*fs` must each be ≥ 1 sample.")
    if N > len(levels):
        raise ValueError("Signal is shorter than the analysis window.")

    # 3) Sliding Windows (zero-copy view)
    frames = sliding_window_view(levels, N)[::H]  # shape (n_frames, N)

    leq = np.empty(frames.shape[0])
    for i, frame in enumerate(frames):
        valid = np.isfinite(frame)
        if not valid.any():
            leq[i] = np.nan
        else:
            power = 10.0 ** (frame[valid] / 10.0)
            leq[i] = 10.0 * np.log10(power.mean())

    return leq

# ------------------------
#### HELPER FUNCTIONS ####
# ------------------------

def _integrator(insig: np.ndarray, fs: int | float, mode: str) -> np.ndarray:
    """IEC-61672 leaky integrator for fast/slow/impulse modes."""
    mode = mode.lower()
    tau = {"f": 125e-3, "s": 1.0, "i": 35e-3}.get(mode)
    if tau is None:
        raise ValueError("weight_time must be 'f', 's' or 'i'")

    e = np.exp(-1.0 / (tau * fs))
    b = np.array([1.0 - e])          # numerator
    a = np.array([1.0, -e])          # denominator
    return lfilter(b, a, np.abs(insig))

def _leq(level_dB: np.ndarray) -> float:
    """Equivalent continuous SPL (Leq) of a dB vector."""
    return 10.0 * np.log10(np.mean(10 ** (level_dB / 10.0)))

# ----------------------------
#### VALIDATION FUNCTIONS ####
# ----------------------------

# # Do_OB13_ISO532_1.m
# if __name__ == "__main__":
#     # -------- locate the reference WAV file -------------------------------
#     # Adjust 'basepath_SQAT' if your folder differs:
#     wav_path = "sound_files\RefSignal_Loudness_ISO532_1.wav"

#     # -------- read audio & run the filter bank ----------------------------
#     insig, fs = sf.read(wav_path)
#     if insig.ndim != 1:
#         raise RuntimeError("The reference file must be mono.")

#     dBFS = 94                                      # same assumption as MATLAB
#     OB_filt, fc = ob13_iso532_1(insig, fs)         # filtering

#     # -------- levels per band + consistency check -------------------------
#     rms = lambda x: np.sqrt(np.mean(x**2, axis=0))
#     OB_lvls = 20 * np.log10(rms(OB_filt)) + dBFS
#     lvl_orig = 20 * np.log10(rms(insig)) + dBFS
#     lvl_from_outsig = 10 * np.log10(np.sum(10 ** (OB_lvls / 10)))
#     delta = lvl_from_outsig - lvl_orig

#     print("----------------------------------------------------------------")
#     print(f"Level from input signal:            {lvl_orig:7.2f}  dB SPL")
#     print(f"Level from summed band powers:      {lvl_from_outsig:7.2f}  dB SPL")
#     print(f"Difference (should be ≈0 dB):       {delta:7.3f}  dB")
#     print("----------------------------------------------------------------")

#     # -------- reproduce MATLAB figure -------------------------------------
#     plt.semilogx(fc, OB_lvls, "bo-")
#     plt.xlim(fc.min() - 1, fc.max() + 1_000)
#     plt.xticks(fc, [f"{f:.0f}" for f in fc], rotation=45)
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Band level (dB SPL)")
#     plt.title("One-third-octave band levels – Python port")
#     plt.grid(True, which="both", linestyle=":")
#     plt.tight_layout()
#     plt.show()

# # Gen_weighting_filters.m
# if __name__ == "__main__":
#     fs = 44100
#     b, a = gen_weighting_filters(fs, "C", plot=True)
#     print("b =", b)
#     print("a =", a)

# # Do_SLM.m
# if __name__ == "__main__":
#     # 1-kHz sine, 60 dB SPL (A-weighted, fast)
#     fs = 44_100
#     dur = 1.0
#     t = np.arange(0, dur, 1 / fs)
#     dBFS = 94
#     target = 60
#     cal = 10 ** ((target - dBFS) / 20.0)
#     sig = cal * np.sqrt(2) * np.sin(2 * np.pi * 1000 * t)

#     outsig_dB, _ = do_slm(sig, fs, "A", "f", dBFS, plot=True)
#     print(f"Leq = {_leq(outsig_dB):.1f} dB(A)")

# # Get_Leq.m
# if __name__ == "__main__":
#     # Example 1 – single overall Leq from a 1-s trace
#     fs = 48000  # Hz
#     lvls_1s = np.random.normal(60.0, 3.0, fs)  # synthetic 1-s SPL record
#     overall_leq = get_leq(lvls_1s)
#     print(f"Example 1 – overall Leq: {overall_leq[0]:.2f} dB")

#     # Example 2 – running 1-s Leq every 0.1 s over a 5-s signal
#     duration_s = 5
#     lvls_5s = np.random.normal(60.0, 3.0, int(fs * duration_s))
#     leq_running = get_leq(lvls_5s, fs=fs, dt=0.1, framelen_s=1.0)
#     print(f"Example 2 – running Leq array shape: {leq_running.shape}")
#     print("First ten values:", np.round(leq_running[:10], 2))

# test_Gen_weighting.m (Gen_weighting_filters.m)
if __name__ == "__main__":
    fs = 44_100                             # sampling rate
    K  = fs // 2                            # ⇒ df ≈ 1 Hz below Nyquist

    # 1) Generate All Filters
    bA, aA = gen_weighting_filters(fs, "A")
    bB, aB = gen_weighting_filters(fs, "B")
    bC, aC = gen_weighting_filters(fs, "C")
    bD, aD = gen_weighting_filters(fs, "D")
    bR, aR = gen_weighting_filters(fs, "R")

    # 2 Frequency responses
    f, hA = freqz(bA, aA, K, fs=fs)
    _, hB  = freqz(bB, aB, K, fs=fs)
    _, hC  = freqz(bC, aC, K, fs=fs)
    _, hD  = freqz(bD, aD, K, fs=fs)
    _, hR  = freqz(bR, aR, K, fs=fs)

    # 3 Plot
    plt.figure()
    plt.semilogx(f, 20 * np.log10(abs(hA)), "b", label="A")
    plt.semilogx(f, 20 * np.log10(abs(hB)), "g", label="B")
    plt.semilogx(f, 20 * np.log10(abs(hC)), "r", label="C")
    plt.semilogx(f, 20 * np.log10(abs(hD)), "k", label="D")
    plt.semilogx(f, 20 * np.log10(abs(hR)), "m", label="R")

    plt.xlim(20, 20_000)
    plt.grid(True, which="both")
    plt.legend(loc="lower right")
    plt.title("Frequency-weighting curves\n"
              "(note that B and D are no longer standardised)")
    xt = [32, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    plt.xticks(xt, xt)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Relative magnitude (dB)")
    plt.ylim([-60, 20])
    plt.show()


