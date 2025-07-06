
##############################
#### SOUND METRICS MODULE ####
##############################

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, resample_poly, bilinear_zpk, zpk2tf, freqz
from typing import Tuple, Sequence
import soundfile as sf
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import warnings

# ----------------------
#### MAIN FUNCTIONS ####
# ----------------------

__all__ = ["ob13_iso532_1", "gen_weighting_filters", "do_slm", "get_leq"]

def ob13_iso532_1(insig, fs, fmin=None, fmax=None):
    """
    One-third-octave filter-bank as specified in **ISO 532-1:2017 (Annex B)**.

    The routine designs and applies a bank of 28 digital IIR filters whose
    nominal centre frequencies span **25 Hz – 12.5 kHz** in ⅓-oct steps
    (base-10 formulation).  
    Each band is realised by **three cascaded biquads** whose coefficients
    are tabulated in the Standard; an overall linear *gain* aligns the pass-
    band to 0 dB.  If the input sampling rate is not *48 kHz* the signal is
    up/down-sampled to that rate with polyphase filtering before processing,
    because the published coefficients have only been validated at
    48 kHz.

    A smaller subset of bands can be selected with *fmin* / *fmax*; the
    filterbank and its gains are trimmed accordingly.

    Parameters
    ----------
    insig : array_like
        Input waveform (mono).  Any shape is accepted but squeezed to 1-D.
    fs : int or float
        Sampling frequency of *insig* in hertz.
    fmin : float, optional
        Lowest band centre frequency **≥ 25 Hz** to retain.  Defaults to
        *25 Hz* (no lower limit).
    fmax : float, optional
        Highest band centre frequency **≤ 12 500 Hz** to retain.  Defaults
        to *12 500 Hz* (no upper limit).
    Returns
    -------
    outsig : numpy.ndarray
        Filtered signal, shape ``(N, B)`` where *N* is the sample count and
        *B* the number of retained bands.
    fc : numpy.ndarray
        Centre frequencies of the columns in *outsig* (hertz).

    Warns
    -----
    RuntimeWarning
        If *fmin* < 25 Hz or *fmax* > 12.5 kHz, the values are clamped to
        the valid range and a warning is emitted.

    Notes
    -----
    * The coefficient tables reproduce ISO 532-1 Annex B (Tables A.1/A.2).
    * Filter order per band is six (three biquads, **maximally flat**).
    * Resampling uses :pyfunc:`scipy.signal.resample_poly`.
    """


    # Handle input arguments
    if fmin is None:
        bLimit_range = 0
    else:
        # Then fmin has been specified
        if fmax is None:
            # Then only fmin has been specified as input
            fmax = 12500  # Hz
        bLimit_range = 1
        if fmax > 12500:
            warnings.warn('The input parameter fmax cannot be greater than 12500 Hz, setting fmax to this value')
            fmax = 12500  # Hz, maximum possible frequency of the filterbank
        if fmin < 25:
            warnings.warn('The input parameter fmin cannot be lower than 25 Hz, setting fmin to this value')
            fmin = 25  # Hz, minimum possible frequency of the filterbank

    # resample to 48 kHz if necessary
    if fs != 48000:
        gcd_fs = np.gcd(48000, fs) # greatest common denominator
        insig = resample_poly(insig, 48000 // gcd_fs, fs // gcd_fs)
        fs = 48000
        print(f'Do_OB13_ISO532_1: This script has only been validated at a sampling frequency fs=48 kHz, resampling to this fs value')
    
    # Ensure insig is 1D for consistent filtering operations
    insig = insig.squeeze()
    len_sig = insig.shape[0]

    # Create filter bank and filter the signal

    # reference
    br = np.array([[1, 2, 1],
                   [1, 0, -1],
                   [1, -2, 1]])
    ar = np.array([[1, -2, 1],
                   [1, -2, 1],
                   [1, -2, 1]])

    # filter 'a' coefficient offsets TABLES A.1 A.2
    ad = np.array([
        # 25 Hz
        [[0, -6.70260e-004, 6.59453e-004],
         [0, -3.75071e-004, 3.61926e-004],
         [0, -3.06523e-004, 2.97634e-004]],
        # 31.5 Hz
        [[0, -8.47258e-004, 8.30131e-004],
         [0, -4.76448e-004, 4.55616e-004],
         [0, -3.88773e-004, 3.74685e-004]],
        # 40 Hz
        [[0, -1.07210e-003, 1.04496e-003],
         [0, -6.06567e-004, 5.73553e-004],
         [0, -4.94004e-004, 4.71677e-004]],
        # 50 Hz
        [[0, -1.35836e-003, 1.31535e-003],
         [0, -7.74327e-004, 7.22007e-004],
         [0, -6.29154e-004, 5.93771e-004]],
        # 63 Hz
        [[0, -1.72380e-003, 1.65564e-003],
         [0, -9.91780e-004, 9.08866e-004],
         [0, -8.03529e-004, 7.47455e-004]],
        # 80 Hz
        [[0, -2.19188e-003, 2.08388e-003],
         [0, -1.27545e-003, 1.14406e-003],
         [0, -1.02976e-003, 9.40900e-004]],
        # 100 Hz
        [[0, -2.79386e-003, 2.62274e-003],
         [0, -1.64828e-003, 1.44006e-003],
         [0, -1.32520e-003, 1.18438e-003]],
        # 125 Hz
        [[0, -3.57182e-003, 3.30071e-003],
         [0, -2.14252e-003, 1.81258e-003],
         [0, -1.71397e-003, 1.49082e-003]],
        # 160 Hz
        [[0, -4.58305e-003, 4.15355e-003],
         [0, -2.80413e-003, 2.28135e-003],
         [0, -2.23006e-003, 1.87646e-003]],
        # 200 Hz
        [[0, -5.90655e-003, 5.22622e-003],
         [0, -3.69947e-003, 2.87118e-003],
         [0, -2.92205e-003, 2.36178e-003]],
        # 250 Hz
        [[0, -7.65243e-003, 6.57493e-003],
         [0, -4.92540e-003, 3.61318e-003],
         [0, -3.86007e-003, 2.97240e-003]],
        # 315 Hz
        [[0, -1.00023e-002, 8.29610e-003],
         [0, -6.63788e-003, 4.55999e-003],
         [0, -5.15982e-003, 3.75306e-003]],
        # 400 Hz
        [[0, -1.31230e-002, 1.04220e-002],
         [0, -9.02274e-003, 5.73132e-003],
         [0, -6.94543e-003, 4.71734e-003]],
        # 500 Hz
        [[0, -1.73693e-002, 1.30947e-002],
         [0, -1.24176e-002, 7.20526e-003],
         [0, -9.46002e-003, 5.93145e-003]],
        # 630 Hz
        [[0, -2.31934e-002, 1.64308e-002],
         [0, -1.73009e-002, 9.04761e-003],
         [0, -1.30358e-002, 7.44926e-003]],
        # 800 Hz
        [[0, -3.13292e-002, 2.06370e-002],
         [0, -2.44342e-002, 1.13731e-002],
         [0, -1.82108e-002, 9.36778e-003]],
        # 1000 Hz
        [[0, -4.28261e-002, 2.59325e-002],
         [0, -3.49619e-002, 1.43046e-002],
         [0, -2.57855e-002, 1.17912e-002]],
        # 1250 Hz
        [[0, -5.91733e-002, 3.25054e-002],
         [0, -5.06072e-002, 1.79513e-002],
         [0, -3.69401e-002, 1.48094e-002]],
        # 1600 Hz
        [[0, -8.26348e-002, 4.05894e-002],
         [0, -7.40348e-002, 2.24476e-002],
         [0, -5.34977e-002, 1.85371e-002]],
        # 2000 Hz
        [[0, -1.17018e-001, 5.08116e-002],
         [0, -1.09516e-001, 2.81387e-002],
         [0, -7.85097e-002, 2.32872e-002]],
        # 2500 Hz
        [[0, -1.67714e-001, 6.37872e-002],
         [0, -1.63378e-001, 3.53729e-002],
         [0, -1.16419e-001, 2.93723e-002]],
        # 3150 Hz
        [[0, -2.42528e-001, 7.98576e-002],
         [0, -2.45161e-001, 4.43370e-002],
         [0, -1.73972e-001, 3.70015e-002]],
        # 4000 Hz
        [[0, -3.53142e-001, 9.96330e-002],
         [0, -3.69163e-001, 5.53535e-002],
         [0, -2.61399e-001, 4.65428e-002]],
        # 5000 Hz
        [[0, -5.16316e-001, 1.24177e-001],
         [0, -5.55473e-001, 6.89403e-002],
         [0, -3.93998e-001, 5.86715e-002]],
        # 6300 Hz
        [[0, -7.56635e-001, 1.55023e-001],
         [0, -8.34281e-001, 8.58123e-002],
         [0, -5.94547e-001, 7.43960e-002]],
        # 8000 Hz
        [[0, -1.10165e+000, 1.91713e-001],
         [0, -1.23939e+000, 1.05243e-001],
         [0, -8.91666e-001, 9.40354e-002]],
        # 10000 Hz
        [[0, -1.58477e+000, 2.39049e-001],
         [0, -1.80505e+000, 1.28794e-001],
         [0, -1.32500e+000, 1.21333e-001]],
        # 12500 Hz
        [[0, -2.50630e+000, 1.42308e-001],
         [0, -2.19464e+000, 2.76470e-001],
         [0, -1.90231e+000, 1.47304e-001]]
    ])

    # Transpose to match MATLAB dimensions (3, 3, 28)
    ad = np.transpose(ad, (1, 2, 0))

    # filter gains
    filtgain = np.array([
        4.30764e-011,  # 25 Hz
        8.59340e-011,  # 31.5 Hz
        1.71424e-010,  # 40 Hz
        3.41944e-010,  # 50 Hz
        6.82035e-010,  # 63 Hz
        1.36026e-009,  # 80 Hz
        2.71261e-009,  # 100 Hz
        5.40870e-009,  # 125 Hz
        1.07826e-008,  # 160 Hz
        2.14910e-008,  # 200 Hz
        4.28228e-008,  # 250 Hz
        8.54316e-008,  # 315 Hz
        1.70009e-007,  # 400 Hz
        3.38215e-007,  # 500 Hz
        6.71990e-007,  # 630 Hz
        1.33531e-006,  # 800 Hz
        2.65172e-006,  # 1000 Hz
        5.25477e-006,  # 1250 Hz
        1.03780e-005,  # 1600 Hz
        2.04870e-005,  # 2000 Hz
        4.05198e-005,  # 2500 Hz
        7.97914e-005,  # 3150 Hz
        1.56511e-004,  # 4000 Hz
        3.04954e-004,  # 5000 Hz
        5.99157e-004,  # 6300 Hz
        1.16544e-003,  # 8000 Hz
        2.27488e-003,  # 10000 Hz
        3.91006e-003   # 12500 Hz
    ])

    # Calculate centre frequencies (25 Hz - 12.5 kHz)
    N_bands = 28  # number of bands
    CenterFrequency = np.zeros(N_bands)

    for i in range(N_bands):
        CenterFrequency[i] = 10**(((i)-16)/10.) * 1000  # calculate centre frequencies

    # limit freq range if required
    if bLimit_range:
        # find idx of fmin and fmax and fix freq range before filtering
        # fmin and fmax are only used if they are specified as inputs:
        # Given the input validation, these indices will always be found.
        idx_fmin = np.where(CenterFrequency >= fmin)[0][0]
        idx_fmax = np.where(CenterFrequency >= fmax)[0][0]
            
        filtgain = filtgain[idx_fmin:idx_fmax+1]  # adjust filtgain to fmin and fmax 
        ad = ad[:, :, idx_fmin:idx_fmax+1]  # adjust ad to fmin and fmax

    # Filter signal
    N_bands = filtgain.shape[0]
    outsig = np.zeros((len_sig, N_bands))

    for n in range(N_bands):
        # Three cascaded filters:
        temp1 = lfilter(br[0, :], ar[0, :] - ad[0, :, n], insig)
        temp2 = lfilter(br[1, :], ar[1, :] - ad[1, :, n], temp1)
        outsig[:, n] = filtgain[n] * lfilter(br[2, :], ar[2, :] - ad[2, :, n], temp2)

    fc = CenterFrequency.copy()

    if bLimit_range:
        fc = CenterFrequency[idx_fmin:idx_fmax+1]

    return outsig, fc

def gen_weighting_filters(
        fs: int | float,
        weighting: str = "A",
        plot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Design a digital frequency-weighting filter (*A*, *B*, *C*, *D*, *R*, or *Z*)
    and return its transfer-function coefficients.

    The analogue prototypes for *A–D* follow **IEC 61672-1 : 2013**  
    (see Annex A for pole/zero values).  They are transformed to digital
    form with the **bilinear (Tustin) mapping** at the requested sampling
    rate *fs*.  Weightings *R* (RLB high-pass from ITU-R BS.1770) and *Z*
    (unity, i.e. no weighting) are supplied directly in discrete-time form.
    An optional log-magnitude plot up to *fs/2* can be displayed.

    Parameters
    ----------
    fs : int | float
        Sampling frequency in hertz.
    weighting : {'A', 'B', 'C', 'D', 'R', 'Z'}, default ``'A'``
        Desired weighting curve (case-insensitive).
    plot : bool, default ``False``
        If ``True`` show the magnitude response using
        :pyfunc:`scipy.signal.freqz` and Matplotlib.

    Returns
    -------
    b : numpy.ndarray
        Numerator coefficients of the **IIR** filter (direct form I).
    a : numpy.ndarray
        Denominator coefficients (``a[0] == 1``).

    Raises
    ------
    ValueError
        If *weighting* is not one of the recognised letters.

    Notes
    -----
    * Filter order varies: A–C (5th), D (7th), R (2nd), Z (0th, i.e. FIR[0]).
    * The analogue constants are taken from IEC 61672 except for the
      additional break frequency *w₅ ≈ 996 rad/s* used by the **B-curve** in
      Fastl & Zwicker (2010).
    """


    # 1) IEC 61672-1:2013 Weighting Filter Design (rad s-¹)
    w1 = 129.4273156550629
    w2 = 676.4015402329549
    w3 = 4636.125126885012
    w4 = 76618.52601685845
    w5 = 995.88 # Source: Osses2010

    # 2) Select Analogue Prototype
    weightingType = weighting.upper()
    if weightingType == "A":
        K = 7.3901e9
        zrs = [0, 0, 0, 0]
        pls = [-w1, -w1, -w2, -w3, -w4, -w4]

    elif weightingType == "B":
        K = 5.9862e9
        zrs = [0, 0, 0]
        pls = [-w1, -w1, -w5, -w4, -w4]

    elif weightingType == "C":
        K = 5.9124e9
        zrs = [0, 0]
        pls = [-w1, -w1, -w4, -w4]

    elif weightingType == "D":
        K = 91103.49
        zrs = [0] + np.roots([1, 6532, 4.0975e7]).tolist()
        pls = [-1773.6, -7288.5] + np.roots([1, 21514, 3.8836e8]).tolist()

    elif weightingType == "R":
        b = [1, -2, 1]
        a = [1, -1.99004745483398, 0.99007225036621]
        return b, a

    elif weightingType == "Z":
        return np.array([1.0]), np.array([1.0])

    else:
        raise ValueError("weighting must be 'A', 'B', 'C', 'D' or 'Z'.")

    # 3) Bilinear Transform (Tustin) 
    Zd, Pd, Kd = bilinear_zpk(np.array(zrs), np.array(pls), K, fs)

    # 4) Convert to Digital Filter Coefficients
    b, a = zpk2tf(Zd, Pd, Kd)

    # 5) Plotting (if requested)
    if plot:
            # Make the *upper* half of the FFT grid: N/2 points between 0-fs/2
            N = 16384
            w, H = freqz(b, a, N//2, fs=fs)

            plt.figure()
            plt.semilogx(w, 20 * np.log10(np.abs(H)))

            # match the MATLAB axes and tick marks
            plt.title(f"{weightingType}-weighting frequency response (fs = {fs:g} Hz)")
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
    Simulate a basic **sound-level meter** compliant with IEC 61672-1 by
    applying frequency and time weightings to an audio signal.

    Parameters
    ----------
    insig : numpy.ndarray
        Mono input waveform.
    fs : int | float
        Sampling frequency in hertz.
    weight_freq : {'A', 'B', 'C', 'D', 'R', 'Z'}, default ``'A'``
        Frequency weighting curve; ``'Z'`` means flat (zero weighting).
    weight_time : {'f', 's', 'i'}, default ``'f'``
        Time weighting – **f**ast, **s**low, or **i**mpulse (case-insensitive).
    dBFS : float, default ``100.0``
        Sound-pressure level, in dB SPL, represented by a full-scale sine.
    plot : bool, default ``False``
        When ``True`` show the instantaneous level trace and *L\ :sub:`eq`*.

    Returns
    -------
    outsig_dB : numpy.ndarray
        Instantaneous sound level in decibels after both weightings.
    dBFS : float
        The *dBFS* value actually used for calibration (echo of the input).

    Raises
    ------
    ValueError
        If *insig* is not one-dimensional.

    Notes
    -----
    * Levels below 0 dB are set to exactly 0 dB for practical display.
    * The calibration offset ``+0.93 dB`` aligns the response to typical SLM
      readings at 1 kHz with A-weighting engaged.
    """


    # 1) Ensure Mono Vector
    insig = np.asarray(insig, dtype=float).squeeze()
    if insig.ndim != 1:
        raise ValueError("Input signal must be a 1-D (mono) array")

    # 2) Get Frequency-weighting IIR
    b, a = gen_weighting_filters(fs, weight_freq)

    # 3) Calibrate to Pascals
    dBoffset = 0.93
    calCoeff = 10 ** ((dBFS + dBoffset - 94) / 20.0)
    insig *= calCoeff        # Pa

    # 4) Apply IEC Time Integrator
    outsig = lfilter(b, a, insig)
    outsig = _integrator(outsig, fs, weight_time)

    # 5) Convert to dB SPL & Clamp
    outsig_dB = 20.0 * np.log10(np.abs(outsig) / 2e-5)
    outsig_dB = np.maximum(outsig_dB, 0.0)   # set < 0 dB to 0 dB

    # 6) Plotting (if requested)
    if plot:
        Leq = 10.0 * np.log10(np.mean(10 ** (outsig_dB / 10.0)))
        t = np.arange(len(outsig_dB)) / fs
        plt.figure()
        plt.plot(t, outsig_dB)
        plt.grid(True)
        unit = "dB" if weight_freq.upper() == "Z" else f"dB({weight_freq.upper()})"
        plt.xlabel("Time [s]")
        plt.ylabel(f"Amplitude [{unit}]")
        plt.title(f"Level Leq = {Leq:0.1f} {unit}")
        plt.show()

    return outsig_dB, dBFS

def get_leq(
    levels: Sequence[float] | np.ndarray,
    fs: float | None = None,
    dt: float | None = None,
    framelen_s: float | None = None
) -> np.ndarray:
    """
    Compute the **equivalent continuous sound-pressure level** (*L\ :sub:`eq`*)
    from a sequence of dB values.

    Two operating modes are supported:

    * **Whole-signal mode** – if *fs* is ``None`` the function returns a
      single scalar *Leq* for the entire input, ignoring any non-finite
      samples (``NaN``, ``Inf``).
    * **Running-Leq mode** – when *fs* is given, the signal is analysed in
      overlapping windows of length *framelen_s* seconds with a hop size of
      *dt* seconds (``framelen_s`` defaults to *dt*).  The result is a
      vector of frame-by-frame *Leq* values.

    Parameters
    ----------
    levels : Sequence[float] | numpy.ndarray
        One-dimensional array of sound levels in **dB** (any reference).
    fs : float, optional
        Sampling frequency of *levels* in hertz.  Omit to compute a single
        overall *Leq* (**whole-signal mode**).
    dt : float, optional
        Hop size between consecutive windows in seconds.  Required for
        **running-Leq mode**.
    framelen_s : float, optional
        Window length in seconds; defaults to *dt*.  Must be ≥ *dt*.

    Returns
    -------
    numpy.ndarray
        *Leq* value(s) in decibels:

        * shape ``(1,)`` for whole-signal mode,
        * shape ``(n_frames,)`` for running-Leq mode.

    Raises
    ------
    ValueError
        If *fs* is supplied without *dt*, if ``dt*fs`` or
        ``framelen_s*fs`` evaluates to < 1 sample, or if the signal is
        shorter than the requested analysis window.

    Notes
    -----
    * *Leq* is computed as

      ``Leq = 10 · log10( mean( 10 ** (L / 10) ) )``

      where the mean is taken over the finite samples in each window.
    * Windows containing only non-finite values yield ``NaN`` in the output.
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
# if __name__ == "__main__":
#     fs = 44_100                             # sampling rate
#     K  = fs // 2                            # ⇒ df ≈ 1 Hz below Nyquist

#     # 1) Generate All Filters
#     bA, aA = gen_weighting_filters(fs, "A")
#     bB, aB = gen_weighting_filters(fs, "B")
#     bC, aC = gen_weighting_filters(fs, "C")
#     bD, aD = gen_weighting_filters(fs, "D")
#     bR, aR = gen_weighting_filters(fs, "R")

#     # 2 Frequency responses
#     f, hA = freqz(bA, aA, K, fs=fs)
#     _, hB  = freqz(bB, aB, K, fs=fs)
#     _, hC  = freqz(bC, aC, K, fs=fs)
#     _, hD  = freqz(bD, aD, K, fs=fs)
#     _, hR  = freqz(bR, aR, K, fs=fs)

#     # 3 Plot
#     plt.figure()
#     plt.semilogx(f, 20 * np.log10(abs(hA)), "b", label="A")
#     plt.semilogx(f, 20 * np.log10(abs(hB)), "g", label="B")
#     plt.semilogx(f, 20 * np.log10(abs(hC)), "r", label="C")
#     plt.semilogx(f, 20 * np.log10(abs(hD)), "k", label="D")
#     plt.semilogx(f, 20 * np.log10(abs(hR)), "m", label="R")

#     plt.xlim(20, 20_000)
#     plt.grid(True, which="both")
#     plt.legend(loc="lower right")
#     plt.title("Frequency-weighting curves\n"
#               "(note that B and D are no longer standardised)")
#     xt = [32, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
#     plt.xticks(xt, xt)
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Relative magnitude (dB)")
#     plt.ylim([-60, 20])
#     plt.show()

