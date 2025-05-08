
##########################
#### UTILITIES MODULE ####
##########################

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import firwin2, freqz, lfilter, sosfilt, sosfreqz
from scipy.io import wavfile
from scipy.special import comb
import soundfile as sf
from IPython.display import Audio, display
from pathlib import Path
from typing import Union, Sequence, Any, Tuple, Optional, Dict, Literal

FloatArrayLike = Union[float, Sequence[float], np.ndarray]
ArrayLikeInt   = Union[Sequence[int], np.ndarray]

# ----------------------
#### MAIN FUNCTIONS ####
# ----------------------

__all__ = ["see", "hz2bark", "bark2hz", "phon2sone",
           "sone2phon", "get_exceeded_value", "get_statistics",
           "get_bark", "from_db", "create_a0_FIR", "calculate_a0",
           "calibrate", "get_defaults", "shm_auditory_filt_bank",
           "shm_basis_loudness","shm_noise_red_lowpass"]

def see(file_path: str) -> None:
    """Visualise a WAV file (waveform + spectrogram) and play the audio.

    Parameters
    ----------
    file_path : str
        Path to the `.wav` file (must exist).

    Returns
    -------
    None
    """
    plt.close("all")

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    # Load the WAV file
    sample_rate, data = wavfile.read(file_path)

    # Handle stereo by converting to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Time axis in seconds
    time = np.linspace(0, len(data) / sample_rate, num=len(data))

    # Plot waveform
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, data)
    max_amp = np.max(np.abs(data))
    plt.ylim(-max_amp, max_amp)
    plt.title("Waveform")
    plt.xlabel("Time [s]")
    plt.xlim(0, time.max())
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.ylabel("Amplitude")
    plt.minorticks_on()

    # Plot spectrogram
    plt.subplot(2, 1, 2)
    Pxx, freqs, bins, im = plt.specgram(
        data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis'
    )
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.ylim(freqs[1], freqs.max())  # Avoid zero for log scale
    plt.title("Spectrogram")
    plt.xlabel("Time [s]")
    plt.xlim(0, time.max())
    plt.ylabel("Frequency [Hz]")
    plt.minorticks_on()
    

    plt.tight_layout()
    plt.show()

def hz2bark(f: FloatArrayLike) -> np.ndarray:
    """Convert frequency *f* from Hertz to the Bark critical‑band scale.

    Parameters
    ----------
    f : float or array‑like
        Frequency in Hertz.

    Returns
    -------
    ndarray
        Bark values, same shape as *f*.
    """
    f = np.asarray(f)  # Support scalar or array input
    z = 13 * np.arctan(0.76 * (f / 1000)) + 3.5 * np.arctan((f / (1000 * 7.5))**2)
    return z

def bark2hz(z: FloatArrayLike) -> np.ndarray:
    """Convert Bark numbers back to Hertz (piece‑wise linear interpolation).

    Parameters
    ----------
    z : float or array‑like
        Bark scale values (0 – ≈24.5).

    Returns
    -------
    ndarray
        Hertz values corresponding to *z*.
    """
    f0 = 1000
    k = np.arange(-20, 13)  # Equivalent to MATLAB -20:12
    f = f0 * 2 ** (k / 3)

    zt = hz2bark(f)  # Use your defined function

    interpolator = interp1d(zt, f, kind='linear', fill_value="extrapolate")
    f_interp = interpolator(z)

    return f_interp

def phon2sone(phon: FloatArrayLike) -> np.ndarray:
    """ISO 532‑1 mapping from **phon** to **sone**.

    Parameters
    ----------
    phon : array‑like
        Loudness level in phon.

    Returns
    -------
    ndarray
        Loudness in sone.
    """
    phon = np.asarray(phon).flatten()  # Ensure 1D array
    phon = phon[:, np.newaxis]  # Convert to column vector [nTime, 1]
    
    sone = np.zeros_like(phon)

    idx = phon >= 40

    # Calculate sone values for phon >= 40
    sone[idx] = 2 ** (0.1 * (phon[idx] - 40))

    # Calculate sone values for phon < 40
    sone[~idx] = (phon[~idx] / 40) ** (1 / 0.35)

    return sone

def sone2phon(sone: FloatArrayLike) -> np.ndarray:
    """Inverse of :pyfunc:`phon2sone`.

    Parameters
    ----------
    sone : array‑like
        Loudness in sone (≥ 0).

    Returns
    -------
    ndarray
        Loudness level in phon.
    """
    sone = np.atleast_1d(sone).astype(float)
    sone = sone.reshape(-1, 1) if sone.ndim == 1 else sone

    phon = np.zeros_like(sone)

    idx = sone >= 1
    phon[idx] = 40 + 33.22 * np.log10(sone[idx])
    phon[~idx] = 40 * np.power(sone[~idx] + 0.0005, 0.35)

    return phon

def get_exceeded_value(x: FloatArrayLike, percent: float) -> np.ndarray:
    """Return the value exceeded *percent* % of the time (per channel).

    Parameters
    ----------
    x : array‑like
        Input data, shape (T,) or (T, C≤3).
    percent : float
        Percentage of exceedance, 1 ≤ value ≤ 99.

    Returns
    -------
    ndarray
        Exceeded values for each channel.
    """
    x = np.asarray(x)

    # Accept row-vector input and transpose so that time is axis-0
    if x.ndim == 1:
        x = x[:, None]
    elif x.shape[1] > 3 and x.shape[0] <= 3:
        x = x.T

    if x.shape[1] > 3:
        raise ValueError("Input has more than three channels.")

    n = x.shape[0]
    idx = int(np.floor((100 - percent) / 100 * n))
    idx = max(idx - 1, 0)  # shift to 0-based, ensure non-negative

    values = np.sort(x, axis=0)[idx, :]

    return values

def get_statistics(data: FloatArrayLike, metric: str) -> Dict[str, np.ndarray]:
    """Compute descriptive statistics for a psycho‑acoustic time‑series.

    Parameters
    ----------
    data : array‑like
        Time‑series data.
    metric : str
        Verbose metric identifier.

    Returns
    -------
    dict[str, ndarray]
        Mapping ``<prefix><label> → statistic``.
    """

    metric_map = {
        "Loudness_ISO532_1":                "N",
        "Sharpness_DIN45692":               "S",
        "Roughness_Daniel1997":             "R",
        "FluctuationStrength_Osses2016":    "FS",
        "Tonality_Aures1985":               "K",
        "PsychoacousticAnnoyance_Di2016":   "PA",
        "PsychoacousticAnnoyance_More2010": "PA",
        "PsychoacousticAnnoyance_Zwicker1999": "PA",
        "Loudness_ECMA418_2":               "N",
        "Tonality_ECMA418_2":               "T",
        "Roughness_ECMA418_2":              "R",
    }

    try:
        prefix = metric_map[metric]
    except KeyError as exc:
        prefix = "X"
        warnings.warn(
            f"Unknown metric {metric!r}. Defaulting to 'X'.",
            category=RuntimeWarning,   # or UserWarning
            stacklevel=2               # show caller’s line number
        )

    x = np.asarray(data)
    if x.ndim == 1:
        x = x[:, None]
    elif x.shape[1] > 3 and x.shape[0] <= 3:
        x = x.T

    if x.shape[1] > 3:
        raise ValueError("Input has more than three channels.")

    labels = [
        "max", "min", "mean", "std",
        "1", "2", "3", "4", "5",
        "10", "20", "30", "40", "50",
        "60", "70", "80", "90", "95",
    ]

    out = {}
    for lbl in labels:
        if lbl == "max":
            val = np.max(x, axis=0)
        elif lbl == "min":
            val = np.min(x, axis=0)
        elif lbl == "mean":
            val = np.mean(x, axis=0)
        elif lbl == "std":
            val = np.std(x, axis=0, ddof=1)
        elif lbl == "50":
            val = np.median(x, axis=0)
        else:
            val = get_exceeded_value(x, int(lbl))

        out[prefix + lbl] = val

    return out

def get_bark(N: int, qb: ArrayLikeInt, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert FFT‑bin frequencies to Bark numbers.

    Parameters
    ----------
    N : int
        FFT length.
    qb : array‑like of int
        Bin indices to convert.
    freqs : ndarray
        Frequencies in Hertz corresponding to ``qb``.

    Returns
    -------
    Bark : ndarray
        Bark numbers at ``qb`` (zeros elsewhere).
    Bark_raw : ndarray
        Zwicker Bark reference table.
    """

    Bark_raw = np.array([
        [ 0,     0,     50,     0.5],
        [ 1,   100,    150,     1.5],
        [ 2,   200,    250,     2.5],
        [ 3,   300,    350,     3.5],
        [ 4,   400,    450,     4.5],
        [ 5,   510,    570,     5.5],
        [ 6,   630,    700,     6.5],
        [ 7,   770,    840,     7.5],
        [ 8,   920,   1000,     8.5],
        [ 9,  1080,   1170,     9.5],
        [10,  1270,   1370,    10.5],
        [11,  1480,   1600,    11.5],
        [12,  1720,   1850,    12.5],
        [13,  2000,   2150,    13.5],
        [14,  2320,   2500,    14.5],
        [15,  2700,   2900,    15.5],
        [16,  3150,   3400,    16.5],
        [17,  3700,   4000,    17.5],
        [18,  4400,   4800,    18.5],
        [19,  5300,   5800,    19.5],
        [20,  6400,   7000,    20.5],
        [21,  7700,   8500,    21.5],
        [22,  9500,  10500,    22.5],
        [23, 12000,  13500,    23.5],
        [24, 15500,  20000,    24.5],
    ], dtype=float)

    bark_sorted = np.column_stack([
        np.sort(np.r_[Bark_raw[:, 1], Bark_raw[:, 2]]),
        np.sort(np.r_[Bark_raw[:, 0], Bark_raw[:, 3]])
    ])

    bark = np.zeros(int(round(N / 2 + 1)), dtype=float)

    bark[qb] = np.interp(freqs, bark_sorted[:, 0], bark_sorted[:, 1])

    return bark, Bark_raw

def from_db(gain_db: FloatArrayLike, divisor: float = 20.0) -> np.ndarray:
    """Convert decibel magnitude to linear gain.

    Parameters
    ----------
    gain_db : float or array‑like
        Magnitude in decibels.
    divisor : float, default 20
        Denominator in dB definition (20 → amplitude, 10 → power).

    Returns
    -------
    ndarray
        Linear‑gain values.
    """
    gain_db = np.asarray(gain_db, dtype=float)
    out = 10.0 ** (gain_db / divisor)
    return out

def create_a0_FIR(
    f: np.ndarray,
    a0: np.ndarray,
    N: int,
    fs: float,
    *,
    plot: bool = False
) -> Optional[np.ndarray]:
    """Design an FIR filter from a breakpoint‑magnitude curve.

    Parameters
    ----------
    f : ndarray
        Break‑point frequencies in Hz (< fs/2).
    a0 : ndarray
        Desired magnitudes at the same break points.
    N : int
        Filter order (``num_taps = N + 1``).
    fs : float
        Sampling rate in Hz.
    plot : bool, default False
        If ``True`` only plots are produced and the function returns ``None``.

    Returns
    -------
    ndarray | None
        FIR taps or ``None`` when ``plot is True``.
    """

    f = np.asarray(f, dtype=float)
    a0 = np.asarray(a0, dtype=float)

    if f.ndim != 1 or a0.ndim != 1:
        raise ValueError("f and a0 must be 1-D arrays.")
    if f.size != a0.size:
        raise ValueError("f and a0 must have the same length.")
    if f[0] <= 0:
        raise ValueError("First break-point frequency must be > 0 Hz.")
    if np.any(np.diff(f) <= 0):
        raise ValueError("f must be strictly increasing.")
    if f[-1] >= fs / 2:
        raise ValueError("All break-point frequencies must be below fs/2.")

    f_ext  = np.hstack(([0.0],  f,  [fs / 2.0]))
    a0_ext = np.hstack(([a0[0]], a0, [a0[-1]]))

    B = firwin2(numtaps=N + 1, freq=f_ext, gain=a0_ext, fs=fs)

    if plot:
        w, h = freqz(B, worN=N // 2, fs=fs)
        plt.figure()
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.title("FIR filter resulting from the input curve a0")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.xlim([0, fs / 2])
        plt.grid(True, which="both", ls=":")
        plt.legend([f"{N} taps"])
        plt.show()
        return

    return B

def calculate_a0(
    fs: float,
    N: int,
    a0_type: str = "fastl2007",
    *,
    plot: bool = False
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Generate the outer/middle‑ear transfer filter *a₀*.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    N : int
        FFT length (filter length = ``N + 1`` taps).
    a0_type : {'fastl2007', 'fluctuationstrength_osses2016'}
        Reference curve identifier.
    plot : bool, default False
        If ``True`` only plots are produced.

    Returns
    -------
    B : ndarray | None
        FIR taps or ``None`` when ``plot=True``.
    freqs : ndarray
        Frequency grid (Hz).
    a0_lin : ndarray
        Linear magnitude reference at ``freqs``.
    """

    # 1) frequency grid   (20 Hz – 20 kHz, positive FFT bins only)
    df     = fs / N
    k_min  = int(round(20.0    / df))
    k_max  = int(round(20_000.0 / df))
    k_max  = min(k_max, N // 2)
    qb     = np.arange(k_min, k_max + 1)
    freqs  = qb * df

    # 2) Bark scale
    bark, _ = get_bark(N, qb, freqs)

    # 3) choose breakpoint table  (Bark, dB)
    a0_type = a0_type.lower()
    if a0_type == 'fastl2007':
        a0tab = np.array([
            [0,    0], [10,  0], [12,  1.15], [13,  2.31], [14,  3.85],
            [15,  5.62], [16,  6.92], [16.5, 7.38], [17,  6.92], [18, 4.23],
            [18.5, 2.31], [19, 0], [20, -1.43], [21, -2.59], [21.5, -3.57],
            [22, -5.19], [22.5, -7.41], [23, -11.3], [23.5, -20],
            [24, -40], [25, -130], [26, -999]
        ], dtype=float)
    elif a0_type == 'fluctuationstrength_osses2016':
        a0tab = np.array([
            [0, 0], [10, 0], [19, 0], [20, -1.43], [21, -2.59],
            [21.5, -3.57], [22, -5.19], [22.5, -7.41], [23, -11.3],
            [23.5, -20], [24, -40], [25, -130], [26, -999]
        ], dtype=float)
    else:
        raise ValueError("a0_type must be 'fastl2007' or 'fluctuationstrength_osses2016'")

    # 4) interpolate → dB → linear
    a0_lin_full = np.zeros(int(round(N/2 + 1)))
    db_interp   = np.interp(bark[qb], a0tab[:, 0], a0tab[:, 1], left=np.nan, right=np.nan)
    a0_lin_full[qb] = from_db(db_interp)
    a0_lin_full[np.isnan(a0_lin_full)] = 0.0

    a0_lin = a0_lin_full[qb]     # slice actually used

    # 5) design FIR (or plot only)
    B = create_a0_FIR(freqs, a0_lin, N, fs, plot=plot)

    return B, freqs, a0_lin

def calibrate(
    input_signal: np.ndarray,
    ref_signal: np.ndarray,
    reference_level: float,
    *,
    return_dbfs: bool = False
) -> Tuple[np.ndarray, float, Optional[float]]:
    """Scale *input_signal* to a known SPL reference.

    Parameters
    ----------
    input_signal : ndarray
        Signal to be calibrated.
    ref_signal : ndarray
        Reference recording at known level.
    reference_level : float
        SPL of ``ref_signal`` in dB (rms).
    return_dbfs : bool, default False
        If ``True`` also return the SPL equivalent of 0 dBFS.

    Returns
    -------
    calibrated_signal : ndarray
    cal_factor : float
    dBFS : float, optional
    """
    # Ensure floating-point math
    input_signal = np.asarray(input_signal, dtype=float)
    ref_signal = np.asarray(ref_signal, dtype=float)

    # --- 1. Calibration factor -----------------------------------------------
    #     CalFactor = √(10^(L_ref/10) * 4e-10 / mean(ref_signal²))
    mean_sq_ref = np.mean(ref_signal ** 2)
    cal_factor = np.sqrt((10.0 ** (reference_level / 10.0)) * 4e-10 / mean_sq_ref)

    # --- 2. Apply scaling ------------------------------------------------------
    calibrated_signal = cal_factor * input_signal

    # --- 3. Optional dBFS ------------------------------------------------------
    if return_dbfs:
        rms_ref = np.sqrt(mean_sq_ref)
        dbfs = reference_level - 20.0 * np.log10(rms_ref)
        return calibrated_signal, cal_factor, dbfs

    return calibrated_signal, cal_factor

def get_defaults(model_name: str) -> Dict[str, Any]:
    """Return default‑parameter dictionary for a psycho‑acoustic model.

    Parameters
    ----------
    model_name : str
        Model identifier (case‑sensitive).

    Returns
    -------
    dict
        Default parameter dictionary; each key has a matching ``*_description``.
    """

    # --------- Group 1: Fluctuation Strength (Osses 2016) ---------
    if model_name in {
        "FluctuationStrength_Osses2016",
        "FluctuationStrength_Osses2016_from_wavfile",
    }:
        return {
            "method": 1,
            "method_description": (
                "0 = stationary method (win size = length of sound); "
                "1 = time varying (window size = 2 s)"
            ),
            "time_skip": 0,
            "time_skip_description": "time_skip, in seconds for statistical calculations",
            "show": 1,
            "show_description": "Plots the outputs (optional parameter)",
        }

    # --------- Group 2: Loudness ISO 532-1 ------------------------
    if model_name in {
        "Loudness_ISO532_1",
        "Loudness_ISO532_1_from_wavfile",
    }:
        return {
            "field": 0,
            "field_description": "0 = free field; 1 = diffuse field",
            "method": 2,
            "method_description": "1 = stationary method; 2 = time varying",
            "time_skip": 0.5,
            "time_skip_description": (
                "time_skip, in seconds for level (stationary signals) and "
                "statistics (stationary and time-varying signals) calculations"
            ),
            "show": 1,
            "show_description": "Plots the outputs (optional parameter)",
        }

    # --------- Group 3: Sharpness DIN 45692 -----------------------
    if model_name in {
        "Sharpness_DIN45692",
        "Sharpness_DIN45692_from_loudness",
    }:
        return {
            "field": 0,
            "field_description": "0 = free field; 1 = diffuse field for loudness",
            "method": 2,
            "method_description": "1 = stationary method; 2 = time varying for loudness",
            "time_skip": 0.5,
            "time_skip_description": (
                "time_skip, in seconds for level (stationary signals) and "
                "statistics (stationary and time-varying signals) calculations"
            ),
            "show_loudness": 0,
            "show_loudness_description": "Plots the loudness outputs (optional parameter)",
            "show": 1,
            "show_description": "Plots the outputs (optional parameter)",
            "weight_type": "DIN45692",
            "weight_type_description": (
                "Type of sharpness models. Options: 'DIN45692', 'aures', 'bismarck'"
            ),
        }

    # --------- Group 4: Roughness (Daniel 1997) -------------------
    if model_name in {
        "Roughness_Daniel1997",
        "Roughness_Daniel1997_from_wavfile",
    }:
        return {
            "time_skip": 0,
            "time_skip_description": "time_skip",
            "show": 0,
            "show_description": "Plots the outputs (optional parameter)",
        }

    # --------- Group 5: Tonality (Aures 1985) ---------------------
    if model_name in {
        "Tonality_Aures1985",
        "Tonality_Aures1985_from_wavfile",
    }:
        return {
            "Loudness_field": 0,
            "Loudness_field_description": "Loudness: 0 = free field; 1 = diffuse field",
            "time_skip": 0,
            "time_skip_description": "time_skip",
            "show": 0,
            "show_description": "Plots the outputs (optional parameter)",
        }

    # --------- Group 6: Psycho-acoustic Annoyance -----------------
    if model_name in {
        "PsychoacousticAnnoyance_Di2016",
        "PsychoacousticAnnoyance_More2010",
        "PsychoacousticAnnoyance_Zwicker1999",
    }:
        return {
            "Loudness_field": 0,
            "Loudness_field_description": "Loudness: 0 = free field; 1 = diffuse field",
            "time_skip": 0.2,
            "time_skip_description": "time_skip",
            "showPA": 0,
            "showPA_description": (
                "Plots the outputs for psychoacoustic annoyance (optional parameter)"
            ),
            "show": 0,
            "show_description": "Plots the outputs (optional parameter)",
        }

    # --------- Unknown model -------------------------------------
    raise ValueError("Unrecognised model name: '{}'".format(model_name))

# ---------------------------
#### ECMA418_2 FUNCTIONS ####
# ---------------------------

def shm_auditory_filt_bank(signal: np.ndarray, outplot: bool = False) -> np.ndarray:
    """Apply the ECMA‑418‑2 auditory filter bank.

    Parameters
    ----------
    signal : np.ndarray
        Mono audio time‑series **sampled at 48 kHz**. Shape ``(N,)``.
    outplot : bool, default ``False``
        If *True*, draw the combined magnitude/phase response of the 53 filters.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 53)``, with one column per ½‑Bark band.
    """

    # Argument validation
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1‑D (mono) array")
    if not np.isrealobj(signal):
        raise ValueError("signal must be real‑valued")

    sample_rate: int = 48_000  # Hz

    # Constants (§ 5.1.4.1)
    delta_freq0: float = 81.9289
    c: float = 0.1618

    half_bark = np.arange(0.5, 26.5 + 0.5, 0.5)  # 0.5 … 26.5 inclusive (53 bands)
    band_centre_freqs = (delta_freq0 / c) * np.sinh(c * half_bark)
    dfz = np.sqrt(delta_freq0**2 + (c * band_centre_freqs) ** 2)

    k: int = 5  # filter order (footnote 5 ECMA‑418‑2)
    e_i = np.array([0, 1, 11, 11, 1], dtype=float)
    out = np.empty((signal.size, half_bark.size), dtype=float)
    if outplot:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        ax_mag.set_xscale("log")
        ax_phase.set_xscale("log")

    # Main loop: highest → lowest band (to mimic MATLAB for reproducibility)
    for z_idx in range(half_bark.size - 1, -1, -1):
        tau = (
            (1 / (2 ** (2 * k - 1)))
            * comb(2 * k - 2, k - 1)
            * (1 / dfz[z_idx])
        )
        d = np.exp(-1 / (sample_rate * tau))

        bp = np.exp(
            1j
            * 2
            * np.pi
            * band_centre_freqs[z_idx]
            * np.arange(k + 2)
            / sample_rate
        )

        m_fb = np.arange(1, k + 1)
        a_m = np.concatenate(([1.0], (-d) ** m_fb * comb(k, m_fb))) * bp[: k + 1]

        m_ff = np.arange(0, k)
        denom = np.sum(e_i[1:] * d ** np.arange(1, k))
        b_m = (((1 - d) ** k) / denom) * (d ** m_ff) * e_i[:k] * bp[:k]

        filtered = lfilter(b_m, a_m, signal)
        out[:, z_idx] = 2.0 * np.real(filtered)
        if outplot:
            w, h = freqz(b_m, a_m, worN=10_000, fs=sample_rate, whole=True)
            ax_mag.semilogx(w, 20 * np.log10(np.abs(h)))
            ax_phase.semilogx(w, np.unwrap(np.angle(h)) * 180 / np.pi)

    if outplot:
        for ax in (ax_mag, ax_phase):
            ax.grid(True, which="both", ls=":")
            ax.set_xlim(20, 20_000)
            ax.set_xticks(
                [31.5, 63, 125, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
            )
            ax.set_xticklabels(
                [
                    "31.5",
                    "63",
                    "125",
                    "250",
                    "500",
                    "1k",
                    "2k",
                    "4k",
                    "8k",
                    "16k",
                ]
            )
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_phase.set_ylabel("Phase (°)")
        ax_phase.set_xlabel("Frequency (Hz)")
        plt.tight_layout()

    return out

def shm_basis_loudness(signal_segmented: np.ndarray,
                       band_centre_freq: float | None = None,
                       tol: float = 1.0):
    """Self‑contained Sottek Hearing Model basis loudness (ECMA‑418‑2:2024)

   signal_segmented : ndarray
        Segmented pressure signal with shape ``(n_samples, n_blocks[, n_bands])``.
        Must be real‑valued.
    band_centre_freq : float, optional
        Centre frequency (Hz) *only* when *signal_segmented* is 2‑D (single
        band).  Omit for 3‑D input that already contains all 53 half‑Bark
        bands on its third axis.
    tol : float, default 1.0
        Matching tolerance (Hz) when validating *band_centre_freq* against
        the 53 standard half‑Bark centres.  ``tol=0`` enforces exact match.

    Returns
    -------
    signal_rect_seg : ndarray  – rectified signal (same shape as input)
    basis_loudness  : ndarray  – basis loudness (same shape as input)
    block_rms       : ndarray  – block RMS (1‑D if input has one band, else 2‑D)
    """

    # Constants
    DELTA_F0 = 81.9289
    C = 0.1618
    half_bark = np.arange(0.5, 26.5 + 0.0001, 0.5)  # 0.5 … 26.5 inclusive
    band_centres = (DELTA_F0 / C) * np.sinh(C * half_bark)  # Eq. 9

    CAL_N = 0.0211668
    CAL_NX = 1.00132
    ALPHA = 1.5  # a in Eq. 23

    p_threshold = 2e-5 * 10 ** (np.arange(15, 86, 10) / 20)  # Pa (8‑vals)
    v = np.array([1, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434])
    diff_v_over_a = np.diff(v) / ALPHA  # shape (8,)

    ltqz = np.array([
        0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453, 0.0365, 0.0298,
        0.0247, 0.0207, 0.0176, 0.0151, 0.0131, 0.0115, 0.0103, 0.0093,
        0.0086, 0.0081, 0.0077, 0.0074, 0.0073, 0.0072, 0.0071, 0.0072,
        0.0073, 0.0074, 0.0076, 0.0079, 0.0082, 0.0086, 0.0092, 0.0100,
        0.0109, 0.0122, 0.0138, 0.0157, 0.0172, 0.0180, 0.0180, 0.0177,
        0.0176, 0.0177, 0.0182, 0.0190, 0.0202, 0.0217, 0.0237, 0.0263,
        0.0296, 0.0339, 0.0398, 0.0485, 0.0622
    ])

    # Validation
    if not np.isrealobj(signal_segmented):
        raise TypeError("signal_segmented must be real‑valued")

    if signal_segmented.ndim not in (2, 3):
        raise ValueError("signal_segmented must be 2‑D or 3‑D")

    if band_centre_freq is None and signal_segmented.ndim == 2:
        raise ValueError("band_centre_freq required for 2‑D input")

    if band_centre_freq is not None and signal_segmented.ndim == 3:
        raise ValueError("band_centre_freq should be omitted for 3‑D input")

    # Centre‑frequency handling
    if band_centre_freq is not None:
        idx = int(np.abs(band_centres - band_centre_freq).argmin())
        if abs(band_centres[idx] - band_centre_freq) > tol:
            raise ValueError(
                f"{band_centre_freq} Hz is not within ±{tol} Hz of any standard half‑Bark centre; "
                f"closest is {band_centres[idx]:.2f} Hz.")

    ## Core processing ##
    signal_rect_seg = np.maximum(signal_segmented, 0.0)

    # Block RMS (Eq. 22) – factor 2 because rectified signal is positive‑only
    n_samples = signal_rect_seg.shape[0]
    block_rms = np.sqrt((2.0 / n_samples) * np.sum(signal_rect_seg ** 2, axis=0))

    # Loudness transform (Eqs. 23–24)
    term = (1 + (block_rms[..., None] / p_threshold) ** ALPHA) ** diff_v_over_a
    band_loudness = CAL_N * CAL_NX * (block_rms / 20e-6) * np.prod(term, axis=-1)

    # Threshold‑in‑quiet correction (Eq. 25)
    if band_centre_freq is not None:  # 2‑D input, single band
        basis_loudness = band_loudness - ltqz[idx]
    else:                             # 3‑D input, all 53 bands
        ltqz_reshaped = ltqz.reshape((1,) * (block_rms.ndim - 1) + (53,))
        basis_loudness = band_loudness - ltqz_reshaped

    basis_loudness = np.maximum(basis_loudness, 0.0)

    # Squeeze singleton dims in RMS for MATLAB‑like behaviour
    block_rms = np.squeeze(block_rms)

    return signal_rect_seg, basis_loudness, block_rms

def shm_noise_red_lowpass(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply the ECMA‑418‑2 low‑pass noise‑reduction filter.

    Parameters
    ----------
    signal : np.ndarray
        Audio samples with shape (N,) or (N, C). Time must be on axis 0.
    fs : float
        Sample‑rate in Hz (> 0).

    Returns
    -------
    np.ndarray
        Filtered signal of identical shape and `float64` dtype.
    """

    # Validation
    if not isinstance(signal, np.ndarray):
        raise TypeError("`signal` must be a NumPy array")

    if signal.ndim not in (1, 2):
        raise ValueError("`signal` must be 1‑D (mono) or 2‑D (time × channels)")

    if not np.isrealobj(signal):
        raise ValueError("`signal` must contain real‑valued samples")

    if not (isinstance(fs, (float, int)) and fs > 0):
        raise ValueError("`fs` must be a positive scalar sample‑rate (Hz)")

    # Coefficient design (ECMA‑418‑2:2024, Equations 14‑15)
    k = 3
    e_i = np.array([0.0, 1.0, 1.0])          # Footnote 21
    tau = (1 / 32) * (6 / 7)                 # Footnote 20
    d = np.exp(-1.0 / (fs * tau))            # §5.1.4.2

    # Denominator (a): Equation 14
    m = np.arange(1, k + 1)
    nck = np.array([3, 3, 1], dtype=float)   # nchoosek(3,m) for m = 1‥3
    a = np.concatenate(([1.0], ((-d) ** m) * nck))

    # Numerator (b): Equation 15
    m = np.arange(0, k)          # 0‥k‑1
    i = np.arange(1, k)          # 1‥k‑1
    denom = np.sum(e_i[i] * (d ** i))
    gain = ((1 - d) ** k) / denom
    b = gain * (d ** m) * e_i     # note e_i[0] == 0 → b[0] == 0

    # Filtering
    was_1d = (signal.ndim == 1)
    x = signal[:, None] if was_1d else signal  # shape (N, C)

    y = lfilter(b, a, x, axis=0)

    return y.ravel() if was_1d else y

def shm_out_mid_ear_filter(
    signal: Union[np.ndarray, list],
    fieldtype: Literal["free-frontal", "diffuse"] = "free-frontal",
    outplot: bool = False,
):
    """Outer‑ & middle‑ear filter (ECMA‑418‑2:2024) for calibrated signals.

    Parameters
    ----------
    signal : array‑like (N,) or (N, C)
        Time‑series sound‑pressure waveform sampled at **48 kHz**. Rows = time; columns = channels.
    fieldtype : {'free-frontal', 'diffuse'}, default 'free-frontal'
        * 'free-frontal' – applies all 8 biquad sections (free‑field + middle‑ear).
        * 'diffuse'       – omits the first two free‑field stages (6 sections total).
    outplot : bool, default False
        If *True*, plots magnitude (dB) and unwrapped phase (°) of the overall filter.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape and dtype as *signal*.

    Notes
    -----
    • Implementation follows ECMA‑418‑2:2024 § 5.1.3.2 (Hearing Model of Sottek).
    • Uses SciPy's *second‑order‑section* (`sosfilt`, `sosfreqz`) API – no custom files required.
    • Causal processing along axis 0 enables block‑wise streaming via ``sosfilt_zi``/``sosfilt`` if needed.
    """

    # Validation
    if isinstance(signal, list):
        signal = np.asarray(signal, dtype=float)
    elif not isinstance(signal, np.ndarray):
        raise TypeError("'signal' must be a NumPy array or a list of numbers")
    if not np.isrealobj(signal):
        raise ValueError("'signal' must contain real values only")

    fieldtype = str(fieldtype).lower()
    if fieldtype not in {"free-frontal", "diffuse"}:
        raise ValueError("fieldtype must be 'free-frontal' or 'diffuse'")

    signal_was_1d = False
    if signal.ndim == 1:
        signal = signal[:, None]
        signal_was_1d = True
    elif signal.ndim != 2:
        raise ValueError("'signal' must be 1‑D or 2‑D (time × channels)")


    # ECMA‑418‑2 biquad coefficients
    b0 = np.array([
        1.015896020255593,
        0.958943219304445,
        0.961371976333197,
        2.225803503609735,
        0.471735128494163,
        0.115267139824401,
        0.988029297230954,
        1.952237687301361,
    ])
    b1 = np.array([
        -1.925298877776079,
        -1.806088011849494,
        -1.763632154338248,
        -1.434650484792157,
        -0.366091796830044,
        0.0,
        -1.912433802933870,
        0.162319983017519,
    ])
    b2 = np.array([
        0.922118060364679,
        0.876438777856084,
        0.821787991845146,
        -0.498204282194628,
        0.244144703885020,
        -0.115267139824401,
        0.926131550180785,
        -0.667994113035186,
    ])
    a0 = np.ones_like(b0)
    a1 = np.array([
        -1.925298877776079,
        -1.806088011849494,
        -1.763632154338248,
        -1.434650484792157,
        -0.366091796830044,
        -1.796002566692014,
        -1.912433802933871,
        0.162319983017519,
    ])
    a2 = np.array([
        0.938014080620272,
        0.835381997160530,
        0.783159968178343,
        0.727599221415107,
        -0.284120167620817,
        0.805837815618546,
        0.914160847411739,
        0.284243574266175,
    ])

    if fieldtype == "free-frontal":
        idx = slice(None)  # all 8 sections
    else:  # 'diffuse'
        idx = slice(2, None)  # omit first two (free‑field) stages

    sos = np.column_stack([b0[idx], b1[idx], b2[idx], a0[idx], a1[idx], a2[idx]])


    # Filtering (axis 0 = time)
    filtered = sosfilt(sos, signal, axis=0)

    if signal_was_1d:
        filtered = filtered.ravel()

    # Plotting (optional)
    if outplot:
        w, h = sosfreqz(sos, worN=10_000, fs=48_000)
        mag_db = 20 * np.log10(np.abs(h))
        phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
        ax1.semilogx(w, mag_db)
        ax1.set(xlabel="Frequency (Hz)", ylabel="|H| (dB)", xlim=(20, 20e3))
        ax1.set_xticks([31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
        ax1.grid(True, which="both")
        ax1.set_title(fieldtype)

        ax2.semilogx(w, phase_deg)
        ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (°)", xlim=(20, 20e3))
        ax2.set_xticks([31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
        ax2.grid(True, which="both")

        plt.show()

    return filtered

# ------------------
#### VALIDATION ####
# ------------------
if __name__ == "__main__":
    #see("sound_files\ExSignal_A320_auralized_departure_104dBFS.wav")

    # z = hz2bark(1000)
    # print(z)
    # f = bark2hz(z)
    # print(f)

    # x = np.linspace(0, 10, 101)
    # y = np.exp(-((x - 5)**2) / (2 * 2**2))  # same as gaussmf(x, [2, 5])
    # p5 = get_exceeded_value(y, 5)
    # p90 = get_exceeded_value(y, 90)
    # print(p5,p90)
    # plt.plot(x, y, label='Signal')
    # plt.axhline(p5, color='r', linestyle='--', label='5% exceedance')
    # plt.axhline(p90, color='g', linestyle='--', label='90% exceedance')
    # plt.legend()
    # plt.title("Exceedance Thresholds")
    # plt.show()

    # sr, input = wavfile.read('sound_files\ExSignal_A320_auralized_departure_104dBFS.wav')
    # print(sr, input.shape)
    # print(input)
    # output = get_statistics(input, 'test')
    # print(output)

    # N       = 1024
    # qb      = np.arange(1, 11)         # pretend we want the first ten bins
    # freqs   = qb * (44100 / N)         # Hz for those bins at 44.1 kHz Fs
    # bark, table = get_bark(N, qb, freqs)
    # print("Bark values at qb:\n", bark[qb])

    #print(from_db(123,10))

    # fs, N = 44_100, 4096
    # B1, f, a0_fastl = calculate_a0(fs, N)
    # B2, _, a0_osses = calculate_a0(fs, N, 'fluctuationstrength_osses2016')
    # import matplotlib.pyplot as plt
    # plt.plot(f, 20*np.log10(a0_fastl), label='Fastl 2007')
    # plt.plot(f, 20*np.log10(a0_osses), '--', label='Osses 2016')
    # plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    # plt.ylim([-100, 10]); plt.grid(True); plt.legend(); plt.show()

    # fs = 48000                                # 48 kHz sample rate
    # t  = np.arange(0, 1.0, 1/fs)              # 1-s test tone
    # ref = 0.1 * np.sin(2*np.pi*1000*t)        # rms = 0.1/√2 ≈ 0.0707
    # inp = 0.05 * np.sin(2*np.pi*500*t)      # random “measurement” signal
    # cal_sig, g, dbfs = calibrate(inp, ref, 94, return_dbfs=True)
    # print(f"Gain applied: {g:.3f}")
    # print(f"0 dBFS equals {dbfs:.2f} dB SPL")
    # print(f"Calibrated signal rms: {np.sqrt(np.mean(cal_sig**2)):.4f}")

    # model = "Sharpness_DIN45692"
    # parameters = get_defaults(model)
    # for key, value in parameters.items():
    #     print(f"{key}: {value}")

    # sr = 48_000
    # duration = 1.0
    # x = np.random.randn(int(sr * duration))
    # y = shm_auditory_filt_bank(x, outplot=True)
    # plt.show()
    # sf.write("band01.wav", y[:, 0] / np.max(np.abs(y[:, 0])) * 0.99, sr)

    # np.random.seed(0)
    # sig2d = np.random.randn(1024, 8) * 0.02
    # rect, loud, rms = shm_basis_loudness(sig2d, 1027.02470862)
    # print("2‑D test shapes:", rect.shape, loud.shape, rms.shape)
    # print("2‑D loudness ≥ 0:", np.all(loud >= 0))
    # sig3d = np.random.randn(1024, 8, 53) * 0.02
    # rect3, loud3, rms3 = shm_basis_loudness(sig3d)
    # print("3‑D test shapes:", rect3.shape, loud3.shape, rms3.shape)
    # print("3‑D loudness ≥ 0:", np.all(loud3 >= 0))
    # assert rect.shape == sig2d.shape and rect3.shape == sig3d.shape
    # assert loud.min() >= 0 and loud3.min() >= 0
    # print("Basic sanity checks passed.")

    # fs = 48_000
    # t = np.linspace(0, 0.5, int(0.5 * fs), endpoint=False)
    # tone = 0.5 * np.sin(2 * np.pi * 6_000 * t)
    # noisy = tone + 0.1 * np.random.randn(t.size)
    # clean = shm_noise_red_lowpass(noisy, fs)
    # plt.figure()
    # plt.title("ECMA‑418‑2 low‑pass noise reduction")
    # plt.plot(t, noisy, alpha=0.4, label="Noisy input")
    # plt.plot(t, clean, linewidth=1.2, label="Filtered output")
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.tight_layout()
    # plt.show()

    fs = 48_000
    x = np.random.randn(fs)
    y = shm_out_mid_ear_filter(x, fieldtype="free-frontal", outplot=True)
    print("Processed", y.shape, "samples")
    