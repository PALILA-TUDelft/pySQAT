
##########################
#### UTILITIES MODULE ####
##########################

import os
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import firwin2, freqz, lfilter, sosfilt, sosfreqz, resample_poly
from scipy.io import wavfile
from scipy.special import comb
import soundfile as sf
from IPython.display import Audio, display
from pathlib import Path
from typing import Union, Sequence, Any, Tuple, Optional, Dict, Literal

FloatArrayLike = Union[float, Sequence[float], np.ndarray]
ArrayLikeInt   = Union[Sequence[int], np.ndarray]
ArrayLike = Union[np.ndarray, float, int]

# ----------------------
#### MAIN FUNCTIONS ####
# ----------------------

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
    f = np.asarray(f)
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
    k = np.arange(-20, 13)
    f = f0 * 2 ** (k / 3)

    zt = hz2bark(f)

    interpolator = interp1d(zt, f, kind='linear', fill_value="extrapolate")
    f = interpolator(z)

    return f

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
    phon = np.asarray(phon).flatten()
    phon = phon[:, np.newaxis]
    
    sone = np.zeros_like(phon)

    idx = phon >= 40

    sone[idx] = 2 ** (0.1 * (phon[idx] - 40))

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
    phon[~idx] = 40 * (sone[~idx] + 0.0005) ** 0.35

    return phon

def get_exceeded_value(input: FloatArrayLike, PercentValue: float) -> np.ndarray:
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
    input = np.asarray(input)

    if input.ndim == 1:
        input = input[:, None]
    elif input.shape[1] > 3 and input.shape[0] <= 3:
        input = input.T

    if input.shape[1] > 3:
        raise ValueError("Input has more than three channels.")

    n = input.shape[0]
    X_index = int(np.floor((100 - PercentValue) / 100 * n))
    X_index = max(X_index - 1, 0)

    sort_input = np.sort(input, axis=0)[X_index, :]

    return sort_input

def get_statistics(input: FloatArrayLike, metric: str) -> Dict[str, np.ndarray]:
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
        var_string = metric_map[metric]
    except KeyError as exc:
        var_string = "X"
        warnings.warn(
            f"Unknown metric {metric!r}. Defaulting to 'X'.",
            category=RuntimeWarning,   # or UserWarning
            stacklevel=2               # show caller’s line number
        )

    input = np.asarray(input)
    if input.ndim == 1:
        input = input[:, None]
    elif input.shape[1] > 3 and input.shape[0] <= 3:
        input = input.T

    if input.shape[1] > 3:
        raise ValueError("Input has more than three channels.")

    string_vector = [
        "max", "min", "mean", "std",
        "1", "2", "3", "4", "5",
        "10", "20", "30", "40", "50",
        "60", "70", "80", "90", "95",
    ]

    temp_varName = {}
    for k in string_vector:
        if k == "max":
            temp_val = np.max(input, axis=0)
        elif k == "min":
            temp_val = np.min(input, axis=0)
        elif k == "mean":
            temp_val = np.mean(input, axis=0)
        elif k == "std":
            temp_val = np.std(input, axis=0, ddof=1)
        elif k == "50":
            temp_val = np.median(input, axis=0)
        else:
            temp_val = get_exceeded_value(input, int(k))

        temp_varName[var_string + k] = temp_val

    return temp_varName

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

def from_db(gain_dB: FloatArrayLike, divisor: float = 20.0) -> np.ndarray:
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
    gain_dB = np.asarray(gain_dB, dtype=float)
    gain = 10.0 ** (gain_dB / divisor)
    return gain

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

    f  = np.hstack(([0.0],  f,  [fs / 2.0]))
    a0 = np.hstack(([a0[0]], a0, [a0[-1]]))

    B = firwin2(numtaps=N + 1, freq=f, gain=a0, fs=fs)

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

    # 1) Frequency Grid
    df     = fs / N
    k_min  = int(round(20.0    / df))
    k_max  = int(round(20_000.0 / df))
    k_max  = min(k_max, N // 2)
    qb     = np.arange(k_min, k_max + 1)
    freqs  = qb * df

    # 2) Bark Scale
    bark, _ = get_bark(N, qb, freqs)

    # 3) Choose Breakpoint Table
    a0_type = a0_type.lower()
    if a0_type == 'fastl2007':
        a0tab = np.array([
            [0,    0],
            [10,  0],
            [12,  1.15],
            [13,  2.31],
            [14,  3.85],
            [15,  5.62],
            [16,  6.92],
            [16.5, 7.38],
            [17,  6.92],
            [18, 4.23],
            [18.5, 2.31],
            [19, 0],
            [20, -1.43],
            [21, -2.59],
            [21.5, -3.57],
            [22, -5.19],
            [22.5, -7.41],
            [23, -11.3],
            [23.5, -20],
            [24, -40],
            [25, -130],
            [26, -999]
        ], dtype=float)
    elif a0_type == 'fluctuationstrength_osses2016':
        a0tab = np.array([
            [0, 0],
            [10, 0],
            [19, 0],
            [20, -1.43],
            [21, -2.59],
            [21.5, -3.57],
            [22, -5.19],
            [22.5, -7.41],
            [23, -11.3],
            [23.5, -20],
            [24, -40],
            [25, -130],
            [26, -999]
        ], dtype=float)
    else:
        raise ValueError("a0_type must be 'fastl2007' or 'fluctuationstrength_osses2016'")

    # 4) Interpolate
    a0 = np.zeros(int(round(N/2 + 1)))
    a0[qb] = from_db(np.interp(bark[qb], a0tab[:, 0], a0tab[:, 1], left=np.nan, right=np.nan))
    a0[np.isnan(a0)] = 0.0

    B = create_a0_FIR(freqs, a0[qb], N, fs, plot=plot)

    return B, freqs, a0[qb]

def calibrate(
    InputSignal: np.ndarray,
    RefSignal: np.ndarray,
    ReferenceLevel: float,
    *,
    return_dbfs: bool = False
) -> Tuple[np.ndarray, float, Optional[float]]:
    """Scale *InputSignal* to a known SPL reference.

    Parameters
    ----------
    InputSignal : ndarray
        Signal to be calibrated.
    RefSignal : ndarray
        Reference recording at known level.
    ReferenceLevel : float
        SPL of ``RefSignal`` in dB (rms).
    return_dbfs : bool, default False
        If ``True`` also return the SPL equivalent of 0 dBFS.

    Returns
    -------
    calibrated_signal : ndarray
    cal_factor : float
    dBFS : float, optional
    """
    # Ensure floating-point math
    InputSignal = np.asarray(InputSignal, dtype=float)
    RefSignal = np.asarray(RefSignal, dtype=float)

    CalFactor = np.sqrt((10.0 ** (ReferenceLevel / 10.0)) * 4e-10 / np.mean(RefSignal ** 2))
    CalibratedSignal = CalFactor * InputSignal

    if return_dbfs:
        dbfs = ReferenceLevel - 20.0 * np.log10(np.sqrt(np.mean(RefSignal ** 2)))
        return CalibratedSignal, CalFactor, dbfs

    return CalibratedSignal, CalFactor

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
    """
    Apply the ECMA-418-2 auditory filter bank to a mono audio signal.

    Parameters
    ----------
    signal : array_like of float, shape (N,)
        Mono audio time-series sampled at 48 kHz.
    outplot : bool, default False
        If True, plot the combined magnitude and phase responses of all 53 half-Bark filters.

    Returns
    -------
    filtered : ndarray, shape (N, 53)
        Filtered signal with one column per half-Bark band.
    """

    # Argument Validation
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1‑D (mono) array")
    if not np.isrealobj(signal):
        raise ValueError("signal must be real‑valued")

    sampleRate48k = 48e3  # Hz
    deltaFreq0 = 81.9289
    c = 0.1618

    halfBark = np.arange(0.5, 26.5 + 0.5, 0.5)  # 0.5 … 26.5 inclusive (53 bands)
    bandCentreFreqs = (deltaFreq0 / c) * np.sinh(c * halfBark)
    dfz = np.sqrt(deltaFreq0**2 + (c * bandCentreFreqs) ** 2)

    # Signal Processing
    k = 5  # filter order (footnote 5 ECMA‑418‑2)
    e_i = np.array([0, 1, 11, 11, 1], dtype=float)
    signalFiltered = np.empty((signal.size, halfBark.size), dtype=float)
    if outplot:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        ax_mag.set_xscale("log")
        ax_phase.set_xscale("log")

    # Main loop: highest → lowest band (to mimic MATLAB for reproducibility)
    for zBand in range(halfBark.size - 1, -1, -1):
        tau = (
            (1 / (2 ** (2 * k - 1)))
            * comb(2 * k - 2, k - 1)
            * (1 / dfz[zBand])
        )
        d = np.exp(-1 / (sampleRate48k * tau))

        bp = np.exp(
            1j
            * 2
            * np.pi
            * bandCentreFreqs[zBand]
            * np.arange(k + 2)
            / sampleRate48k
        )

        m = np.arange(1, k + 1)
        a_m = np.concatenate(([1.0], (-d) ** m * comb(k, m))) * bp[: k + 1]

        m = np.arange(0, k)
        b_m = (((1 - d) ** k) / np.sum(e_i[1:] * d ** np.arange(1, k))) * (d ** m) * e_i[:k] * bp[:k]

        signalFiltered[:, zBand] = 2.0 * np.real(lfilter(b_m, a_m, signal))
        if outplot:
            w, h = freqz(b_m, a_m, worN=10_000, fs=sampleRate48k, whole=True)
            ax_mag.semilogx(w, 20 * np.log10(np.abs(h)))
            ax_phase.semilogx(w, np.unwrap(np.angle(h)) * 180 / np.pi)

    # Plot Figures
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

    return signalFiltered

# cross-check compatibility readability checkpoint
def shm_basis_loudness(
    signalSegmented: np.ndarray,
    bandCentreFreqs: float | None = None,
    tol: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute basis loudness using the Sottek Hearing Model (ECMA-418-2).

    Parameters
    ----------
    signalSegmented : ndarray of float, shape (n_samples, n_blocks[, n_bands])
        Segmented pressure signal. Must be real-valued.
    bandCentreFreqs : float, optional
        Centre frequency in Hz when input is 2-D (single band). Omit for 3-D input.
    tol : float, default 1.0
        Frequency tolerance in Hz for matching bandCentreFreqs to standard half-Bark centres.

    Returns
    -------
    signal_rect_seg : ndarray
        Half-wave rectified signal, same shape as input.
    basis_loudness : ndarray
        Basis loudness values, same shape as input.
    block_rms : ndarray
        Block RMS values, shape (n_blocks,) or (n_blocks, n_bands).
    """

    # Constants
    deltaFreq0 = 81.9289
    c = 0.1618
    halfBark = np.arange(0.5, 27, 0.5)  # 0.5 … 26.5 inclusive
    bandCentreFreqs = (deltaFreq0 / c) * np.sinh(c * halfBark)  # Eq. 9

    cal_N = 0.0211668
    cal_Nx = 1.00132
    a = 1.5  # a in Eq. 23

    p_threshold = 2e-5 * 10 ** (np.arange(15, 86, 10) / 20)  # Pa (8‑vals)
    v = np.array([1, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434])

    LTQz = np.array([
        0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453, 0.0365, 0.0298,
        0.0247, 0.0207, 0.0176, 0.0151, 0.0131, 0.0115, 0.0103, 0.0093,
        0.0086, 0.0081, 0.0077, 0.0074, 0.0073, 0.0072, 0.0071, 0.0072,
        0.0073, 0.0074, 0.0076, 0.0079, 0.0082, 0.0086, 0.0092, 0.0100,
        0.0109, 0.0122, 0.0138, 0.0157, 0.0172, 0.0180, 0.0180, 0.0177,
        0.0176, 0.0177, 0.0182, 0.0190, 0.0202, 0.0217, 0.0237, 0.0263,
        0.0296, 0.0339, 0.0398, 0.0485, 0.0622
    ])

    # Validation
    if not np.isrealobj(signalSegmented):
        raise TypeError("signal_segmented must be real‑valued")

    if signalSegmented.ndim not in (2, 3):
        raise ValueError("signal_segmented must be 2‑D or 3‑D")

    if bandCentreFreqs is None and signalSegmented.ndim == 2:
        raise ValueError("bandCentreFreqs required for 2‑D input")

    if bandCentreFreqs is not None and signalSegmented.ndim == 3:
        raise ValueError("bandCentreFreqs should be omitted for 3‑D input")

    # Centre‑frequency handling
    if bandCentreFreqs is not None:
        idx = int(np.abs(bandCentreFreqs - bandCentreFreqs).argmin())
        if abs(bandCentreFreqs[idx] - bandCentreFreqs) > tol:
            raise ValueError(
                f"{bandCentreFreqs} Hz is not within ±{tol} Hz of any standard half‑Bark centre; "
                f"closest is {bandCentreFreqs[idx]:.2f} Hz.")

    ## Core processing ##
    # Half Wave Rectification
    signalRectSeg = np.maximum(signalSegmented, 0.0)

    # Block RMS (Eq. 22) – factor 2 because rectified signal is positive‑only
    blockRMS = np.sqrt((2.0 / signalRectSeg.shape[0]) * np.sum(signalRectSeg ** 2, axis=0))

    # Loudness transform (Eqs. 23–24)
    bandLoudness = cal_N * cal_Nx * (blockRMS / 20e-6) * np.prod((1 + (blockRMS[..., None] / p_threshold) ** a) ** (np.diff(v) / a), axis=-1)

    blockRMS = np.squeeze(blockRMS)

    # Threshold‑in‑quiet correction (Eq. 25)
    if bandCentreFreqs is not None:  # 2‑D input, single band
        basisLoudness = bandLoudness - LTQz[idx]
    else:                             # 3‑D input, all 53 bands
        basisLoudness = bandLoudness - LTQz.reshape((1,) * (blockRMS.ndim - 1) + (53,))

    basisLoudness = np.maximum(basisLoudness, 0.0)

    return signalRectSeg, basisLoudness, blockRMS

def shm_noise_red_lowpass(signal: np.ndarray, sampleRatein: float) -> np.ndarray:
    """
    Apply a low-pass noise-reduction filter (ECMA-418-2).

    Parameters
    ----------
    signal : ndarray of float, shape (N,) or (N, C)
        Input audio samples (time × channels).
    sampleRatein : float
        Sampling rate in Hz.

    Returns
    -------
    filtered : ndarray of float, same shape as signal
        Noise-reduced signal.
    """

    # Validation
    if not isinstance(signal, np.ndarray):
        raise TypeError("`signal` must be a NumPy array")

    if signal.ndim not in (1, 2):
        raise ValueError("`signal` must be 1‑D (mono) or 2‑D (time × channels)")

    if not np.isrealobj(signal):
        raise ValueError("`signal` must contain real‑valued samples")

    if not (isinstance(sampleRatein, (float, int)) and sampleRatein > 0):
        raise ValueError("`sampleRatein` must be a positive scalar sample‑rate (Hz)")

    # Coefficient design (ECMA‑418‑2:2024, Equations 14‑15)
    k = 3
    e_i = np.array([0.0, 1.0, 1.0])          # Footnote 21
    tau = (1 / 32) * (6 / 7)                 # Footnote 20
    d = np.exp(-1.0 / (sampleRatein * tau))            # §5.1.4.2

    # Denominator (a): Equation 14
    m = np.arange(1, k + 1)
    a = np.concatenate(([1.0], ((-d) ** m) * np.array([3, 3, 1], dtype=float)))

    # Numerator (b): Equation 15
    m = np.arange(0, k)          # 0‥k‑1
    i = np.arange(1, k)          # 1‥k‑1
    b = (((1 - d) ** k) / (np.sum(e_i[i] * (d ** i)))) * (d ** m) * e_i     # note e_i[0] == 0 → b[0] == 0

    # Filtering
    was_1d = (signal.ndim == 1)
    x = signal[:, None] if was_1d else signal  # shape (N, C)

    y = lfilter(b, a, x, axis=0)

    return y.ravel() if was_1d else y

def shm_out_mid_ear_filter(
    signal: np.ndarray,
    fieldtype: str = "free-frontal",
    outplot: bool = False,
) -> np.ndarray:
    """
    Apply outer- and middle-ear filtering to a calibrated signal (ECMA-418-2).

    Parameters
    ----------
    signal : array_like of float, shape (N,) or (N, C)
        Sound-pressure waveform sampled at 48 kHz.
    fieldtype : {'free-frontal', 'diffuse'}, default 'free-frontal'
        Filter type: 'free-frontal' applies all 8 sections; 'diffuse' omits the first two.
    outplot : bool, default False
        If True, plot magnitude (dB) and phase (deg) of the overall filter response.

    Returns
    -------
    filtered : ndarray of float, same shape and dtype as signal
        Filtered output.
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
    b_0k = np.array([
        1.015896020255593,
        0.958943219304445,
        0.961371976333197,
        2.225803503609735,
        0.471735128494163,
        0.115267139824401,
        0.988029297230954,
        1.952237687301361,
    ])
    b_1k = np.array([
        -1.925298877776079,
        -1.806088011849494,
        -1.763632154338248,
        -1.434650484792157,
        -0.366091796830044,
        0.0,
        -1.912433802933870,
        0.162319983017519,
    ])
    b_2k = np.array([
        0.922118060364679,
        0.876438777856084,
        0.821787991845146,
        -0.498204282194628,
        0.244144703885020,
        -0.115267139824401,
        0.926131550180785,
        -0.667994113035186,
    ])
    a_0k = np.ones_like(b_0k)
    a_1k = np.array([
        -1.925298877776079,
        -1.806088011849494,
        -1.763632154338248,
        -1.434650484792157,
        -0.366091796830044,
        -1.796002566692014,
        -1.912433802933871,
        0.162319983017519,
    ])
    a_2k = np.array([
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

    sos = np.column_stack([b_0k[idx], b_1k[idx], b_2k[idx], a_0k[idx], a_1k[idx], a_2k[idx]])


    # Filtering (axis 0 = time)
    signalFiltered = sosfilt(sos, signal, axis=0)

    if signal_was_1d:
        signalFiltered = signalFiltered.ravel()

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

    return signalFiltered

def shm_preproc(
    signal: np.ndarray,
    blockSize: int,
    hopSize: int,
    padStart: bool = True,
    padEnd: bool = True
) -> np.ndarray:
    """
    Pre-process signal with raised-cosine fade-in and zero-padding (ECMA-418-2).

    Parameters
    ----------
    signal : array_like of float, shape (N,) or (N, C)
        Input time-series samples.
    block_size : int
        Segmentation block size in samples.
    hop_size : int
        Hop size in samples.
    pad_start : bool, default True
        If True, prepend zeros equal to block_size.
    pad_end : bool, default True
        If True, append zeros so total length matches block_size + k * hop_size.

    Returns
    -------
    processed : ndarray of float
        Faded and padded signal, shape (N',) or (N', C).
    """

    # Validate
    sig = np.asarray(signal, dtype=float)
    if sig.ndim == 1:
        sig = sig[:, None]

    if sig.ndim != 2:
        raise ValueError("`signal` must be 1-D or 2-D (time x channels).")
    if blockSize <= 0 or hopSize <= 0:
        raise ValueError("`blockSize` and `hopSize` must be positive integers.")

    n_ch = sig.shape[1]

    # Fade-in
    fadeWeight = 0.5 - 0.5 * np.cos(np.pi * np.arange(240) / 240)
    fadeWeight = fadeWeight[:, None]
    signalFade = np.vstack((fadeWeight * sig[:240, :], sig[240:, :]))

    # Padding
    n_zeross = blockSize if padStart else 0

    if padEnd:
        n_samples = sig.shape[0]
        n_new = hopSize * (int(np.ceil((n_samples + hopSize + n_zeross)
                                        / hopSize)) - 1)
        n_zerose = n_new - n_samples
    else:
        n_zerose = 0

    # Assemble
    signalOut = np.vstack((
        np.zeros((n_zeross, n_ch)),
        signalFade,
        np.zeros((n_zerose, n_ch))
    ))

    # Restore
    if isinstance(signal, (list, np.ndarray)) and np.ndim(signal) == 1:
        signalOut = signalOut[:, 0]

    return signalOut

def shm_resample(signal: np.ndarray, sampleRatein: int) -> tuple[np.ndarray, int]:
    """
    Resample a signal to 48 kHz (ECMA-418-2 target rate).

    Parameters
    ----------
    signal : ndarray of float, shape (N,) or (N, C)
        Input signal at sampleRatein.
    sampleRatein : int
        Original sampling rate in Hz.

    Returns
    -------
    resampledSignal : ndarray of float
        Signal resampled to 48 kHz.
    resampledRate : int
        Always 48000.
    """

    # Validation
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a NumPy ndarray")
    if signal.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("signal must contain real numbers")
    if signal.ndim > 2:
        raise ValueError("signal must be 1-D or 2-D (time[, channels])")
    if not isinstance(sampleRatein, int) or sampleRatein <= 0:
        raise ValueError("sampleRatein must be a positive integer")

    resampledRate = 48_000

    if sampleRatein == resampledRate:
        return signal, resampledRate

    # Compute Resampling
    up = resampledRate // np.gcd(resampledRate, sampleRatein)
    down = sampleRatein // np.gcd(resampledRate, sampleRatein)
    resampledSignal = resample_poly(signal, up, down, axis=0)

    return resampledSignal, resampledRate

def shm_rough_low_pass(
    specRoughEstTform: np.ndarray,
    sampleRate: float,
    riseTime: float,
    fallTime: float
) -> np.ndarray:
    """
    Smooth specific roughness estimates with a low-pass IIR filter (ECMA-418-2).

    Parameters
    ----------
    specRoughEstTform : ndarray, shape (T, B)
        Specific-roughness estimates (time × bands).
    sampleRate : float
        Frame rate of the input in Hz.
    riseTime : float
        Attack time constant in seconds.
    fallTime : float
        Release time constant in seconds.

    Returns
    -------
    spec_roughness : ndarray, shape (T, B)
        Smoothed specific roughness.
    """

    # Validation
    specRoughEstTform = np.asanyarray(specRoughEstTform, dtype=float)
    if specRoughEstTform.ndim != 2:
        raise ValueError("specRoughEstTform must be a 2-D array (time × bands)")
    if sampleRate <= 0 or riseTime <= 0 or fallTime <= 0:
        raise ValueError("sampleRate, riseTime and fallTime must be positive")

    # IIR coefficients
    riseExponent = np.exp(-1.0 / (sampleRate * riseTime))
    fallExponent = np.exp(-1.0 / (sampleRate * fallTime))

    # Filtering
    specRoughness = np.empty_like(specRoughEstTform)
    specRoughness[0, :] = specRoughEstTform[0, :]

    for llBlock in range(1, specRoughEstTform.shape[0]):
        riseMask = specRoughEstTform[llBlock, :] >= specRoughness[llBlock - 1, :]
        fallMask = ~riseMask

        if np.any(riseMask):
            specRoughness[llBlock, riseMask] = (
                specRoughEstTform[llBlock, riseMask] * (1.0 - riseExponent)
                + specRoughness[llBlock - 1, riseMask] * riseExponent
            )

        if np.any(fallMask):
            specRoughness[llBlock, fallMask] = (
                specRoughEstTform[llBlock, fallMask] * (1.0 - fallExponent)
                + specRoughness[llBlock - 1, fallMask] * fallExponent
            )

    return specRoughness

def shm_rough_weight(
    modRate: np.ndarray,
    modfreqMaxWeight: np.ndarray,
    roughWeightParams: np.ndarray
) -> np.ndarray:
    """
    Compute roughness weighting factors (ECMA-418-2, Eq. 85).

    Parameters
    ----------
    modRate : array_like of float
        Modulation rates in Hz.
    modfreqMaxWeight : array_like of float
        Modulation rate at which the weight peaks (value=1).
    roughWeightParams : array_like of float, shape (2, ...)
        Weighting parameters: [alpha (gain), beta (sharpness)].

    Returns
    -------
    roughWeight : ndarray
        Weighting factors, same shape as broadcasted inputs.
    """

    modRate = np.asarray(modRate, dtype=float)
    modfreqMaxWeight = np.asarray(modfreqMaxWeight, dtype=float)
    roughWeightParams = np.asarray(roughWeightParams, dtype=float)

    roughWeight = 1.0 /(1.0 + ((((modRate / modfreqMaxWeight) - (modfreqMaxWeight / modRate)) * roughWeightParams[0]) ** 2)) ** roughWeightParams[1]
    return roughWeight

def shm_signal_segment(
    signal: np.ndarray,
    axisn: int = 0,
    blockSize: int = 1024,
    overlap: float = 0.0,
    i_start: int = 0,
    endShrink: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment a signal into overlapping blocks (ECMA-418-2).

    Parameters
    ----------
    signal : array_like, shape (N,) or (N, C)
        Input samples and optional channels.
    axisn : {0, 1}, default 0
        Axis along which to segment. Axis 1 input is transposed internally.
    blockSize : int, default 1024
        Number of samples per block.
    overlap : float, default 0.0
        Fractional overlap between successive blocks (0 ≤ overlap < 1).
    i_start : int, default 0
        Starting index for the first block.
    endShrink : bool, default False
        If True, include a final block capturing the signal tail.

    Returns
    -------
    signal_segmented : ndarray
        Segmented signal, shape (blockSize, n_blocks, n_channels) or transposed if axisn==1.
    i_blocks_out : ndarray of int
        Starting indices of each block relative to i_start.
    """

    # Validation
    if not isinstance(signal, np.ndarray):
        signal = np.asanyarray(signal)

    if signal.ndim == 1:
        signal = signal[:, None]

    if axisn not in (0, 1):
        raise ValueError("axisn must be 0 or 1")

    if not (0 <= overlap < 1):
        raise ValueError("overlap must satisfy 0 ≤ overlap < 1")

    if blockSize <= 0:
        raise ValueError("blockSize must be positive")

    # Re-orient
    axisFlip = False
    if axisn == 1:
        signal = signal.T
        axisFlip = True

    n_total, nchans = signal.shape

    if i_start < 0 or i_start >= n_total:
        raise ValueError("i_start falls outside the signal")

    # Truncate
    hopSize = round(blockSize * (1.0 - overlap))
    if hopSize <= 0:
        raise ValueError("overlap too large: hop size becomes zero")

    signalTrunc = signal[i_start:]
    if signalTrunc.shape[0] <= blockSize:
        raise ValueError("Signal is too short for the requested blockSize")

    n_blocks = int(np.floor((signalTrunc.shape[0] - overlap * blockSize) / hopSize))
    i_end = n_blocks * hopSize + int(overlap * blockSize)
    signalTrunc = signalTrunc[:i_end]

    # Build Block Matrix
    windows = sliding_window_view(signalTrunc, window_shape=(blockSize,), axis=0)
    windows = windows[::hopSize]
    windows = windows.reshape(-1, blockSize, nchans)

    if windows.shape[0] != n_blocks:
        windows = windows[:n_blocks]

    # Tail Block  (optional)
    iBlocksOut = np.arange(0, n_blocks * hopSize, hopSize, dtype=int)
    if endShrink and signalTrunc.shape[0] > signalTrunc.shape[0]:
        tail_start = signalTrunc.shape[0] - blockSize
        tail_block = signalTrunc[tail_start : tail_start + blockSize]
        tail_block = tail_block[:, :, None] if tail_block.ndim == 1 else tail_block
        windows = np.concatenate((windows, tail_block[None, ...]), axis=0)
        iBlocksOut = np.concatenate((iBlocksOut, [tail_start]))

    # Re-orient
    # current shape: (n_blocks, block_size, n_chans)
    if axisFlip:
        signalSegmented = windows.transpose(1, 0, 2)  # (block, n_blocks, chan)
    else:
        signalSegmented = windows.transpose(1, 0, 2)  # same orientation
    # If axisFlip==True, caller expects (n_blocks, block, chan)
    if axisFlip:
        signalSegmented = signalSegmented.swapaxes(0, 1)

    return signalSegmented, iBlocksOut

# ------------------
#### VALIDATION ####
# ------------------

if __name__ == "__main__":
    print("Validating")
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

    # fs = 48_000
    # x = np.random.randn(fs)
    # y = shm_out_mid_ear_filter(x, fieldtype="free-frontal", outplot=True)
    # print("Processed", y.shape, "samples")

    # np.random.seed(42)
    # x = np.random.randn(10_000)
    # y = shm_preproc(x, block_size=2048, hop_size=1024)
    # print(f"Input  shape : {x.shape}")
    # print(f"Output shape : {y.shape}")
    # print(f"First 5 out samples : {y[:5]}")
    # print(f"Last  5 out samples : {y[-5:]}")
    # print(x)
    # print(y)
    # from scipy.io import savemat
    # savemat("shm_preproc_test.mat", {"x_py": x, "y_py": y})
    # print("\nSaved shm_preproc_test.mat for MATLAB cross-check")

    # audio, sr = sf.read("sound_files\ExSignal_A320_auralized_departure_104dBFS.wav")
    # audio_48k, _ = shm_resample(audio, sr)
    # sf.write("output_48k.wav", audio_48k, 48_000)
    # audio2, sr2 = sf.read("output_48k.wav")
    # print(sr,sr2)

    # spec = np.array([[0.0, 0.0],
    #                 [1.0, 0.5],
    #                 [0.5, 0.2],
    #                 [1.2, 0.1]])
    # out = shm_rough_low_pass(spec, 50.0, 0.02, 0.10)
    # print(out,out.shape)

    # f_p = np.linspace(1, 300, 500)
    # f_max = 70.0
    # alpha_vec = np.full_like(f_p, 1.5)
    # beta_vec = np.full_like(f_p, 0.8)
    # params = np.stack((alpha_vec, beta_vec)) 
    # w = shm_rough_weight(f_p, f_max, params)
    # print("First five weights:", w[:5])

    # fs = 48_000
    # t = np.arange(0, 1.0, 1 / fs)
    # x  = np.stack((np.sin(2 * np.pi * 440 * t),
    #                np.sin(2 * np.pi * 880 * t)), axis=-1)
    # blocks, idx = shm_signal_segment(x, axisn=0, block_size=4096,
    #                              overlap=0.75, i_start=0, end_shrink=False)
    # print("blocks shape:", blocks.shape)
    # print("block starts:", idx[:4], "…")