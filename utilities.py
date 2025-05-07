
##########################
#### UTILITIES MODULE ####
##########################

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import firwin2, freqz
from scipy.io import wavfile
from IPython.display import Audio, display
from pathlib import Path
from typing import Union, Sequence, Any, Tuple, Optional, Dict

FloatArrayLike = Union[float, Sequence[float], np.ndarray]
ArrayLikeInt   = Union[Sequence[int], np.ndarray]

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

# ----------------------------
#### VALIDATION FUNCTIONS ####
# ----------------------------
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