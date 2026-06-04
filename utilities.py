
##########################
#### UTILITIES MODULE ####
##########################

import os
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import firwin2, freqz, lfilter, sosfilt, sosfreqz, resample_poly, iirdesign
from scipy.io import wavfile
from scipy.special import comb
import soundfile as sf
from pathlib import Path
from typing import Union, Sequence, Any, Tuple, Optional, Dict, Literal
import itertools

FloatArrayLike = Union[float, Sequence[float], np.ndarray]
ArrayLikeInt   = Union[Sequence[int], np.ndarray]
ArrayLike = Union[np.ndarray, float, int]

# ----------------------
#### MAIN FUNCTIONS ####
# ----------------------

def acquire(variable, name):
        import pandas as pd
        if len(variable) > 1:
            if (np.iscomplex(variable[0]) or np.iscomplex(variable[-1]) or np.iscomplex(variable[len(variable)//3])) == True:
                df = pd.DataFrame({
                    f'{name} (Real)':  np.real(variable).ravel(),
                    f'{name} (Imag)':  np.imag(variable).ravel()
                })
                df.to_excel(f'{name}_complex.xlsx', index=False)
            else:
                df = pd.DataFrame({
                    f'{name}':  variable.ravel()
                })
                df.to_excel(f'{name}.xlsx', index=False)

def see(file_path: str) -> None:
    """
    Visualize the waveform and log-frequency spectrogram of a WAV file.

    The function first closes any previously opened Matplotlib figures,
    verifies that *file_path* exists, and then loads the audio using
    :pyfunc:`scipy.io.wavfile.read`.  Stereo signals are down-mixed to mono
    by averaging the two channels.  A two-panel figure is created:

    #. **Waveform** – amplitude versus time (seconds).
    #. **Spectrogram** – power spectral density computed with a 1024-point
       FFT and 50 % (512-sample) overlap, shown on a logarithmic frequency
       axis and rendered with the *viridis* colormap.

    If *file_path* is not found, a warning is printed and the function
    returns without raising an exception.

    Parameters
    ----------
    file_path : str
        Path to the ``*.wav`` file to visualise.

    Returns
    -------
    None
        The primary effect is the display of a Matplotlib window.
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
    """
    Convert frequency from hertz to the Bark psychoacoustic scale.

    The Bark scale models the auditory system’s critical bands.  This
    function implements the widely used Zwicker approximation:

    ``z = 13 * arctan(0.76 * f / 1000) + 3.5 * arctan((f / (7.5 * 1000))**2)``

    where *f* is the frequency in hertz and *z* is the corresponding Bark
    value.

    Parameters
    ----------
    f : FloatArrayLike
        Scalar or array-like of frequencies in hertz.  Any input accepted by
        :pyfunc:`numpy.asarray` (e.g., list, tuple, NumPy array).

    Returns
    -------
    numpy.ndarray
        Bark-scaled frequencies with the same shape as *f*.

    Notes
    -----
    * The calculation is purely numerical; negative frequencies are not
      checked and will yield mathematically valid—though acoustically
      meaningless—results.
    * A value of ``z ≈ 24`` corresponds to approximately 15.5 kHz.
    """
    f = np.asarray(f)
    z = 13 * np.arctan(0.76 * (f / 1000)) + 3.5 * np.arctan((f / (1000 * 7.5))**2)
    return z

def bark2hz(z: FloatArrayLike) -> np.ndarray:
    """
    Convert Bark scale values back to frequency in hertz.

    This routine provides an approximate numerical inverse of
    :pyfunc:`hz2bark` by linearly interpolating a lookup table that spans
    roughly 20 Hz–15 kHz.  The table is built from the Zwicker formula,
    evaluated at one-third-octave points

    ``f = 1000 * 2**(k / 3),  k ∈ [-20, 12]``,

    and interpolated with :pyclass:`scipy.interpolate.interp1d`
    (``kind='linear', fill_value='extrapolate'``).

    Parameters
    ----------
    z : FloatArrayLike
        Scalar or array-like of Bark values.  Any input accepted by
        :pyfunc:`numpy.asarray`.

    Returns
    -------
    numpy.ndarray
        Frequencies in hertz with the same shape as *z*.

    Notes
    -----
    * Results outside the calibrated 0–24 Bark range rely on linear
      extrapolation and may be inaccurate.
    * For typical audio work (20 Hz–15 kHz) the error is << 1 %.  The
      inversion is monotonic but not analytic.
    """

    f0 = 1000
    k = np.arange(-20, 13)
    f = f0 * 2 ** (k / 3)

    zt = hz2bark(f)

    interpolator = interp1d(zt, f, kind='linear', fill_value="extrapolate")
    f = interpolator(z)

    return f

def phon2sone(phon: FloatArrayLike) -> np.ndarray:
    """
    Convert loudness level from phons to perceived loudness in sones.

    The mapping follows ISO-226 conventions:

    * **≥ 40 phon** — loudness doubles every 10 phon:
      ``sone = 2 ** (0.1 · (phon − 40))``.
    * **< 40 phon** — a steeper, power-law growth:
      ``sone = (phon / 40) ** (1 / 0.35)``.

    Input values are flattened to a 1-D column vector before the conversion,
    and the returned array preserves that shape.

    Parameters
    ----------
    phon : FloatArrayLike
        Loudness level(s) in phons.  Accepts any object convertible with
        :pyfunc:`numpy.asarray` (scalar, list, tuple, ndarray, etc.).

    Returns
    -------
    numpy.ndarray
        Loudness in sones with the same length as *phon*.

    Notes
    -----
    * A value of 40 phon is defined as 1 sone by convention.
    * For each 10 phon increase above 40, perceived loudness (sones)
      doubles; below 40 phon, the relationship becomes non-linear.
    """

    phon = np.asarray(phon).flatten()
    phon = phon[:, np.newaxis]
    
    sone = np.zeros_like(phon)

    idx = phon >= 40

    sone[idx] = 2 ** (0.1 * (phon[idx] - 40))

    sone[~idx] = (phon[~idx] / 40) ** (1 / 0.35)

    return sone

def sone2phon(sone: FloatArrayLike) -> np.ndarray:
    """
    Convert perceived loudness from **sones** to loudness level in **phons**.

    This is the inverse of :pyfunc:`phon2sone`.  Two empirical regimes are
    applied:

    * **≥ 1 sone** – loudness grows logarithmically:  
      ``phon = 40 + 33.22 · log10(sone)``.
    * **< 1 sone** – a steeper power-law relation with a tiny bias to avoid
      zero:  
      ``phon = 40 · (sone + 5 × 10⁻⁴)^{0.35}``.

    Parameters
    ----------
    sone : FloatArrayLike
        Perceived loudness value(s) in sones.  Any object accepted by
        :pyfunc:`numpy.asarray` (scalar, list, tuple, ndarray, etc.).

    Returns
    -------
    numpy.ndarray
        Loudness level(s) in phons, preserving the shape of *sone*.

    Notes
    -----
    * By convention, **1 sone = 40 phon**.
    * The small offset ``5 × 10⁻⁴`` ensures finite results as *sone*
      approaches zero.
    """

    sone = np.atleast_1d(sone).astype(float)
    sone = sone.reshape(-1, 1) if sone.ndim == 1 else sone

    phon = np.zeros_like(sone)

    idx = sone >= 1
    phon[idx] = 40 + 33.22 * np.log10(sone[idx])
    phon[~idx] = 40 * (sone[~idx] + 0.0005) ** 0.35

    return phon

def get_exceeded_value(input: FloatArrayLike, PercentValue: float) -> np.ndarray:
    """
    Return the value that is **exceeded by a given percentage** of the data.

    The input array is sorted along its first dimension and the element at

    ``i = ceil((1 − PercentValue / 100) · N) − 1``

    is chosen, where *N* is the number of rows (samples).  
    This statistic is often called an **exceedance level**:  
    the result is greater than or equal to *PercentValue* % of the
    observations.

    * **1-D input** → a single scalar is returned.  
    * **2-D input** (shape ``(N, M)``) → a row vector of length *M*,
      computed independently for each column.

    Parameters
    ----------
    input : FloatArrayLike
        Array of samples to analyse.  Converted with
        :pyfunc:`numpy.asarray`.
    PercentValue : float
        Percentage of the dataset that should lie **below** the returned
        value (0 ≤ *PercentValue* ≤ 100).

    Returns
    -------
    numpy.ndarray
        Exceedance value(s): a scalar for 1-D input or a 1 × *M* array for
        2-D input.

    Notes
    -----
    * *PercentValue* = 0 gives the maximum; 100 gives the minimum.
    * The computed index is clamped to ``[0, N − 1]`` to prevent
      out-of-bounds errors.
    """

    # Sort the input array
    sort_input = np.sort(input, axis=0)

    # Calculate the index corresponding to the exceeded value.
    # Faithful to MATLAB get_exceeded_value:
    #   X_index = floor((100-PercentValue)/100 * N);  if X_index==0, X_index=1
    # (1-based); converted here to 0-based. Using floor (not ceil-1) matters
    # whenever the index is non-integer, where the two conventions otherwise
    # differ by one sorted element.
    X_index = int(np.floor((100 - PercentValue) / 100 * input.shape[0]))
    if X_index == 0:
        X_index = 1
    X_index = X_index - 1  # MATLAB 1-based -> Python 0-based

    # Clamp X_index to the valid range
    X_index = max(0, min(X_index, input.shape[0] - 1))

    # Handle edge cases for 1D input
    if input.ndim == 1:
        return sort_input[X_index]
    else:
        return sort_input[X_index, :]

def get_statistics(input: FloatArrayLike, metric: str) -> Dict[str, np.ndarray]:
    """
    Compute a suite of descriptive statistics for a psychoacoustic time-series
    and package the results in a labelled dictionary.

    The routine supports up to three parallel channels.  For each channel it
    evaluates:

    * **Extrema and moments** – ``max``, ``min``, arithmetic ``mean``, and
      unbiased sample standard deviation ``std``.
    * **Exceedance levels** – values that are exceeded by the top
      ``{1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95} %`` of the
      observations (with the 50 % level equal to the median).

    Results are stored in a dictionary whose keys are formed by concatenating
    a one– or two–letter metric identifier with the statistic label
    (e.g. ``"Nmax"``, ``"S10"``, ``"PA50"``).

    The identifier is derived from *metric* via an internal lookup:

    ===== Metric string ======================= Identifier
    Loudness_ISO532_1, Loudness_ECMA418_2       ``N``
    Sharpness_DIN45692                          ``S``
    Roughness_Daniel1997, Roughness_ECMA418_2   ``R``
    FluctuationStrength_Osses2016               ``FS``
    Tonality_Aures1985                          ``K``
    Tonality_ECMA418_2                          ``T``
    PsychoacousticAnnoyance_*                   ``PA``
    ==========================================  =========

    An unknown *metric* issues a :pyclass:`RuntimeWarning` and uses the
    fallback identifier ``"X"``.

    Parameters
    ----------
    input : FloatArrayLike
        Array of samples with shape ``(N,)`` or ``(N, C)`` where
        *N* ≥ 1 and 1 ≤ *C* ≤ 3.  One-dimensional input is promoted to a
        column vector.  If the array has more than three columns a
        :class:`ValueError` is raised.
    metric : str
        Name of the psychoacoustic metric that dictates the key prefix.

    Returns
    -------
    Dict[str, numpy.ndarray]
        Dictionary mapping statistic labels to NumPy arrays of length *C*
        (or scalars for single-channel input).

    Raises
    ------
    ValueError
        If *input* contains more than three channels.

    Warns
    -----
    RuntimeWarning
        When *metric* is not recognised; the identifier defaults to ``"X"``.
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
    """
    Map FFT bin frequencies to the Bark psychoacoustic scale.

    The function builds a 25-band reference table—``Bark_raw``—whose rows
    contain

    ``[band-index, lower-Hz, upper-Hz, centre-Bark]``

    after Zwicker & Terhardt.  A monotonic, piece-wise linear
    interpolator is then created between frequency (Hz) and Bark number
    and evaluated for the frequency vector *freqs*.  The resulting Bark
    coordinates are written into a zero-initialised array of length
    ``round(N / 2) + 1`` at the positions indicated by *qb*.

    Parameters
    ----------
    N : int
        FFT size (used to size the output array ``(N // 2 + 1,)``).
    qb : ArrayLikeInt
        Indices of frequency bins to convert; typically the same length as
        *freqs*.
    freqs : numpy.ndarray
        Frequency of the selected bins in hertz.

    Returns
    -------
    bark : numpy.ndarray
        One-dimensional array of Bark values (length ``N // 2 + 1``) with
        zeros everywhere except at *qb*.
    Bark_raw : numpy.ndarray
        The 25 × 4 lookup table described above.  Provided for reference.

    Notes
    -----
    * Frequencies outside 0–20 kHz are handled by linear extrapolation.
    * The mapping is **many-to-one**; multiple FFT bins may share the same
      Bark value depending on frequency resolution.
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
    """
    Convert level from decibels (dB) to a linear gain factor.

    The transformation is performed element-wise using

    ``gain = 10 ** (gain_dB / divisor)``,

    where *divisor* is typically **20** for amplitude (voltage, sound-
    pressure) ratios and **10** for power ratios.

    Parameters
    ----------
    gain_dB : FloatArrayLike
        Scalar or array-like decibel values to be converted.  Any input
        accepted by :pyfunc:`numpy.asarray`.
    divisor : float, default 20
        Denominator of the exponent; set to **10** to convert dB of power.

    Returns
    -------
    numpy.ndarray
        Linear gain(s) with the same shape as *gain_dB*.

    Notes
    -----
    * Positive dB yield gains **> 1**, negative dB produce gains **< 1**.
    * The function is numerical only; no checks are made for physical
      plausibility of the decibel values.
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
    """
    Design an **FIR correction filter** that follows an arbitrary amplitude
    target ``a0(f)`` and, optionally, display its frequency response.

    The target curve is supplied as *break-point* samples—frequency–gain
    pairs that form a piece-wise linear magnitude response.  The filter is
    created with :pyfunc:`scipy.signal.firwin2` using

    ``B = firwin2(numtaps=N + 1, freq=f_ext, gain=a0_ext, fs=fs)``,

    where two extra points are prepended/appended so that the response is
    defined at 0 Hz and the Nyquist frequency.

    Parameters
    ----------
    f : numpy.ndarray, shape (*K*,)
        Strictly increasing break-point frequencies in hertz  
        (*0 < f[0] < … < f[-1] < fs/2*).
    a0 : numpy.ndarray, shape (*K*,)
        Desired linear gain at each break-point frequency.
    N : int
        Filter **order**.  The resulting filter has *N + 1* taps.
    fs : float
        Sampling frequency in hertz.
    plot : bool, keyword-only, default ``False``
        If ``True``, plot the magnitude response in decibels and **return
        ``None``** instead of the coefficient vector.

    Returns
    -------
    numpy.ndarray | None
        FIR coefficients ``B`` (length *N + 1*) when *plot* is ``False``;
        otherwise ``None``.

    Raises
    ------
    ValueError
        If the shape, ordering, or range of *f* and *a0* violate the
        requirements (see *Parameters*).

    Notes
    -----
    * End-point extension: ``f_ext = [0, *f, fs/2]`` and
      ``a0_ext = [a0[0], *a0, a0[-1]]`` ensure a well-defined response over
      the entire 0–Nyquist band.
    * The filter is linear-phase (symmetrical coefficients) and therefore
      introduces a pure delay of *N/2* samples.
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
    """
    Generate the **Fastl (2007)** or **Osses & van de Par (2016)**
    equal-loudness pre-emphasis curve *a₀(f)*, optionally design an FIR
    filter that realises it, and return both the curve and the underlying
    frequency grid.

    Parameters
    ----------
    fs : float
        Sampling rate of the intended audio system in hertz.
    N : int
        Size of the frequency grid *and* the order of the FIR filter
        (resulting in *N + 1* taps).
    a0_type : {'fastl2007', 'fluctuationstrength_osses2016'}, default
        ``'fastl2007'``
        Selects the underlying equal-loudness table.
    plot : bool, keyword-only, default ``False``
        When ``True`` the FIR’s magnitude response is plotted and *B* is
        returned as ``None``.

    Returns
    -------
    B : numpy.ndarray | None
        FIR coefficients of length *N + 1* (linear phase) or ``None`` when
        *plot* is ``True``.
    freqs : numpy.ndarray
        Frequency grid in hertz used for the design (20 Hz … 20 kHz).
    a0 : numpy.ndarray
        Linear gain samples of the equal-loudness curve at *freqs*.

    Raises
    ------
    ValueError
        If *a0_type* is not one of the recognised strings.
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
    """
    Calibrate an audio signal to a given **reference SPL** and, optionally,
    report its peak level in **dBFS**.

    The routine scales *InputSignal* such that the **RMS** of *RefSignal*
    corresponds to *ReferenceLevel* decibels re **20 µPa**.  The linear
    calibration factor is

    ``CalFactor = √[(10^{ReferenceLevel/10} · 4 × 10⁻¹⁰) / mean(RefSignal²)]``

    which converts the pressure‐like samples in *InputSignal* to physical
    pascals when multiplied element-wise.

    Parameters
    ----------
    InputSignal : numpy.ndarray
        Signal to be calibrated (any length or shape); cast to
        floating-point.
    RefSignal : numpy.ndarray
        Reference recording whose RMS defines the target level.
    ReferenceLevel : float
        Sound-pressure level in **dB SPL** that the RMS of *RefSignal* shall
        represent.
    return_dbfs : bool, keyword-only, default ``False``
        If ``True``, compute and return the full-scale digital level
        (*dBFS*) of *RefSignal* alongside the calibration factor.

    Returns
    -------
    CalibratedSignal : numpy.ndarray
        *InputSignal* scaled by *CalFactor*; same shape as the input.
    CalFactor : float
        Linear multiplier that maps raw sample units to pascals.
    dbfs : float, optional
        Full-scale level of *RefSignal* in **dBFS**  
        (*returned only when* ``return_dbfs=True``).
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
    """
    Return a dictionary of **default keyword arguments** for a supported
    psycho-acoustic model.

    The dictionary contains both the parameter *value* and a concise
    explanatory string for GUI widgets or autogenerated documentation.  For
    every setting you will therefore find a pair of keys:

    ``<parameter>``  
    ``<parameter>_description``


    Parameters
    ----------
    model_name : str
        Exact name of the model for which the defaults are requested (see
        *Supported model groups*).

    Returns
    -------
    Dict[str, Any]
        Dictionary with default settings.  Each parameter is accompanied by
        a ``<parameter>_description`` key that explains its purpose.

    Raises
    ------
    ValueError
        If *model_name* does not match any supported identifier.
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

def export_dict_to_excel(data_dict, filename="output.xlsx"):
    """
    Export a dictionary of simple data objects to an Excel workbook.

    Each key in *data_dict* becomes a worksheet whose name is truncated to
    **31 characters** (Excel’s limit).  The value is converted to a
    :class:`pandas.DataFrame` as follows:

    * **Scalar** *(int, float, str)* – wrapped in a single-row DataFrame
      with the key as the column label.
    * **1-D list / ndarray** – written as a single column labelled with the
      key.
    * **2-D list / ndarray** – written verbatim; column labels are the
      default integer range.
    * Values that are *None*, higher-dimensional, or of unsupported type
      are silently skipped.

    The workbook is created with :pyclass:`pandas.ExcelWriter`; if
    *filename* already exists it is overwritten.  I/O errors encountered
    while writing a sheet are caught and reported to ``stdout``.

    Parameters
    ----------
    data_dict : Mapping[str, Any]
        Dictionary whose values are scalars, 1-D/2-D sequences, or
        :class:`numpy.ndarray` objects to be exported.
    filename : str, default ``"output.xlsx"``
        Destination file name or path.

    Returns
    -------
    None
        The primary effect is the creation of *filename* on disk.
    """


    import pandas as pd
    with pd.ExcelWriter(filename) as writer:
        for key, value in data_dict.items():
            try:
                if isinstance(value, (int, float, str)):
                    # Wrap scalar in DataFrame
                    df = pd.DataFrame({key: [value]})
                elif isinstance(value, (list, np.ndarray)):
                    value = np.array(value)
                    if value.ndim == 1:
                        df = pd.DataFrame({key: value})
                    elif value.ndim == 2:
                        df = pd.DataFrame(value)
                    else:
                        continue  # skip 3D or higher
                else:
                    continue  # skip unsupported types

                df.to_excel(writer, sheet_name=key[:31], index=False)
            except Exception as e:
                print(f"Could not write {key}: {e}")

def wav2sig(insig, fs=None, dBFS=94):
    """
    Load a WAV file and return a calibrated, **mono, floating-point** signal.

    Parameters
    ----------
    insig : str | pathlib.Path
        Path to the input WAV file.
    fs : None, optional
        Reserved for backward compatibility—ignored, as the sample rate is
        obtained from the file header.
    dBFS : float, default ``94``
        Reference level that defines how many decibels full scale represent
        (see step 2 above).

    Returns
    -------
    numpy.ndarray
        Mono signal (shape ``(N,)``) in pascals if *dBFS* refers to SPL.
    int
        Sampling frequency in hertz.

    Notes
    -----
    * If the input file is already floating-point, only the dBFS scaling is
      applied.
    * Simple channel averaging may alter perceived loudness for correlated
      stereo material.
    """


    fs, insig_raw = wavfile.read(insig)

    # Convert to float if needed
    if insig_raw.dtype.kind in 'iu':
        max_val = np.iinfo(insig_raw.dtype).max
        insig = insig_raw.astype(np.float32) / max_val
    else:
        insig = insig_raw.astype(np.float32)

    # dBFS scaling (default: 94 dB SPL full scale = 1 Pa)
    gain_factor = 10 ** ((dBFS - 94) / 20)
    insig = gain_factor * insig

    # If stereo, convert to mono
    if insig.ndim > 1:
        insig = np.mean(insig, axis=1)

    insig = np.asarray(insig)

    return insig, fs

def buffer(x, n, p=0, opt='nodelay'):
    """
    Segment a 1-D signal into (possibly overlapping) frames, emulating
    MATLAB’s :func:`buffer` with the ``'nodelay'`` option.

    The input vector *x* is sliced into columns of length *n*.  
    If an **overlap** of *p* samples is specified, successive frames start
    every ``n − p`` samples; otherwise the frames are non-overlapping.  
    Insufficient samples in the last frame are zero-padded.

    Parameters
    ----------
    x : array_like
        Input signal (1-D).  Any type accepted by
        :pyfunc:`numpy.asarray`.
    n : int
        Length of each frame in samples.
    p : int, default ``0``
        Number of samples that **overlap** between consecutive frames
        (``0 ≤ p < n``).
    opt : {'nodelay'}, default ``'nodelay'``
        Only the *nodelay* behaviour is implemented; other MATLAB options
        raise :class:`NotImplementedError`.

    Returns
    -------
    numpy.ndarray
        2-D array with shape ``(n, cols)``, where ``cols`` is the number of
        generated frames.  The signal is arranged **column-wise** (Fortran
        order), matching MATLAB output.
    
    Raises
    ------
    ValueError
        If *p* ≥ *n*.
    NotImplementedError
        For any *opt* value other than ``'nodelay'``.
    """

    if opt == 'nodelay':
        # Calculate number of columns
        if p == 0:
            # No overlap
            cols = len(x) // n
            # Truncate x to fit exactly
            x_truncated = x[:cols * n]
            return x_truncated.reshape((n, cols), order='F')
        else:
            # With overlap
            step = n - p
            if step <= 0:
                raise ValueError("Overlap must be less than frame length")
            
            # Calculate number of frames
            cols = (len(x) - p) // step
            if cols <= 0:
                cols = 1
            
            # Create buffered array
            buffered = np.zeros((n, cols))
            for i in range(cols):
                start = i * step
                end = start + n
                if end <= len(x):
                    buffered[:, i] = x[start:end]
                else:
                    # Pad with zeros if necessary
                    remaining = len(x) - start
                    buffered[:remaining, i] = x[start:]
            
            return buffered
    else:
        raise NotImplementedError("Only 'nodelay' option is implemented")

def cos_ramp(sig_len=None, fs=None, attack=None, release=None, plot_result=False):
    """
    Create a **raised-cosine ramp envelope** with optional fade-in (*attack*)
    and fade-out (*release*) segments.

    The envelope is initialised to unity and the first *attack* samples and
    final *release* samples are multiplied by

    ``0.5 · (1 − cos(π · n / L))``

    where *L* is the respective ramp length in samples and
    ``n = 1 … L``.  If *release* is omitted it defaults to the *attack*
    duration.  Ramp lengths that exceed half the signal are clipped to
    ``⌊(sig_len − 1) / 2⌋`` samples.

    Parameters
    ----------
    sig_len : int, optional
        Total number of samples in the envelope.  Defaults to ``44100``.
    fs : float, optional
        Sampling frequency in hertz (used only to interpret *attack* and
        *release*).  Defaults to ``44100`` Hz.
    attack : float, optional
        Fade-in duration in **milliseconds**.  Defaults to ``25`` ms.
    release : float, optional
        Fade-out duration in **milliseconds**.  If ``None`` the envelope
        uses the *attack* duration.  Defaults to ``25`` ms.
    plot_result : bool, default ``False``
        When ``True`` display the envelope versus time using Matplotlib and
        still return the sample array.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of length *sig_len* containing the
        sample-accurate ramp (range 0–1).

    Notes
    -----
    * Internally, durations are rounded to the nearest integer number of
      samples.
    * For zero ramp lengths the function returns an all-ones vector.
    * Only the MATLAB-compatible ``'nodelay'`` behaviour is implemented; no
      initial zero-padding delay is added.
    """

    
    nargin = 0
    if sig_len is not None:
        nargin += 1
    if fs is not None:
        nargin += 1
    if attack is not None:
        nargin += 1
    if release is not None:
        nargin += 1
    
    if nargin < 1 or sig_len is None:
        sig_len = 44100  # only for demo
    
    if nargin < 2 or fs is None:
        fs = 44100
    
    if nargin < 3 or attack is None:
        attack = 25
    
    if nargin < 4 or release is None:
        release = 25
    
    attack = round(fs * attack / 1000)
    ramp = np.ones(sig_len)
    
    if attack >= sig_len:
        print('Attack: %i, Signal: %i' % (attack, len(ramp)))
        attack = round((sig_len - 1) / 2)
        print('Attack changed to: %i, %.4f (msec)' % (attack, 1000/fs*attack))
    
    if nargin == 4:  # attack and release time are different
        for i in range(1, attack + 1):
            ramp[i-1] = ramp[i-1] * 0.5 * (1 - np.cos(np.pi * i / attack))
        
        release = round(fs * release / 1000)
        for i in range(sig_len - release, sig_len + 1):
            if i <= sig_len:  # bounds check
                ramp[i-1] = ramp[i-1] * 0.5 * (1 - np.cos(np.pi * (i - sig_len) / release))
    else:  # attack and release time are the same
        for i in range(1, attack + 1):
            ramp[i-1] = ramp[i-1] * 0.5 * (1 - np.cos(np.pi * i / attack))
        
        for i in range(sig_len - attack, sig_len + 1):
            if i <= sig_len:  # bounds check
                ramp[i-1] = ramp[i-1] * 0.5 * (1 - np.cos(np.pi * (i - sig_len) / attack))
    
    if attack == 0 or release == 0:
        ramp[-1] = 1  # ramp(end) in Matlab becomes ramp[-1] in Python
    
    # Plotting functionality (equivalent to nargout == 0 check)
    if plot_result:
        t = np.arange(1, sig_len + 1) / fs
        plt.figure()
        plt.plot(t, ramp)
        plt.xlabel('time [s]')
        plt.ylabel('Amplitude')
        plt.show()
    
    return ramp

def rmsdb(x, fs=None, ti=None, tf=None):
    """
    Compute the **root-mean-square (RMS) level** of an audio signal—or WAV
    file segment—in decibels.

    The level for the selected time window ``[ti, tf)`` is

    ``L = 10 · log10( Σ x_i² / N )``  [dB],

    where *N* is the number of samples in the segment.  Multi-channel inputs
    are collapsed to one scalar by summing the squared samples across all
    channels before averaging.

    Parameters
    ----------
    x : str | array_like
        Path to a WAV file **or** an array of audio samples.  When a
        filename is given the signal and its sampling rate are obtained via
        :pyfunc:`scipy.io.wavfile.read`.
    fs : int | float, optional
        Sampling frequency in hertz.  Required only when *x* is an array;
        ignored for filename input.
    ti : float, optional
        Start time of the analysis window in seconds.  Defaults to the
        beginning of the signal.
    tf : float, optional
        End time of the analysis window in seconds (non-inclusive).  Defaults
        to the end of the signal.

    Returns
    -------
    float
        RMS level in decibels relative to an amplitude of **1**.

    Raises
    ------
    ValueError
        If *x* is a string but no readable WAV file is found.

    Notes
    -----
    * Row and column vectors are both treated as single-channel signals.
    * If *ti* and *tf* delimit an empty slice the function uses all
      available samples and therefore matches MATLAB’s ``rmsdb`` behaviour.
    """

    if isinstance(x, str):
        try:
            fs, x = wavfile.read(x)
            x = x.astype(np.float64)
        except:
            raise ValueError('variable x interpreted as char, but no wav file with such a name was found')
    
    # Convert to numpy array for consistent handling
    x = np.array(x)
    
    if tf is None:
        Nf = len(x)
    else:
        Nf = round(tf * fs)
    
    if ti is None:
        Ni = 0  # Python uses 0-based indexing, so equivalent to Matlab's 1
    else:
        Ni = int(np.ceil(ti * fs + 1e-6)) - 1  # Convert from 1-based to 0-based indexing
        Ni = max(0, Ni)  # Ensure minimum index is 0
    
    # Get dimensions equivalent to Matlab's [r,c]=size(x)
    if x.ndim == 1:
        r = len(x)
        c = 1
        x = x.reshape(-1, 1)  # Make it a column vector for consistency
    else:
        r, c = x.shape
    
    if c == 1:
        # Column vector case: y = 10*log10( x(Ni:Nf)'*x(Ni:Nf)/length(x(Ni:Nf)) )
        x_segment = x[Ni:Nf].flatten()
        y = 10 * np.log10(np.dot(x_segment, x_segment) / len(x_segment))
    elif r == 1:
        # Row vector case: Nf = c; y = 10*log10( x(Ni:Nf)*x(Ni:Nf)'/length(x(Ni:Nf)) )
        Nf = c
        x_segment = x.flatten()[Ni:Nf]
        y = 10 * np.log10(np.dot(x_segment, x_segment) / len(x_segment))
    else:
        # Generic case: y = 10*log10( sum(x(Ni:Nf,:).*x(Ni:Nf,:))/length(x(Ni:Nf,:)) )
        x_segment = x[Ni:Nf, :]
        y = 10 * np.log10(np.sum(x_segment * x_segment) / len(x_segment))
    
    return y

# ----------------------
#### EPNL FUNCTIONS ####
# ----------------------

def get_Duration_Correction(PNLT, PNLTM, PNLTM_idx, dt, threshold):
    """
    Calculate the **duration-correction, *D***, used in the ICAO procedure
    for Effective Perceived Noise Level (EPNL).

    The algorithm follows Annex 16, Vol. I, App. 2 logic.

    Parameters
    ----------
    PNLT : numpy.ndarray
        Array of instantaneous perceived-noise levels in decibels.
    PNLTM : float
        Maximum of *PNLT* (peak perceived-noise level).
    PNLTM_idx : int
        Index of *PNLTM* within *PNLT*.
    dt : float
        Time step between consecutive *PNLT* samples in seconds.
    threshold : float
        Decibel offset below *PNLTM* that defines the integration window
        (commonly **10 dB**).

    Returns
    -------
    D : float
        Duration-correction in decibels to be **added** to *PNLTM* to obtain
        EPNL.
    idx_t1 : int or None
        Index of the first sample inside the integration window (*t₁*), or
        ``None`` if no such sample exists before *PNLTM*.
    idx_t2 : int
        Index of the last sample inside the window (*t₂*).  If *PNLT* never
        decays below the threshold, this equals ``len(PNLT) − 1`` and a
        warning is emitted.

    Warns
    -----
    RuntimeWarning
        When *PNLT* does not decay past the threshold within the available
        data; *D* is then indicative only.

    Notes
    -----
    * The summation in step 3 includes both limits: ``PNLT[idx_t1:idx_t2+1]``.
    * All indices are **0-based**, matching NumPy conventions.
    """


    # Find the PNLTM-threshold down points (t1 and t2)
    Decay = PNLTM - threshold
    K = 1  # find only first idx
    
    # t(1) is the first point of time after which PNLT becomes greater than PNLTM minus threshold
    indices = np.where(PNLT[0:PNLTM_idx] > Decay)[0]  # idx_t1 is the first point where PNLT becomes >(PNLTM - threshold)
    if len(indices) > 0:
        idx_t1 = indices[0]  # take first occurrence (K=1)
    else:
        idx_t1 = None
    
    # t(2) is the point of time after which PNLT remains constantly less than PNLTM minus threshold
    indices = np.where(PNLT[PNLTM_idx:] < Decay)[0]  # idx_t2 is the first point where PNLT becomes <(PNLTM - threshold)
    if len(indices) > 0:
        idx_t2 = indices[0] + PNLTM_idx  # correct for PNLTM_idx number because full vector is trimmed in the previous line
    else:
        idx_t2 = None
    
    # if case for idx_t2 not found (PNLT never becomes lower than Decay)
    if idx_t2 is None:
        idx_t2 = PNLT.shape[0] - 1  # take idx_t2 as last index in PNLT
        warnings.warn("The signal does not decay by more than the threshold within the available duration. An indicative EPNL value is calculated from the available duration, but should not be used for aircraft noise certification.")
    
    # Calculate duration correction factor
    D = 10 * np.log10(np.sum(10**(PNLT[idx_t1:idx_t2+1]/10))) - PNLTM + 10 * np.log10(dt/10)
    
    return D, idx_t1, idx_t2

def get_PNL(input):
    """
    Compute **Perceived Noisiness** (*PN*) and **Perceived Noise Level**
    (*PNL*) from octave- or ⅓-octave-band sound-pressure levels.

    The implementation follows the ICAO Annex 16, Vol. I,  
    *Aircraft Noise* – Appendix 2 algorithm.

    The peak value *PNLM* and its index *PNLM_idx* are returned for
    convenience.

    Parameters
    ----------
    input : numpy.ndarray, shape (T, B)
        Matrix of octave-band SPLs in decibels where *T* is the number of
        time samples and *B* ≤ 24 is the number of frequency bands
        (31.5 Hz … 10 kHz as per Table A36-3).

    Returns
    -------
    PN : numpy.ndarray, shape (T,)
        Instantaneous perceived noisiness in **Noys**.
    PNL : numpy.ndarray, shape (T,)
        Instantaneous perceived noise level in **PNdB**.
    PNLM : float
        Maximum PNL value (peak level).
    PNLM_idx : int
        Index of *PNLM* within *PNL* (0-based).

    Notes
    -----
    * Bands with *SPL* below *SPL₍d₎* contribute **0 noy**.
    * For bands 10–22 the upper break point *SPL₍a₎* is ∞, matching ICAO’s
      tabulated values.
    * All logarithms are base-10.
    """

    
    num_times = input.shape[0]  # number of time steps
    num_freqs = input.shape[1]  # number of freq bands

    # Read Table A36-3 from Ref. [1], which provides the constants for mathematically formulated NOY values
    noy_tab = np.array([
    [1, 50, 91, 64, 52, 49, 55, 0.043478, 0.030103, 0.07952, 0.058098],
    [2, 63, 85.9, 60, 51, 44, 51, 0.040570, 0.030103, 0.06816, 0.058098],
    [3, 80, 87.3, 56, 49, 39, 46, 0.036831, 0.030103, 0.06816, 0.052288],
    [4, 100, 79.9, 53, 47, 34, 42, 0.036831, 0.030103, 0.05964, 0.047534],
    [5, 125, 79.8, 51, 46, 30, 39, 0.035336, 0.030103, 0.053013, 0.043573],
    [6, 160, 76, 48, 45, 27, 36, 0.033333, 0.030103, 0.053013, 0.043573],
    [7, 200, 74, 46, 43, 24, 33, 0.033333, 0.030103, 0.053013, 0.040221],
    [8, 250, 74.9, 44, 42, 21, 30, 0.032051, 0.030103, 0.053013, 0.037349],
    [9, 315, 94.6, 42, 41, 18, 27, 0.030675, 0.030103, 0.053013, 0.034859],
    [10, 400, np.inf, 40, 40, 16, 25, 0.030103, 0, 0.053013, 0.034859],
    [11, 500, np.inf, 40, 40, 16, 25, 0.030103, 0, 0.053013, 0.034859],
    [12, 630, np.inf, 40, 40, 16, 25, 0.030103, 0, 0.053013, 0.034859],
    [13, 800, np.inf, 40, 40, 16, 25, 0.030103, 0, 0.053013, 0.034859],
    [14, 1000, np.inf, 40, 40, 16, 25, 0.030103, 0, 0.053013, 0.034859],
    [15, 1250, np.inf, 38, 38, 15, 23, 0.030103, 0, 0.05964, 0.034859],
    [16, 1600, np.inf, 34, 34, 12, 21, 0.02996, 0, 0.053013, 0.040221],
    [17, 2000, np.inf, 32, 32, 9, 18, 0.02996, 0, 0.053013, 0.037349],
    [18, 2500, np.inf, 30, 30, 5, 15, 0.02996, 0, 0.047712, 0.034859],
    [19, 3150, np.inf, 29, 29, 4, 14, 0.02996, 0, 0.047712, 0.034859],
    [20, 4000, np.inf, 29, 29, 5, 14, 0.02996, 0, 0.053013, 0.034859],
    [21, 5000, np.inf, 30, 30, 6, 15, 0.02996, 0, 0.053013, 0.034859],
    [22, 6300, np.inf, 31, 31, 10, 17, 0.02996, 0, 0.06816, 0.037349],
    [23, 8000, 44.3, 37, 34, 17, 23, 0.042285, 0.02996, 0.07952, 0.037349],
    [24, 10000, 50.7, 41, 37, 21, 29, 0.042285, 0.02996, 0.05964, 0.043573]
    ])
    # Band = noy_tab[:, 0]
    # f = noy_tab[:, 1]
    SPLa = noy_tab[:, 2]
    SPLb = noy_tab[:, 3]
    SPLc = noy_tab[:, 4]
    SPLd = noy_tab[:, 5]
    SPLe = noy_tab[:, 6]
    Mb = noy_tab[:, 7]
    Mc = noy_tab[:, 8]
    Md = noy_tab[:, 9]
    Me = noy_tab[:, 10]

    # Convert SPL to Perceived Noisiness, nn

    # DEFINITIONS
    # i is index for the octave bands
    # k is time-index vector

    nn = np.zeros((num_times, num_freqs))

    for i in range(num_freqs):
        for k in range(num_times):
            SPL = input[k, i]
            if SPL >= SPLa[i]:
                nn[k, i] = 10**(Mc[i] * (SPL - SPLc[i]))
            elif SPL >= SPLb[i] and SPL < SPLa[i]:
                nn[k, i] = 10**(Mb[i] * (SPL - SPLb[i]))
            elif SPL >= SPLe[i] and SPL < SPLb[i]:
                nn[k, i] = 0.3 * 10**(Me[i] * (SPL - SPLe[i]))
            elif SPL >= SPLd[i] and SPL < SPLe[i]:
                nn[k, i] = 0.1 * 10**(Md[i] * (SPL - SPLd[i]))
            else:
                nn[k, i] = 0

    # Combine the Perceived Noisiness, nn, to get PN and PNL

    PNL = np.zeros(num_times)
    PN = np.zeros(num_times)

    for k in range(num_times):
        nmax = np.max(nn[k, :])
        PN[k] = 0.85 * nmax + 0.15 * np.sum(nn[k, :])  # Perceived Noisiness, unit is Noys
        
        # Convert the total Perceived Noisiness, N(k) into Perceived Noise Level, PNL(k):
        PNL[k] = 40 + (10/np.log10(2)) * np.log10(PN[k])  # Perceived Noise Level, unit is PNdB

    PNLM_idx = np.argmax(PNL)  # Index of Maximum Perceived Noise Level
    PNLM = PNL[PNLM_idx]  # Maximum Perceived Noise Level (PNLM), unit is PNdB

    return PN, PNL, PNLM, PNLM_idx

def get_PNLT(input, freq_bands, PNL):
    """
    Apply the **ICAO Annex 16, Vol. I, App. 2 tone-correction** algorithm to a
    sequence of octave- or third-octave-band spectra and return the tone-
    corrected perceived-noise level *PNLT*.

    Parameters
    ----------
    input : ndarray, shape ``(T, B)``
        Spectral sound-pressure levels in **dB** where *T* is the number of
        time steps and *B* ≥ 24 are the octave/⅓-octave bands
        50 Hz … 10 kHz.
    freq_bands : ndarray, shape ``(B,)``
        Centre frequency of each band in **hertz**.
    PNL : ndarray, shape ``(T,)``
        Previously computed perceived-noise levels (see :pyfunc:`get_PNL`).

    Returns
    -------
    PNLT : ndarray, shape ``(T,)``
        Tone-corrected perceived-noise level (TPNdB) for every time step.
    PNLTM : float
        Maximum value of *PNLT*.
    PNLTM_idx : int
        Index of *PNLTM* within *PNLT* (0-based).
    OUT : dict
        Intermediate arrays useful for validation:

        ``'S', 'delS', 'SPLP', 'SP', 'SB', 'SPLPP', 'F', 'C', 'diff'``.

    Warns
    -----
    RuntimeWarning
        If the **band-sharing adjustment** is impossible because *PNLTM* is
        too close to the record’s edges; *ΔB* is then set to 0 dB.

    Notes
    -----
    * All logarithmic operations are base-10.
    * The algorithm assumes **linear** time spacing between spectra.
    * Bands below 80 Hz are passed through unchanged.
    """


    # DEFINITIONS
    # i is index for the octave bands
    # k is time-index vector
    
    num_times = input.shape[0]  # number of time steps
    num_freqs = input.shape[1]  # number of freq bands
    
    ## Step 1: start with the SPL in the 80 Hz TOB (band number 3 in this case),
    # and calculate the changes in SPL (or "SLOPES) in the remainder TOB
    
    index_80 = np.where(freq_bands >= 80)[0][0]
    
    S = np.zeros((num_freqs, num_times))
    for k in range(num_times):
        for i in range(index_80 + 1, num_freqs):
            S[i, k] = input[k, i] - input[k, i-1]
    
    ## Step 2: Encircle the value of the SLOPE, S(i,k),
    # where the absolute value of the change in slope is greater than 5 dB
    # STEP 3 is also included here
    
    delS = np.zeros(S.shape)
    SPLs = np.zeros(delS.shape)
    diff = np.zeros(num_freqs)
    
    for k in range(num_times):
        for i in range(index_80, num_freqs):
            diff[i] = abs(S[i, k] - S[i-1, k])
            if diff[i] > 5:  # <--Step 2
                delS[i, k] = S[i, k]
                if S[i, k] > 0 and (S[i, k] > S[i-1, k]):  # Step 3.1: If the encircled value of S(i,k) is positive and algebraically greater than S(i-1,k), then encircle SPL(i,k)
                    SPLs[i, k] = input[k, i]
                elif S[i, k] <= 0 and (S[i-1, k] > 0):  # Step 3.2: if the encircled value of S(i,k) is zero or negative and S(i-1,k) is positive, then encircle SPL(i-1,k)
                    SPLs[i-1, k] = input[k, i-1]
            # Step 3.3: for all other cases, no SPL value is encircled
    
    ## Step 4: Omit all SPL(i,k) encircled in Step 3 and
    # compute the new SPL levels, SPLP(i,k):
    
    SPLP = np.zeros((num_freqs, num_times))
    
    for k in range(num_times):
        for i in range(num_freqs):
            if SPLs[i, k] == 0:  # Step 4.1: for non-encircled SPL, set SPLP equal to the original SPL, SPLP(i,k) = SPL(i,k)
                SPLP[i, k] = input[k, i]
            elif SPLs[i, k] > 0 and i < num_freqs - 1:  # Step 4.2: for encircled SPL in bands 1 (50 Hz) till 23 (8 kHz) inclusive, set SPLP equal to the arithmetic average of the preceding and following SPL
                SPLP[i, k] = 0.5 * (input[k, i-1] + input[k, i+1])
            elif SPLs[i, k] > 0 and i == num_freqs - 1:  # Step 4.3: if the SPL in the highest freq band (10 kHz) is encircled, set the SPLP in that band equal to SPLP (k,24) = SPL (k,23) + S(k,23).
                SPLP[i, k] = (input[k, i-1] + S[i-1, k])
            elif SPLs[i, k] <= 0 and i == num_freqs - 1:
                SPLP[i, k] = input[k, i]
    
    ## STEP 5: Recompute the new SLOPE, SP(i,k),
    # including one for an imaginary 25th freq band (i.e. 12.5 kHz)
    
    SP = np.zeros((num_freqs + 1, num_times))
    
    for k in range(num_times):
        for i in range(index_80 + 1, num_freqs - 1):
            SP[i, k] = SPLP[i, k] - SPLP[i-1, k]
        SP[index_80, k] = SP[index_80 + 1, k]
        SP[num_freqs - 1, k] = SPLP[num_freqs - 1, k] - SPLP[num_freqs - 2, k]
        SP[num_freqs, k] = SP[num_freqs - 1, k]
    
    ## STEP 6: for i, from 3 (80 Hz) till 23 (8 kHz),
    # compute the arithmetic average of the three adjacent slopes (i.e. next 2)
    
    SB = np.zeros((num_freqs + 1, num_times))
    for k in range(num_times):
        for i in range(index_80, num_freqs - 1):
            SB[i, k] = (1/3) * (SP[i, k] + SP[i+1, k] + SP[i+2, k])
    
    ## STEP 7: compute the final adjusted TOB SPL, SPLPP(i,k),
    # by beginning with the band number 3 (80 Hz) and proceeding to band number 24 (10 kHz)
    
    SPLPP = np.zeros((num_freqs, num_times))
    
    for k in range(num_times):
        SPLPP[index_80, k] = input[k, index_80]
        for i in range(index_80 + 1, num_freqs - 1):
            SPLPP[i, k] = (SPLPP[i-1, k] + SB[i-1, k])
        SPLPP[num_freqs - 1, k] = SPLPP[num_freqs - 2, k] + SB[num_freqs - 2, k]
    
    ## Step 8: calculate the differences, F(i,k),
    # between the original SPL and the final background SPL, SPLPP(i,k)
    
    F = np.zeros((num_freqs, num_times))
    
    for k in range(num_times):
        for i in range(index_80, num_freqs):
            F[i, k] = (input[k, i] - SPLPP[i, k])
            if F[i, k] <= 1.5:  # note only values equal to or greater than 1.5.
                F[i, k] = 0
    
    ## Step 9: for each TOB from 80 Hz till 10 kHz (i.e. band 3 through 24),
    # determine the tone correction factor C(i,k) from the SPL differences F(i,k), and Table A36-2
    # Step 10 is included here also
    
    C = np.zeros((num_freqs, num_times))
    Cmax = np.zeros(num_times)
    
    for k in range(num_times):
        for i in range(index_80, num_freqs):
            if (freq_bands[i] >= 50 and freq_bands[i] < 500 and F[i, k] >= 1.5 and F[i, k] < 3):
                C[i, k] = (F[i, k]/3 - 0.5)
            elif (freq_bands[i] >= 50 and freq_bands[i] < 500 and F[i, k] >= 3 and F[i, k] < 20):
                C[i, k] = (F[i, k]/6)
            elif (freq_bands[i] >= 50 and freq_bands[i] < 500 and F[i, k] >= 20):
                C[i, k] = 10/3
            elif (freq_bands[i] >= 500 and freq_bands[i] <= 5000 and F[i, k] >= 1.5 and F[i, k] < 3):
                C[i, k] = ((2 * F[i, k]/3) - 1)
            elif (freq_bands[i] >= 500 and freq_bands[i] <= 5000 and F[i, k] >= 3 and F[i, k] < 20):
                C[i, k] = (F[i, k]/3)
            elif (freq_bands[i] >= 500 and freq_bands[i] <= 5000 and F[i, k] >= 20):
                C[i, k] = (20/3)
            elif (freq_bands[i] > 5000 and freq_bands[i] <= 10000 and F[i, k] >= 1.5 and F[i, k] < 3):
                C[i, k] = (F[i, k]/3 - 0.5)
            elif (freq_bands[i] > 5000 and freq_bands[i] <= 10000 and F[i, k] >= 3 and F[i, k] < 20):
                C[i, k] = (F[i, k]/6)
            elif (freq_bands[i] > 5000 and freq_bands[i] <= 10000 and F[i, k] >= 20):
                C[i, k] = (10/3)
        Cmax[k] = np.max(C[:, k])  # <--- STEP 10: designate the largest of the tone correction factors determined in STEP 9 as Cmax(k).
    
    ## PNLT calculation
    # The Tone-corrected perceived noise levels, PNLT(k), must be determined by
    # adding Cmax(k) values to corresponding PNL(k) values
    
    PNLT = np.zeros(num_times)
    
    for k in range(num_times):
        PNLT[k] = PNL[k] + Cmax[k]  # TONE-CORRECTED PERCEIVED NOISE LEVEL, unit is TPNdB
    
    PNLTM_idx = np.argmax(PNLT)
    PNLTM = PNLT[PNLTM_idx]  # MAXIMUM TONE-CORRECTED PERCEIVED NOISE LEVEL (PNLTM)
    
    ## Bandsharing adjustment to PNLTM
    
    if Cmax.shape[0] != 1:  # workaround to run the <run_validation_tone_correction.m> code, where only one time-step is considered
        
        # in case <PNLTM_idx> is closer from the lower or higher boundaries of
        # the time vector, the bandsharing adjustment may not possible
        
        indicesToAccess = np.array([PNLTM_idx - 2, PNLTM_idx - 1, PNLTM_idx + 1, PNLTM_idx + 2])  # get indices to access
        
        isValid = (indicesToAccess >= 0) & (indicesToAccess < len(PNLT))  # Logical condition to check for valid indices (0-based)
        
        if np.all(isValid):  # runs only if all indices are valid
            
            Cavg = np.sum([Cmax[PNLTM_idx - 2], Cmax[PNLTM_idx - 1], Cmax[PNLTM_idx],
                          Cmax[PNLTM_idx + 1], Cmax[PNLTM_idx + 2]]) / 5
            
            if Cavg > Cmax[PNLTM_idx]:
                DeltaB = Cavg * Cmax[PNLTM_idx]
            else:
                DeltaB = 0
        
        else:  # there are empty indices: Bandsharing adjustment to PNLTM not possible
            DeltaB = 0
            warnings.warn('Bandsharing adjustment to PNLTM not possible. DeltaB truncated to zero.')
        
        # apply adjustment
        PNLTM = PNLTM + DeltaB
    
    ## Output variables for verification of the tone correction implementation
    
    OUT = {} # Initialize OUT as a dictionary
    
    OUT['S'] = S
    OUT['delS'] = delS
    OUT['SPLP'] = SPLP
    OUT['SP'] = SP
    OUT['SB'] = SB
    OUT['SPLPP'] = SPLPP
    OUT['F'] = F
    OUT['C'] = C
    OUT['diff'] = diff
    
    return PNLT, PNLTM, PNLTM_idx, OUT

# ---------------------------
#### ECMA418_2 FUNCTIONS ####
# ---------------------------

# def shm_auditory_filt_bank(signal: np.ndarray, outplot: bool = False) -> np.ndarray:
#     """
#     Apply the ECMA-418-2 auditory filter bank to a mono audio signal.

#     Parameters
#     ----------
#     signal : array_like of float, shape (N,)
#         Mono audio time-series sampled at 48 kHz.
#     outplot : bool, default False
#         If True, plot the combined magnitude and phase responses of all 53 half-Bark filters.

#     Returns
#     -------
#     filtered : ndarray, shape (N, 53)
#         Filtered signal with one column per half-Bark band.
#     """

#     # Argument Validation
#     signal = np.asarray(signal, dtype=float)
#     if signal.ndim != 1:
#         raise ValueError("signal must be a 1‑D (mono) array")
#     if not np.isrealobj(signal):
#         raise ValueError("signal must be real‑valued")

#     sampleRate48k = 48e3  # Hz
#     deltaFreq0 = 81.9289
#     c = 0.1618

#     halfBark = np.arange(0.5, 26.5 + 0.5, 0.5)  # 0.5 … 26.5 inclusive (53 bands)
#     bandCentreFreqs = (deltaFreq0 / c) * np.sinh(c * halfBark)
#     dfz = np.sqrt(deltaFreq0**2 + (c * bandCentreFreqs) ** 2)

#     # Signal Processing
#     k = 5  # filter order (footnote 5 ECMA‑418‑2)
#     e_i = np.array([0, 1, 11, 11, 1], dtype=float)
#     signalFiltered = np.empty((signal.size, halfBark.size), dtype=float).T
#     if outplot:
#         fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
#         ax_mag.set_xscale("log")
#         ax_phase.set_xscale("log")

#     # Main loop: highest → lowest band (to mimic MATLAB for reproducibility)
#     for zBand in range(halfBark.size - 1, -1, -1):
#         tau = (
#             (1 / (2 ** (2 * k - 1)))
#             * comb(2 * k - 2, k - 1)
#             * (1 / dfz[zBand])
#         )
#         d = np.exp(-1 / (sampleRate48k * tau))

#         bp = np.exp(
#             1j
#             * 2
#             * np.pi
#             * bandCentreFreqs[zBand]
#             * np.arange(k + 2)
#             / sampleRate48k
#         )

#         m = np.arange(1, k + 1)
#         a_m = np.concatenate(([1.0], (-d) ** m * comb(k, m))) * bp[: k + 1]

#         m = np.arange(0, k)
#         b_m = (((1 - d) ** k) / np.sum(e_i[1:] * d ** np.arange(1, k))) * (d ** m) * e_i[:k] * bp[:k]

#         signalFiltered[zBand,:] = 2.0 * np.real(lfilter(b_m, a_m, signal))
#         if outplot:
#             w, h = freqz(b_m, a_m, worN=10_000, fs=sampleRate48k, whole=True)
#             ax_mag.semilogx(w, 20 * np.log10(np.abs(h)))
#             ax_phase.semilogx(w, np.unwrap(np.angle(h)) * 180 / np.pi)

#     # Plot Figures
#     if outplot:
#         for ax in (ax_mag, ax_phase):
#             ax.grid(True, which="both", ls=":")
#             ax.set_xlim(20, 20_000)
#             ax.set_xticks(
#                 [31.5, 63, 125, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
#             )
#             ax.set_xticklabels(
#                 [
#                     "31.5",
#                     "63",
#                     "125",
#                     "250",
#                     "500",
#                     "1k",
#                     "2k",
#                     "4k",
#                     "8k",
#                     "16k",
#                 ]
#             )
#         ax_mag.set_ylabel("Magnitude (dB)")
#         ax_phase.set_ylabel("Phase (°)")
#         ax_phase.set_xlabel("Frequency (Hz)")
#         plt.tight_layout()

#     return signalFiltered

# def shm_basis_loudness(
#     signalSegmented: np.ndarray,
#     bandCentreFreq: float | None = None,
#     tol: float = 1.0
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute basis loudness using the Sottek Hearing Model (ECMA-418-2).

#     Parameters
#     ----------
#     signalSegmented : ndarray of float, shape (n_samples, n_blocks[, n_bands])
#         Segmented pressure signal. Must be real-valued.
#     bandCentreFreqs : float, optional
#         Centre frequency in Hz when input is 2-D (single band). Omit for 3-D input.
#     tol : float, default 1.0
#         Frequency tolerance in Hz for matching bandCentreFreqs to standard half-Bark centres.

#     Returns
#     -------
#     signal_rect_seg : ndarray
#         Half-wave rectified signal, same shape as input.
#     basis_loudness : ndarray
#         Basis loudness values, same shape as input.
#     block_rms : ndarray
#         Block RMS values, shape (n_blocks,) or (n_blocks, n_bands).
#     """

#     # Constants
#     deltaFreq0 = 81.9289
#     c = 0.1618
#     halfBark = np.arange(0.5, 27, 0.5)  # 0.5 … 26.5 inclusive
#     bandCentreFreqs = (deltaFreq0 / c) * np.sinh(c * halfBark)  # Eq. 9

#     cal_N = 0.0211668
#     cal_Nx = 1.00132
#     a = 1.5  # a in Eq. 23

#     p_threshold = 2e-5 * 10 ** (np.arange(15, 86, 10) / 20)  # Pa (8‑vals)
#     v = np.array([1, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434])

#     LTQz = np.array([
#         0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453, 0.0365, 0.0298,
#         0.0247, 0.0207, 0.0176, 0.0151, 0.0131, 0.0115, 0.0103, 0.0093,
#         0.0086, 0.0081, 0.0077, 0.0074, 0.0073, 0.0072, 0.0071, 0.0072,
#         0.0073, 0.0074, 0.0076, 0.0079, 0.0082, 0.0086, 0.0092, 0.0100,
#         0.0109, 0.0122, 0.0138, 0.0157, 0.0172, 0.0180, 0.0180, 0.0177,
#         0.0176, 0.0177, 0.0182, 0.0190, 0.0202, 0.0217, 0.0237, 0.0263,
#         0.0296, 0.0339, 0.0398, 0.0485, 0.0622
#     ])

#     # Validation
#     if not np.isrealobj(signalSegmented):
#         raise TypeError("signal_segmented must be real‑valued")

#     if signalSegmented.ndim not in (2, 3):
#         raise ValueError("signal_segmented must be 2‑D or 3‑D")

#     if bandCentreFreqs is None and signalSegmented.ndim == 2:
#         raise ValueError("bandCentreFreqs required for 2‑D input")

#     if bandCentreFreqs is not None and signalSegmented.ndim == 3:
#         raise ValueError("bandCentreFreqs should be omitted for 3‑D input")

#     # Centre‑frequency handling
#     if bandCentreFreqs is not None:
#         idx = int(np.abs(bandCentreFreqs - bandCentreFreqs).argmin())
#         if abs(bandCentreFreqs[idx] - bandCentreFreqs[idx]) > tol:
#             raise ValueError(
#                 f"{bandCentreFreqs} Hz is not within ±{tol} Hz of any standard half‑Bark centre; "
#                 f"closest is {bandCentreFreqs[idx]:.2f} Hz.")

#     ## Core processing ##
#     # Half Wave Rectification
#     signalRectSeg = np.maximum(signalSegmented, 0.0)

#     # Block RMS (Eq. 22) – factor 2 because rectified signal is positive‑only
#     sumRMS = np.sum(signalRectSeg ** 2, axis=0)
#     factorRMS = (2.0 / signalRectSeg.shape[0])
#     blockRMS = np.sqrt((2.0 / signalRectSeg.shape[0]) * np.sum(signalRectSeg ** 2, axis=0))

#     # Loudness transform (Eqs. 23–24) # TODO: this might be the issue
#     bandLoudness = cal_N * cal_Nx * (blockRMS / 20e-6) * np.prod(
#     (1 + (blockRMS[np.newaxis, :] / p_threshold[:, np.newaxis]) ** a) ** 
#     ((np.diff(v) / a)[:, np.newaxis]),
#     axis=0)

#     blockRMS = np.squeeze(blockRMS)

#     # Threshold‑in‑quiet correction (Eq. 25)
#     if bandCentreFreqs is not None:  # 2‑D input, single band
#         D1 = LTQz[bandCentreFreq == bandCentreFreqs]
#         basisLoudness = bandLoudness - LTQz[bandCentreFreq == bandCentreFreqs]
#     else:                             # 3‑D input, all 53 bands
#         basisLoudness = bandLoudness - LTQz.reshape((1,) * (blockRMS.ndim - 1) + (53,))

#     basisLoudness = np.maximum(basisLoudness, 0.0)

#     return signalRectSeg, basisLoudness, blockRMS

# def shm_noise_red_lowpass(signal: np.ndarray, sampleRatein: float) -> np.ndarray:
#     """
#     Apply a low-pass noise-reduction filter (ECMA-418-2).

#     Parameters
#     ----------
#     signal : ndarray of float, shape (N,) or (N, C)
#         Input audio samples (time × channels).
#     sampleRatein : float
#         Sampling rate in Hz.

#     Returns
#     -------
#     filtered : ndarray of float, same shape as signal
#         Noise-reduced signal.
#     """

#     # Validation
#     if not isinstance(signal, np.ndarray):
#         raise TypeError("`signal` must be a NumPy array")

#     if signal.ndim not in (1, 2):
#         raise ValueError("`signal` must be 1‑D (mono) or 2‑D (time × channels)")

#     if not np.isrealobj(signal):
#         raise ValueError("`signal` must contain real‑valued samples")

#     if not (isinstance(sampleRatein, (float, int)) and sampleRatein > 0):
#         raise ValueError("`sampleRatein` must be a positive scalar sample‑rate (Hz)")

#     # Coefficient design (ECMA‑418‑2:2024, Equations 14‑15)
#     k = 3
#     e_i = np.array([0.0, 1.0, 1.0])          # Footnote 21
#     tau = (1 / 32) * (6 / 7)                 # Footnote 20
#     d = np.exp(-1.0 / (sampleRatein * tau))            # §5.1.4.2

#     # Denominator (a): Equation 14
#     m = np.arange(1, k + 1)
#     a = np.concatenate(([1.0], ((-d) ** m) * np.array([3, 3, 1], dtype=float)))

#     # Numerator (b): Equation 15
#     m = np.arange(0, k)          # 0‥k‑1
#     i = np.arange(1, k)          # 1‥k‑1
#     b = (((1 - d) ** k) / (np.sum(e_i[i] * (d ** i)))) * (d ** m) * e_i     # note e_i[0] == 0 → b[0] == 0

#     # Filtering
#     was_1d = (signal.ndim == 1)
#     x = signal[:, None] if was_1d else signal  # shape (N, C)

#     y = lfilter(b, a, x, axis=0)

#     return y.ravel() if was_1d else y

# def shm_out_mid_ear_filter(
#     signal: np.ndarray,
#     fieldtype: str = "free-frontal",
#     outplot: bool = False,
# ) -> np.ndarray:
#     """
#     Apply outer- and middle-ear filtering to a calibrated signal (ECMA-418-2).

#     Parameters
#     ----------
#     signal : array_like of float, shape (N,) or (N, C)
#         Sound-pressure waveform sampled at 48 kHz.
#     fieldtype : {'free-frontal', 'diffuse'}, default 'free-frontal'
#         Filter type: 'free-frontal' applies all 8 sections; 'diffuse' omits the first two.
#     outplot : bool, default False
#         If True, plot magnitude (dB) and phase (deg) of the overall filter response.

#     Returns
#     -------
#     filtered : ndarray of float, same shape and dtype as signal
#         Filtered output.
#     """

#     # Validation
#     if isinstance(signal, list):
#         signal = np.asarray(signal, dtype=float)
#     elif not isinstance(signal, np.ndarray):
#         raise TypeError("'signal' must be a NumPy array or a list of numbers")
#     if not np.isrealobj(signal):
#         raise ValueError("'signal' must contain real values only")

#     fieldtype = str(fieldtype).lower()
#     if fieldtype not in {"free-frontal", "diffuse"}:
#         raise ValueError("fieldtype must be 'free-frontal' or 'diffuse'")

#     signal_was_1d = False
#     if signal.ndim == 1:
#         signal = signal[:, None]
#         signal_was_1d = True
#     elif signal.ndim != 2:
#         raise ValueError("'signal' must be 1‑D or 2‑D (time × channels)")


#     # ECMA‑418‑2 biquad coefficients
#     b_0k = np.array([
#         1.015896020255593,
#         0.958943219304445,
#         0.961371976333197,
#         2.225803503609735,
#         0.471735128494163,
#         0.115267139824401,
#         0.988029297230954,
#         1.952237687301361,
#     ])
#     b_1k = np.array([
#         -1.925298877776079,
#         -1.806088011849494,
#         -1.763632154338248,
#         -1.434650484792157,
#         -0.366091796830044,
#         0.0,
#         -1.912433802933870,
#         0.162319983017519,
#     ])
#     b_2k = np.array([
#         0.922118060364679,
#         0.876438777856084,
#         0.821787991845146,
#         -0.498204282194628,
#         0.244144703885020,
#         -0.115267139824401,
#         0.926131550180785,
#         -0.667994113035186,
#     ])
#     a_0k = np.ones_like(b_0k)
#     a_1k = np.array([
#         -1.925298877776079,
#         -1.806088011849494,
#         -1.763632154338248,
#         -1.434650484792157,
#         -0.366091796830044,
#         -1.796002566692014,
#         -1.912433802933871,
#         0.162319983017519,
#     ])
#     a_2k = np.array([
#         0.938014080620272,
#         0.835381997160530,
#         0.783159968178343,
#         0.727599221415107,
#         -0.284120167620817,
#         0.805837815618546,
#         0.914160847411739,
#         0.284243574266175,
#     ])

#     if fieldtype == "free-frontal":
#         idx = slice(None)  # all 8 sections
#     else:  # 'diffuse'
#         idx = slice(2, None)  # omit first two (free‑field) stages

#     sos = np.column_stack([b_0k[idx], b_1k[idx], b_2k[idx], a_0k[idx], a_1k[idx], a_2k[idx]])


#     # Filtering (axis 0 = time)
#     signalFiltered = sosfilt(sos, signal, axis=0)

#     if signal_was_1d:
#         signalFiltered = signalFiltered.ravel()

#     # Plotting (optional)
#     if outplot:
#         w, h = sosfreqz(sos, worN=10_000, fs=48_000)
#         mag_db = 20 * np.log10(np.abs(h))
#         phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi

#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
#         ax1.semilogx(w, mag_db)
#         ax1.set(xlabel="Frequency (Hz)", ylabel="|H| (dB)", xlim=(20, 20e3))
#         ax1.set_xticks([31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
#         ax1.grid(True, which="both")
#         ax1.set_title(fieldtype)

#         ax2.semilogx(w, phase_deg)
#         ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (°)", xlim=(20, 20e3))
#         ax2.set_xticks([31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
#         ax2.grid(True, which="both")

#         plt.show()

#     return signalFiltered

# def shm_preproc(
#     signal: np.ndarray,
#     blockSize: int,
#     hopSize: int,
#     padStart: bool = True,
#     padEnd: bool = True
# ) -> np.ndarray:
#     """
#     Pre-process signal with raised-cosine fade-in and zero-padding (ECMA-418-2).

#     Parameters
#     ----------
#     signal : array_like of float, shape (N,) or (N, C)
#         Input time-series samples.
#     block_size : int
#         Segmentation block size in samples.
#     hop_size : int
#         Hop size in samples.
#     pad_start : bool, default True
#         If True, prepend zeros equal to block_size.
#     pad_end : bool, default True
#         If True, append zeros so total length matches block_size + k * hop_size.

#     Returns
#     -------
#     processed : ndarray of float
#         Faded and padded signal, shape (N',) or (N', C).
#     """

#     # Validate
#     sig = np.asarray(signal, dtype=float)
#     if sig.ndim == 1:
#         sig = sig[:, None]

#     if sig.ndim != 2:
#         raise ValueError("`signal` must be 1-D or 2-D (time x channels).")
#     if blockSize <= 0 or hopSize <= 0:
#         raise ValueError("`blockSize` and `hopSize` must be positive integers.")

#     n_ch = sig.shape[1]

#     # Fade-in
#     fadeWeight = 0.5 - 0.5 * np.cos(np.pi * np.arange(240) / 240)
#     fadeWeight = fadeWeight[:, None]
#     signalFade = np.vstack((fadeWeight * sig[:240, :], sig[240:, :]))

#     # Padding
#     n_zeross = blockSize if padStart else 0

#     if padEnd:
#         n_samples = sig.shape[0]
#         n_new = hopSize * (int(np.ceil((n_samples + hopSize + n_zeross)
#                                         / hopSize)) - 1)
#         n_zerose = n_new - n_samples
#     else:
#         n_zerose = 0

#     # Assemble
#     signalOut = np.vstack((
#         np.zeros((n_zeross, n_ch)),
#         signalFade,
#         np.zeros((int(n_zerose), n_ch))
#     ))

#     # Restore
#     if isinstance(signal, (list, np.ndarray)) and np.ndim(signal) == 1:
#         signalOut = signalOut[:, 0]

#     return signalOut

# def shm_resample(signal: np.ndarray, sampleRatein: int) -> tuple[np.ndarray, int]:
#     """
#     Resample a signal to 48 kHz (ECMA-418-2 target rate).

#     Parameters
#     ----------
#     signal : ndarray of float, shape (N,) or (N, C)
#         Input signal at sampleRatein.
#     sampleRatein : int
#         Original sampling rate in Hz.

#     Returns
#     -------
#     resampledSignal : ndarray of float
#         Signal resampled to 48 kHz.
#     resampledRate : int
#         Always 48000.
#     """

#     # Validation
#     if not isinstance(signal, np.ndarray):
#         raise TypeError("signal must be a NumPy ndarray")
#     if signal.dtype.kind not in {"f", "i", "u"}:
#         raise TypeError("signal must contain real numbers")
#     if signal.ndim > 2:
#         raise ValueError("signal must be 1-D or 2-D (time[, channels])")
#     if not isinstance(sampleRatein, int) or sampleRatein <= 0:
#         raise ValueError("sampleRatein must be a positive integer")

#     resampledRate = 48_000

#     if sampleRatein == resampledRate:
#         return signal, resampledRate

#     # Compute Resampling
#     up = resampledRate // np.gcd(resampledRate, sampleRatein)
#     down = sampleRatein // np.gcd(resampledRate, sampleRatein)
#     resampledSignal = resample_poly(signal, up, down, axis=0)

#     return resampledSignal, resampledRate

# def shm_rough_low_pass(
#     specRoughEstTform: np.ndarray,
#     sampleRate: float,
#     riseTime: float,
#     fallTime: float
# ) -> np.ndarray:
#     """
#     Smooth specific roughness estimates with a low-pass IIR filter (ECMA-418-2).

#     Parameters
#     ----------
#     specRoughEstTform : ndarray, shape (T, B)
#         Specific-roughness estimates (time × bands).
#     sampleRate : float
#         Frame rate of the input in Hz.
#     riseTime : float
#         Attack time constant in seconds.
#     fallTime : float
#         Release time constant in seconds.

#     Returns
#     -------
#     spec_roughness : ndarray, shape (T, B)
#         Smoothed specific roughness.
#     """

#     # Validation
#     specRoughEstTform = np.asanyarray(specRoughEstTform, dtype=float)
#     if specRoughEstTform.ndim != 2:
#         raise ValueError("specRoughEstTform must be a 2-D array (time × bands)")
#     if sampleRate <= 0 or riseTime <= 0 or fallTime <= 0:
#         raise ValueError("sampleRate, riseTime and fallTime must be positive")

#     # IIR coefficients
#     riseExponent = np.exp(-1.0 / (sampleRate * riseTime))
#     fallExponent = np.exp(-1.0 / (sampleRate * fallTime))

#     # Filtering
#     specRoughness = np.empty_like(specRoughEstTform)
#     specRoughness[0, :] = specRoughEstTform[0, :]

#     for llBlock in range(1, specRoughEstTform.shape[0]):
#         riseMask = specRoughEstTform[llBlock, :] >= specRoughness[llBlock - 1, :]
#         fallMask = ~riseMask

#         if np.any(riseMask):
#             specRoughness[llBlock, riseMask] = (
#                 specRoughEstTform[llBlock, riseMask] * (1.0 - riseExponent)
#                 + specRoughness[llBlock - 1, riseMask] * riseExponent
#             )

#         if np.any(fallMask):
#             specRoughness[llBlock, fallMask] = (
#                 specRoughEstTform[llBlock, fallMask] * (1.0 - fallExponent)
#                 + specRoughness[llBlock - 1, fallMask] * fallExponent
#             )

#     return specRoughness

# def shm_rough_weight(
#     modRate: np.ndarray,
#     modfreqMaxWeight: np.ndarray,
#     roughWeightParams: np.ndarray
# ) -> np.ndarray:
#     """
#     Compute roughness weighting factors (ECMA-418-2, Eq. 85).

#     Parameters
#     ----------
#     modRate : array_like of float
#         Modulation rates in Hz.
#     modfreqMaxWeight : array_like of float
#         Modulation rate at which the weight peaks (value=1).
#     roughWeightParams : array_like of float, shape (2, ...)
#         Weighting parameters: [alpha (gain), beta (sharpness)].

#     Returns
#     -------
#     roughWeight : ndarray
#         Weighting factors, same shape as broadcasted inputs.
#     """

#     modRate = np.asarray(modRate, dtype=float)
#     modfreqMaxWeight = np.asarray(modfreqMaxWeight, dtype=float)
#     roughWeightParams = np.asarray(roughWeightParams, dtype=float)

#     roughWeight = 1.0 /(1.0 + ((((modRate / modfreqMaxWeight) - (modfreqMaxWeight / modRate)) * roughWeightParams[0]) ** 2)) ** roughWeightParams[1]
#     return roughWeight

# def shm_signal_segment(
#     signal: np.ndarray,
#     axisn: int = 0,
#     blockSize: int = 1024,
#     overlap: float = 0.0,
#     i_start: int = 0,
#     endShrink: bool = False,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Segment a signal into overlapping blocks (ECMA-418-2).

#     Parameters
#     ----------
#     signal : array_like, shape (N,) or (N, C)
#         Input samples and optional channels.
#     axisn : {0, 1}, default 0
#         Axis along which to segment. Axis 1 input is transposed internally.
#     blockSize : int, default 1024
#         Number of samples per block.
#     overlap : float, default 0.0
#         Fractional overlap between successive blocks (0 ≤ overlap < 1).
#     i_start : int, default 0
#         Starting index for the first block.
#     endShrink : bool, default False
#         If True, include a final block capturing the signal tail.

#     Returns
#     -------
#     signal_segmented : ndarray
#         Segmented signal, shape (blockSize, n_blocks, n_channels) or transposed if axisn==1.
#     i_blocks_out : ndarray of int
#         Starting indices of each block relative to i_start.
#     """

#     # Validation
#     if not isinstance(signal, np.ndarray):
#         signal = np.asanyarray(signal)

#     if signal.ndim == 1:
#         signal = signal[:, None]

#     if axisn not in (0, 1):
#         raise ValueError("axisn must be 0 or 1")

#     if not (0 <= overlap < 1):
#         raise ValueError("overlap must satisfy 0 ≤ overlap < 1")

#     if blockSize <= 0:
#         raise ValueError("blockSize must be positive")

#     # Re-orient
#     axisFlip = False
#     if axisn == 1:
#         signal = signal.T
#         axisFlip = True

#     n_total, nchans = signal.shape

#     if i_start < 0:
#         raise ValueError("i_start falls outside the signal")

#     # Truncate
#     hopSize = round(blockSize * (1.0 - overlap))
#     if hopSize <= 0:
#         raise ValueError("overlap too large: hop size becomes zero")

#     signalTrunc = signal[i_start:]
#     if signalTrunc.shape[0] <= blockSize:
#         raise ValueError("Signal is too short for the requested blockSize")

#     n_blocks = int(np.floor((signalTrunc.shape[0] - overlap * blockSize) / hopSize))
#     i_end = (n_blocks * hopSize + int(overlap * blockSize))-1
#     signalTrunc = signalTrunc[:i_end+1]

#     signalSegmented_list = []

#     for chan in range(nchans, 0, -1):
        
#         signalSegmentedChan = np.concatenate((np.zeros((hopSize, 3)),
#                                               np.reshape(signalTrunc, (hopSize, -1))), axis=1)

#         # Apply circular shifts and stack like MATLAB circshift
#         signalSegmentedChan = np.concatenate(
#             [
#                 np.roll(signalSegmentedChan, 3, axis=1),
#                 np.roll(signalSegmentedChan, 2, axis=1),
#                 np.roll(signalSegmentedChan, 1, axis=1),
#                 np.roll(signalSegmentedChan, 0, axis=1),
#             ],
#             axis=0,
#         )
#         signalSegmentedChan = signalSegmentedChan[:, 6:]

#         # Optionally include block of end data
#         if endShrink and signal[i_start:].shape[0] > signalTrunc.shape[0]:
#             tail_start = signal.shape[0] - blockSize
#             tail_block = signal[tail_start:, chan][:, np.newaxis]
#             signalSegmentedChanOut = np.concatenate((signalSegmentedChan, tail_block), axis=1)
#             iBlocksOut = np.concatenate(
#                 (np.arange(0, n_blocks * hopSize, hopSize), [tail_start])
#             )
#         else:
#             signalSegmentedChanOut = signalSegmentedChan
#             iBlocksOut = np.arange(0, n_blocks * hopSize, hopSize)

#         signalSegmented_list.append(signalSegmentedChanOut)

#     # Stack into 3D array: (blockSize, n_blocks, nchans)
#     signalSegmented = np.stack(signalSegmented_list, axis=-1)

#     # Re-orient segmented signal to match input
#     if axisFlip:
#         signalSegmented = np.transpose(signalSegmented, (1, 0, 2))

#     return signalSegmented, iBlocksOut

# ------------------
#### VALIDATION ####
# ------------------

if __name__ == "__main__":
    print("utilities.py")