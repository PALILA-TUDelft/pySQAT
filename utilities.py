
##########################
#### UTILITIES MODULE ####
##########################

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio, display
import os
import warnings

# ----------------------
#### MAIN FUNCTIONS ####
# ----------------------

def see(file_path):
    """
    Loads a WAV file, displays its waveform and spectrogram (with min/max frequency range), 
    and plays the audio.

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 07/05/2025

    Parameters:
¡       file_path (str): Path to the WAV file.

    Returns:
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

def hz2bark(f):
    """
    Converts a frequency f in Hz to a frequency z in Barks.

    ORIGINAL MATLAB CODE:
    hz2bark_local.m   (Osses, 2014) [Last Update: 15/08/2014]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 06/05/2025

    Parameters:
    f (float or array-like): Frequency in Hz.

    Returns:
    z (float or ndarray): Frequency in Barks.
    """
    f = np.asarray(f)  # Support scalar or array input
    z = 13 * np.arctan(0.76 * (f / 1000)) + 3.5 * np.arctan((f / (1000 * 7.5))**2)
    return z

def bark2hz(z):
    """
    Converts Bark values to Hertz using interpolation.

    ORIGINAL MATLAB CODE:
    bark2hz_local.m   (Osses, 2014) [Last Update: 25/09/2014]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 06/05/2025

    Parameters:
        z (float or array-like): Bark scale values to be converted to Hz.

    Returns:
        f_interp (float or ndarray): Frequency in Hz corresponding to the given Bark values.
    """
    f0 = 1000
    k = np.arange(-20, 13)  # Equivalent to MATLAB -20:12
    f = f0 * 2 ** (k / 3)

    zt = hz2bark(f)  # Use your defined function

    interpolator = interp1d(zt, f, kind='linear', fill_value="extrapolate")
    f_interp = interpolator(z)

    return f_interp

def phon2sone(phon):
    """
    Converts loudness from phon to sone according to ISO 532-1:2017.

    ORIGINAL MATLAB CODE:
    phon2sone_local.m   (Osses, 2014) [Last Update: 25/09/2014] (?)

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 06/05/2025

    Parameters:
        phon (np.ndarray): Loudness level in phons. Can be a list or array-like.
    
    Returns:
        sone (np.ndarray): Loudness in sones, as a column vector.
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

def sone2phon(sone):
    """
    Converts loudness from sone to phon according to ISO 532-1:2017.
    
    ORIGINAL MATLAB CODE:
    sone2phon_local.m   (Osses, 2014) [Last Update: 25/09/2014] (?)

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 06/05/2025

    Parameters:
        sone (array-like): Loudness in sone (1D array).
        
    Returns:
        phon (np.ndarray): Loudness level in phon.
    """
    sone = np.atleast_1d(sone).astype(float)
    sone = sone.reshape(-1, 1) if sone.ndim == 1 else sone

    phon = np.zeros_like(sone)

    idx = sone >= 1
    phon[idx] = 40 + 33.22 * np.log10(sone[idx])
    phon[~idx] = 40 * np.power(sone[~idx] + 0.0005, 0.35)

    return phon

def get_exceeded_value(x, percent):
    """
    Value that is exceeded *percent* % of the time for each channel in x.

    ORIGINAL MATLAB CODE:
    get_exceeded_value.m [Last Update: 24/04/2025]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 07/05/2025

    Parameters
    ----------
    x (array-like): Data to be analyzed (1-D or 2-D array).
    percent (int or float): Percentage of exceedance (1–99).

    Returns
    -------
    values (ndarray): 1-D array of values that are exceeded *percent* % of the time.
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

def get_statistics(data, metric):
    """
    Compute a suite of statistics for *data* and return them in a dict whose
    keys mirror the MATLAB field-names (e.g. 'Nmax', 'N1', …).

    ORIGINAL MATLAB CODE:
    get_exceeded_value.m [Last Update: 19/02/2025]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 07/05/2025

    Parameters
    ----------
    data (array-like): Data to be analyzed (1-D or 2-D array).
    metric (str): One of the verbose metric labels, e.g. 'Loudness_ISO532_1'.

    Returns
    -------
    out (dict): Keys are <prefix><statLabel>; values are 1-D NumPy arrays (length C).
    For a single-channel input the arrays have length 1, so you can safely cast them to float with `.item()`.
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