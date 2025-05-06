import numpy as np
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio, display
import os

def see(file_path):
    """
    Loads a WAV file, displays its waveform and spectrogram (with min/max frequency range), 
    and plays the audio.

    Parameters:
        file_path (str): Path to the WAV file.

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

    # Play audio
    print("Playing audio:")
    display(Audio(data, rate=sample_rate))

def hz2bark(f):
    """
    Converts a frequency f in Hz to a frequency z in Barks.

    ORIGINAL MATLAB CODE:
    hz2bark_local.m   (Osses, 2014) [Last Update: 15/08/2014]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 6/05/2025

    Parameters:
    -----------
    f : float or array-like
        Frequency in Hz.

    Returns:
    --------
    z : float or ndarray
        Frequency in Barks.
    """
    f = np.asarray(f)  # Support scalar or array input
    z = 13 * np.arctan(0.76 * (f / 1000)) + 3.5 * np.arctan((f / (1000 * 7.5))**2)
    return z

def bark2hz_local(z):
    """
    Converts Bark values to Hertz using interpolation.

    ORIGINAL MATLAB CODE:
    hz2bark_local.m   (Osses, 2014) [Last Update: 15/08/2014]

    PYTHON IMPLEMENTATION:
    Gerard Mendoza Ferrandis - 6/05/2025

    Parameters:
    -----------
    z : float or array-like
        Bark scale values to be converted to Hz.

    Returns:
    --------
    f_interp : float or ndarray
        Frequency in Hz corresponding to the given Bark values.
    """
    f0 = 1000
    k = np.arange(-20, 13)  # Equivalent to MATLAB -20:12
    f = f0 * 2 ** (k / 3)

    zt = hz2bark(f)  # Use your defined function

    interpolator = interp1d(zt, f, kind='linear', fill_value="extrapolate")
    f_interp = interpolator(z)

    return f_interp

if __name__ == "__main__":
    #see("sound_files\ExStereo_TrainStation7-0100-0130.wav")
    z = hz2bark(1000)
    print(z)
    f = bark2hz_local(z)
    print(f)