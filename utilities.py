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

if __name__ == "__main__":
    see("sound_files\ExStereo_TrainStation7-0100-0130.wav")