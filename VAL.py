import numpy as np
from loudness_tools import Loudness_ISO532_1

# --- Generate 1 kHz sine wave at 40 dB SPL ---
fs = 48000  # Sampling rate in Hz
duration = 1.0  # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

amplitude = 0.02  # 40 dB SPL = 20 µPa * 10^(40/20) = 0.02 Pa
signal = amplitude * np.sin(2 * np.pi * 1000 * t)

# --- Run loudness calculation ---
field = 0  # free field
method = 2  # time-varying from audio signal
time_skip = 0.2  # skip transient
show = True  # show plots

OUT = Loudness_ISO532_1(signal, fs, field, method, time_skip, show)

# --- Display key results ---
print("Mean Loudness (sone):", OUT['Nmean'])
print("Max Loudness Level (phon):", np.max(OUT['InstantaneousLoudnessLevel']))