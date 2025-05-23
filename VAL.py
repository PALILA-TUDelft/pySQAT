import numpy as np
import matplotlib.pyplot as plt
from psychoacoustic_metrics import Tonality_Aures1985

# Define parameters
fs = 44100              # Sampling rate in Hz
duration = 100          # Duration in seconds
f0 = 1000               # Frequency of pure tone (Hz)
Lp = 60                 # Desired sound pressure level (dB SPL)
pref = 20e-6            # Reference pressure in Pa

# Create time vector
t = np.arange(0, duration, 1/fs)

# Generate sine wave with RMS level corresponding to 60 dB SPL
rms_target = pref * 10**(Lp / 20)
amp = rms_target * np.sqrt(2)
signal = amp * np.sin(2 * np.pi * f0 * t)

# Compute tonality using Aures 1985 model
result = Tonality_Aures1985(signal, fs=fs, LoudnessField=0, time_skip=0.5, show=True)

print("Mean Tonal Weighting (w_tonal):", np.mean(result['TonalWeighting']))
print("Mean Loudness Weighting (w_gr):", np.mean(result['LoudnessWeighting']))
print("Mean Tonality:", np.mean(result['InstantaneousTonality']))

# Print statistics
print("Tonality Statistics:")
for k, v in result.items():
    if isinstance(v, float) or isinstance(v, np.ndarray) and np.ndim(v) == 0:
        print(f"{k}: {v:.4f}")
