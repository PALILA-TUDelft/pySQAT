import soundfile as sf
from sound_metrics import do_slm, get_leq, do_ob13_iso532_1

# Load audio file
file_path = 'sound_files\ExSignal_A320_auralized_departure_104dBFS.wav'
signal, fs = sf.read(file_path)

# Ensure mono
if signal.ndim > 1:
    signal = signal.mean(axis=1)

# Step 1: Apply A-weighting and fast time-weighting
levels_dB, dBFS = do_slm(signal, fs, weight_freq='A', weight_time='f', dBFS=94)

# Step 2: Compute equivalent continuous sound level (Leq) over entire file
leq_total = get_leq(levels_dB)

# Step 3: Filter into one-third octave bands
filtered_bands, center_freqs = do_ob13_iso532_1(signal, fs)

# Optional: Compute band levels
import numpy as np
band_levels = 20 * np.log10(np.sqrt(np.mean(filtered_bands**2, axis=0))) + dBFS

# Display results
print(f"Leq (A-weighted, fast): {leq_total[0]:.2f} dB(A)")
print("Band levels (dB SPL):")
for fc, lvl in zip(center_freqs, band_levels):
    print(f"{fc:.1f} Hz: {lvl:.2f} dB")