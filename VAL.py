import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loudness_tools import Loudness_ISO532_1

# --- Common signal settings ---
fs = 48000
duration = 1.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
amp = 0.02
signal = amp * np.sin(2 * np.pi * 1000 * t)
field = 0
time_skip = 0.2
show = False

# ========== Method 2 ==========
print("=== METHOD 2 ===")
m2_loud_matlab = pd.read_csv("out_m2_loudness.csv", header=None).values
m2_level_matlab = pd.read_csv("out_m2_loudness_level.csv", header=None).values

OUT2 = Loudness_ISO532_1(signal, fs, field, 2, time_skip, show)
N2 = min(len(OUT2['time']), len(m2_loud_matlab))
time2 = OUT2['time'][:N2]
loud_py2 = OUT2['InstantaneousLoudness'][:N2]
loud_mat2 = m2_loud_matlab[:N2, 1]
level_py2 = OUT2['InstantaneousLoudnessLevel'][:N2]
level_mat2 = m2_level_matlab[:N2, 1]

mae_loud2 = np.mean(np.abs(loud_py2 - loud_mat2))
mae_level2 = np.mean(np.abs(level_py2 - level_mat2))

# Plot: Loudness
plt.figure()
plt.plot(time2, loud_py2, label="Python")
plt.plot(time2, loud_mat2, '--', label="MATLAB")
plt.title("Method 2: Instantaneous Loudness")
plt.xlabel("Time [s]")
plt.ylabel("Loudness [sone]")
plt.legend()
plt.grid(True)

# Plot: Loudness Difference
plt.figure()
plt.plot(time2, loud_py2 - loud_mat2)
plt.title("Method 2: Loudness Difference (Python - MATLAB)")
plt.xlabel("Time [s]")
plt.ylabel("Δ Loudness [sone]")
plt.grid(True)

# Plot: Loudness Level
plt.figure()
plt.plot(time2, level_py2, label="Python")
plt.plot(time2, level_mat2, '--', label="MATLAB")
plt.title("Method 2: Instantaneous Loudness Level")
plt.xlabel("Time [s]")
plt.ylabel("Loudness Level [phon]")
plt.legend()
plt.grid(True)

# Plot: Loudness Level Difference
plt.figure()
plt.plot(time2, level_py2 - level_mat2)
plt.title("Method 2: Loudness Level Difference (Python - MATLAB)")
plt.xlabel("Time [s]")
plt.ylabel("Δ Level [phon]")
plt.grid(True)

# ========== Method 1 ==========
print("=== METHOD 1 ===")
m1 = pd.read_csv("out_m1_stationary.csv", header=None).values
loud_mat1, level_mat1 = m1[0, 0], m1[0, 1]

OUT1 = Loudness_ISO532_1(signal, fs, field, 1, time_skip, show)
loud_py1 = OUT1['Loudness']
level_py1 = OUT1['LoudnessLevel']
mae_loud1 = abs(loud_py1 - loud_mat1)
mae_level1 = abs(level_py1 - level_mat1)

# Plot: Stationary Loudness Comparison
plt.figure()
plt.bar(['Python', 'MATLAB'], [loud_py1, loud_mat1], color=['tab:blue', 'tab:orange'])
plt.title("Method 1: Loudness (Stationary)")
plt.ylabel("Loudness [sone]")

# Plot: Stationary Loudness Level Comparison
plt.figure()
plt.bar(['Python', 'MATLAB'], [level_py1, level_mat1], color=['tab:blue', 'tab:orange'])
plt.title("Method 1: Loudness Level (Stationary)")
plt.ylabel("Loudness Level [phon]")

# ========== Method 0 ==========
print("=== METHOD 0 ===")
m0 = pd.read_csv("out_m0_spl.csv", header=None).values
loud_mat0, level_mat0 = m0[0, 0], m0[0, 1]

spl = np.zeros(28)
spl[15] = 40  # Simulate 1 kHz tone
OUT0 = Loudness_ISO532_1(spl, 1, field, 0, 0, show)
loud_py0 = OUT0['Loudness']
level_py0 = OUT0['LoudnessLevel']
mae_loud0 = abs(loud_py0 - loud_mat0)
mae_level0 = abs(level_py0 - level_mat0)

# Plot: SPL-Based Loudness Comparison
plt.figure()
plt.bar(['Python', 'MATLAB'], [loud_py0, loud_mat0], color=['tab:blue', 'tab:orange'])
plt.title("Method 0: Loudness (SPL Input)")
plt.ylabel("Loudness [sone]")

# Plot: SPL-Based Loudness Level Comparison
plt.figure()
plt.bar(['Python', 'MATLAB'], [level_py0, level_mat0], color=['tab:blue', 'tab:orange'])
plt.title("Method 0: Loudness Level (SPL Input)")
plt.ylabel("Loudness Level [phon]")

# ========== Summary ==========
print("\n=== Mean Absolute Errors ===")
print(f"Method 0 - Loudness MAE: {mae_loud0:.4f}, Level MAE: {mae_level0:.4f}")
print(f"Method 1 - Loudness MAE: {mae_loud1:.4f}, Level MAE: {mae_level1:.4f}")
print(f"Method 2 - Loudness MAE: {mae_loud2:.4f}, Level MAE: {mae_level2:.4f}")

plt.show()