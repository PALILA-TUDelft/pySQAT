"""
validate_tonality_ecma418_2.py
==============================

Validation script for the Python port of `tonality_ecma418_2`.

Reproduces the MATLAB “Tonality_ECMA418_2_software_comparison”:
 1. Load binaural WAV excerpt “TrainStation7-0100-0130.wav”.
 2. Compute ECMA‑418‑2 tonality metrics via the Python port.
 3. Load reference results from ./reference_results/*.asc.
 4. Plot comparisons (time‑series, average specific, single values, spectrograms).

Usage:
    cd VAL_tonality_ecma418_2
    python validate_tonality_ecma418_2.py

"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import tonality.py
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# import the Python implementation from tonality.py
from tonality import tonality_ecma418_2

# Toggle figure saving
SAVE_FIGS = False

# Determine paths
dir_script = script_dir
ndir_ref    = dir_script / 'reference_results'

# WAV file must be located under project_root/sound_files
wav_path = project_root / 'sound_files' / 'ExStereo_TrainStation7-0100-0130.wav'

# Load signal
def load_wav(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"WAV not found: {path}")
    sig, fs = sf.read(str(path))
    return sig, fs

insig, fs = load_wav(wav_path)

# Analyze with Python port
fieldtype = 'free-frontal'
time_skip = 304e-3
show = False
OUT = tonality_ecma418_2(insig, fs, fieldtype, time_skip, show)

def load_asc(fname):
    with open(ndir_ref / fname, 'r') as file:
        lines = file.readlines()

    data = []
    max_cols = 0

    # First pass: read numeric values and determine max number of columns
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            numeric_parts = [float(part) for part in parts if is_float(part)]
            data.append(numeric_parts)
            if len(numeric_parts) > max_cols:
                max_cols = len(numeric_parts)

    # Pad shorter rows with NaN
    padded_data = [row + [np.nan] * (max_cols - len(row)) for row in data]

    return np.array(padded_data)

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

ref = {}
base = 'TrainStation7-0100-0130'
ref['AvgSpec'] = load_asc(f"{base}.Specific Tonality (Hearing Model).asc")
ref['TDep']    = load_asc(f"{base}.Tonality (Hearing Model) vs. Time.asc")
ref['Spec1']   = load_asc(f"{base}.Specific Tonality (Hearing Model) vs. Time_channel_1.asc")
ref['Spec2']   = load_asc(f"{base}.Specific Tonality (Hearing Model) vs. Time_channel_2.asc")
ref['single']  = np.array([0.660, 0.314])

# 1) Time‑dependent overall tonality
for ch in [0, 1]:
    t_ref = ref['TDep'][:, 0]
    y_ref = ref['TDep'][:, ch+1]
    t_py  = OUT['timeOut']
    y_py  = OUT['tonalityTDep'][:, ch]

    plt.figure()
    plt.plot(t_ref, y_ref, label='Reference')
    plt.plot(t_py,  y_py, '--', label='Implementation')
    plt.xlabel('Time (s)')
    plt.ylabel('Tonality (tu_HMS)')
    plt.title(f'Channel {ch+1} – Time-dependent tonality')
    plt.legend()
    if SAVE_FIGS:
        plt.savefig(dir_script / f'ch{ch+1}_tDep.png')

# 2) Average specific tonality per Bark band
bark = np.linspace(0.5, 26.5, ref['AvgSpec'].shape[0])
for ch in [0, 1]:
    y_ref = ref['AvgSpec'][:, ch+1]
    y_py  = OUT['specTonalityAvg'][:, ch]

    plt.figure()
    width = 0.4
    plt.bar(bark - width/2, y_ref, width, alpha=0.6, label='Reference')
    plt.bar(bark + width/2, y_py,  width, alpha=0.6, label='Implementation')
    plt.xlabel('Half-critical band rate (Bark_HMS)')
    plt.ylabel('Specific tonality (tu_HMS/Bark_HMS)')
    plt.title(f'Channel {ch+1} – Avg specific tonality')
    plt.legend()
    if SAVE_FIGS:
        plt.savefig(dir_script / f'ch{ch+1}_avgSpec.png')

# 3) Single-value overall tonality
labels = ['Ch1', 'Ch2']
x = np.arange(2)
fig, ax = plt.subplots()
ax.bar(x - 0.2, ref['single'], 0.4, label='Reference')
ax.bar(x + 0.2, OUT['tonalityAvg'], 0.4, label='Implementation')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Tonality (tu_HMS)')
ax.set_title('Time-averaged overall tonality')
ax.legend()
if SAVE_FIGS:
    fig.savefig(dir_script / 'single_values.png')

# 4) Specific‑tonality spectrograms
def plot_spec(mat, times, freqs, title, fname_suffix):
    plt.figure()
    plt.pcolormesh(times, freqs, mat.T, shading='auto')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.colorbar(label='Specific tonality (tu_HMS/Bark_HMS)')
    if SAVE_FIGS:
        plt.savefig(dir_script / fname_suffix)

# Reference spectrograms
# note: first column is time, first row is freq
times = ref['Spec1'][1:, 0]
freqs = ref['Spec1'][0, 1:]
z1    = ref['Spec1'][1:, 1:]
plot_spec(z1, times, freqs, 'Ref – Ch1 specific tonality', 'ref_ch1_spec.png')

times = ref['Spec2'][1:, 0]
freqs = ref['Spec2'][0, 1:]
z2    = ref['Spec2'][1:, 1:]
plot_spec(z2, times, freqs, 'Ref – Ch2 specific tonality', 'ref_ch2_spec.png')

# Implementation spectrograms
times = OUT['timeOut']
freqs = OUT['bandCentreFreqs']
plot_spec(OUT['specTonality'][:, :, 0], times, freqs,
          'Impl – Ch1 specific tonality', 'impl_ch1_spec.png')
plot_spec(OUT['specTonality'][:, :, 1], times, freqs,
          'Impl – Ch2 specific tonality', 'impl_ch2_spec.png')

plt.show()