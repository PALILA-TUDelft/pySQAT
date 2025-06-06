from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.interpolate import interp1d 
from scipy.fft import fft, ifft
from scipy.signal.windows import hann, blackman
from matplotlib import pyplot as plt

from sound_metrics import *
from utilities import *

from metrics_loudness import Loudness_ISO532_1

__all__ = ["Tonality_Aures1985"]
FloatArray = NDArray[np.floating]


if __name__ == "__main__":

    fs = 44156              # Sampling rate in Hz
    duration = 4.0          # Duration in seconds
    f0 = 1050               # Frequency of pure tone (Hz)
    Lp = 69                 # Desired sound pressure level (dB SPL)
    pref = 20e-6            # Reference pressure in Pa

    # Create time vector
    t = np.arange(0, duration, 1/fs)

    # Generate sine wave with RMS level corresponding to 60 dB SPL
    rms_target = pref * 10**(Lp / 20)
    amp = rms_target * np.sqrt(2)
    signal = amp * np.sin(2 * np.pi * f0 * t)

    # Compute tonality using Aures 1985 model
    result = Tonality_Aures1985(signal, fs=fs, LoudnessField=0, time_skip=0.123, show=True)

    # Print statistics
    print("Tonality Statistics:")
    print(f"InstantaneousTonality: {np.mean(result['InstantaneousTonality']):.3g}")
    print(f"LoudnessWeighting: {np.mean(result['LoudnessWeighting']):.3g}")
    print(f"TonalWeighting: {np.mean(result['TonalWeighting']):.3g}")