from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample_poly, lfilter, sosfilt, iirfilter, butter
from scipy.interpolate import interp1d 
from scipy.fft import fft, ifft
from scipy.signal.windows import hann, blackman
from matplotlib import pyplot as plt
import warnings, sys, os, inspect
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from SQAT4PY.sound_metrics import *
from utilities import *
from metrics_loudness import Loudness_ISO532_1

__all__ = ["Sharpness_DIN45692"]
FloatArray = NDArray[np.floating]
# --- Sharpness_DIN45692 function (updated) ---

def Sharpness_DIN45692(insig=None, fs=None, weight_type=None, LoudnessField=None, LoudnessMethod=None, time_skip=None, show_sharpness=None, show_loudness=None, SpecificLoudness_input=None, time_input=None):
    """
    Stationary and time-varying sharpness calculation according to DIN 45692 (2009).
    Includes loudness calculation as pre-processing or accepts pre-calculated specific loudness.

    INPUT ARGUMENTS:
    insig : [Nx1] array, calibrated audio signal (Pa), 1 channel only. Required if SpecificLoudness_input is None.
    fs : integer, sampling frequency (Hz). Required if insig is provided.
    weight_type : string, sharpness calculation using weighting function ('DIN45692', 'bismarck', 'aures').
    LoudnessField : integer, type of field for loudness (0=free field, 1=diffuse field). Required if insig is provided.
    LoudnessMethod : integer, method for loudness (1=stationary, 2=time varying). Required if insig is provided.
    time_skip : integer, skip start of signal in seconds for statistics (time-varying only).
    show_loudness : logical(boolean), display loudness results (default: False).
    show_sharpness : logical(boolean), display sharpness results (default: False).
    SpecificLoudness_input : [MxN] array, optional. Direct input of specific loudness (sones/bark).
                             If provided, insig, fs, LoudnessField, LoudnessMethod are ignored.
                             M is number of time steps, N is number of bark bands.
    time_input : [Mx1] array, optional. Time vector corresponding to SpecificLoudness_input.
                 Required if SpecificLoudness_input is time-varying.

    OUTPUTS:
    OUT : dict containing sharpness results and statistics.
    """
    
    # Handle case when no primary arguments are provided
    if insig is None and SpecificLoudness_input is None:
        help(Sharpness_DIN45692)
        return None # Return None to prevent further errors
    
    # Handle default arguments for show_loudness and show_sharpness
    if show_loudness is None:
        show_loudness = False
    
    if show_sharpness is None:
        show_sharpness = False

    L = None # Initialize L to None, will be populated later

    # Determine if SpecificLoudness is provided directly or needs to be calculated
    if SpecificLoudness_input is not None:
        # Path for direct SpecificLoudness input
        SpecificLoudness = SpecificLoudness_input
        
        # Ensure SpecificLoudness is 2D (time_steps x bark_bands)
        if SpecificLoudness.ndim == 1:
            SpecificLoudness = SpecificLoudness.reshape(1, -1) # Make it (1, N) for stationary
        
        n = SpecificLoudness.shape[1] # Number of bark bands
        
        # Determine if stationary or time-varying based on number of time steps
        if SpecificLoudness.shape[0] == 1: # Stationary
            LoudnessMethod_internal = 1
            # Create a dummy L dict for consistency with the original flow
            L = {'SpecificLoudness': SpecificLoudness, 'time': np.array([0])}
        else: # Time-varying
            LoudnessMethod_internal = 2
            if time_input is None:
                raise ValueError("time_input must be provided when SpecificLoudness_input is time-varying.")
            if len(time_input) != SpecificLoudness.shape[0]:
                raise ValueError("Length of time_input must match the number of time steps in SpecificLoudness_input.")
            # Create a dummy L dict for consistency with the original flow
            L = {'InstantaneousSpecificLoudness': SpecificLoudness, 'time': time_input}
        
        # Calculate loudness_sones from the provided SpecificLoudness
        loudness_sones = np.zeros((SpecificLoudness.shape[0], 1))
        for i in range(SpecificLoudness.shape[0]):
            loudness_sones[i] = np.sum(SpecificLoudness[i, :], axis=0) * 0.10
            
    else:
        # Original path: calculate loudness from insig
        # Call Loudness_ISO532_1 with correct parameter names
        L = Loudness_ISO532_1(insig, fs=fs, field=LoudnessField, method=LoudnessMethod, time_skip=time_skip, show=show_loudness)
        
        if LoudnessMethod == 1: # stationary loudness calculation
            n = L['SpecificLoudness'].shape[1]
            loudness_sones = np.zeros((L['SpecificLoudness'].shape[0], 1)) # pre allocate memory
            SpecificLoudness = L['SpecificLoudness']
            LoudnessMethod_internal = 1
            
        elif LoudnessMethod == 2: # time-varying loudness calculation
            n = L['InstantaneousSpecificLoudness'].shape[1]
            loudness_sones = np.zeros((L['InstantaneousSpecificLoudness'].shape[0], 1)) # pre allocate memory
            SpecificLoudness = L['InstantaneousSpecificLoudness']
            LoudnessMethod_internal = 2
            
        else:
            # This case means insig was provided, but LoudnessMethod is not 1 or 2.
            raise ValueError("LoudnessMethod must be 1 or 2 when insig is provided.")
    
    z = np.linspace(0.1, 24, n)  # create bark axis
    
    ## Sharpness calculation ##
    s = np.zeros(SpecificLoudness.shape[0]) # Initialize s array
    
    if weight_type == 'DIN45692':  # Widmann model
        
        g = il_sharpWeights(z, 'standard', None)  # calculate sharpness weighting factors
        k = 0.11  # adjusted to yield 1 acum using SQAT - DIN45692 allows 0.105<=k<=0.0115 for this weighting function
        
        for i in range(SpecificLoudness.shape[0]):
            # Ensure loudness_sones[i] is not zero to avoid division by zero
            if loudness_sones[i] == 0:
                s[i] = 0
            else:
                s[i] = k * np.sum(SpecificLoudness[i, :] * g * z * 0.10, axis=0) / loudness_sones[i]
        
    elif weight_type == 'aures':  # Aures model
        
        g = np.zeros((SpecificLoudness.shape[0], len(z))) # g can be time-varying for 'aures'
        for i in range(SpecificLoudness.shape[0]):
            # il_sharpWeights for 'aures' expects a scalar N, so pass loudness_sones[i,0]
            if loudness_sones[i,0] == 0:
                g[i, :] = np.zeros_like(z)
                s[i] = 0
            else:
                g[i, :] = il_sharpWeights(z, 'aures', loudness_sones[i,0]) 
                s[i] = 0.11 * np.sum(SpecificLoudness[i, :] * g[i, :] * z * 0.10, axis=0) / loudness_sones[i]
        
    elif weight_type == 'bismarck':  # von Bismarck
        g = il_sharpWeights(z, 'bismarck', None)  # calculate sharpness weighting factor
        
        for i in range(SpecificLoudness.shape[0]):
            # Ensure loudness_sones[i] is not zero to avoid division by zero
            if loudness_sones[i] == 0:
                s[i] = 0
            else:
                s[i] = 0.11 * np.sum(SpecificLoudness[i, :] * g * z * 0.10, axis=0) / loudness_sones[i]
    else:
        raise ValueError("Invalid weight_type. Choose 'DIN45692', 'bismarck', or 'aures'.")
    
    # Output dictionary
    OUT = {}
    
    if LoudnessMethod_internal == 2:  # (time-varying sharpness)
        
        OUT['InstantaneousSharpness'] = s  # instantaneous sharpness
        OUT['time'] = L['time']           # time vector
        OUT['loudness'] = L               # output dict from the loudness calculation
        
        # get statistics from Time-varying sharpness (acum)
        if time_skip is not None and time_skip > 0:
            if OUT['time'].size > 0: # Ensure time vector is not empty
                idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector
                data_for_stats = s[idx:]
            else:
                data_for_stats = s # Fallback if time vector is empty
        else:
            data_for_stats = s # Use all data if no skip or skip is 0
        
        metric_statistics = 'Sharpness_DIN45692'
        OUT_statistics = get_statistics(data_for_stats, metric_statistics)  # get statistics
        
        # copy fields of <OUT_statistics> dict into the <OUT> dict
        fields_OUT_statistics = list(OUT_statistics.keys())  # Get all field names in OUT_statistics
        
        for fieldName in fields_OUT_statistics:
            if fieldName not in OUT: # Only copy if OUT does NOT already have this field
                OUT[fieldName] = OUT_statistics[fieldName]
        
        # Show plots (time-varying)
        if show_sharpness == True:
            plt.figure()
            plt.gcf().canvas.manager.set_window_title('Sharpness analysis (time-varying)')
            
            # Ensure S5 is a scalar for plotting label
            s5_value = OUT['S5'][0] if isinstance(OUT['S5'], np.ndarray) else OUT['S5']
            
            plt.plot(OUT['time'], s5_value * np.ones(OUT['time'].shape), 'r--', label=f'$S_5$={s5_value:.3f}')
            plt.plot(OUT['time'], s)
            
            plt.xlabel('Time, $t$ (s)', fontsize=12)
            plt.ylabel('Sharpness, $S$ (acum)', fontsize=12)
            
            plt.legend(loc='best', fontsize=10)
            plt.gca().legend().set_frame_on(False)
            
            plt.gcf().patch.set_facecolor('white')
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()
            plt.show()
    
    elif LoudnessMethod_internal == 1:  # (stationary sharpness)
        # For stationary, s will be a 1-element array, so return the scalar value
        OUT['Sharpness'] = s[0] if s.size == 1 else s 
    
    return OUT

def il_sharpWeights(z, type, N):
    """
    Computes sharpness weighting functions according to required model type.
    """
    
    g = np.zeros(len(z))
    
    if type == 'standard':  # Widmann model according to DIN 45692 (2009)
        g[z < 15.8] = 1
        mask = z >= 15.8
        g[mask] = 0.15 * np.exp(0.42 * (z[mask] - 15.8)) + 0.85
    
    elif type == 'bismarck':  # von bismark's model according to DIN 45692 (2009)
        g[z < 15] = 1
        mask = z >= 15
        g[mask] = 0.2 * np.exp(0.308 * (z[mask] - 15)) + 0.8
    
    elif type == 'aures':    # Aure's model according to DIN 45692 (2009)
        # N is expected to be a scalar loudness_sones value for 'aures'
        if N is None or N <= 0:
            return np.zeros_like(z)
            
        N_scalar = N if np.isscalar(N) else N[0] 
        
        log_arg = 0.05 * N_scalar + 1
        if log_arg <= 0: 
            return np.zeros_like(z)
        
        log_term = np.log(log_arg)
        
        if log_term == 0: 
            return np.zeros_like(z)
            
        g = 0.078 * (np.exp(0.171 * z) / z) * (N_scalar / log_term)
    
    return g

check_which = 1

if __name__ == "__main__":
    if check_which == 0: # NO TEST

        print("metrics_sharpness.py")
    
    elif check_which == 1: # Sharpness_DIN45692 (regular / wavfile)
        with_wavfile = 0

        """
        Validation clip for Sharpness_DIN45692
        -----------------------------------

        Generates a narrowband, 1 kHz sinusoid with 160 Hz bandwidth at 60 dB SPL, sampled at 48 kHz.
        """

        print("Running Sharpness_DIN45692 test (from WAV/signal)...")


        # Define parameters for the narrowband noise
        fs = 48_000
        fc = 1_000.0  # Center frequency = 1 kHz
        bw = 160      # Bandwidth = 160 Hz
        Lp_dB = 60    # Sound Pressure Level = 60 dB SPL
        duration = 5  # Duration = 2 seconds

        num_samples = int(fs * duration)
        white_noise = np.random.randn(num_samples)

        # Design a Butterworth bandpass filter
        lowcut = fc - bw / 2
        highcut = fc + bw / 2
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Using a 4th order Butterworth filter for good rolloff
        b, a = butter(4, [low, high], btype='band')

        # Apply the filter
        narrowband_noise = lfilter(b, a, white_noise)

        # Calibrate to desired SPL
        P_ref = 20e-6  # Reference sound pressure in Pascals for 0 dB SPL
        P_rms_target = P_ref * (10**(Lp_dB / 20))
        current_rms = np.sqrt(np.mean(narrowband_noise**2))

        if current_rms > 0:
            calibrated_noise = narrowband_noise * (P_rms_target / current_rms)
        else:
            calibrated_noise = np.zeros_like(narrowband_noise) # Handle case where noise is zero

        insig = calibrated_noise
        insig = insig.astype(np.float32) # Ensure float32 as in your example

        if with_wavfile == 1:
            wavfile.write("test_S1a.wav", fs, insig)
            OUT = Sharpness_DIN45692(
                insig="test_S1a.wav",
                fs=fs,
                weight_type='DIN45692', # Example: 'DIN45692', 'bismarck', or 'aures'
                LoudnessField=0,        # 0 for free field, 1 for diffuse field
                LoudnessMethod=2,       # 1 for stationary, 2 for time-varying
                time_skip=0.5,          # Skip first 0.5 seconds for statistics (if LoudnessMethod=2)
                show_sharpness=True,    # Display sharpness results
                show_loudness=False     # Display loudness results
            )
            os.remove("test_S1a.wav") if os.path.exists("test_S1a.wav") else None
        else:
            OUT = Sharpness_DIN45692(
                insig=insig,
                fs=fs,
                weight_type='DIN45692', # Example: 'DIN45692', 'bismarck', or 'aures'
                LoudnessField=0,        # 0 for free field, 1 for diffuse field
                LoudnessMethod=2,       # 1 for stationary, 2 for time-varying
                time_skip=0.5,          # Skip first 0.5 seconds for statistics (if LoudnessMethod=2)
                show_sharpness=True,    # Display sharpness results
                show_loudness=False     # Display loudness results
            )

        print("\nSharpness Calculation Results (OUT):")
        if 'InstantaneousSharpness' in OUT:
            print(f"  Instantaneous Sharpness (first 5 values): {OUT['InstantaneousSharpness'][:5]}")
            print(f"  Mean Sharpness (Smean): {OUT['Smean'][0]:.3f} acum")
            print(f"  S5 Sharpness: {OUT['S5'][0]:.3f} acum")
        elif 'Sharpness' in OUT:
            # For stationary, s is a 1-element array, so [0] is needed
            print(f"  Stationary Sharpness: {OUT['Sharpness'][0]:.3f} acum")
        else:
            print("  No sharpness output found (check LoudnessMethod).")
