
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

from sound_metrics import *
from utilities import *
from metrics_loudness import Loudness_ISO532_1

__all__ = ["Sharpness_DIN45692"]
FloatArray = NDArray[np.floating]

def Sharpness_DIN45692(insig=None, fs=None, weight_type=None, LoudnessField=None, 
                      LoudnessMethod=None, time_skip=None, show_sharpness=None, 
                      show_loudness=None, SpecificLoudness=None, time=None):
    """
    function OUT = Sharpness_DIN45692(insig, fs, weight_type, LoudnessField, LoudnessMethod, time_skip, show_sharpness, show_loudness)
    or
    function OUT = Sharpness_DIN45692_from_loudness(SpecificLoudness, weight_type, time, time_skip, show_sharpness)

     Stationary and time-varying sharpness calculation according to DIN 45692
       (2009) from an input signal. The loudness calculation, required as pre-
       processing for sharpness, is included in this code.

     Loudness calculation is conducted according to ISO 532:1-2017
     (type <help Loudness_ISO532_1> for more info)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    INPUT ARGUMENTS (Main function)
      insig : [Nx1] array
      calibrated audio signal (Pa), 1 channel only

      fs : integer
      sampling frequency (Hz). For method = 3, provide a dummy scalar

      weight_type : string
      sharpness calculation using weighting function according to:
          - 'DIN45692'
          - 'bismarck'
          - 'aures' (dependent on the specific loudness level)

      LoudnessField : integer
      type of field used for loudness calculation; free field = 0; diffuse field = 1;

      LoudnessMethod : integer
      method used for loudness calculation - method used for loudness 
          calculation: stationary (from input 1/3 octave unweighted SPL)=0 (not 
          accepted in this context); stationary = 1; time varying = 2;

      time_skip : integer
      skip start of the signal in <time_skip> seconds for statistics 
           calculations (method=1 (time-varying) only)

      show_loudness : logical(boolean)
      optional parameter to display loudness results (only method=1)
      'false' (disable, default value) or 'true' (enable).

      show_sharpness : logical(boolean)
      optional parameter to display sharpness results (only method=1)
      'false' (disable, default value) or 'true' (enable).

    INPUT ARGUMENTS (From loudness function)
      SpecificLoudness : array
      if method = 0 (stationary) - Specific loudness [1,sone/Bark]
      if method = 1 (time-varying) - Instantaneous specific loudness [nTimeSteps,sone/Bark]

      weight_type : string
          weighting function used for sharpness calculation, according to:
          - 'DIN45692'
          - 'bismarck'
          - 'aures' (dependent on the specific loudness level)

      time : array
          time vector of the specific loudness [1,nTimeSteps] - used only for
          plot purposes if method = 1 (time-varying)

      time_skip : integer
      skip start of the signal in <time_skip> seconds for statistics 
          calculations (method=1 (time-varying) only)

      show_sharpness : logical(boolean)
      optional parameter for figures (results) display (only method=1)
      'false' (disable, default value) or 'true' (enable).

    OUTPUTS (method==0; stationary)
      OUT : struct containing the following fields

          * Sharpness: sharpness (acum)

    OUTPUTS (method==1; time-varying)
      OUT : struct containing the following fields

          * loudness: output struct from loudness calculation (type 
            <help loudness_ISO532_1> for more info)
          * InstantaneousSharpness: instantaneous sharpness (acum) vs time
          * time : time vector in seconds
          * Several statistics based on the InstantaneousSharpness (acum)
            ** Smean : mean value of InstantaneousSharpness (acum)
            ** Sstd : standard deviation of InstantaneousSharpness (acum)
            ** Smax : maximum of InstantaneousSharpness (acum)
            ** Smin : minimum of InstantaneousSharpness (acum)
            ** Sx : sharpness value exceeded during x percent of the time (acum)

              *** HINT: time-varying loudness calculation takes some time to
                        have a steady-response (thus sharpness too!).
                        Therefore, it is a good practice to consider a 
                        time_skip to compute the statistics

    Author: Gil Felix Greco, Braunschweig 09.03.2023
    Author: Gil Felix Greco, Braunschweig 16.02.2025 - introduced get_statistics function
    """
    
    # Handle no arguments case (help functionality)
    if (insig is None and fs is None and weight_type is None and LoudnessField is None and 
        LoudnessMethod is None and time_skip is None and show_sharpness is None and 
        show_loudness is None and SpecificLoudness is None and time is None):
        print(__doc__)
        return None
    
    # Determine which mode we're in based on inputs
    if SpecificLoudness is not None:
        # From loudness mode
        from_loudness_mode = True
        
        # Handle default parameters for from_loudness mode
        if show_sharpness is None:
            show_sharpness = False  # Assuming nargout > 0 equivalent
            
    else:
        # Main mode (includes loudness calculation)
        from_loudness_mode = False
        
        # Handle default parameters for main mode
        if show_loudness is None:
            show_loudness = False  # Assuming nargout > 0 equivalent
            
        if show_sharpness is None:
            show_sharpness = False  # Assuming nargout > 0 equivalent

    if from_loudness_mode:
        # Sharpness calculation from pre-computed loudness
        n = SpecificLoudness.shape[1]
        z = np.linspace(0.1, 24, n)   # create bark axis
        
        if SpecificLoudness.shape[0] == 1:  # define method based on the size of the input specific loudness
            method = 0  # (stationary) - Specific loudness [1,sone/Bark]
        else:
            method = 1  # (time-varying) - Instantaneous specific loudness [nTimeSteps,sone/Bark]
        
        loudness_sones = np.zeros((SpecificLoudness.shape[0], 1))  # pre allocate memory
        
        for i in range(SpecificLoudness.shape[0]):
            loudness_sones[i] = np.sum(SpecificLoudness[i, :], axis=0) * 0.10
            
    else:
        # Main mode with loudness calculation
        # Call the actual Loudness_ISO532_1 function
        L = Loudness_ISO532_1(insig, fs=fs, field=LoudnessField, method=LoudnessMethod, 
                              time_skip=time_skip, show=show_loudness)
        
        if LoudnessMethod == 1:  # stationary loudness calculation
            n = L['SpecificLoudness'].shape[1]
            loudness_sones = np.zeros((L['SpecificLoudness'].shape[0], 1))  # pre allocate memory
            SpecificLoudness = L['SpecificLoudness']
            
            for i in range(L['SpecificLoudness'].shape[0]):
                loudness_sones[i] = np.sum(L['SpecificLoudness'][i, :], axis=0) * 0.10
                
        elif LoudnessMethod == 2:  # time-varying loudness calculation
            n = L['InstantaneousSpecificLoudness'].shape[1]
            loudness_sones = np.zeros((L['InstantaneousSpecificLoudness'].shape[0], 1))  # pre allocate memory
            SpecificLoudness = L['InstantaneousSpecificLoudness']
            
            for i in range(L['InstantaneousSpecificLoudness'].shape[0]):
                loudness_sones[i] = np.sum(L['InstantaneousSpecificLoudness'][i, :], axis=0) * 0.10
        
        method = LoudnessMethod

    z = np.linspace(0.1, 24, n)  # create bark axis

    ## Sharpness calculation ##########################################################

    if weight_type == 'DIN45692':   # Widmann model
        
        g = il_sharpWeights(z, 'standard', None)  # calculate sharpness weighting factors
        k = 0.11  # adjusted to yield 1 acum using SQAT - DIN45692 allows 0.105<=k<=0.0115 for this weighting function
        
        s = np.zeros(SpecificLoudness.shape[0])
        for i in range(SpecificLoudness.shape[0]):
            # Handle division by zero if loudness_sones[i] is zero
            if loudness_sones[i] == 0:
                s[i] = np.nan # Consistent with Matlab's behavior for 0/0
            else:
                s[i] = k * np.sum(SpecificLoudness[i, :] * g * z * 0.10, axis=0) / loudness_sones[i]
        
        ###############################################################################
    elif weight_type == 'aures':  # Aures model
        
        g = np.zeros((SpecificLoudness.shape[0], len(z)))
        s = np.zeros(SpecificLoudness.shape[0])
        for i in range(SpecificLoudness.shape[0]):
            # Handle division by zero if loudness_sones[i] is zero
            if loudness_sones[i] == 0:
                g[i, :] = np.full_like(z, np.nan) # Consistent with Matlab's NaN for 0/0
                s[i] = np.nan # Consistent with Matlab's NaN for 0/0
            else:
                g[i, :] = il_sharpWeights(z, 'aures', loudness_sones[i])  # calculate sharpness weighting factor
                s[i] = 0.11 * np.sum(SpecificLoudness[i, :] * g[i, :] * z * 0.10, axis=0) / loudness_sones[i]
        
        ###############################################################################
    elif weight_type == 'bismarck':  # von Bismarck
        g = il_sharpWeights(z, 'bismarck', None)  # calculate sharpness weighting factor
        
        s = np.zeros(SpecificLoudness.shape[0])
        for i in range(SpecificLoudness.shape[0]):
            # Handle division by zero if loudness_sones[i] is zero
            if loudness_sones[i] == 0:
                s[i] = np.nan # Consistent with Matlab's NaN for 0/0
            else:
                s[i] = 0.11 * np.sum(SpecificLoudness[i, :] * g * z * 0.10, axis=0) / loudness_sones[i]

    ###############################################################################
    # Output struct for time-varying signals

    OUT = {}
    
    if method == 2 or (from_loudness_mode and method == 1):  # (time-varying sharpness)
        
        OUT['InstantaneousSharpness'] = s  # instantaneous sharpness
        if from_loudness_mode:
            OUT['time'] = time              # time vector
        else:
            OUT['time'] = L['time']         # time vector
            OUT['loudness'] = L             # output struct from the loudness calculation
        
        # get statistics from Time-varying sharpness (acum)
        #####################################################

        # Filter out NaNs from s before calculating statistics
        s_filtered = s[~np.isnan(s)]
        
        if len(s_filtered) > 0: # Only calculate statistics if there's valid data
            # Find idx of time_skip on the original time vector
            original_time_vector = OUT['time']
            idx_original_time_skip = np.argmin(np.abs(original_time_vector - time_skip))
            
            # Filter s and time based on NaNs and then apply time_skip
            valid_indices = ~np.isnan(s)
            s_valid_and_after_skip = s[valid_indices & (original_time_vector >= time_skip)]
            
            metric_statistics = 'Sharpness_DIN45692'
            OUT_statistics = get_statistics(s_valid_and_after_skip, metric_statistics)  # get statistics

            # copy fields of <OUT_statistics> struct into the <OUT> struct
            for fieldName in OUT_statistics.keys():  # Get all field names in OUT_statistics
                if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
                    OUT[fieldName] = OUT_statistics[fieldName]
        else:
            # If all sharpness values are NaN, populate statistics with NaN
            OUT['Smean'] = np.nan
            OUT['Sstd'] = np.nan
            OUT['Smax'] = np.nan
            OUT['Smin'] = np.nan
            percentiles = [1, 2, 3, 4, 5, 10, 50, 90, 95, 96, 97, 98, 99]
            for p in percentiles:
                OUT[f'S{p}'] = np.nan
        
        #####################################################
        
        #####################################################
        # Show plots (time-varying)
        #####################################################
        
        if show_sharpness == True:
            
            plt.figure()
            plt.gcf().canvas.manager.set_window_title('Sharpness analysis (time-varying)')
            
            plt.plot(OUT['time'], OUT['S5'] * np.ones_like(OUT['time']), 'r--', label=f'$S_5$={OUT["S5"][0]:.3g}')
            plt.plot(OUT['time'], s)
            
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Sharpness, $S$ (acum)')
            
            plt.legend(loc='best')
            
            plt.gcf().patch.set_facecolor('white')
            plt.show()
        
    elif method == 1 or (from_loudness_mode and method == 0):  # (stationary sharpness)
        
        OUT['Sharpness'] = s  # sharpness
    
    return OUT

def il_sharpWeights(z, type_name, N):
    """
    Embedded function (compute weighting functions according to required model type)
    """
    g = np.zeros(len(z))

    if type_name == 'standard':  # Widmann model according to DIN 45692 (2009)
        g[z < 15.8] = 1
        mask = z >= 15.8
        g[mask] = 0.15 * np.exp(0.42 * (z[mask] - 15.8)) + 0.85

    elif type_name == 'bismarck':  # von bismark's model according to DIN 45692 (2009)
        g[z < 15] = 1
        mask = z >= 15
        g[mask] = 0.2 * np.exp(0.308 * (z[mask] - 15)) + 0.8

    elif type_name == 'aures':    # Aures' model according to DIN 45692 (2009)
        # N can be a scalar or an array (loudness_sones)
        if hasattr(N, '__len__') and np.ndim(N) > 0:  # N is array-like (e.g., multiple time steps)
            g = np.zeros((len(N), len(z)))
            for nt in range(len(N)):
                N_val = N[nt]
                if N_val <= 0: # Handle non-positive loudness values, log(1)=0, division by zero
                    g[nt, :] = np.full_like(z, np.nan) # Consistent with Matlab's NaN for 0/0
                else:
                    log_term = np.log(0.05 * N_val + 1)
                    if log_term == 0: # Should not happen if N_val > 0, but as a safeguard
                        g[nt, :] = np.full_like(z, np.nan)
                    else:
                        g[nt, :] = 0.078 * (np.exp(0.171 * z) / z) * (N_val / log_term)
        else:  # N is scalar
            if N <= 0:
                g = np.full_like(z, np.nan) # Consistent with Matlab's NaN for 0/0
            else:
                log_term = np.log(0.05 * N + 1)
                if log_term == 0:
                    g = np.full_like(z, np.nan)
                else:
                    g = 0.078 * (np.exp(0.171 * z) / z) * (N / log_term)

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