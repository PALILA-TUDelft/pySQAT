
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
                      LoudnessMethod=None, time_skip=None, show_sharpness=None, show_loudness=None, 
                      dBFS=94, export_excel=None, SpecificLoudness=None, time=None):
    """
    Calculate **Zwicker sharpness** in accordance with *DIN 45692:2009*.

    The routine offers two workflows
    (selected automatically by the presence of *SpecificLoudness*):

    * **From audio** – the input waveform is converted to stationary or
      time-varying loudness via :pyfunc:`metrics_loudness.Loudness_ISO532_1`;
      sharpness is then derived from the resulting specific-loudness
      pattern.
    * **From specific loudness** – skip the loudness stage and supply a
      Bark-band specific-loudness matrix directly.

    Four weighting functions are available:

    ==============  ==========================================================
    **Function**    **Description**
    --------------  ----------------------------------------------------------
    ``'DIN45692'``  Widmann’s *standard* curve (recommended by the norm)
    ``'aures'``     Aures (1985) loudness-dependent curve
    ``'bismarck'``  von Bismarck (1974) constant curve
    ``None``        Raises :class:`ValueError`
    ==============  ==========================================================


    Parameters
    ----------
    insig : str | array_like, optional
        Path to a WAV file **or** mono pressure signal in pascals.  
        Ignored when *SpecificLoudness* is given.
    fs : float, optional
        Sampling rate of *insig* in hertz (required when *insig* is an
        array).
    weight_type : {'DIN45692', 'aures', 'bismarck'}, default ``'DIN45692'``
        Sharpness weighting curve.
    LoudnessField : int, {0, 1}, optional
        Acoustic field for loudness (0 = free, 1 = diffuse).
    LoudnessMethod : int, {1, 2}, optional
        1 = stationary, 2 = time-varying loudness (ignored in
        *from-loudness* mode).
    time_skip : float, optional
        Seconds to discard at the beginning before computing statistics.
    show_sharpness, show_loudness : bool, optional
        Display diagnostic plots.
    dBFS : float, default ``94``
        SPL represented by a full-scale sine (used when reading WAV).
    export_excel : str, optional
        If provided, write output dictionary to *filename* via
        :pyfunc:`utilities.export_dict_to_excel`.
    SpecificLoudness : numpy.ndarray, optional
        Bark-band loudness – shape ``(T, 240)`` for time-varying or
        ``(1, 240)`` for stationary analysis.
    time : numpy.ndarray, optional
        Time vector that matches the rows of *SpecificLoudness*.

    Returns
    -------
    dict
        Dictionary containing instantaneous data and summary statistics.

    Notes
    -----
    * The scaling constant *k = 0.11* is chosen so that a 1-kHz sine at
      60 dB SPL yields **1 acum** with the DIN weighting.
    * For *from-audio* workflows the signal is resampled to 48 kHz before
      loudness processing (per ISO 532-1 filterbank requirements).
    """

    # Determine operation mode based on input
    if SpecificLoudness is not None:
        # Mode 2: From specific loudness
        mode = 'from_loudness'
    else:
        # Mode 1: From audio signal
        mode = 'from_signal'
    
    # Handle mode 1: From audio signal
    if mode == 'from_signal':
        # --- WAV file interface ---
        if isinstance(insig, str):
            insig, fs = wav2sig(insig, fs, dBFS)

        elif fs is None:
            raise ValueError("If insig is not a filename, fs must be provided.")

        if show_loudness is None:
            # Heuristic: if called without assignment, assume display is desired
            if 'return' not in str(inspect.currentframe().f_back.f_code.co_names):
                show_loudness = True
            else:
                show_loudness = False
        
        if show_sharpness is None:
            if 'return' not in str(inspect.currentframe().f_back.f_code.co_names):
                show_sharpness = True
            else:
                show_sharpness = False

        if LoudnessMethod == 1:  # stationary loudness calculation
            
            L = Loudness_ISO532_1(insig, fs,          # input signal and sampling freq.
                                 LoudnessField,       # free field = 0; diffuse field = 1;
                                 LoudnessMethod,      # method used for loudness calculation: stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2;
                                 time_skip,           # time_skip
                                 show_loudness)       # show loudness results
            
            n = L['SpecificLoudness'].shape[1]
            loudness_sones = np.zeros((L['SpecificLoudness'].shape[0], 1))  # pre allocate memory
            SpecificLoudness = L['SpecificLoudness']
            
            for i in range(L['SpecificLoudness'].shape[0]):
                loudness_sones[i] = np.sum(L['SpecificLoudness'][i, :]) * 0.10
            
        elif LoudnessMethod == 2:  # time-varying loudness calculation
            
            L = Loudness_ISO532_1(insig, fs,          # input signal and sampling freq.
                                 LoudnessField,       # free field = 0; diffuse field = 1;
                                 LoudnessMethod,      # method used for loudness calculation: stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2;
                                 time_skip,           # time_skip
                                 show_loudness)       # show loudness results
            
            n = L['InstantaneousSpecificLoudness'].shape[1]
            loudness_sones = np.zeros((L['InstantaneousSpecificLoudness'].shape[0], 1))  # pre allocate memory
            SpecificLoudness = L['InstantaneousSpecificLoudness']
            
            for i in range(L['InstantaneousSpecificLoudness'].shape[0]):
                loudness_sones[i] = np.sum(L['InstantaneousSpecificLoudness'][i, :]) * 0.10

        z = np.linspace(0.1, 24, n)  # create bark axis
       
    else:
        # Mode 2: From specific loudness
        if show_sharpness is None:
            show_sharpness = False # Default to False when SpecificLoudness is provided
        
        n = SpecificLoudness.shape[1]
        z = np.linspace(0.1, 24, n)  # create bark axis

        if SpecificLoudness.shape[0] == 1:  # define method based on the size of the input specific loudness
            method = 0  # (stationary) - Specific loudness [1,sone/Bark]
        else:
            method = 1  # (time-varying) - Instantaneous specific loudness [nTimeSteps,sone/Bark]

        loudness_sones = np.zeros((SpecificLoudness.shape[0], 1))  # pre allocate memory

        for i in range(SpecificLoudness.shape[0]):
            loudness_sones[i] = np.sum(SpecificLoudness[i, :]) * 0.10

    ## Sharpness calculation ##########################################################

    if weight_type == 'DIN45692':   # Widmann model
        
        g = il_sharpWeights(z, 'standard', [])  # calculate sharpness weighting factors
        k = 0.11  # adjusted to yield 1 acum using SQAT - DIN45692 allows 0.105<=k<=0.0115 for this weighting function
        
        s = np.zeros(SpecificLoudness.shape[0])
        for i in range(SpecificLoudness.shape[0]):
            # Fix: Use .item() to get scalar from 1-element array
            s[i] = k * np.sum(SpecificLoudness[i, :] * g * z * 0.10) / loudness_sones[i].item()
        
        ###############################################################################
    elif weight_type == 'aures':  # Aures model
        
        s = np.zeros(SpecificLoudness.shape[0])
        g_sharpness_weights = np.zeros((SpecificLoudness.shape[0], len(z))) # This 'g' is for Sharpness_DIN45692 scope
        for i in range(SpecificLoudness.shape[0]):
            # il_sharpWeights returns a 1D array for a single loudness value
            g_sharpness_weights[i, :] = il_sharpWeights(z, 'aures', loudness_sones[i].item())  # calculate sharpness weighting factor
            # Fix: Use .item() to get scalar from 1-element array
            s[i] = 0.11 * np.sum(SpecificLoudness[i, :] * g_sharpness_weights[i, :] * z * 0.10) / loudness_sones[i].item()
        
        ###############################################################################
    elif weight_type == 'bismarck':  # von Bismarck
        g = il_sharpWeights(z, 'bismarck', [])  # calculate sharpness weighting factor
        
        s = np.zeros(SpecificLoudness.shape[0])
        for i in range(SpecificLoudness.shape[0]):
            # Fix: Use .item() to get scalar from 1-element array
            s[i] = 0.11 * np.sum(SpecificLoudness[i, :] * g * z * 0.10) / loudness_sones[i].item()

    ###############################################################################
    # Output struct for time-varying signals

    # Determine if we're dealing with time-varying or stationary analysis
    if mode == 'from_signal':
        is_time_varying = (LoudnessMethod == 2)
        time_vector = L['time'] if LoudnessMethod == 2 else None
    else:
        is_time_varying = (method == 1)
        time_vector = time

    if is_time_varying:  # (time-varying sharpness)
        
        OUT = {}
        OUT['InstantaneousSharpness'] = s  # instantaneous sharpness
        OUT['time'] = time_vector            # time vector
        
        if mode == 'from_signal':
            OUT['loudness'] = L                # output struct from the loudness calculation
           
        # get statistics from Time-varying sharpness (acum)
        #############################################

        # Set default time_skip if not provided
        if time_skip is None:
            time_skip = 0

        idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector

        metric_statistics = 'Sharpness_DIN45692'
        OUT_statistics = get_statistics(s[idx:], metric_statistics)  # get statistics

        # copy fields of <OUT_statistics> struct into the <OUT> struct
        fields_OUT_statistics = list(OUT_statistics.keys())  # Get all field names in OUT_statistics

        for i in range(len(fields_OUT_statistics)):
            fieldName = fields_OUT_statistics[i]
            if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
                OUT[fieldName] = OUT_statistics[fieldName]
        #############################################

          
        #############################################
        # Show plots (time-varying)
        #############################################
        
        if show_sharpness == True:
            
            plt.figure()
            plt.gcf().canvas.manager.set_window_title('Sharpness analysis (time-varying)')
            
            # Ensure S5 is treated correctly whether it's a scalar or array
            s5_val = OUT['S5'][0] if isinstance(OUT['S5'], np.ndarray) else OUT['S5']
            plt.plot(OUT['time'], s5_val * np.ones(len(OUT['time'])), 'r--', label=f'$S_5$={s5_val:.3g}')
            plt.plot(OUT['time'], s)
            
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Sharpness, $S$ (acum)')
            plt.xlim([0, OUT['time'][-1]])
            
            plt.legend(loc='best')
            plt.legend().set_frame_on(False)
            
            plt.gcf().patch.set_facecolor('white')
            plt.show()
        
    else:  # (stationary sharpness)
        
        OUT = {}
        if mode == 'from_signal':
            OUT['Sharpness'] = s  # sharpness (will be a 1-element array)
        else:
            OUT['Sharpness'] = s[0]  # sharpness (scalar from 1-element array)
    
    if export_excel is not None:
        export_dict_to_excel(OUT, filename=f"{export_excel}")

    return OUT
    
def il_sharpWeights(z, type, N):
    """
    Calculate sharpness weighting factors according to different models.
    
    Parameters:
    -----------
    z : array-like
        Bark frequency scale
    type : str
        Type of weighting ('standard', 'bismarck', 'aures')
    N : scalar, list, or None
        Loudness values. For 'aures' type:
        - If scalar (or 1-element array/list): Calculates 1D weighting factors.
        - If multi-element list/array: Calculates 2D weighting factors (time-varying).
    
    Returns:
    --------
    g : np.ndarray
        Weighting factors (1D or 2D depending on 'type' and 'N' for 'aures').
    """

    # Initial g is 1D, but might be reassigned to 2D later for 'aures'
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
        # Normalize N to a list of scalar values for consistent iteration
        N_vals = []
        if np.isscalar(N) or N is None:
            N_vals = [N] if N is not None else [0] # Treat None as 0 loudness
        elif isinstance(N, np.ndarray):
            if N.ndim == 0: # 0-dim array (scalar)
                N_vals = [N.item()]
            elif N.ndim == 1 and N.shape[0] == 1: # 1-element 1D array
                N_vals = [N.item()]
            else: # Multi-element 1D array (time-varying)
                N_vals = N.tolist() # Convert to list of scalars
        elif isinstance(N, list):
            N_vals = N
        else: # Fallback for unexpected N types
            N_vals = [0] # Default to 0 loudness

        # Determine if the output 'g' should be 1D or 2D based on N_vals length
        if len(N_vals) > 1: # Time-varying loudness, g_output should be 2D
            g_output = np.zeros((len(N_vals), len(z)))
            for nt_idx, N_val in enumerate(N_vals):
                # Avoid division by zero or log of non-positive values
                if N_val is not None and N_val > 0:
                    log_term = np.log(0.05 * N_val + 1)
                    if log_term > 0:
                        g_output[nt_idx, :] = 0.078 * (np.exp(0.171 * z) / z) * (N_val / log_term)
                    else:
                        g_output[nt_idx, :] = 0 # Handle log_term <= 0
                else:
                    g_output[nt_idx, :] = 0 # Handle N_val <= 0 or None
            g = g_output # Reassign g to the 2D output
        else: # Stationary loudness (N_vals has one element), g remains 1D
            N_val = N_vals[0] if N_vals else 0 # Get the single value, default to 0 if list is empty
            if N_val is not None and N_val > 0:
                log_term = np.log(0.05 * N_val + 1)
                if log_term > 0:
                    g = 0.078 * (np.exp(0.171 * z) / z) * (N_val / log_term)
                else:
                    g = np.zeros_like(z)
            else:
                g = np.zeros_like(z)
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

    elif check_which == 2: # Sharpness_DIN45692 (loudness)

        """
        Validation clip for Sharpness_DIN45692 (loudness)
        -----------------------------------
        Tests the function when SpecificLoudness and time are provided directly.
        """
        print("Running Sharpness_DIN45692 test (from SpecificLoudness)...")

        # --- Test Case 1: Stationary Specific Loudness ---
        print("\n--- Test Case 1: Stationary Sharpness ---")
        # Simulate SpecificLoudness for a stationary signal (1 time step, 24 bark bands)
        # Values are arbitrary for testing, but should be positive
        stationary_specific_loudness = np.array([[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
            2.1, 2.2, 2.3, 2.4
        ]]) # Shape (1, 24)

        out_stationary = Sharpness_DIN45692(
            SpecificLoudness=stationary_specific_loudness,
            weight_type='DIN45692', # Test with DIN45692
            show_sharpness=False # No plot for stationary
        )
        print("Stationary Sharpness Results (OUT):")
        if 'Sharpness' in out_stationary:
            print(f"  Calculated Stationary Sharpness: {out_stationary['Sharpness']:.3f} acum")
        else:
            print("  Error: Stationary sharpness not found in output.")

        # --- Test Case 2: Time-Varying Specific Loudness ---
        print("\n--- Test Case 2: Time-Varying Sharpness ---")
        # Simulate SpecificLoudness for a time-varying signal
        num_time_steps = 50
        num_bark_bands = 24
        time_duration = 5.0 # seconds
        time_vector = np.linspace(0, time_duration, num_time_steps)

        # Create a specific loudness that changes over time
        # For example, increasing loudness then decreasing
        time_varying_specific_loudness = np.zeros((num_time_steps, num_bark_bands))
        for i in range(num_time_steps):
            # Simulate a peak around the middle of the time series
            factor = 1 + np.sin(time_vector[i] / time_duration * 2 * np.pi) * 0.5
            time_varying_specific_loudness[i, :] = (np.random.rand(num_bark_bands) * 0.5 + 0.1) * factor
        
        # Ensure no zero or negative values for robustness, especially for 'aures'
        time_varying_specific_loudness[time_varying_specific_loudness <= 0] = 0.01

        out_time_varying = Sharpness_DIN45692(
            SpecificLoudness=time_varying_specific_loudness,
            weight_type='aures', # Test with Aures model for time-varying
            time=time_vector,
            time_skip=1.0, # Skip first 1 second for statistics
            show_sharpness=True # Show plot for time-varying
        )

        print("Time-Varying Sharpness Results (OUT):")
        if 'InstantaneousSharpness' in out_time_varying:
            print(f"  Instantaneous Sharpness (first 5 values): {out_time_varying['InstantaneousSharpness'][:5]}")
            print(f"  Mean Sharpness (Smean): {out_time_varying['Smean'][0]:.3f} acum")
            print(f"  S5 Sharpness: {out_time_varying['S5'][0]:.3f} acum")
            print(f"  Time vector length: {len(out_time_varying['time'])}")
        else:
            print("  Error: Time-varying sharpness not found in output.")