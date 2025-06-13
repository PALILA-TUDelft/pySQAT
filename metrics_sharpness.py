
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
import warnings
import sys

from sound_metrics import *
from utilities import *
from metrics_loudness import Loudness_ISO532_1

__all__ = ["Roughness_Daniel1997"]
FloatArray = NDArray[np.floating]

def Sharpness_DIN45692(insig=None, fs=None, weight_type=None, LoudnessField=None, 
                      LoudnessMethod=None, time_skip=None, show_sharpness=None, show_loudness=None, dBFS=94, export_excel=None):
    
    # --- WAV file interface ---
    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)

    elif fs is None:
        raise ValueError("If insig is not a filename, fs must be provided.")

    # Get number of arguments passed
    frame = sys._getframe()
    args = frame.f_locals
    nargin = sum(1 for v in [insig, fs, weight_type, LoudnessField, LoudnessMethod, time_skip, show_sharpness, show_loudness] if v is not None)
    
    if nargin == 0:
        print(__doc__)
        return
    
    if nargin < 8:
        if 'return' not in str(frame.f_back.f_code.co_names):  # equivalent to nargout == 0 check
            show_loudness = 1
        else:
            show_loudness = 0
    
    if nargin < 7:
        if 'return' not in str(frame.f_back.f_code.co_names):  # equivalent to nargout == 0 check
            show_sharpness = 1
        else:
            show_sharpness = 0

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

    ## Sharpness calculation ##########################################################

    if weight_type == 'DIN45692':   # Widmann model
        
        g = il_sharpWeights(z, 'standard', [])  # calculate sharpness weighting factors
        k = 0.11  # adjusted to yield 1 acum using SQAT - DIN45692 allows 0.105<=k<=0.0115 for this weighting function
        
        s = np.zeros(SpecificLoudness.shape[0])
        for i in range(SpecificLoudness.shape[0]):
            s[i] = k * np.sum(SpecificLoudness[i, :] * g * z * 0.10) / loudness_sones[i]
        
        ###############################################################################
    elif weight_type == 'aures':  # Aures model
        
        s = np.zeros(SpecificLoudness.shape[0])
        g = np.zeros((SpecificLoudness.shape[0], len(z)))
        for i in range(SpecificLoudness.shape[0]):
            g[i, :] = il_sharpWeights(z, 'aures', loudness_sones[i])  # calculate sharpness weighting factor
            s[i] = 0.11 * np.sum(SpecificLoudness[i, :] * g[i, :] * z * 0.10) / loudness_sones[i]
        
        ###############################################################################
    elif weight_type == 'bismarck':  # von Bismarck
        g = il_sharpWeights(z, 'bismarck', [])  # calculate sharpness weighting factor
        
        s = np.zeros(SpecificLoudness.shape[0])
        for i in range(SpecificLoudness.shape[0]):
            s[i] = 0.11 * np.sum(SpecificLoudness[i, :] * g * z * 0.10) / loudness_sones[i]

    ###############################################################################
    # Output struct for time-varying signals

    if LoudnessMethod == 2:  # (time-varying sharpness)
        
        OUT = {}
        OUT['InstantaneousSharpness'] = s  # instantaneous sharpness
        OUT['time'] = L['time']            # time vector
        OUT['loudness'] = L                # output struct from the loudness calculation
           
       
        # get statistics from Time-varying sharpness (acum)
        #############################################

        idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector

        metric_statistics = 'Sharpness_DIN45692'
        OUT_statistics = get_statistics(s[idx:], metric_statistics)  # get statistics

        # copy fields of <OUT_statistics> struct into the <OUT> struct
        fields_OUT_statistics = list(OUT_statistics.keys())  # Get all field names in OUT_statistics

        for i in range(len(fields_OUT_statistics)):
            fieldName = fields_OUT_statistics[i]
            if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
                OUT[fieldName] = OUT_statistics[fieldName]
        
        del OUT_statistics, metric_statistics, fields_OUT_statistics, fieldName
        #############################################

          
        #############################################
        # Show plots (time-varying)
        #############################################
        
        if show_sharpness == True:
            
            plt.figure()
            plt.gcf().canvas.manager.set_window_title('Sharpness analysis (time-varying)')
            
            plt.plot(L['time'], OUT['S5'] * np.ones(len(L['time'])), 'r--', label=f'$S_5$={OUT["S5"][0]:.3g}')
            plt.plot(L['time'], s)
            
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Sharpness, $S$ (acum)')
            plt.xlim([0, L['time'][-1]])
            
            plt.legend(loc='best')
            plt.legend().set_frame_on(False)
            
            plt.gcf().patch.set_facecolor('white')
            plt.show()
        
    elif LoudnessMethod == 1:  # (stationary sharpness)
        
        OUT = {}
        OUT['Sharpness'] = s  # sharpness
    
    if export_excel is not None:
        export_dict_to_excel(OUT, filename=f"{export_excel}")

    return OUT

# End of Sharpness_DIN45692

###############################################################################
# Embedded function (compute weighting functions according to required model type)

def il_sharpWeights(z, type, N):

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
        if np.isscalar(N):
            N = [N]
        for nt in range(len(N)):
            g[nt, :] = 0.078 * (np.exp(0.171 * z) / z) * (N[nt] / np.log(0.05 * N[nt] + 1))

    return g

# end of il_sharpWeights

check_which = 1

if __name__ == "__main__":
    if check_which == 0: # NO TEST

        print("metrics_sharpness.py")
    
    elif check_which == 1: # Sharpness_DIN45692
        with_wavfile = 0

        """
        Validation clip for Sharpness_DIN45692
        -----------------------------------

        Generates a narrowband, 1 kHz sinusoid with 160 Hz bandwidth at 60 dB SPL, sampled at 48 kHz.
        """

        print("Running Sharpness_DIN45692 test...")


        # Define parameters for the narrowband noise
        fs = 48_000
        fc = 1_000.0  # Center frequency = 1 kHz
        bw = 160      # Bandwidth = 160 Hz
        Lp_dB = 60    # Sound Pressure Level = 60 dB SPL
        duration = 2  # Duration = 2 seconds

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
            wavfile.write("test_S1.wav", fs, insig)
            OUT = Sharpness_DIN45692(
                insig="test_S1.wav",
                fs=fs,
                weight_type='DIN45692', # Example: 'DIN45692', 'bismarck', or 'aures'
                LoudnessField=0,        # 0 for free field, 1 for diffuse field
                LoudnessMethod=2,       # 1 for stationary, 2 for time-varying
                time_skip=0.5,          # Skip first 0.5 seconds for statistics (if LoudnessMethod=2)
                show_sharpness=True,    # Display sharpness results
                show_loudness=False     # Display loudness results
            )
        else:
            os.remove("test_S1.wav") if os.path.exists("test_S1.wav") else None
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
            print(f"  Stationary Sharpness: {OUT['Sharpness'][0]:.3f} acum")
        else:
            print("  No sharpness output found (check LoudnessMethod).")

