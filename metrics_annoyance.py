
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
from metrics_sharpness import Sharpness_DIN45692
from metrics_roughness import Roughness_Daniel1997
from metrics_fluctuation import FluctuationStrength_Osses2016
from metrics_tonality import Tonality_Aures1985

def PsychoacousticAnnoyance_Di2016(insig=None, fs=None, LoudnessField=None, time_skip=None, showPA=None, show=None, dBFS=94, percentiles=None):
    """
    function OUT = PsychoacousticAnnoyance_Di2016(insig,fs,LoudnessField,time_skip,showPA,show)
    
    This function calculates the Di et al's modified psychoacoustic annoyance 
      model from an input acoustic signal
    
    The modified psychoacoustic annoyance model is according to: (page 201)
     [1] Di et al., Improvement of Zwicker's psychoacoustic annoyance model 
         aiming at tonal noises, Applied Acoustics 105 (2016) 164-170
    
    - This metric combines five psychoacoustic metrics to quantitatively 
      describe annoyance:
     1) Loudness, N (sone) - calculated hereafter following ISO 532-1:2017
        type <help Loudness_ISO532_1> for more info
    
     2) Sharpness, S (acum) - calculated hereafter following DIN 45692:2009
        NOTE: uses DIN 45692 weighting function by default, please change code if
        the use of a different withgitng function is desired).
        type <help Sharpness_DIN45692_from_loudness>
    
     3) Roughness, R (asper) - calculated hereafter following Daniel & Weber model
        type <help Roughness_Daniel1997> for more info
    
     4) Fluctuation strength, FS (vacil) - calculated hereafter following 
        Osses et al. model type <help FluctuationStrength_Osses2016> for more info
    
     4) Tonality, K (t.u.) - calculated hereafter following Aures' model
        type <help Tonality_Aures1985> for more info
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    if percentiles is None:

        # --- WAV file interface ---
        if isinstance(insig, str):
            insig, fs = wav2sig(insig, fs, dBFS)

        elif fs is None:
            raise ValueError("If insig is not a filename, fs must be provided.")

        # Handle default parameters based on number of arguments provided
        if show is None:
            if showPA is None and time_skip is None and LoudnessField is None:
                show = True
            else:
                show = False
        
        if showPA is None:
            if time_skip is None and LoudnessField is None:
                showPA = True
            else:
                showPA = False
        
        insig = np.array(insig)
        
        time_insig = np.arange(len(insig)) / fs
        
        if time_insig[-1] < 2:
            warnings.warn('WARNING: the signal is shorter than 2 seconds. Due to the minimum window size used for the fluctuation strength, the computation of a time-varying psychoacoustic annoyance is not possible ! Only scalar psychoacoustic annoyance number will be calculated for this signal!')
            method_FS = 0
        else:
            method_FS = 1
        
        alpha = 0.52
        beta = 6.41
        
        L = Loudness_ISO532_1(insig, fs, field=LoudnessField, method=2, time_skip=time_skip, show=False)
        
        OUT = {}
        OUT['L'] = L
        
        S = Sharpness_DIN45692(SpecificLoudness=L['InstantaneousSpecificLoudness'], weight_type='DIN45692', time=L['time'], time_skip=time_skip, show_sharpness=False)
        
        OUT['S'] = S
        
        R = Roughness_Daniel1997(insig, fs, time_skip=time_skip, show=False)
        
        OUT['R'] = R
        
        FS = FluctuationStrength_Osses2016(insig, fs, method=method_FS, time_skip=time_skip, show=False)
        
        OUT['FS'] = FS
        
        K = Tonality_Aures1985(insig, fs, LoudnessField=LoudnessField, time_skip=0, show=False)
        
        OUT['K'] = K
        
        if time_insig[-1] < 2:
            if S['S5'] > 1.75:
                ws = (S['S5'] - 1.75) * (np.log10(L['N5'] + 10)) / 4
            else:
                ws = 0
            
            if np.isinf(ws) or np.isnan(ws):
                ws = 0
            
            wfr = (2.18 / (L['N5'] ** 0.4)) * (0.4 * FS['FS5'] + 0.6 * R['R5'])
            
            if np.isinf(wfr) or np.isnan(wfr):
                wfr = 0
            
            wt = (beta / (L['N5'] ** alpha)) * K['K5']
            
            if np.isinf(wt) or np.isnan(wt):
                wt = 0
            
            PA_scalar = L['N5'] * (1 + np.sqrt(ws**2 + wfr**2 + wt**2))
            
            OUT['ScalarPA'] = PA_scalar
            
            if show == True:
                print('No plots are shown because the signal is too short, but:')
                print(f"\tThe mean loudness is {OUT['L']['Nmean'].item():.2f} (sones)")
                print(f"\tThe mean fluctuation {OUT['R']['Rmean'].item():.2f} (asper)")
                print(f"\tThe mean roughness is {OUT['FS']['FSmean'].item():.2f} (vacil)")
                print(f"\tThe mean tonality is {OUT['K']['Kmean'].item():.2f} (t.u.)")
            
            if showPA == True:
                print('\nThe obtained psychoacoustic annoyance (PA) using the previous metrics is:')
                print(f'\tPA is equal to {PA_scalar.item():.1f}')
        
        else:
            LastTime = FS['time'][-1]
            
            idx_L = np.argmin(np.abs(L['time'] - LastTime))
            L['time'] = L['time'][:idx_L+1]
            L['InstantaneousLoudness'] = L['InstantaneousLoudness'][:idx_L+1]
            
            idx_S = idx_L
            S['time'] = S['time'][:idx_S+1]
            S['InstantaneousSharpness'] = S['InstantaneousSharpness'][:idx_S+1]
            
            idx_R = np.argmin(np.abs(R['time'] - LastTime))
            R['time'] = R['time'][:idx_R+1]
            R['InstantaneousRoughness'] = R['InstantaneousRoughness'][:idx_R+1]
            
            R['time'] = R['time'].T
            R['InstantaneousRoughness'] = R['InstantaneousRoughness'].T
            
            idx_K = np.argmin(np.abs(K['time'] - LastTime))
            K['time'] = K['time'][:idx_K+1]
            K['InstantaneousTonality'] = K['InstantaneousTonality'][:idx_K+1]
            
            del idx_R, idx_L, idx_S, idx_K
            
            roughness = interp1d(R['time'], R['InstantaneousRoughness'], kind='cubic', bounds_error=False, fill_value='extrapolate')(L['time'])
            fluctuation = interp1d(FS['time'], FS['InstantaneousFluctuationStrength'], kind='cubic', bounds_error=False, fill_value='extrapolate')(L['time'])
            tonality = interp1d(K['time'], K['InstantaneousTonality'], kind='cubic', bounds_error=False, fill_value='extrapolate')(L['time'])
            
            PA = np.zeros(len(L['time']))
            ws = np.zeros(len(L['time']))
            wfr = np.zeros(len(L['time']))
            wt = np.zeros(len(L['time']))
            
            for i in range(len(L['time'])):
                if S['InstantaneousSharpness'][i] > 1.75:
                    ws[i] = (S['InstantaneousSharpness'][i] - 1.75) * (np.log10(L['InstantaneousLoudness'][i] + 10)) / 4
                else:
                    ws[i] = 0
                
                if np.isinf(ws[i]) or np.isnan(ws[i]):
                    ws[i] = 0
                
                wfr[i] = (2.18 / (L['InstantaneousLoudness'][i] ** 0.4)) * (0.4 * fluctuation[i] + 0.6 * roughness[i])
                
                if np.isinf(wfr[i]) or np.isnan(wfr[i]):
                    wfr[i] = 0
                
                wt[i] = (beta / (L['InstantaneousLoudness'][i] ** alpha)) * tonality[i]
                
                if np.isinf(wt[i]) or np.isnan(wt[i]):
                    wt[i] = 0
                
                PA[i] = L['InstantaneousLoudness'][i] * (1 + np.sqrt(ws[i]**2 + wfr[i]**2 + wt[i]**2))
            
            OUT['wt'] = np.sqrt(wt)
            OUT['wfr'] = wfr
            OUT['ws'] = ws
            
            if S['S5'] > 1.75:
                ws = (S['S5'] - 1.75) * (np.log10(L['N5'] + 10)) / 4
            else:
                ws = 0
            
            if np.isinf(ws) or np.isnan(ws):
                ws = 0
            
            wfr = (2.18 / (L['N5'] ** 0.4)) * (0.4 * FS['FS5'] + 0.6 * R['R5'])
            
            if np.isinf(wfr) or np.isnan(wfr):
                wfr = 0
            
            wt = (beta / (L['N5'] ** alpha)) * K['K5']
            
            if np.isinf(wt) or np.isnan(wt):
                wt = 0
            
            PA_scalar = L['N5'] * (1 + np.sqrt(ws**2 + wfr**2 + wt**2))
            
            OUT['InstantaneousPA'] = PA
            OUT['ScalarPA'] = PA_scalar
            OUT['time'] = L['time']
            
            idx = np.argmin(np.abs(OUT['time'] - time_skip))
            
            metric_statistics = 'PsychoacousticAnnoyance_Di2016'
            OUT_statistics = get_statistics(PA[idx:], metric_statistics)
            
            for fieldName in OUT_statistics.keys():
                if fieldName not in OUT:
                    OUT[fieldName] = OUT_statistics[fieldName]
            
            del OUT_statistics, metric_statistics
            
            if show == True or showPA == True:
                num_plots = 0
                if show:
                    num_plots += 5 # L, S, R, FS, K
                if showPA:
                    num_plots += 1 # PA

                if num_plots > 0:
                    # Determine grid size (e.g., 2 columns, calculate rows)
                    cols = 2
                    rows = int(np.ceil(num_plots / cols))
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows)) # Adjust figsize as needed
                    axes = axes.flatten() # Flatten the axes array for easy iteration

                    plot_idx = 0
                    if show == True:
                        il_plotter(OUT['time'], L['InstantaneousLoudness'], L['N5'], 'loudness', ax=axes[plot_idx])
                        plot_idx += 1
                        il_plotter(OUT['time'], S['InstantaneousSharpness'], S['S5'], 'sharpness', ax=axes[plot_idx])
                        plot_idx += 1
                        il_plotter(OUT['time'], roughness, R['R5'], 'roughness', ax=axes[plot_idx])
                        plot_idx += 1
                        il_plotter(OUT['time'], fluctuation, FS['FS5'], 'fluctuation', ax=axes[plot_idx])
                        plot_idx += 1
                        il_plotter(OUT['time'], tonality, K['K5'], 'tonality', ax=axes[plot_idx])
                        plot_idx += 1
                    
                    if showPA == True:
                        il_plotter(OUT['time'], OUT['InstantaneousPA'], OUT['PA5'], 'annoyance', ax=axes[plot_idx])
                        plot_idx += 1

                    # Turn off any unused subplots if the grid has more axes than plots
                    for i in range(plot_idx, len(axes)):
                        fig.delaxes(axes[i]) # Remove empty subplots

                    plt.tight_layout() # Adjust layout to make space for suptitle
                    plt.show()

    elif percentiles is not None and insig is None:

        N, S, R, FS, K = percentiles

        ## modified PA model constants (Ref. [1] pg. 168, eq (9))
    
        alpha = 0.52
        beta = 6.41
        
        ## (scalar) modified psychoacoustic annoyance - computed directly from percentile values
        
        # sharpness and loudness influence
        if S > 1.75:
            ws = (S - 1.75) * (np.log10(N + 10)) / 4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
        else:
            ws = 0
        
        if np.isinf(ws) or np.isnan(ws):  # replace inf and NaN with zeros
            ws = 0
        
        # influence of roughness and fluctuation strength
        wfr = (2.18 / (N ** 0.4)) * (0.4 * FS + 0.6 * R)
        
        if np.isinf(wfr) or np.isnan(wfr):  # replace inf and NaN with zeros
            wfr = 0
        
        # Tonality influence
        wt = (beta / (N ** alpha)) * K
        
        if np.isinf(wt) or np.isnan(wt):  # replace inf and NaN with zeros
            wt = 0
        
        # Di's modified psychoacoustic annoyance
        PA_scalar = N * (1 + np.sqrt(ws**2 + wfr**2 + wt**2))
        
        OUT = PA_scalar  # Annoyance calculated from the percentiles of each variable

    return OUT

def PsychoacousticAnnoyance_Zwicker1999(insig=None, fs=None, LoudnessField=None, time_skip=None, showPA=None, show=None, dBFS = 94, percentiles=None):
    """
    function OUT = PsychoacousticAnnoyance_Zwicker1999(insig,fs,LoudnessField,time_skip,showPA,show)
    
    This function calculates the Zwicker's psychoacoustic annoyance model from an input acoustic signal
    
    The psychoacoustic annoyance model is according to: (page 327) Zwicker, E. and Fastl, H. Second ed,
    Psychoacoustics, Facts and Models, 2nd ed. M.R. Schroeder. Springer-Verlag, Berlin, 1999.
    
    - This metric combines 4 psychoacoustic metrics to quantitatively describe annoyance:
    
       1) Loudness (sone) - calculated hereafter following ISO 532-1:2017
          type <help Loudness_ISO532_1> for more info
    
       2) Sharpness (acum) - calculated hereafter following DIN 45692:2009
          NOTE: uses DIN 45692 weighting function by default, please change code if
          the use of a different withgitng function is desired).
          type <help Sharpness_DIN45692_from_loudness>
    
       3) Roughness (asper) - calculated hereafter following Daniel & Weber model
          type <help Roughness_Daniel1997> for more info
    
       4) Fluctuation strength (vacil) - calculated hereafter following Osses et al. model
          type <help FluctuationStrength_Osses2016> for more info
    
    ##########################################################################
    
    INPUT:
      insig : array
      acoustic signal [1,nTimeSteps], monophonic (Pa)
    
      fs : integer
      sampling frequency (Hz) - preferible 48 kHz or 44.1 kHz (default by the authors and takes less time to compute)
    
      time_skip : integer
      skip start of the signal in <time_skip> seconds for statistics calculations
    
      LoudnessField : integer
      chose field for loudness calculation; free field = 0; diffuse field = 1;
      type <help Loudness_ISO532_1> for more info
    
      show : logical(boolean)
      optional parameter, display results of loudness, sharpness, roughness and fluctuation strength
      'false' (disable, default value) or 'true' (enable).
    
      showPA : logical(boolean)
      optional parameter, display results of psychoacoustic annoyance
      'false' (disable, default value) or 'true' (enable).
    
    OUTPUTS:
      OUT: struct
         * include results from the psychoacoustic annoyance
                ** InstantaneousPA: instantaneous quantity (unity) vs time
                ** ScalarPA : PA (scalar value) computed using the percentile values of each metric.
                              NOTE: if the signal's length is smaller than 2s, this is the only output as no time-varying PA is calculated
                ** time : time vector in seconds
                ** wfr : fluctuation strength and roughness weighting function (not squared)
                ** ws : sharpness and loudness weighting function (not squared)
    
                ** Statistics
                  *** PAmean : mean value of psychoacoustic annoyance (unit)
                  *** PAstd : standard deviation of instantaneous psychoacoustic annoyance (unit)
                  *** PAmax : maximum of instantaneous psychoacoustic annoyance (unit)
                  *** PAmin : minimum of instantaneous psychoacoustic annoyance (unit)
                  *** PAx : value exceeded x percent of the time
    
         * include structs with the results from the other metrics computed
           **  L : struct with Loudness results, type <help Loudness_ISO532_1> for more info
           **  S : struct with Sharpness, type <help Sharpness_DIN45692_from_loudness>
           **  R : strcut with roughness results, type <help Roughness_Daniel1997> for more info
           ** FS : struct with fluctuation strength results, type <help FluctuationStrength_Osses2016> for more info
    
    
     NOTE: 1) Input signals should be in pascal values or calibrated .wav files
    
           2) Fluctuation strength window has length of 2s. If the signal is less than 2s long, the FS calculation will be automatically
              changed to stationary (i.e. uses a window with length equal to signal's size) . in this case, no time-varying PA is available.
    
           3) Be aware that, because of item 2), if the signal is more than 2s long, the last 2 seconds of the input signal are LOST !!!!
    
           4) is a best practice to compute percentile values following a time_skip (s) after the signal's beginning to avoid misleading results caused by possible transient effects caused by digital filtering
    
           5) because of item 2), the PA(t) outputs are also 2s smaller, but the percentile values are calculed inside each function before this cut
    
           6) Loudness and sharpness have the same time vector, but roughness and FS differ because of their window lengths of 200ms and 2s, respectively.
              Therefore, in order to have the same time vector, after each respective metric calculation, the outputs are interpolated with respect to the loudness time vector and all cutted in the end
              to the final time corresponding to the FS metric
    
    Author: Gil Felix Greco, Braunschweig 04.03.2020 (updated 14.03.2023)
    Author: Gil Felix Greco, Braunschweig 16.02.2025 - introduced get_statistics function
    ###########################################################################
    """
    
    if percentiles is None:

        # --- WAV file interface ---
        if isinstance(insig, str):
            insig, fs = wav2sig(insig, fs, dBFS)

        elif fs is None:
            raise ValueError("If insig is not a filename, fs must be provided.")
        
        # Handle default arguments
        if show is None:
            show = 1 if showPA is None else 0
        
        if showPA is None:
            showPA = 1 if show is None else 0
        
        time_insig = np.arange(0, len(insig)) / fs  # time vector of the audio input, in seconds
        
        if time_insig[-1] < 2:
            print('\nWARNING: the signal\'s length is smaller than 2 seconds.\nDue to the minimum window size used for the fluctuation strength, the computation of a time-varying psychoacoustic annoyance is not possible !!!\nOnly scalar psychoacoustic annoyance number will be calculated for this signal!\n')
            method_FS = 0  # stationary method used for the fluctuation strength
        else:
            method_FS = 1  # time-varying method used for the fluctuation strength
        
        ## Loudness (according to ISO 531-1:2017)
        
        L = Loudness_ISO532_1(insig, fs,         # input signal and sampling freq.
                            LoudnessField,      # field; free field = 0; diffuse field = 1;
                            2,                  # method; stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2; 
                            time_skip,          # time_skip, in seconds for level (stationary signals) and statistics (stationary and time-varying signals) calculations
                            0)                  # show results, 'false' (disable, default value) or 'true' (enable)
        
        OUT = {}
        OUT['L'] = L  # output loudness results
        
        ## Sharpness (according to DIN 45692) from loudness input 

        S = Sharpness_DIN45692(SpecificLoudness=L['InstantaneousSpecificLoudness'],  # input (time-varying) specific loudness
                            weight_type='DIN45692',                           # type of weighting function used for sharpness calculation
                            time=L['time'],                            # time vector of the loudness calculation
                            time_skip=time_skip,                            # time_skip (second) for statistics calculation
                            show_sharpness=False)                                    # show sharpness results; true or false
        
        OUT['S'] = S  # output sharpness results
        
        ## Roughness (according to Daniel & Weber model)
        
        R = Roughness_Daniel1997(insig, fs,      # input signal and sampling freq.
                                time_skip,       # time_skip, in seconds for statistical calculations
                                0)               # show results, 'false' (disable, default value) or 'true' (enable)  
        
        OUT['R'] = R  # output roughness results
        
        ## Fluctuation strength (according to Osses et al. model)
        
        # the output signal will be 2s smaller due to the windown length
        FS = FluctuationStrength_Osses2016(insig, fs,     # input signal and sampling freq.
                                        method_FS,       # method, stationary analysis =0 - window size=length(insig), time_varying analysis - window size=2s
                                        time_skip,       # time_skip, in seconds for statistical calculations
                                        0)               # show results, 'false' (disable, default value) or 'true' (enable)  
        
        OUT['FS'] = FS  # output fluctuation strength results
        
        ## for signal with length smaller than 2 s, only scalar psychoacoustic annoyance can be computed
        
        if time_insig[-1] < 2:
            
            ## (scalar) psychoacoustic annoyance - computed directly from percentile values
            
            # sharpness and loudness influence
            if S['S5'] > 1.75:
                ws = (S['S5']-1.75)*(np.log10(L['N5']+10))/4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
            else:
                ws = 0
                
            # Handle inf and NaN values
            if np.isinf(ws) or np.isnan(ws):
                ws = 0  # replace inf and NaN with zeros
            
            # influence of roughness and fluctuation strength
            wfr = (2.18/(L['N5']**0.4))*(0.4*FS['FS5'] + 0.6*R['R5'])
            
            # Handle inf and NaN values
            if np.isinf(wfr) or np.isnan(wfr):
                wfr = 0  # replace inf and NaN with zeros
            
            # psychoacoustic annoyance
            PA_scalar = L['N5']*(1 + np.sqrt(ws**2 + wfr**2))
            
            ## ##################################################################
            #   output struct for time-varying signals
            #####################################################################
            
            # main output results
            OUT['ScalarPA'] = PA_scalar               # Annoyance calculated from the percentiles of each variable
            
        else:  # for signals larger than 2 seconds
            
            ## interpolation due to different output lengths of the different metrics
            
            ############################################################################################################
            # step 1) find idx related to the last time step of output from the fluctuation strength function (shorter output signal)
            # step 2) cut instaneous quantities from 1st idx to index related to the last time step of fluctuation strength
            ############################################################################################################
            
            LastTime = FS['time'][-1]          # take last time of the fluctuation strength
            
            # loudness
            idx_L = np.argmin(np.abs(L['time'] - LastTime))  # step 1) find idx
            
            L['time'] = L['time'][:idx_L+1]  # step 2) cut signal's end according to idx from step 1)
            L['InstantaneousLoudness'] = L['InstantaneousLoudness'][:idx_L+1]  # step 2)
            
            # sharpness
            idx_S = idx_L    # indice is the same as the loudness
            
            S['time'] = S['time'][:idx_S+1]  # step 2)
            S['InstantaneousSharpness'] = S['InstantaneousSharpness'][:idx_S+1]  # step 2)
            
            # roughness
            idx_R = np.argmin(np.abs(R['time'] - LastTime))  # step 1) find idx
            
            R['time'] = R['time'][:idx_R+1]  # step 2)
            R['InstantaneousRoughness'] = R['InstantaneousRoughness'][:idx_R+1]  # step 2)
            
            R['time'] = R['time'].T  # step 2)
            R['InstantaneousRoughness'] = R['InstantaneousRoughness'].T  # step 2)
            
            ############################################################################
            
            roughness = np.interp(L['time'], R['time'], R['InstantaneousRoughness'])  # interpolation to have the same time vector as loudness metric
            
            fluctuation = np.interp(L['time'], FS['time'], FS['InstantaneousFluctuationStrength'])  # interpolation to have the same time vector as loudness metric
            
            ## Time-varying psychoacoustic annoyance
            
            # declaring variables for pre allocating memory
            PA = np.zeros(len(L['time']))
            ws = np.zeros(len(L['time']))
            wfr = np.zeros(len(L['time']))
            
            for i in range(len(L['time'])):
                
                # sharpness influence
                if S['InstantaneousSharpness'][i] > 1.75:
                    ws[i] = (S['InstantaneousSharpness'][i]-1.75)*(np.log10(L['InstantaneousLoudness'][i]+10))/4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
                else:
                    ws[i] = 0
                
                # Handle inf and NaN values
                ws[np.isinf(ws) | np.isnan(ws)] = 0  # replace inf and NaN with zeros
                
                # influence of roughness and fluctuation strength
                wfr[i] = (2.18/(L['InstantaneousLoudness'][i]**0.4))*(0.4*fluctuation[i]+0.6*roughness[i])
                
                # Handle inf and NaN values
                wfr[np.isinf(wfr) | np.isnan(wfr)] = 0  # replace inf and NaN with zeros
                
                # psychoacoustic annoyance
                PA[i] = L['InstantaneousLoudness'][i]*(1 + np.sqrt(ws[i]**2 + wfr[i]**2))
            
            OUT['wfr'] = wfr     # OUTPUT: fluctuation strength and sharpness weighting function (not squared)
            OUT['ws'] = ws       # OUTPUT: sharpness and loudness weighting function (not squared)
            
            ## (scalar) psychoacoustic annoyance - computed directly from percentile values
            
            # sharpness influence
            if S['S5'] > 1.75:
                ws_scalar = (S['S5']-1.75)*(np.log10(L['N5']+10))/4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
            else:
                ws_scalar = 0
            
            # Handle inf and NaN values
            if np.isinf(ws_scalar) or np.isnan(ws_scalar):
                ws_scalar = 0  # replace inf and NaN with zeros
            
            # influence of roughness and fluctuation strength
            wfr_scalar = (2.18/(L['N5']**0.4))*(0.4*FS['FS5'] + 0.6*R['R5'])
            
            # Handle inf and NaN values
            if np.isinf(wfr_scalar) or np.isnan(wfr_scalar):
                wfr_scalar = 0  # replace inf and NaN with zeros
            
            # psychoacoustic annoyance
            PA_scalar = L['N5']*(1 + np.sqrt(ws_scalar**2 + wfr_scalar**2))
            
            ## ##################################################################
            #   output struct for time-varying signals
            #####################################################################
            
            # main output results
            OUT['InstantaneousPA'] = PA               # instantaneous Annoyance
            OUT['ScalarPA'] = PA_scalar               # Annoyance calculated from the percentiles of each variable
            OUT['time'] = L['time']                   # time vector
            
            #######################################################################
            # get statistics from Time-varying PA
            #####################################################################
            
            idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector
            
            metric_statistics = 'PsychoacousticAnnoyance_Zwicker1999'
            OUT_statistics = get_statistics(PA[idx:], metric_statistics)  # get statistics
            
            # copy fields of <OUT_statistics> struct into the <OUT> struct
            for fieldName in OUT_statistics.keys():  # Get all field names in OUT_statistics
                if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
                    OUT[fieldName] = OUT_statistics[fieldName]
            
            #####################################################
            
            ## plot
            
            if show == True or showPA == True:
                num_plots = 0
                if show:
                    num_plots += 5 # L, S, R, FS, K
                if showPA:
                    num_plots += 1 # PA

                if num_plots > 0:
                    # Determine grid size (e.g., 2 columns, calculate rows)
                    cols = 2
                    rows = int(np.ceil(num_plots / cols))
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows)) # Adjust figsize as needed
                    axes = axes.flatten() # Flatten the axes array for easy iteration

                    plot_idx = 0
                    if show == True:
                        il_plotter(OUT['time'], L['InstantaneousLoudness'], L['N5'], 'loudness', ax=axes[plot_idx])
                        plot_idx += 1
                        il_plotter(OUT['time'], S['InstantaneousSharpness'], S['S5'], 'sharpness', ax=axes[plot_idx])
                        plot_idx += 1
                        il_plotter(OUT['time'], roughness, R['R5'], 'roughness', ax=axes[plot_idx])
                        plot_idx += 1
                        il_plotter(OUT['time'], fluctuation, FS['FS5'], 'fluctuation', ax=axes[plot_idx])
                        plot_idx += 1
                    
                    if showPA == True:
                        il_plotter(OUT['time'], OUT['InstantaneousPA'], OUT['PA5'], 'annoyance', ax=axes[plot_idx])
                        plot_idx += 1

                    # Turn off any unused subplots if the grid has more axes than plots
                    for i in range(plot_idx, len(axes)):
                        fig.delaxes(axes[i]) # Remove empty subplots

                    plt.tight_layout() # Adjust layout to make space for suptitle
                    plt.show()
    
    elif percentiles is not None and insig is None:

        N, S, R, FS = percentiles

        # sharpness and loudness influence
        if S > 1.75:
            ws = (S-1.75)*(np.log10(N+10))/4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
        else:
            ws = 0
        
        # replace inf and NaN with zeros
        if np.isinf(ws) or np.isnan(ws):
            ws = 0
        
        # influence of roughness and fluctuation strength
        wfr = (2.18/(N**(0.4)))*(0.4*FS + 0.6*R)
        
        # replace inf and NaN with zeros
        if np.isinf(wfr) or np.isnan(wfr):
            wfr = 0
        
        # psychoacoustic annoyance
        PA_scalar = N*(1 + np.sqrt(ws**2 + wfr**2))
        
        OUT = PA_scalar  # Annoyance calculated from the percentiles of each variable

    return OUT

########################
### HELPER FUNCTIONS ###
########################

def il_plotter(time, Instantaneous, percentile, variable, ax): # Added 'ax' argument
    x_axis = 'Time, $t$ (s)'

    # Define plot properties using a dictionary for cleaner access
    plot_props = {
        'loudness': {'p': '$N', 'y_axis': 'Loudness, $N$ (sone)', 'title': 'Loudness'},
        'sharpness': {'p': '$S', 'y_axis': 'Sharpness, $S$ (acum)', 'title': 'Sharpness'},
        'roughness': {'p': '$R', 'y_axis': 'Roughness, $R$ (asper)', 'title': 'Roughness'},
        'fluctuation': {'p': 'FS$', 'y_axis': 'Fluctuation strength, $F$ (vacil)', 'title': 'Fluctuation strength'},
        'tonality': {'p': 'K$', 'y_axis': 'Aures tonality, $K$ (t.u.)', 'title': 'Aures tonality'},
        'annoyance': {'p': 'PA$', 'y_axis': 'Psychoacoustic annoyance, PA (-)', 'title': 'Psychoacoustic annoyance'},
    }

    props = plot_props.get(variable, {'p': '', 'y_axis': '', 'title': variable.capitalize()})

    # Ensure percentile is a scalar if it's a numpy array with one element
    if isinstance(percentile, np.ndarray):
        if percentile.size == 1:
            percentile_scalar = percentile.item()
        else:
            warnings.warn(f"Percentile for {variable} is an array with multiple elements. Using the first element.")
            percentile_scalar = percentile[0]
    else:
        percentile_scalar = percentile

    # Plotting on the provided axes object 'ax'
    ax.plot(time, Instantaneous, 'k', linewidth=0.5, label='_nolegend_')
    
    # Plotting the percentile line
    ax.plot(time, percentile_scalar * np.ones(len(time)), 'r--', linewidth=0.5,
            label=f'{props["p"]}_5={percentile_scalar:.2f}$')

    # Set legend, labels, grid, and background color for the specific subplot
    ax.legend(loc='upper right', fancybox=False, framealpha=1, edgecolor='black')
    ax.set_ylabel(props['y_axis'])
    ax.set_xlabel(x_axis)
    ax.set_xlim([time[0], time[-1]])
    ax.set_title(props['title']) # Set title for the individual subplot
    ax.grid(False)
    ax.set_facecolor('white')

check_which = 1

if __name__ == "__main__":
    if check_which == 0: # NO TEST
        print("metrics_annoyance.py")
    
    elif check_which == 1: # PsychoacousticAnnoyance_Di2016

        with_wavfile = 0 # 0 = no wavfile, 1 = wavfile
        type_wave = 5 # 0 = pure, 1 = AM, 2 = FM, 3 = noise, 4 = short, 5 = percentiles

        if type_wave == 0: # Pure Sine Wave
            fs = 48000
            duration = 5.0
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_sine = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_sine = PsychoacousticAnnoyance_Di2016(insig_sine, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_sine['PAmean'].item())

        if type_wave == 1: # Amplitude Modulated Sine Wave
            fs = 48000
            duration = 5.0
            carrier_freq = 1000
            mod_freq = 4
            amplitude = 0.1
            mod_depth = 1.0
            t = np.arange(0, duration, 1/fs)

            carrier = np.sin(2 * np.pi * carrier_freq * t)
            modulator = 0.5 * (1 + mod_depth * np.sin(2 * np.pi * mod_freq * t))
            insig_am = amplitude * modulator * carrier

            OUT_am = PsychoacousticAnnoyance_Di2016(insig_am, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_am['PAmean'].item())

        if type_wave == 2: # Frequency Modulated Sine Wave 
            fs = 48000
            duration = 5.0
            carrier_freq = 1000
            mod_freq = 70
            freq_deviation = 100
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)

            phase = 2 * np.pi * carrier_freq * t + (freq_deviation / mod_freq) * np.cos(2 * np.pi * mod_freq * t)
            insig_fm = amplitude * np.sin(phase)

            OUT_fm = PsychoacousticAnnoyance_Di2016(insig_fm, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_fm['PAmean'].item())

        if type_wave == 3: # Noise Signal
            
            fs = 48000
            duration = 5.0
            amplitude = 0.1
            insig_noise = amplitude * np.random.randn(int(fs * duration))

            OUT_noise = PsychoacousticAnnoyance_Di2016(insig_noise, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_noise['PAmean'].item())

        if type_wave == 4: # Short Signal

            fs = 48000
            duration = 1.5
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_short = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_short = PsychoacousticAnnoyance_Di2016(insig_short, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_short['ScalarPA'])

        if type_wave == 5: # Percentile-based computation
            
            N_val = 10.0  # Example Loudness (sone)
            S_val = 2.5   # Example Sharpness (acum)
            R_val = 0.8   # Example Roughness (asper)
            FS_val = 0.5  # Example Fluctuation Strength (vacil)
            K_val = 2.5   # Example Tonality (t.u.)

            print(f"Input Percentiles: N={N_val}, S={S_val}, R={R_val}, FS={FS_val}, K={K_val}")
            
            PA_percentile = PsychoacousticAnnoyance_Di2016(percentiles = (N_val, S_val, R_val, FS_val, K_val))
            
            print(f"Calculated Psychoacoustic Annoyance (from percentiles): {PA_percentile.item():.4f}")

    elif check_which == 2: # PsychoacousticAnnoyance_Zwicker1999

        with_wavfile = 0 # 0 = no wavfile, 1 = wavfile
        type_wave = 5 # 0 = pure, 1 = AM, 2 = FM, 3 = noise, 4 = short, 5 = percentiles

        if type_wave == 0: # Pure Sine Wave
            fs = 48000
            duration = 5.0
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_sine = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_sine = PsychoacousticAnnoyance_Zwicker1999(insig_sine, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_sine['PAmean'].item())

        if type_wave == 1: # Amplitude Modulated Sine Wave
            fs = 48000
            duration = 5.0
            carrier_freq = 1000
            mod_freq = 4
            amplitude = 0.1
            mod_depth = 1.0
            t = np.arange(0, duration, 1/fs)

            carrier = np.sin(2 * np.pi * carrier_freq * t)
            modulator = 0.5 * (1 + mod_depth * np.sin(2 * np.pi * mod_freq * t))
            insig_am = amplitude * modulator * carrier

            OUT_am = PsychoacousticAnnoyance_Zwicker1999(insig_am, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_am['PAmean'].item())

        if type_wave == 2: # Frequency Modulated Sine Wave
            fs = 48000
            duration = 5.0
            carrier_freq = 1000
            mod_freq = 70
            freq_deviation = 100
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)

            phase = 2 * np.pi * carrier_freq * t + (freq_deviation / mod_freq) * np.cos(2 * np.pi * mod_freq * t)
            insig_fm = amplitude * np.sin(phase)

            OUT_fm = PsychoacousticAnnoyance_Zwicker1999(insig_fm, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_fm['PAmean'].item())

        if type_wave == 3: # Noise Signal
            
            fs = 48000
            duration = 5.0
            amplitude = 0.1
            insig_noise = amplitude * np.random.randn(int(fs * duration))

            OUT_noise = PsychoacousticAnnoyance_Zwicker1999(insig_noise, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_noise['PAmean'].item())

        if type_wave == 4: # Short Signal

            fs = 48000
            duration = 1.5
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_short = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_short = PsychoacousticAnnoyance_Zwicker1999(insig_short, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
            print(OUT_short['ScalarPA'].item())

        if type_wave == 5: # Percentile-based computation
            
            N_val = 10.0  # Example Loudness (sone)
            S_val = 2.5   # Example Sharpness (acum)
            R_val = 0.8   # Example Roughness (asper)
            FS_val = 0.5  # Example Fluctuation Strength (vacil)

            print(f"Input Percentiles: N={N_val}, S={S_val}, R={R_val}, FS={FS_val}")
            
            PA_percentile = PsychoacousticAnnoyance_Zwicker1999(percentiles = (N_val, S_val, R_val, FS_val))
            
            print(f"Calculated Psychoacoustic Annoyance (from percentiles): {PA_percentile.item():.4f}")
