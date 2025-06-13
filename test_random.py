
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

def PsychoacousticAnnoyance_More2010(insig=None, fs=None, LoudnessField=None, time_skip=None, showPA=None, show=None, dBFS = 94, percentiles=None):
    """
    function OUT = PsychoacousticAnnoyance_More2010(insig,fs,LoudnessField,time_skip,showPA,show)
    
    This function calculates the More's modified psychoacoustic annoyance 
    model from an input acoustic signal
    
    The modified psychoacoustic annoyance model is according to: (page 201)
    [1] More, Shashikant. Aircraft noise characteristics and metrics. 
        PhD Thesis, Purdue University, 2010
    
    - This metric combines 5 psychoacoustic metrics to quantitatively describe annoyance:
    
       1) Loudness, N (sone) - calculated hereafter following ISO 532-1:2017
          type <help Loudness_ISO532_1> for more info
    
       2) Sharpness, S (acum) - calculated hereafter following DIN 45692:2009
          NOTE: uses DIN 45692 weighting function by default, please change code if
          the use of a different withgitng function is desired).
          type <help Sharpness_DIN45692_from_loudness>
    
       3) Roughness, R (asper) - calculated hereafter following Daniel & Weber model
          type <help Roughness_Daniel1997> for more info
    
       4) Fluctuation strength, FS (vacil) - calculated hereafter following 
          Osses et al. model, type <help FluctuationStrength_Osses2016> for more info
    
       5) Tonality, K (t.u.) - calculated hereafter following Aures' model
          type <help Tonality_Aures1985> for more info
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    INPUT:
      insig : array
      acoustic signal [1,nTimeSteps], monophonic (Pa)
    
      fs : integer
      sampling frequency (Hz) - preferible 48 kHz or 44.1 kHz (default by the authors and takes less time to compute)
    
      time_skip : integer
      skip start of the signal in <time_skip> seconds for statistics calculations
    
      LoudnessField : integer
      chose field for loudness calculation; free field = 0; diffuse field = 1; (used in the loudness and tonality codes)
      type <help Loudness_ISO532_1> for more info
    
      show : logical(boolean)
      optional parameter, display results of loudness, sharpness, roughness, fluctuation strength and tonality
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
                ** wt : tonality and loudness weighting function (not squared)
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
           **  K : struct with tonality results, type <help Tonality_Aures1985> for more info
    
     NOTE: 1) Input signals should be in pascal values or calibrated .wav files
    
           2) Fluctuation strength window has length of 2s. If the signal is 
              less than 2s long, the FS calculation will be automatically
              changed to stationary (i.e. uses a window with length equal to 
              signal's size). in this case, no time-varying PA is available.
    
           3) Be aware that, because of item 2), if the signal is more than 2s 
              long, the last 2 seconds of the input signal are LOST !!!!
    
           4) is a best practice to compute percentile values following a 
              time_skip (s) after the signal's beginning to avoid misleading 
              results caused by possible transient effects caused by digital filtering
    
           5) because of item 2), the PA(t) outputs are also 2s smaller, but 
              the percentile values are calculed inside each function before this cut
    
           6) Loudness and sharpness have the same time vector, but roughness,
              FS and tonality differ because of their window lengths.
              Therefore, in order to have the same time vector, after each 
              respective metric calculation, the outputs are interpolated 
              with respect to the loudness time vector and all cutted in the 
              end to the final time corresponding to the FS metric
    
    Author: Gil Felix Greco, Braunschweig 05.04.2023
    Author: Gil Felix Greco, Braunschweig 16.02.2025 - introduced get_statistics function
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    if percentiles is None:

        # --- WAV file interface ---
        if isinstance(insig, str):
            insig, fs = wav2sig(insig, fs, dBFS)

        elif fs is None:
            raise ValueError("If insig is not a filename, fs must be provided.")
        
        if show is None:
            show = 1 if len([]) == 0 else 0  # Equivalent to nargout == 0 check
        
        if showPA is None:
            showPA = 1 if len([]) == 0 else 0  # Equivalent to nargout == 0 check
        
        time_insig = np.arange(len(insig)) / fs  # time vector of the audio input, in seconds
        
        if time_insig[-1] < 2:
            print('\nWARNING: the signal\'s length is smaller than 2 seconds.\nDue to the minimum window size used for the fluctuation strength, the computation of a time-varying psychoacoustic annoyance is not possible !!!\nOnly scalar psychoacoustic annoyance number will be calculated for this signal!\n')
            method_FS = 0  # stationary method used for the fluctuation strength
        else:
            method_FS = 1  # time-varying method used for the fluctuation strength
        
        ## modified PA model constants (Ref. [1] pg. 204)
        
        gamma_0 = -0.16
        gamma_1 = 11.48
        gamma_2 = 0.84
        gamma_3 = 1.25
        gamma_4 = 0.29
        gamma_5 = 5.49
        
        ## Loudness (according to ISO 531-1:2017)
        
        L = Loudness_ISO532_1(insig, fs,                    # input signal and sampling freq.
                            LoudnessField,                # field; free field = 0; diffuse field = 1;
                            2,                            # method; stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2; 
                            time_skip,                    # time_skip, in seconds for level (stationary signals) and statistics (stationary and time-varying signals) calculations
                            0)                            # show results, 'false' (disable, default value) or 'true' (enable)
        
        OUT = {}
        OUT['L'] = L  # output loudness results
        
        ## Sharpness (according to DIN 45692) from loudness input 
        
        S = Sharpness_DIN45692(SpecificLoudness=L['InstantaneousSpecificLoudness'],  # input (time-varying) specific loudness
                            weight_type='DIN45692',                          # type of weighting function used for sharpness calculation 
                            time=L['time'],                           # time vector of the loudness calculation
                            time_skip=time_skip,                           # time_skip (second) for statistics calculation
                            show_sharpness=False)                                   # show sharpness results; true or false
        
        OUT['S'] = S  # output sharpness results
        
        ## Roughness (according to Daniel & Weber model)
        
        R = Roughness_Daniel1997(insig, fs,        # input signal and sampling freq.
                                time_skip,        # time_skip, in seconds for statistical calculations
                                0)                # show results, 'false' (disable, default value) or 'true' (enable)  
        
        OUT['R'] = R  # output roughness results
        
        ## Fluctuation strength (according to Osses et al. model)
        
        # the output signal will be 2s smaller due to the windown length
        FS = FluctuationStrength_Osses2016(insig, fs,      # input signal and sampling freq.
                                        method_FS,       # method, stationary analysis =0 - window size=length(insig), time_varying analysis - window size=2s
                                        time_skip,       # time_skip, in seconds for statistical calculations
                                        0)               # show results, 'false' (disable, default value) or 'true' (enable)  
        
        OUT['FS'] = FS  # output fluctuation strength results
        
        ## Tonality (according to Aures' model)
        
        K = Tonality_Aures1985(insig, fs,          # input signal and sampling freq.
                            LoudnessField,      # field for loudness calculation; free field = 0; diffuse field = 1;
                            0,                  # time_skip, in seconds for level (stationary signals) and statistics (stationary and time-varying signals) calculations
                            0)                  # show results, 'false' (disable, default value) or 'true' (enable)
        
        OUT['K'] = K  # output fluctuation strength results
        
        ## for signal with length smaller than 2 s, only scalar psychoacoustic annoyance can be computed
        
        if time_insig[-1] < 2:
            
            ## (scalar) psychoacoustic annoyance - computed directly from percentile values
            
            # sharpness influence
            if S['S5'] > 1.75:
                ws = (S['S5'] - 1.75) * (np.log10(L['N5'] + 10)) / 4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
            else:
                ws = 0
            
            # replace inf and NaN with zeros
            if np.isinf(ws) or np.isnan(ws):
                ws = 0
            
            # influence of roughness and fluctuation strength
            wfr = (2.18 / (L['N5'] ** 0.4)) * (0.4 * FS['FS5'] + 0.6 * R['R5'])
            
            # replace inf and NaN with zeros
            if np.isinf(wfr) or np.isnan(wfr):
                wfr = 0
            
            # Tonality influence
            wt = abs((1 - np.exp(-gamma_4 * L['N5'])) ** 2 * (1 - np.exp(-gamma_5 * K['K5'])) ** 2)
            
            # replace inf and NaN with zeros
            if np.isinf(wt) or np.isnan(wt):
                wt = 0
            
            # More's modified psychoacoustic annoyance
            PA_scalar = abs(L['N5'] * (1 + np.sqrt(gamma_0 + (gamma_1 * ws ** 2) + (gamma_2 * wfr ** 2) + (gamma_3 * wt))))
            

            OUT['ScalarPA'] = PA_scalar  # Annoyance calculated from the percentiles of each variable
            
        else:  # for signals larger than 2 seconds
            
            ## interpolation due to different output lengths of the different metrics
            
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # step 1) find idx related to the last time step of output from the 
            #         fluctuation strength function (shorter output signal)
            # step 2) cut instaneous quantities - only 1st idx till index related 
            #         to the last time step of fluctuation strength remain
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            LastTime = FS['time'][-1]  # take last time of the fluctuation strength
            
            # loudness
            idx_L = np.argmin(np.abs(L['time'] - LastTime))  # step 1) find idx
            
            L['time'] = L['time'][:idx_L + 1]  # step 2) cut signal's end according to idx from step 1)
            L['InstantaneousLoudness'] = L['InstantaneousLoudness'][:idx_L + 1]  # step 2)
            
            # sharpness
            idx_S = idx_L  # indice is the same as the loudness
            
            S['time'] = S['time'][:idx_S + 1]  # step 2)
            S['InstantaneousSharpness'] = S['InstantaneousSharpness'][:idx_S + 1]  # step 2)
            
            # roughness
            idx_R = np.argmin(np.abs(R['time'] - LastTime))  # step 1) find idx
            
            R['time'] = R['time'][:idx_R + 1]  # step 2)
            R['InstantaneousRoughness'] = R['InstantaneousRoughness'][:idx_R + 1]  # step 2)
            
            R['time'] = R['time'].T  # step 2)
            R['InstantaneousRoughness'] = R['InstantaneousRoughness'].T  # step 2)
            
            # tonality
            idx_K = np.argmin(np.abs(K['time'] - LastTime))  # step 1) find idx
            
            K['time'] = K['time'][:idx_K + 1]  # step 2) cut signal's end according to idx from step 1)
            K['InstantaneousTonality'] = K['InstantaneousTonality'][:idx_K + 1]  # step 2)
            
            del idx_R, idx_L, idx_S, idx_K
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            roughness = interp1d(R['time'], R['InstantaneousRoughness'], kind='cubic')(L['time'])  # interpolation to have the same time vector as loudness metric
            
            fluctuation = interp1d(FS['time'], FS['InstantaneousFluctuationStrength'], kind='cubic')(L['time'])  # interpolation to have the same time vector as loudness metric
            
            tonality = interp1d(K['time'], K['InstantaneousTonality'], kind='cubic')(L['time'])  # interpolation to have the same time vector as loudness metric
            
            ## Time-varying psychoacoustic annoyance
            
            # declaring variables for pre allocating memory
            PA = np.zeros(len(L['time']))
            ws = np.zeros(len(L['time']))
            wfr = np.zeros(len(L['time']))
            wt = np.zeros(len(L['time']))
            
            for i in range(len(L['time'])):
                
                # sharpness influence
                if S['InstantaneousSharpness'][i] > 1.75:
                    ws[i] = (S['InstantaneousSharpness'][i] - 1.75) * (np.log10(L['InstantaneousLoudness'][i] + 10)) / 4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
                else:
                    ws[i] = 0
                
                # replace inf and NaN with zeros
                ws[np.isinf(ws) | np.isnan(ws)] = 0
                
                # influence of roughness and fluctuation strength
                wfr[i] = (2.18 / (L['InstantaneousLoudness'][i] ** 0.4)) * (0.4 * fluctuation[i] + 0.6 * roughness[i])
                
                # replace inf and NaN with zeros
                wfr[np.isinf(wfr) | np.isnan(wfr)] = 0
                
                # Tonality influence
                wt[i] = abs((1 - np.exp(-gamma_4 * L['InstantaneousLoudness'][i])) ** 2 * (1 - np.exp(-gamma_5 * tonality[i])) ** 2)
                
                # replace inf and NaN with zeros
                wt[np.isinf(wt) | np.isnan(wt)] = 0
                
                # More's modified psychoacoustic annoyance
                PA[i] = abs(L['InstantaneousLoudness'][i] * (1 + np.sqrt(gamma_0 + (gamma_1 * ws[i] ** 2) + (gamma_2 * wfr[i] ** 2) + (gamma_3 * wt[i]))))
            
            OUT['wt'] = np.sqrt(wt)  # OUTPUT: tonality and loudness weighting function (not squared)
            OUT['wfr'] = wfr         # OUTPUT: fluctuation strength and sharpness weighting function (not squared)
            OUT['ws'] = ws           # OUTPUT: sharpness and loudness weighting function (not squared)
            
            ## (scalar) psychoacoustic annoyance - computed directly from percentile values
            
            # sharpness influence
            if S['S5'] > 1.75:
                ws = (S['S5'] - 1.75) * (np.log10(L['N5'] + 10)) / 4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
            else:
                ws = 0
            
            # replace inf and NaN with zeros
            if np.isinf(ws) or np.isnan(ws):
                ws = 0
            
            # influence of roughness and fluctuation strength
            wfr = (2.18 / (L['N5'] ** 0.4)) * (0.4 * FS['FS5'] + 0.6 * R['R5'])
            
            # replace inf and NaN with zeros
            if np.isinf(wfr) or np.isnan(wfr):
                wfr = 0
            
            # Tonality influence
            wt = abs((1 - np.exp(-gamma_4 * L['N5'])) ** 2 * (1 - np.exp(-gamma_5 * K['K5'])) ** 2)
            
            # replace inf and NaN with zeros
            if np.isinf(wt) or np.isnan(wt):
                wt = 0
            
            # More's modified psychoacoustic annoyance
            PA_scalar = abs(L['N5'] * (1 + np.sqrt(gamma_0 + (gamma_1 * ws ** 2) + (gamma_2 * wfr ** 2) + (gamma_3 * wt))))
            
            ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Output struct for time-varying signals
            
            # main output results
            OUT['InstantaneousPA'] = PA       # instantaneous Annoyance
            OUT['ScalarPA'] = PA_scalar       # Annoyance calculated from the percentiles of each variable
            OUT['time'] = L['time']           # time vector
            
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # get statistics from Time-varying PA
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector
            
            metric_statistics = 'PsychoacousticAnnoyance_More2010'
            OUT_statistics = get_statistics(PA[idx:], metric_statistics)  # get statistics
            
            # copy fields of <OUT_statistics> struct into the <OUT> struct
            fields_OUT_statistics = list(OUT_statistics.keys())  # Get all field names in OUT_statistics
            
            for i in range(len(fields_OUT_statistics)):
                fieldName = fields_OUT_statistics[i]
                if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
                    OUT[fieldName] = OUT_statistics[fieldName]
            
            del OUT_statistics, metric_statistics, fields_OUT_statistics, fieldName
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
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
        
        if percentiles is not None and insig is None:
            
            N, S, R, FS, K = percentiles

            # modified PA model constants (Ref. [1] pg. 204)
            gamma_0 = -0.16
            gamma_1 = 11.48
            gamma_2 = 0.84
            gamma_3 = 1.25
            gamma_4 = 0.29
            gamma_5 = 5.49

            # (scalar) modified psychoacoustic annoyance - computed directly from percentile values

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
            wt = abs((1 - np.exp(-gamma_4 * N)) ** 2 * (1 - np.exp(-gamma_5 * K)) ** 2)

            if np.isinf(wt) or np.isnan(wt):  # replace inf and NaN with zeros
                wt = 0

            # More's modified psychoacoustic annoyance
            PA_scalar = abs(N * (1 + np.sqrt(gamma_0 + (gamma_1 * ws ** 2) + (gamma_2 * wfr ** 2) + (gamma_3 * wt))))

            OUT = PA_scalar  # Annoyance calculated from the percentiles of each variable

    return OUT


def il_plotter(time, Instantaneous, percentile, variable, ax):
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

# end plotter function

