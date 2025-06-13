
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

def PsychoacousticAnnoyance_Di2016(insig, fs, LoudnessField=None, time_skip=None, showPA=None, show=None):
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
        
        # --- Start of Subplots Implementation ---
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
        # --- End of Subplots Implementation ---
    
    return OUT

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
        type_wave = 4 # 0 = pure, 1 = AM, 2 = FM, 3 = noise, 4 = short

        if type_wave == 0: # Pure Sine Wave
            fs = 48000
            duration = 3.0
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_sine = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_sine = PsychoacousticAnnoyance_Di2016(insig_sine, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)

        if type_wave == 1: # Amplitude Modulated Sine Wave
            fs = 48000
            duration = 3.0
            carrier_freq = 1000
            mod_freq = 4
            amplitude = 0.1
            mod_depth = 1.0
            t = np.arange(0, duration, 1/fs)

            carrier = np.sin(2 * np.pi * carrier_freq * t)
            modulator = 0.5 * (1 + mod_depth * np.sin(2 * np.pi * mod_freq * t))
            insig_am = amplitude * modulator * carrier

            OUT_am = PsychoacousticAnnoyance_Di2016(insig_am, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)

        if type_wave == 2: # Frequency Modulated Sine Wave # TODO: Check as looks like some metrics are off.
            fs = 48000
            duration = 3.0
            carrier_freq = 1000
            mod_freq = 70
            freq_deviation = 100
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)

            phase = 2 * np.pi * carrier_freq * t + (freq_deviation / mod_freq) * np.cos(2 * np.pi * mod_freq * t)
            insig_fm = amplitude * np.sin(phase)

            OUT_fm = PsychoacousticAnnoyance_Di2016(insig_fm, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)

        if type_wave == 3: # Noise Signal
            
            fs = 48000
            duration = 3.0
            amplitude = 0.1
            insig_noise = amplitude * np.random.randn(int(fs * duration))

            OUT_noise = PsychoacousticAnnoyance_Di2016(insig_noise, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)

        if type_wave == 4: # Short Signal

            fs = 48000
            duration = 1.5
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_short = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_short = PsychoacousticAnnoyance_Di2016(insig_short, fs, LoudnessField=0, time_skip=0.5, showPA=True, show=True)
