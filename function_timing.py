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
import timeit

from sound_metrics import *
from utilities import *
from metrics_loudness import Loudness_ISO532_1, EPNL_FAR_Part36
from metrics_sharpness import Sharpness_DIN45692
#from metrics_roughness import Roughness_Daniel1997
from metrics_fluctuation import FluctuationStrength_Osses2016
from metrics_tonality import Tonality_Aures1985
from metrics_annoyance import PsychoacousticAnnoyance_Di2016, PsychoacousticAnnoyance_Zwicker1999, PsychoacousticAnnoyance_More2010

from test_optimisations import Roughness_Daniel1997

def check_LOUDNESS_1():

        fs = 48_000                      # sampling rate expected by the OB-filter bank
        duration = 5.0                   # seconds
        f_tone = 1_000                   # 1-kHz pure tone
        desired_spl = 40                                   # target acoustic level

        a_rms_pa = 2e-5 * 10**(desired_spl / 20)           # RMS pressure in pascals
        a_peak_pa = a_rms_pa * np.sqrt(2)                  # peak
        fullscale_pa = 2e-5 * 10**(94 / 20)                # 94 dB SPL corresponds to |x| = 1
        amplitude = a_peak_pa / fullscale_pa               # peak value in ±1 full-scale units

        t = np.arange(0, duration, 1/fs)
        tone = amplitude * np.sin(2 * np.pi * f_tone * t)
        tone = tone.astype(np.float32)

        OUT = Loudness_ISO532_1(
            tone,
            fs,
            field=0,            # free-field
            method=2,           # time-varying
            time_skip=0,        # process whole signal
            show=False           # draw summary plots
        )

        print(f"Overall loudness (median of time-series): {np.median(OUT['InstantaneousLoudness']):.2f} sone")
        print(f"Loudness level (median):   {np.median(OUT['InstantaneousLoudnessLevel']):.2f} phon")

def check_LOUDNESS_2():
      
    fs = 48_000          # Hz – the filter bank is validated at 48 kHz
    dur_total   = 20.0            # s   – total length
    tone_freq   = 800.0           # Hz  – a typical fan/blade tone
    spl_broad   = 90.0            # dB  – peak broadband SPL
    spl_tone    = spl_broad - 20  # dB  – tone 20 dB weaker than the overall
    dBFS        = 94.0            # Full-scale reference used by library

    pref        = 2e-5                          # Pa
    FS_pa       = pref * 10**(dBFS/20)         # 1.0 digital  ↔  94 dB SPL (rms)

    t           = np.arange(0, dur_total, 1/fs)
    env         = np.sin(np.pi * t / dur_total)  # 0➜1➜0 half-cos envelope

    target_rms  = pref * 10**(spl_broad/20)      # Pa
    white_raw   = np.random.randn(len(t))
    white_raw  /= np.sqrt(np.mean(white_raw**2)) # unit RMS

    white       = env * white_raw * (target_rms/FS_pa)

    tone_rms    = pref * 10**(spl_tone/20)       # Pa
    tone        = (tone_rms/FS_pa) * np.sin(2*np.pi*tone_freq*t)

    flyover     = (white + tone).reshape(-1, 1)  # [n×1] as expected

    flyover = flyover.astype(np.float32)  # convert to float32

    OUT = EPNL_FAR_Part36(
            insig     = flyover,
            fs        = fs,
            method    = 1,    # audio-signal input
            dt        = 0.5,  # 0.5-s analysis blocks (Part 36 default)
            threshold = 10,   # 10 dB duration threshold
            show      = False  # let the function draw its figures
        )

    print(f"EPNL of validation clip: {OUT['EPNL']} EPNdB")

def check_SHARPNESS():
      
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

        OUT = Sharpness_DIN45692(
                insig=insig,
                fs=fs,
                weight_type='aures', # Example: 'DIN45692', 'bismarck', or 'aures'
                LoudnessField=0,        # 0 for free field, 1 for diffuse field
                LoudnessMethod=2,       # 1 for stationary, 2 for time-varying
                time_skip=0.5,          # Skip first 0.5 seconds for statistics (if LoudnessMethod=2)
                show_sharpness=False,    # Display sharpness results
                show_loudness=False     # Display loudness results
            )

        print(f"  Mean Sharpness (Smean): {OUT['Smean'][0]:.3f} acum")

def check_ROUGHNESS():

        fs = 48_000
        f_mod = 70
        f_carrier = 1_000.0
        L = 60 # dB SPL

        p_rms = 20e-6 * 10**(L / 20)
        A = p_rms * np.sqrt(2)

        t = np.arange(0.0, 5, 1 / fs)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * f_mod * t))
        signal = A * envelope * np.sin(2 * np.pi * f_carrier * t)
        insig = signal.astype(np.float32)

        OUT = Roughness_Daniel1997(insig, fs, time_skip=0.0, show=False)

        print(f"  Mean roughness  : {OUT['Rmean']} asper")

def check_FLUCTUATION():
      
        fs = 48000
        duration = 5.0
        t_py = np.linspace(0, duration, int(fs * duration), endpoint=False)

        f_carrier = 1000  # Hz
        f_mod = 4  # Hz
        mod_index = 1.0

        signal_py = (1 + mod_index * np.sin(2 * np.pi * f_mod * t_py)) * np.sin(2 * np.pi * f_carrier * t_py)

        # Set to 60 dB SPL (0.02 Pa RMS)
        rms_target = 0.02
        rms_current_py = np.sqrt(np.mean(signal_py**2))
        signal_py = signal_py * (rms_target / rms_current_py)
        signal_py = signal_py.astype(np.float32)

        OUT_py = FluctuationStrength_Osses2016(signal_py, fs, method=1, time_skip=2.0, show=False)
        
        print(f'  Mean: {OUT_py["FSmean"][0]:.6f}')

def check_TONALITY():
      
        fs = 48000              # Sampling rate in Hz
        duration = 5.0          # Duration in seconds
        f0 = 1000               # Frequency of pure tone (Hz)
        Lp = 60                 # Desired sound pressure level (dB SPL)
        pref = 20e-6            # Reference pressure in Pa

        # Create time vector
        t = np.arange(0, duration, 1/fs)

        # Generate sine wave with RMS level corresponding to 60 dB SPL
        rms_target = pref * 10**(Lp / 20)
        amp = rms_target * np.sqrt(2)
        signal = amp * np.sin(2 * np.pi * f0 * t)

        # Compute tonality using Aures 1985 model
        result = Tonality_Aures1985(signal, fs=fs, LoudnessField=0, time_skip=0.5, show=False)

        print(f"InstantaneousTonality: {np.mean(result['InstantaneousTonality']):.3g}")

def check_PA_1():
        type_wave = 5 # 0 = pure, 1 = AM, 2 = FM, 3 = noise, 4 = short, 5 = percentiles

        if type_wave == 0: # Pure Sine Wave
            fs = 48000
            duration = 5.0
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_sine = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_sine = PsychoacousticAnnoyance_Di2016(insig_sine, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
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

            OUT_am = PsychoacousticAnnoyance_Di2016(insig_am, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
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

            OUT_fm = PsychoacousticAnnoyance_Di2016(insig_fm, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_fm['PAmean'].item())

        if type_wave == 3: # Noise Signal
            
            fs = 48000
            duration = 5.0
            amplitude = 0.1
            insig_noise = amplitude * np.random.randn(int(fs * duration))

            OUT_noise = PsychoacousticAnnoyance_Di2016(insig_noise, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_noise['PAmean'].item())

        if type_wave == 4: # Short Signal

            fs = 48000
            duration = 1.5
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_short = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_short = PsychoacousticAnnoyance_Di2016(insig_short, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_short['ScalarPA'])

        if type_wave == 5: # Percentile-based computation
            
            N_val = 1  # Example Loudness (sone)
            S_val = 1   # Example Sharpness (acum)
            R_val = 1   # Example Roughness (asper)
            FS_val = 1  # Example Fluctuation Strength (vacil)
            K_val = 1   # Example Tonality (t.u.)

            print(f"Input Percentiles: N={N_val}, S={S_val}, R={R_val}, FS={FS_val}, K={K_val}")
            
            PA_percentile = PsychoacousticAnnoyance_Di2016(percentiles = (N_val, S_val, R_val, FS_val, K_val))
            
            print(f"Calculated Psychoacoustic Annoyance (from percentiles): {PA_percentile.item():.4f}")

def check_PA_2():
      
        type_wave = 4 # 0 = pure, 1 = AM, 2 = FM, 3 = noise, 4 = short, 5 = percentiles

        if type_wave == 0: # Pure Sine Wave
            fs = 48000
            duration = 5.0
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_sine = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_sine = PsychoacousticAnnoyance_Zwicker1999(insig_sine, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
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

            OUT_am = PsychoacousticAnnoyance_Zwicker1999(insig_am, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
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

            OUT_fm = PsychoacousticAnnoyance_Zwicker1999(insig_fm, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_fm['PAmean'].item())

        if type_wave == 3: # Noise Signal
            
            fs = 48000
            duration = 5.0
            amplitude = 0.1
            insig_noise = amplitude * np.random.randn(int(fs * duration))

            OUT_noise = PsychoacousticAnnoyance_Zwicker1999(insig_noise, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_noise['PAmean'].item())

        if type_wave == 4: # Short Signal

            fs = 48000
            duration = 1.5
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_short = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_short = PsychoacousticAnnoyance_Zwicker1999(insig_short, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_short['ScalarPA'].item())

        if type_wave == 5: # Percentile-based computation
            
            N_val = 1  # Example Loudness (sone)
            S_val = 1   # Example Sharpness (acum)
            R_val = 1   # Example Roughness (asper)
            FS_val = 1  # Example Fluctuation Strength (vacil)

            print(f"Input Percentiles: N={N_val}, S={S_val}, R={R_val}, FS={FS_val}")
            
            PA_percentile = PsychoacousticAnnoyance_Zwicker1999(percentiles = (N_val, S_val, R_val, FS_val))
            
            print(f"Calculated Psychoacoustic Annoyance (from percentiles): {PA_percentile.item():.4f}")

def check_PA_3():
     
        type_wave = 3 # 0 = pure, 1 = AM, 2 = FM, 3 = noise, 4 = short, 5 = percentiles

        if type_wave == 0: # Pure Sine Wave
            fs = 48000
            duration = 5.0
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_sine = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_sine = PsychoacousticAnnoyance_More2010(insig_sine, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
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

            OUT_am = PsychoacousticAnnoyance_More2010(insig_am, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
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

            OUT_fm = PsychoacousticAnnoyance_More2010(insig_fm, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_fm['PAmean'].item())

        if type_wave == 3: # Noise Signal
            
            fs = 48000
            duration = 5.0
            amplitude = 0.1
            insig_noise = amplitude * np.random.randn(int(fs * duration))

            OUT_noise = PsychoacousticAnnoyance_More2010(insig_noise, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_noise['PAmean'].item())

        if type_wave == 4: # Short Signal

            fs = 48000
            duration = 1.5
            frequency = 1000
            amplitude = 0.1
            t = np.arange(0, duration, 1/fs)
            insig_short = amplitude * np.sin(2 * np.pi * frequency * t)

            OUT_short = PsychoacousticAnnoyance_More2010(insig_short, fs, LoudnessField=0, time_skip=0.5, showPA=False, show=False)
            print(OUT_short['ScalarPA'].item())

        if type_wave == 5: # Percentile-based computation
            
            N_val = 1  # Example Loudness (sone)
            S_val = 1   # Example Sharpness (acum)
            R_val = 1   # Example Roughness (asper)
            FS_val = 1  # Example Fluctuation Strength (vacil)
            K_val = 1   # Example Tonality (t.u.)

            print(f"Input Percentiles: N={N_val}, S={S_val}, R={R_val}, FS={FS_val}")
            
            PA_percentile = PsychoacousticAnnoyance_More2010(percentiles = (N_val, S_val, R_val, FS_val, K_val))
            
            print(f"Calculated Psychoacoustic Annoyance (from percentiles): {PA_percentile.item():.4f}")


check_which = 3

if __name__ == "__main__":

    if check_which == 1.1:
          
          check_LOUDNESS_1()

          best = min(timeit.repeat(stmt=check_LOUDNESS_1, repeat=10, number=1))
          print(f"Best run: {best:.3f} s")

    if check_which == 1.2:
          
          check_LOUDNESS_2()

          best = min(timeit.repeat(stmt=check_LOUDNESS_2, repeat=10, number=1))
          print(f"Best run: {best:.3f} s")

    if check_which == 2:
          
          check_SHARPNESS()

          best = min(timeit.repeat(stmt=check_SHARPNESS, repeat=10, number=1))
          print(f"Best run: {best:.3f} s")

    if check_which == 3:
          
          check_ROUGHNESS()

          best = min(timeit.repeat(stmt=check_ROUGHNESS, repeat=10, number=1))
          print(f"Best run: {best:.3f} s")

    if check_which == 4:
          
          check_FLUCTUATION()

          best = min(timeit.repeat(stmt=check_FLUCTUATION, repeat=10, number=1))
          print(f"Best run: {best:.3f} s")

    if check_which == 5:
          
          check_TONALITY()

          best = min(timeit.repeat(stmt=check_TONALITY, repeat=10, number=1))
          print(f"Best run: {best:.3f} s")

    if check_which == 6.1:
          
          check_PA_1()

          best = min(timeit.repeat(stmt=check_PA_1, repeat=5, number=1))
          print(f"Best run: {best:.3f} s")

    if check_which == 6.2:
            
            check_PA_2()
    
            best = min(timeit.repeat(stmt=check_PA_2, repeat=5, number=1))
            print(f"Best run: {best:.3f} s")

    if check_which == 6.3:
            
            check_PA_3()
    
            best = min(timeit.repeat(stmt=check_PA_3, repeat=5, number=1))
            print(f"Best run: {best:.3f} s")

