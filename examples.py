from __future__ import annotations
from typing import Dict, Any, Tuple

from sound_metrics import *
from utilities import *
from metrics_loudness import Loudness_ISO532_1, EPNL_FAR_Part36
from metrics_shm_loudness_ecma_fast import shm_loudness_ecma_fast_wrapper as shm_loudness_ecma_wrapper
# from metrics_shm_loudness_ecma import shm_loudness_ecma_wrapper
from metrics_shm_roughness_ecma import shm_roughness_ecma_wrapper
from metrics_shm_tonality_ecma import shm_tonality_ecma_wrapper
from metrics_sharpness import Sharpness_DIN45692
from metrics_roughness import Roughness_Daniel1997   
from metrics_fluctuation import FluctuationStrength_Osses2016
from metrics_tonality import Tonality_Aures1985
from metrics_annoyance import PsychoacousticAnnoyance_Zwicker1999, PsychoacousticAnnoyance_More2010, PsychoacousticAnnoyance_Di2016
from pathlib import Path

# Sound files directory relative to this examples module
SOUND_DIR = Path(__file__).resolve().parent / "sound_files"

def ex_Loudness_ISO532_1():

    L_stationary = Loudness_ISO532_1(insig = str(SOUND_DIR / "RefSignal_Loudness_ISO532_1.wav"),
                                         field = 0,   # field; free field = 0; diffuse field = 1;
                                         method = 1,  # method; stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2; 
                                         time_skip = 0.5, # time_skip, in seconds for level (stationary signals) and statistics (stationary and time-varying signals) calculations
                                         show = 1);     # show results, 'false' (disable, default value) or 'true' (enable)

    L_time_varying = Loudness_ISO532_1(insig = str(SOUND_DIR / "RefSignal_Loudness_ISO532_1.wav"),
                                         field = 0,   # field; free field = 0; diffuse field = 1;
                                         method = 2,  # method; stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2; 
                                         time_skip = 0.5, # time_skip, in seconds for level (stationary signals) and statistics (stationary and time-varying signals) calculations
                                         show = 1);     # show results, 'false' (disable, default value) or 'true' (enable)

    return L_stationary, L_time_varying

def ex_EPNL_FAR_Part36():

    soundfile = str(SOUND_DIR / "ExSignal_A320_auralized_departure_104dBFS.wav")
    dBFS_soundfile = 104 # a priori knowledge

    raw_insig, fs = wav2sig(soundfile, dBFS=dBFS_soundfile)
    cal =  20E-6*10**(dBFS_soundfile/20) # calibration factor

    insig = raw_insig * cal

    EPNL_1 = EPNL_FAR_Part36(insig = insig,
                             fs = fs, #input signal and sampling freq.
                             method = 1, # method = 0, insig is a SPL[nTime,nFreq] matrix; method = 1, insig is a sound file
                             dt = 0.5, # time-step in which the third-octave SPLs are averaged, in seconds.
                             threshold = 10, # threshold value used to calculate the PNLT decay from PNLTM during the calculation of the duration correction
                             show = 1)
    
    input = EPNL_1['SPL_TOB_spectra']
    
    EPNL_2 = EPNL_FAR_Part36(insig = input,
                             method = 0,  # method = 0, insig is a SPL[nTime,nFreq] matrix; method = 1, insig is a sound file
                             dt = 0.5,  # time-step in which the third-octave SPLs are averaged, in seconds.
                             threshold = 10,  # threshold value used to calculate the PNLT decay from PNLTM during the calculation of the duration correction
                             show = 1)  # show results, 'false' (disable, default value) or 'true' (enable)

    return EPNL_1, EPNL_2

def ex_Sharpness_DIN45692():

    soundfile = str(SOUND_DIR / "RefSignal_Sharpness_DIN45692.wav")
    raw_insig, fs = wav2sig(soundfile)
    time = len(raw_insig)/fs

    lvl_cal_signal = 60
    sig_rms = np.sqrt(np.sum(raw_insig**2)/len(raw_insig))

    dBFS_in = lvl_cal_signal-20*np.log10(sig_rms) # difference between target and actual full-scale value
    dBFS_out = 94 # dB full scale convention in SQAT: amplitude of 1 = 1 Pa, or 94 dB SPL
    dB_correction = dBFS_in - dBFS_out

    insig_cal = raw_insig * 10**(dB_correction/20)

    S_stationary = Sharpness_DIN45692(insig = insig_cal, # input signal, 1D array
                                      fs = fs, # input signal and sampling frequency
                                      weight_type = 'DIN45692', # Weight_Type, type of weighting function used for sharpness calculation
                                      LoudnessField = 0, # field used for loudness calculation; free field = 0; diffuse field = 1;
                                      LoudnessMethod = 1, # method used for loudness calculation: stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2;     
                                      time_skip = 0, # time_skip (second) for statistics calculation
                                      show_sharpness = 0, # show sharpness results
                                      show_loudness = 0) # show loudness results
    
    L = Loudness_ISO532_1(insig = insig_cal, # input signal, 1D array
                          fs = fs,
                          field = 0,
                          method = 1,
                          time_skip = 0.5,
                          show = 0)
    
    L2 = Loudness_ISO532_1(insig = insig_cal, # input signal, 1D array
                          fs = fs,
                          field = 0,
                          method = 2,
                          time_skip = 0.5,
                          show = 0)

    S_loudness_stat = Sharpness_DIN45692(SpecificLoudness = L['SpecificLoudness'],
                                    fs = fs,
                                    LoudnessMethod = 1,
                                    time = time,
                                    weight_type = 'DIN45692',
                                    show_sharpness = 0,
                                    show_loudness = 0)
    
    
    S_loudness_time = Sharpness_DIN45692(SpecificLoudness = L2['InstantaneousSpecificLoudness'],
                                    weight_type = 'DIN45692',
                                    time = L2['time'],
                                    time_skip = 0.5,
                                    LoudnessMethod = 2,
                                    show_sharpness = 0,
                                    show_loudness = 0)

    
    print(f"Calculated Stationary Sharpness: {S_stationary['Sharpness']:.3f} acum")
    print(f"Calculated Stationary Sharpness from Loudness (stationary): {S_loudness_stat['Sharpness']:.3f} acum")
    print(f"Calculated Stationary Sharpness from Loudness (varying): {S_loudness_time['Smean'][0]:.3f} acum")

    return S_stationary, S_loudness_stat, S_loudness_time

def ex_Roughness_Daniel1997():

    soundfile = str(SOUND_DIR / "RefSignal_Roughness_Daniel1997.wav")
    raw_insig, fs = wav2sig(soundfile)

    R = Roughness_Daniel1997(insig = raw_insig, # input signal, 1D array
                                fs = fs, # input signal and sampling frequency
                                time_skip = 0, # time_skip (second) for statistics calculation
                                show = 1) # show results, 'false' (disable, default value) or 'true' (enable)
    
    print(f"Calculated Roughness: {R['Rmean'][0]:.3f} asper")
    
    return R

def ex_FluctuationStrength_Osses2016():

    soundfile = str(SOUND_DIR / "RefSignal_FluctuationStrength_Osses2016.wav")
    raw_insig, fs = wav2sig(soundfile)

    F = FluctuationStrength_Osses2016(insig = raw_insig, # input signal, 1D array
                                      fs = fs, # input signal and sampling frequency
                                      method = 1, # method; stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2;
                                      time_skip = 0, # time_skip (second) for statistics calculation
                                      show = 1) # show results, 'false' (disable, default value) or 'true' (enable)
    
    structIn = {"a0_type": "fastl2007"}

    F2 = FluctuationStrength_Osses2016(insig = raw_insig, # input signal, 1D array
                                      fs = fs, # input signal and sampling frequency
                                      method = 1, # method; stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2;
                                      time_skip = 0, # time_skip (second) for statistics calculation
                                      show = 1,
                                      struct_opt=structIn) # show results, 'false' (disable, default value) or 'true' (enable)

    print(f"Calculated Fluctuation Strength: {F['FSmean'][0]:.3f} acum")
    print(f"Calculated Fluctuation Strength (a0): {F2['FSmean'][0]:.3f} acum")
    
    return F, F2

def ex_Tonality_Aures1985():

    soundfile = str(SOUND_DIR / "RefSignal_Tonality_Aures1985.wav")
    raw_insig, fs = wav2sig(soundfile)

    T = Tonality_Aures1985(insig = raw_insig, # input signal, 1D array
                           fs = fs, # input signal and sampling frequency
                           LoudnessField= 0, # field used for loudness calculation; free field = 0; diffuse field = 1;
                           time_skip = 0, # time_skip (second) for statistics calculation
                           show = 1) # show results, 'false' (disable, default value) or 'true' (enable)
    
    print(f"Calculated Tonality: {T['Kmean'][0]:.3f} t.u.")
    
    return T

def ex_PsychoacousticAnnoyance_Zwicker1999():

    soundfile = str(SOUND_DIR / "RefSignal_Loudness_ISO532_1.wav")
    raw_insig, fs = wav2sig(soundfile)
    lvl_cal_signal = 40

    insig_cal, cal_factor, dBFS_out = calibrate(raw_insig, raw_insig, lvl_cal_signal, return_dbfs=True)

    lvl_rms = 20*np.log10(np.sqrt(np.mean(insig_cal*insig_cal))) + dBFS_out

    print(f"\nThe RMS level of the calibrated input signal is {lvl_rms:.1f} dB SPL")
    print(f"\n\t(full scale value = {dBFS_out:.0f} dB SPL)")

    A1 = PsychoacousticAnnoyance_Zwicker1999(insig=insig_cal,
                                            fs = fs,
                                            LoudnessField = 0,
                                            time_skip= 0.2,
                                            showPA=True,
                                            show=True)
    
    PA5 = A1['PA5'][0]

    print(f"Calculated PA (Zwicker): {A1['PAmean'][0]:.3f} annoy")
    
    L_PA = A1['L']['N5'][0]; S_PA = A1['S']['S5'][0]; R_PA = A1['R']['R5'][0]; FS_PA = A1['FS']['FS5'][0]

    A2 = PsychoacousticAnnoyance_Zwicker1999(percentiles = (L_PA, S_PA, R_PA, FS_PA))

    print(f"Calculated PA5 from Loudness (Zwicker): {PA5:.3f} annoy")
    print(f"Compare with: {PA5:.3f} annoy")

    return A1, A2

def ex_PsychoacousticAnnoyance_More2010():

    soundfile = str(SOUND_DIR / "RefSignal_Loudness_ISO532_1.wav")
    raw_insig, fs = wav2sig(soundfile)
    lvl_cal_signal = 40

    insig_cal, cal_factor, dBFS_out = calibrate(raw_insig, raw_insig, lvl_cal_signal, return_dbfs=True)

    lvl_rms = 20*np.log10(np.sqrt(np.mean(insig_cal*insig_cal))) + dBFS_out

    print(f"\nThe RMS level of the calibrated input signal is {lvl_rms:.1f} dB SPL")
    print(f"\n\t(full scale value = {dBFS_out:.0f} dB SPL)")

    A1 = PsychoacousticAnnoyance_More2010(insig=insig_cal,
                                            fs = fs,
                                            LoudnessField = 0,
                                            time_skip= 0.2,
                                            showPA=True,
                                            show=True)
    
    PA5 = A1['PA5'][0]

    print(f"Calculated PA (More): {A1['PAmean'][0]:.3f} annoy")
    
    L_PA = A1['L']['N5'][0]; S_PA = A1['S']['S5'][0]; R_PA = A1['R']['R5'][0]; FS_PA = A1['FS']['FS5'][0]; K_PA = A1['K']['K5'][0]

    A2 = PsychoacousticAnnoyance_More2010(percentiles = (L_PA, S_PA, R_PA, FS_PA, K_PA))

    print(f"Calculated PA5 from Loudness (More): {PA5:.3f} annoy")
    print(f"Compare with: {PA5:.3f} annoy")

    return A1, A2

def ex_PsychoacousticAnnoyance_Di2016():
    
    soundfile = str(SOUND_DIR / "RefSignal_Loudness_ISO532_1.wav")
    raw_insig, fs = wav2sig(soundfile)
    lvl_cal_signal = 40

    insig_cal, cal_factor, dBFS_out = calibrate(raw_insig, raw_insig, lvl_cal_signal, return_dbfs=True)

    lvl_rms = 20*np.log10(np.sqrt(np.mean(insig_cal*insig_cal))) + dBFS_out

    print(f"\nThe RMS level of the calibrated input signal is {lvl_rms:.1f} dB SPL")
    print(f"\n\t(full scale value = {dBFS_out:.0f} dB SPL)")

    A1 = PsychoacousticAnnoyance_Di2016(insig=insig_cal,
                                        fs = fs,
                                        LoudnessField = 0,
                                        time_skip= 0.2,
                                        showPA=True,
                                        show=True)
    
    PA5 = A1['PA5'][0]

    print(f"Calculated PA (Di): {A1['PAmean'][0]:.3f} annoy")

    L_PA = A1['L']['N5'][0]; S_PA = A1['S']['S5'][0]; R_PA = A1['R']['R5'][0]; FS_PA = A1['FS']['FS5'][0]; K_PA = A1['K']['K5'][0]

    A2 = PsychoacousticAnnoyance_Di2016(percentiles = (L_PA, S_PA, R_PA, FS_PA, K_PA))

    print(f"Calculated PA5 from Loudness (Di): {PA5:.3f} annoy")
    print(f"Compare with: {PA5:.3f} annoy")

    return A1, A2


def ex_shm_loudness_ecma():

    # Use the same reference signal as the ISO example for comparability
    soundfile = str(SOUND_DIR / "RefSignal_Loudness_ISO532_1.wav")

    # Run the Sottek ECMA loudness wrapper on the file (audio input)
    OUT_shm = shm_loudness_ecma_wrapper(insig=soundfile,
                                       fs=None,
                                       field=0,
                                       method=1,
                                       time_skip=0.5,
                                       show=True)

    # Print a few key summary values
    if 'Loudness' in OUT_shm:
        print(f"SHM overall loudness (sone): {OUT_shm['Loudness']}")
    if 'InstantaneousLoudness' in OUT_shm:
        import numpy as _np
        print(f"SHM median instantaneous loudness (sone): {_np.median(OUT_shm['InstantaneousLoudness']):.3f}")

    return OUT_shm

def ex_shm_roughness_ecma():
    soundfile = str(SOUND_DIR / "RefSignal_Roughness_ECMA418_2.wav")
    OUT_shm = shm_roughness_ecma_wrapper(insig=soundfile,
                                         fs=None,
                                         field=0,
                                         method=1,
                                         time_skip=0.5,
                                         show=True)
    if 'Roughness' in OUT_shm:
        print(f"SHM overall roughness (asper): {OUT_shm['Roughness']}")
    return OUT_shm

def ex_shm_tonality_ecma():
    soundfile = str(SOUND_DIR / 'RefSignal_Tonality_ECMA418_2.wav')
    OUT_shm = shm_tonality_ecma_wrapper(insig=soundfile,
                                         fs=None,
                                         field=0,
                                         method=1,
                                         time_skip=0.5,
                                         show=True)
    if 'tonalityAvg' in OUT_shm:
        print(f"SHM overall tonality (t.u.): {OUT_shm['tonalityAvg']}")
    return OUT_shm


#example = "shm_loudness_ecma"
#example = "shm_roughness_ecma"
example = "shm_tonality_ecma"
#example = "Loudness_ISO532_1"


if __name__ == "__main__":

    if example == "Loudness_ISO532_1":
        L1, L2 = ex_Loudness_ISO532_1()

    elif example == "EPNL_FAR_Part36":
        E1, E2 = ex_EPNL_FAR_Part36()

    elif example == "Sharpness_DIN45692":
        S1, S2, S3 = ex_Sharpness_DIN45692()

    elif example == "Roughness_Daniel1997":
        R = ex_Roughness_Daniel1997()

    elif example == "FluctuationStrength_Osses2016":
        F1, F2 = ex_FluctuationStrength_Osses2016()

    elif example == "Tonality_Aures1985":
        T = ex_Tonality_Aures1985()

    elif example == "PsychoacousticAnnoyance_Zwicker1999":
        A1, A2 = ex_PsychoacousticAnnoyance_Zwicker1999()

    elif example == "PsychoacousticAnnoyance_More2010":
        A1, A2 = ex_PsychoacousticAnnoyance_More2010()

    elif example == "PsychoacousticAnnoyance_Di2016":
        A1, A2 = ex_PsychoacousticAnnoyance_Di2016()

    elif example == "shm_loudness_ecma":
        OUT_shm = ex_shm_loudness_ecma()

    elif example == "shm_roughness_ecma":
        OUT_shm = ex_shm_roughness_ecma()

    elif example == "shm_tonality_ecma":
        OUT_shm = ex_shm_tonality_ecma()
        

