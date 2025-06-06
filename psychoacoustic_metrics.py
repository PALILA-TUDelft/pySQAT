from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d 
from scipy.signal.windows import hann, blackman
from matplotlib import pyplot as plt

from sound_metrics import ob13_iso532_1

from utilities import (hz2bark, get_statistics, export_dict_to_excel,
                       get_statistics, from_db)
from utilities import (shm_resample, shm_preproc, shm_auditory_filt_bank,
                       shm_basis_loudness, shm_noise_red_lowpass,
                       shm_out_mid_ear_filter, shm_rough_low_pass,
                       shm_rough_weight, shm_signal_segment)

__all__ = ["Loudness_ISO532_1", "Tonality_Aures1985"]
FloatArray = NDArray[np.floating]

"""
metrics.py
------------------
This module contains functions and classes for calculating psychoacoustic metrics.

Functions:
- Loudness_ISO532_1: Implements the Zwicker loudness model.
"""

# ----------------------
#### LOUDNESS METRICS ####
# ----------------------

def Loudness_ISO532_1(insig, fs=None, field=0, method=2, time_skip=0, show=False, dBFS=94, export_excel=None):
    """
    Implements the Zwicker loudness model according to ISO 532-1:2017.

    Parameters:
    - insig: Input signal (array or string for WAV file).
    - fs: Sampling rate (Hz).
    - field: Acoustic field type (0 = free field, 1 = diffuse field).
    - method: Analysis method (0 = stationary, 1 = time-varying).
    - time_skip: Time to skip from start (seconds).
    - show: Whether to display plots.
    - dBFS: Reference full-scale SPL in dB.
    - export_excel: Path to export results to Excel.

    Returns:
    - OUT: Dictionary containing loudness metrics.
    """

    # --- WAV file interface ---
    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)

    elif fs is None:
        raise ValueError("If insig is not a filename, fs must be provided.")

    # --- Input handling ---
    insig = np.asarray(insig)

    if method == 0:
        insig = np.atleast_2d(insig)
        if insig.shape[0] != 1:
            insig = insig.T
    else:
        insig = np.atleast_2d(insig)
        if insig.shape[1] != 1:
            insig = insig.T
    
    if fs is None or fs <= 0:
        raise ValueError("Sampling frequency (fs) must be a positive number.")
    
    # Time constants for non-linear temporal decay
    Tshort = 0.005
    Tlong = 0.015
    Tvar = 0.075

    NL_ITER = 24 # Factors for virtual upsampling/inner iterations
    SR_LEVEL = 2000 # Sampling rate to which third-octave-levels are downsampled
    SR_LOUDNESS = 500 # Sampling rate to which output/total summed loudness is downsampled
    TINY_VALUE = 1e-12 # Tiny value for adjusting intensity levels for stationary signals
    I_REF = 4e-10 # ref value for stationary signals
    pref = np.sqrt(I_REF) # 2e-5 Pa
    barkAxis = np.arange(1, 241) / 10  # bark vector

    # --- Method selection ---
    if method == 0:
        # if method == 0, no need to calculate one-third OB levels.
        SampleRateLevel = 1
        DecFactorLoudness = 1
        NumSamplesLevel = 1

        ThirdOctaveLevel = insig # get 1/3 octave levels from insig if method = 0
    elif method == 1 or method == 2:
        # if different from stationary (from input 1/3 octave unweighted SPL)  

        # **************************************************
        # STEP 1 - resample to 48 kHz if necessary
        # **************************************************
        if fs != 48000:
            gcd_fs = np.gcd(48000, int(fs)) # greatest common denominator
            insig = resample_poly(insig.flatten(), 48000 // gcd_fs, int(fs) // gcd_fs)
            fs = 48000
            insig = insig[:, None]
        len_sig = insig.shape[0]

        # Assign values to global variables according to the selected method
        if method == 1: # stationary from audio signal
            SampleRateLevel = 1
            NumSamplesLevel = 1
            DecFactorLoudness = 1
        elif method == 2: # time_varying from audio signal
            SampleRateLevel = SR_LEVEL
            SampleRateLoudness = SR_LOUDNESS
            DecFactorLevel = fs / SampleRateLevel
            DecFactorLoudness = SampleRateLevel / SampleRateLoudness
            NumSamplesLevel = int(np.ceil(len_sig / DecFactorLevel))
            NumSamplesLoudness = int(np.ceil(NumSamplesLevel / DecFactorLoudness))

        # STEP 2 - Create filter bank and filter the signal
        filteredaudio, fc = ob13_iso532_1(insig.flatten(), fs)

        # STEP 3 - Squaring and smoothing by 3 1st order lowpass filters
        filteredaudio = filteredaudio ** 2

        N_bands = len(fc)
        ThirdOctaveLevel = np.zeros((NumSamplesLevel, N_bands))
        CentreFrequency = fc

        for i in range(N_bands):
            if method == 2: # time-varying from audio signal
                smoothedaudio = np.zeros((len_sig, N_bands))
                Tau = 2 / (3 * CentreFrequency[i]) if CentreFrequency[i] <= 1000 else 2 / (3 * 1000.)
                
                # 3x smoothing 1st order low-pass filters in series
                A1 = np.exp(-1 / (fs * Tau))
                B0 = 1 - A1
                Y1 = 0
                for k in range(3):
                    for j in range(len(filteredaudio)):
                        # smoothedaudio(j,i) = A1*temp(j,i) + B0*Y1;
                        smoothedaudio[j,i] = B0 * filteredaudio[j, i] + (A1 * Y1) # <----- modified from original by gfg
                        Y1 = smoothedaudio[j,i]
                
                c = 0
                for j in range(NumSamplesLevel):
                    idx = int(c)
                    if idx >= len_sig:
                        idx = len_sig - 1
                    ThirdOctaveLevel[j, i] = 10 * np.log10((smoothedaudio[idx,i] + TINY_VALUE) / I_REF)
                    c += DecFactorLevel
            elif method == 1: # stationary from audio signal
                NumSkip = int(np.floor(time_skip * fs))
                smoothedaudio = filteredaudio[NumSkip:len_sig, i]
                if NumSkip > len_sig / 2:
                    raise ValueError('time signal too short')
                if NumSkip == 0:
                    NumSkip = 1
                ThirdOctaveLevel[NumSamplesLevel - 1, i] = 10 * np.log10((np.sum(smoothedaudio[:]) / len_sig + TINY_VALUE) / I_REF)

    else:
        raise ValueError("Invalid method. Choose 0 (stationary), 1 (stationary from audio signal), or 2 (time-varying from audio signal).")
    # ***********************************************************
    # STEP 4 - Apply weighting factor to the first three 1/3 octave bands
    # ***********************************************************

    # WEIGHTING BELLOW 315Hz TABLE A.3

    # Ranges of 1/3 Oct bands for correction at low frequencies according to equal loudness contours
    RAP = [45, 55, 65, 71, 80, 90, 100, 120]

    # Reduction of 1/3 Oct Band levels at low frequencies according to equal loudness contours
    # within the eight ranges defined by RAP (DLL)
    DLL = np.array([
        [-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
        [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
        [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
        [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
        [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
        [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
        [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
        [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0]
    ])
    CorrLevel = np.zeros((NumSamplesLevel, 11))
    Intens = np.zeros((NumSamplesLevel, 11))
    CBI = np.zeros((NumSamplesLevel, 3))
    LCB = np.zeros((NumSamplesLevel, 3))

    for j in range(NumSamplesLevel):
        for i in range(DLL.shape[1]):
            k = 0
            while (k < 7) and (ThirdOctaveLevel[j, i] > RAP[k] - DLL[k, i]):
                k += 1
            CorrLevel[j, i] = ThirdOctaveLevel[j, i] + DLL[k, i] # attenuated levels
            Intens[j, i] = 10 ** (CorrLevel[j, i] / 10) # attenuated 1/3 octave intensities

        # *************************************************************
        # STEP 5 - Sumup intensity values of the first 3 critical bands
        # *************************************************************

        CBI[j, 0] = np.sum(Intens[j, 0:6]) # first critical band (sum of octaves (25Hz to 80Hz))
        CBI[j, 1] = np.sum(Intens[j, 6:9]) # second critical band (sum of octaves (100Hz to 160Hz))
        CBI[j, 2] = np.sum(Intens[j, 9:11]) # third critical band (sum of octaves (200Hz to 250Hz))
        
        FNGi = 10 * np.log10(CBI[j, :]+TINY_VALUE)

        for i in range(3):
            if CBI[j, i] > 0:
                LCB[j, i] = FNGi[i]
            else:
                LCB[j, i] = 0

    # **********************************************************************
    # STEP 6 - Calculate core loudness for each critical band
    # **********************************************************************

    # LEVEL CORRECTIONS TABLE A.5 (LDF0) DDF
    # Level correction to convert from a free field to a diffuse field (last critical band 12.5kHz is not included)
    DDF = [0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3, 2, 0, -1.4, -2, -1.9, -1, 0.5, 3, 4, 4.3, 4]
    
    # LEVEL CORRECTIONS TABLE A.6 (LTQ)
    # Critical band level at absolute threshold without taking into account the
    # transmission characteristics of the ear
    LTQ = [30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] # Threshold due to internal noise
    # Hearing thresholds for the excitation levels (each number corresponds to a critical band 12.5kHz is not included)

    # LEVEL CORRECTIONS TABLE A.7 DCB
    # Correction factor because using third octave band levels (rather than critical bands)
    DCB = [-0.25, -0.6, -0.8, -0.8, -0.5, 0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 0.8, 0.5, 0, -0.5]

    # LEVEL CORRECTIONS TABLE A.4 (A0)
    # Attenuation due to transmission in the middle ear
    A0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6, -3.2, -5.4, -5.6, -4, -1.5, 2, 5, 12]
    # Moore et al disagrees with this being flat for low frequencies

    Le = np.zeros((NumSamplesLevel, 20))
    CoreL = np.zeros((NumSamplesLevel, 21))

    for j in range(NumSamplesLevel):
        for i in range(19):
            Le[j, i] = ThirdOctaveLevel[j, i + 8]
            if i <= 2:
                Le[j, i] = LCB[j, i]
            Le[j, i] = Le[j, i] - A0[i]
            if field == 1:
                Le[j, i] = Le[j, i] + DDF[i]
            if Le[j, i] > LTQ[i]:
                S = 0.25
                Le[j, i] = Le[j, i] - DCB[i]
                MP1 = 0.0635 * 10 ** (0.025 * LTQ[i])
                MP2 = (((1 - S) + S * 10 ** (0.1 * (Le[j, i] - LTQ[i]))) ** 0.25) - 1
                CoreL[j, i] = MP1 * MP2
                if CoreL[j, i] <= 0:
                    CoreL[j, i] = 0

    # *************************************************************************
    # STEP 7 - Correction of specific loudness within the lowest critical band
    # *************************************************************************

    for j in range(NumSamplesLevel):
        CorrCL = 0.4 + 0.32 * CoreL[j, 0] ** 0.2
        if CorrCL > 1:
            CorrCL = 1
        CoreL[j, 0] = CoreL[j, 0] * CorrCL

    # **********************************************************************
    # STEP 8 - Implementation of NL Block
    # ***********************************************************************

    if method == 2: # time-varying from audio signal

        DeltaT = 1 / (SampleRateLevel * NL_ITER)
        P = (Tvar + Tlong) / (Tvar * Tshort)
        Q = 1 / (Tshort * Tvar)
        Lambda1 = -P / 2 + np.sqrt(P * P / 4 - Q)
        Lambda2 = -P / 2 - np.sqrt(P * P / 4 - Q)
        Den = Tvar * (Lambda1 - Lambda2)
        E1 = np.exp(Lambda1 * DeltaT)
        E2 = np.exp(Lambda2 * DeltaT)
        NlLpB = np.zeros(6)
        NlLpB[0] = (E1 - E2) / Den
        NlLpB[1] = ((Tvar * Lambda2 + 1) * E1 - (Tvar * Lambda1 + 1) * E2) / Den
        NlLpB[2] = ((Tvar * Lambda1 + 1) * E1 - (Tvar * Lambda2 + 1) * E2) / Den
        NlLpB[3] = (Tvar * Lambda1 + 1) * (Tvar * Lambda2 + 1) * (E1 - E2) / Den
        NlLpB[4] = np.exp(-DeltaT / Tlong)
        NlLpB[5] = np.exp(-DeltaT / Tvar)

        for i in range(21):
            NlLpUoLast = 0 # At beginning capacitors C1 and C2 are discharged
            NlLpU2Last = 0
            for j in range(NumSamplesLevel - 1):
                NextInput = CoreL[j + 1, i]
                # interpolation steps between current and next sample
                Delta = (NextInput - CoreL[j, i]) / NL_ITER
                Ui = CoreL[j, i]

                # f_nl_lp FUNCTION STARTS
                # case 1
                if Ui < NlLpUoLast:
                    # case 1.1
                    if NlLpUoLast > NlLpU2Last:
                        U2 = NlLpUoLast * NlLpB[0] - NlLpU2Last * NlLpB[1]
                        Uo = NlLpUoLast * NlLpB[2] - NlLpU2Last * NlLpB[3]
                        if Uo < Ui:
                            Uo = Ui
                        if U2 > Uo:
                            U2 = Uo
                    # case 1.2
                    else:
                        Uo = NlLpUoLast * NlLpB[4]
                        if Uo < Ui:
                            Uo = Ui
                        U2 = Uo
                # case 2
                elif Ui == NlLpUoLast:
                    Uo = Ui
                    # case 2.1
                    if Uo > NlLpUoLast:
                        U2 = (NlLpUoLast - Ui) * NlLpB[5] + Ui
                    # case 2.2
                    else:
                        U2 = Ui
                # case 3
                else:
                    Uo = Ui
                    U2 = (NlLpU2Last - Ui) * NlLpB[5] + Ui

                NlLpUoLast = Uo
                NlLpU2Last = U2
                CoreL[j, i] = Uo
                # f_nl_lp FUNCTION ENDS

                Ui += Delta

                # inner iteration
                for k in range(NL_ITER):
                    # f_nl_lp FUNCTION STARTS
                    # case 1
                    if Ui < NlLpUoLast:
                        # case 1.1
                        if NlLpUoLast > NlLpU2Last:
                            U2 = NlLpUoLast * NlLpB[0] - NlLpU2Last * NlLpB[1]
                            Uo = NlLpUoLast * NlLpB[2] - NlLpU2Last * NlLpB[3]
                            if Ui > Uo:
                                Uo = Ui
                            if U2 > Uo:
                                U2 = Uo
                        # case 1.2
                        else:
                            Uo = NlLpUoLast * NlLpB[4]
                            if Ui > Uo:
                                Uo = Ui
                            U2 = Uo
                    # case 2
                    elif Ui == NlLpUoLast:
                        Uo = Ui
                        # case 2.1
                        if Uo > NlLpUoLast:
                            U2 = (NlLpUoLast - Ui) * NlLpB[5] + Ui
                        # case 2.2
                        else:
                            U2 = Ui
                    # case 3
                    else:
                        Uo = Ui
                        U2 = (NlLpU2Last - Ui) * NlLpB[5] + Ui

                    NlLpUoLast = Uo
                    NlLpU2Last = U2
                    CoreL[j, i] = Uo
                    # f_nl_lp FUNCTION ENDS
                    Ui += Delta

    # **********************************************************************
    # STEP 9 - CALCULATE THE SLOPES
    # ***********************************************************************

    # Upper limits of the approximated critical bands in Bark
    # TABLE A.8
    ZUP = np.array([.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7, 18.1, 19.3, 20.6, 21.8, 22.7, 23.6, 24.0]) + 0.0001

    # TABLE A.9
    # Range of specific loudness for the determination of the steepness of the upper slopes in the specific loudness
    # - critical band rate pattern (used to plot the correct USL curve)
    RNS = np.array([21.5, 18, 15.1, 11.5, 9, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30, 0.22, 0.15, 0.10, 0.035, 0])

    # This is used to design the right hand slope of the loudness
    USL = np.array([
        [13, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
        [9, 7.5, 6, 5.1, 4.5, 4.5, 4.5, 4.5],
        [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
        [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
        [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
        [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
        [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
        [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
        [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
        [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
        [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
        [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
        [0.40, 0.33, 0.26, 0.24, 0.24, 0.22, 0.22, 0.22],
        [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
        [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
        [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
        [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
        [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]
    ])

    LN = np.zeros(NumSamplesLevel)
    N_mat = np.zeros(NumSamplesLevel)
    Spec_N = np.zeros(240) 
    ZUP = ZUP+0.0001 # <----- add constant factor to ZUP according to code provided by ISO 532-1
    ns = np.zeros((NumSamplesLevel, 240))

    for l in range(NumSamplesLevel):

        N = 0
        z1 = 0 # critical band rate starts at 0
        n1 = 0 # loudness level starts at 0
        iz = 0
        z = 0.1
        j = 17

        for i in range(21): # specific loudness

            # Determines where to start on the slope
            ig = i - 1

            # steepness of upper slope (USL) for bands above 8th one are identical
            if ig > 7:
                ig = 7

            while z1 < ZUP[i]:

                if n1 <= CoreL[l, i]: # Nm is the main loudness level
                    # contribution of unmasked main loudness to total loudness
                    # and calculation of values
                    if n1 < CoreL[l, i]:
                        j = 0

                        while (RNS[j] > CoreL[l, i]) and (j < 17): # the value of j is used below to build a slope to the range of specific loudness
                            j += 1 # j becomes the index at which Nm(i) 

                    z2 = ZUP[i]
                    n2 = CoreL[l, i]
                    N += n2 * (z2 - z1)
                    k = z # initialisation of k

                    while k <= z2:
                        ns[l, iz] = n2
                        iz += 1
                        k += 0.1

                    z = k

                else: # if N1 > NM(i)
                    # decision wether the critical band in question is completely
                    # or partly masked by accessory loudness

                    n2 = RNS[j]

                    if n2 < CoreL[l, i]:
                        n2 = CoreL[l, i]

                    dz = (n1 - n2) / USL[j, ig]
                    z2 = z1 + dz

                    if z2 > ZUP[i]:
                        z2 = ZUP[i]
                        dz = z2 - z1
                        n2 = n1 - dz * USL[j, ig]

                    N += dz * (n1 + n2) / 2
                    k = z # initialisation of k

                    while k <= z2:
                        ns[l, iz] = n1 - (k - z1) * USL[j, ig]
                        iz += 1
                        k += 0.1

                    z = k

                if (n2 <= RNS[j]) and (j < 17):
                    j += 1

                if (n2 <= RNS[j]) and (j >= 17):
                    j = 17

                z1 = z2 # N1 and Z1 for next loop
                n1 = n2

        if N < 0:
            N = 0

        if N <= 16:
            N = (N * 1000 + .5) / 1000
        else:
            N = (N * 100 + .5) / 100

        if method == 2:  # time-varying
            LN[l] = 40 * (N + .0005) ** .35
            if N >= 1:
                LN[l] = 10 * np.log10(N) / np.log10(2) + 40
            if LN[l] < 3:
                LN[l] = 3

        elif method in (0, 1):  # stationary
            LN = 40 * N ** 0.35
            if N >= 1:
                LN = 40 + 10 * np.log2(N)
            if LN < 0:
                LN = 0
            if LN < 3:
                LN = 3

        N_mat[l] = N

    # specific Loudness as a function of Bark number
    for i in range(240):
        Spec_N[i] = np.mean(ns[:,i])

    # *********************************************************
    # STEP 10 - Apply Temporal Weighting to Arbitrary signals
    # *********************************************************

    if method == 2: # time-varying from audio signal

        Loudness_t1 = np.zeros(NumSamplesLevel)
        Loudness_t2 = np.zeros(NumSamplesLevel)
        Loudness = np.zeros(NumSamplesLevel)

        # 1st order low-pass A
        Tau = 3.5e-3
        A1 = np.exp(-1 / (SampleRateLevel * DecFactorLevel * Tau))
        B0 = 1 - A1
        Y1 = 0

        for i in range(NumSamplesLevel):
            X0 = N_mat[i]
            Y1 = B0 * X0 + A1 * Y1
            Loudness_t1[i] = Y1

            if i < NumSamplesLevel - 1:
                Xd = (N_mat[i] - X0) / DecFactorLevel
                for j in range(int(DecFactorLevel)):
                    X0 = X0 + Xd
                    Y1 = B0 * X0 + A1 * Y1

        # 1st order low-pass B
        Tau = 70e-3
        A1 = np.exp(-1 / (SampleRateLevel * DecFactorLevel * Tau))
        B0 = 1 - A1
        Y1 = 0

        for i in range(NumSamplesLevel):
            X0 = N_mat[i]
            Y1 = B0 * X0 + A1 * Y1
            Loudness_t2[i] = Y1
            if i < NumSamplesLevel - 1:
                Xd = (N_mat[i] - X0) / DecFactorLevel
                for j in range(int(DecFactorLevel)):
                    X0 = X0 + Xd
                    Y1 = B0 * X0 + A1 * Y1

        # Combine the filters
        for i in range(NumSamplesLevel):
            Loudness[i] = (0.47 * Loudness_t1[i]) + (0.53 * Loudness_t2[i])

        # Decimate signal for decreased computation time by factor of 24 (fs = 2 Hz)
        Total_Loudness = np.zeros(NumSamplesLoudness)
        sC = 0
        for i in range(NumSamplesLoudness):
            Total_Loudness[i] = Loudness[int(sC)]
            sC += DecFactorLoudness

        ns_dec = np.zeros((NumSamplesLoudness, 240))
        sC = 0
        for i in range(NumSamplesLoudness):
            ns_dec[i, :] = ns[int(sC), :]
            sC += DecFactorLoudness

        # **********************************************************************
        # Compute loudness level - conversion from sone to phon
        # ***********************************************************************

        LN = 40 * Total_Loudness ** 0.35
        LN[Total_Loudness >= 1] = 40 + 10 * np.log2(Total_Loudness[Total_Loudness >= 1])
        LN[LN < 0] = 0
        LN[LN < 3] = 3

        # **********************************************************************
        # output struct for time-varying signals
        # ***********************************************************************
        OUT = {}
        OUT['barkAxis'] = barkAxis  # Bark vector
        OUT['time'] = np.arange(len(Total_Loudness)) * 2e-3  # time vector of the final loudness calculation, in seconds
        OUT['time_insig'] = np.arange(len(insig)) / fs  # time vector of the audio input, in seconds
        OUT['InstantaneousLoudness'] = Total_Loudness  # Time-varying Loudness, in sone
        OUT['SpecificLoudness'] = Spec_N  # time-averaged specific loudness (sone/Bark)
        OUT['InstantaneousSpecificLoudness'] = ns_dec  # specific loudness (sone/Bark) vs time
        OUT['InstantaneousLoudnessLevel'] = LN  # Time-varying Loudness level (phon)
        OUT['InstantaneousSPL'] = 10 * np.log10(np.sum(10 ** (ThirdOctaveLevel[:, :] / 10), axis=1))  # total SPL (1/3 octave bands) for each time step, in dBSPL

        # get statistics from Time-varying Loudness (sone)
        idx = np.argmin(np.abs(OUT['time'] - time_skip)) # find idx of time_skip on time vector
        metric_statistics = 'Loudness_ISO532_1'
        OUT_statistics = get_statistics(Total_Loudness[idx:], metric_statistics) # get statistics
        
        # copy fields of <OUT_statistics> struct into the <OUT> struct
        for key, value in OUT_statistics.items(): # Get all field names in OUT_statistics
            if key not in OUT: # Only copy if OUT does NOT already have this field
                OUT[key] = value
        OUT['N_ratio'] = OUT['N5'] / OUT['N95']

        # *********************************************************
        # show plots (time-varying)
        # *********************************************************

        if show:
            plt.figure("Loudness analysis (time-varying)", figsize=(16, 8)) # plot fig in full screen
            xmax = OUT['time'][-1]

            # plot input signal
            plt.subplot(2, 6, (1, 2))
            plt.plot(OUT['time_insig'], insig)
            YL = 2 * np.max(np.abs(insig)) * np.array([-1, 1]) # min-max limit for Y axis
            plt.axis([0, xmax, YL[0], YL[1]])
            plt.title('Input signal')
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Sound pressure, $p$ (Pa)')

            # plot instantaneous sound pressure level (dBSPL)
            plt.subplot(2, 6, (3, 4))
            plt.plot(np.linspace(0, OUT['time_insig'][-1], len(OUT['InstantaneousSPL'])), OUT['InstantaneousSPL'])
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3]*1.1])
            plt.title('Instantaneous overall SPL (1/3 octave)')
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('SPL, $L_p$ (dB re 20 μPa)')
            plt.grid(True)

            # plot instantaneous loudness level (phon)
            plt.subplot(2, 6, (5, 6))
            plt.plot(OUT['time'], np.abs(OUT['InstantaneousLoudnessLevel']))
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3]*1.1])
            plt.title('Instantaneous loudness level')
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Loudness level, $L_N$ (phon)')
            plt.grid(True)

            # plot instantaneous loudness (sone)
            plt.subplot(2, 6, (7, 8))
            plt.plot(OUT['time'], OUT['InstantaneousLoudness'])
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3]*1.1])
            plt.title('Instantaneous loudness')
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Loudness, $N$ (sone)')
            plt.grid(True)

            # plot specific loudness (sone/bark)
            plt.subplot(2, 6, (9, 10))
            plt.plot(OUT['barkAxis'], OUT['SpecificLoudness'])
            ax = plt.axis()
            plt.axis([0, 24, ax[2], ax[3]*1.1])
            plt.title('Time-averaged specific loudness')
            plt.xlabel('Critical band, $z$ (Bark)')
            plt.ylabel('Specific loudness, $N\'$ (sone/Bark)')
            plt.grid(True)

            # plot instantaneous specific loudness (sone/bark)
            plt.subplot(2, 6, (11, 12))
            xx, yy = np.meshgrid(OUT['time'], OUT['barkAxis'])
            pcm = plt.pcolormesh(xx, yy, OUT['InstantaneousSpecificLoudness'].T, shading='auto')
            plt.title('Instantaneous specific loudness')
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Critical band, $z$ (Bark)')
            plt.colorbar(pcm, label='Specific Loudness, $N\'$ (sone/Bark)')
            plt.ylim([0, 24])

            plt.tight_layout()
            plt.show()

    elif method == 0 or method == 1: # stationary method
        # *********************************************************
        # output struct for stationary signals
        # *********************************************************

        OUT = {}
        if method == 1: # stationary from audio signal
            OUT['time_insig'] = np.arange(len(insig)) / fs # time vector of the audio input, in seconds
        
        OUT['barkAxis'] = (np.arange(1, 241)) / 10 # bark vector
        OUT['SpecificLoudness'] = Spec_N # time-averaged specific loudness (sone/Bark)
        OUT['Loudness'] = N # loudness (sone)
        OUT['LoudnessLevel'] = LN # loudness level (phon)
        OUT['TimeAveragedSPL'] = 10 * np.log10(np.sum(10 ** (ThirdOctaveLevel[:, :] / 10), axis=1)) # total SPL (1/3 octave bands) for each time step, in dBSPL

        # *********************************************************
        # show plots (stationary)
        # *********************************************************

        if show:
            plt.figure("Loudness analysis (stationary)", figsize=(10, 8)) # plot fig in full screen
            
            xmax = OUT['time_insig'][-1]

            # plot input signal
            insig_rms = np.sqrt(np.mean(insig**2))
            insig_rms_dB = 20 * np.log10(insig_rms / pref)

            plt.subplot(2, 1, 1)
            plt.plot(OUT['time_insig'], insig)
            plt.axhline(insig_rms, color='k', linestyle='--')
            text4legend = f"$p_{{rms}}$={insig_rms:.3g} (Pa)\n$L_{{p}}$={insig_rms_dB:.1f} (dB SPL)"
            plt.legend([text4legend], loc='upper right')
            YL = 2 * np.max(np.abs(insig)) * np.array([-1, 1]) # min-max limit for Y axis
            plt.axis([0, xmax, YL[0], YL[1]])
            plt.title('Input signal')
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Sound pressure, $p$ (Pa)')

            # plot specific loudnes (sone/bark)
            plt.subplot(2, 1, 2)
            plt.plot(OUT['barkAxis'], OUT['SpecificLoudness'])
            text4annotation = f"Loudness, $N$={N:.3f} (sone)\nLoudness level, $L_N$={LN:.1f} (phon)"
            plt.gcf().text(0.78, 0.38, text4annotation, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            ax = plt.axis()
            plt.axis([0, 24, ax[2], ax[3]*1.1])
            plt.title('Specific loudness')
            plt.xlabel('Critical band, $z$ (Bark)')
            plt.ylabel('Specific loudness, $N\'$ (sone/Bark)')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    if export_excel is not None:
        export_dict_to_excel(OUT, filename=f"{export_excel}")


    return OUT


# ----------------------
#### TONALITY METRICS ####
# ----------------------

def Tonality_Aures1985(insig, fs=None, LoudnessField=0, time_skip=0, show=False, dBFS=94):
    """
    Implements the Aures (1985) tonality metric based on Terhardt's virtual pitch theory.

    Parameters:
    - insig: Input signal (array, monophonic in Pa).
    - fs: Sampling frequency (Hz).
    - LoudnessField: Field for loudness calculation (0 = free field, 1 = diffuse field).
    - time_skip: Time to skip at the start of the signal for statistical calculations (seconds).
    - show: Whether to display plots (default: False).

    Returns:
    - OUT: Dictionary containing tonality metrics and statistics.
    """

    # --- WAV file interface ---
    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)

    elif fs is None:
        raise ValueError("If insig is not a filename, fs must be provided.")

    ## resampling ------------------------------------------------------

    # resampling audo to 44.1 kHz or 48kHz
    if fs not in [44100, 48000]:
        gcd_fs = np.gcd(44100, fs) # greatest common denominator
        insig = resample_poly(insig, 44100 // gcd_fs, fs // gcd_fs)
        fs = 44100

    # Window parameters
    time_resolution = 160e-3  # window length fixed in 160 ms, gives a df=6.25 Hz

    N = round(fs * time_resolution)  # define window length, N bins
    window = hann(N)

    fftgain = np.sqrt(2) / (N * np.mean(window))

    ## freq vectors based on window input signals -------------------------

    # from Terhardt [3]: Aurally relevant tonal information of any signal is
    # confined in the frequency region of about 20 Hz to 5 kHz.

    MinFrequency = 20
    MinFrequencyindex = int(np.ceil(1 + (MinFrequency * (N / fs)))-1) # index corresponding to min frequency (20 Hz) for tone extraction

    MaxFrequency = 5000
    MaxFrequencyIndex = int(np.ceil(1 + (MaxFrequency * (N / fs)))-1) # index corresponding to max frequency (5 kHz) for tone extraction

    Freq = fs * (np.arange(1, round(N) + 1) - 1) / N # freq vector
    FreqCrop = Freq[MinFrequencyindex:MaxFrequencyIndex+1] # croped freq vector from MinFrequencyindex till MaxFrequencyIndex
    df = FreqCrop[1] - FreqCrop[0] # freq discretization

    ## Initialize windowed vectors

    t_b = np.arange(1, len(insig) + 1) / fs # time vector

    overlap = round(0.5 * N) # overlap 
    hop_size = N - overlap

    insig = np.lib.stride_tricks.sliding_window_view(insig, N)[::hop_size]
    t_b = np.lib.stride_tricks.sliding_window_view(t_b, N)[::hop_size]

    # Verify the shape of insig
    if insig.ndim != 2:
        raise ValueError(f"Expected insig to be 2D after segmentation, but got shape {insig.shape}")

    nFrames = insig.shape[0] - 1

    tone = [None] * nFrames # Memory allocation: tone cell per time frame
    tonality = np.zeros(nFrames) # Memory allocation for tonality computation
    t = np.zeros(nFrames) # Memory allocation: time vector for iFrames
    w_gr = np.zeros(nFrames) # Memory allocation: loudness weighting function per time frame
    w_tonal = np.zeros(nFrames) # Memory allocation: tonal weighting function per time frame
    TINY_VALUE = 1e-99

    ## Main processing loop

    for iFrame in range(nFrames):

        ## windowed time-frame
        Winsig = insig[iFrame] * window # cut insig for each iFrames & Apply window to frame
        t[iFrame] = t_b[iFrame,0] # output time vector for iFrames

        ## Compute SPL for each time-frame
        SpectralEnergy = np.abs(fft(Winsig * fftgain)) ** 2
        SPL = 10 * np.log10((SpectralEnergy + TINY_VALUE) / 4e-10) # dBSPL 

        ## Find peaks according to Terhard's criteria for each time-frame

        SPLcrop = SPL[MinFrequencyindex:MaxFrequencyIndex+1] # crop SPL vector from MinFrequencyindex to MaxFrequencyIndex

        threshold = 7 # condition for tonal component, in dBSPL

        ToneIdx = [] # initialize vector, tonal components idx
        
        # find tones
        for i in range(4, len(SPLcrop) - 2):
            if (
                SPLcrop[i] > SPLcrop[i - 1] # first condition
                and SPLcrop[i] >= SPLcrop[i + 1]
                and SPLcrop[i] - SPLcrop[i - 3] >= threshold # second condition
                and SPLcrop[i] - SPLcrop[i - 2] >= threshold
                and SPLcrop[i] - SPLcrop[i + 2] >= threshold
                and SPLcrop[i] - SPLcrop[i + 3] >= threshold
            ):
                ToneIdx.append(i)

        # Save tone information
        ToneIdx = np.array(ToneIdx)
        ToneIdx = ToneIdx[ToneIdx > 0]  # Remove zeros from ToneIdx
        ToneL = SPLcrop[ToneIdx]  # SPL of the tones
        ToneF = FreqCrop[ToneIdx]  # Central frequency of the tones

        # Estimate bandwidth of the i-th tone using half-power (-3 dB decay) criteria
        flow = np.zeros(len(ToneIdx))  # Memory allocation for low frequency
        fhigh = np.zeros(len(ToneIdx))  # Memory allocation for high frequency
        BW = np.zeros(len(ToneIdx))  # Memory allocation for bandwidth

        for i in range(len(ToneIdx)):
            ymx = ToneL[i]  # SPL of the i-th tone
            idx = np.argmin(np.abs(Freq - ToneF[i]))  # Index of the i-th tone
            hafmax = ymx * 0.707  # Target value

            # Find the lower frequency index
            idxrng1_candidates = np.where(SPL[:idx] < hafmax)[0]
            idxrng1 = idxrng1_candidates[-1] if len(idxrng1_candidates) > 0 else None

            if idxrng1 is None or idxrng1 < 4:
                idxrng1 = 4  # Truncate idxrng1 to 4 if it's empty or too low

            # Find the upper frequency index
            idxrng2_candidates = np.where(SPL[idx + 1:len(Freq)+1] < hafmax)[0]
            idxrng2 = idxrng2_candidates[0] + idx if len(idxrng2_candidates) > 0 else None

            # Interpolate frequencies for low and high bounds
            flow[i] = np.interp(hafmax, SPL[idxrng1:idxrng1 + 2], Freq[idxrng1:idxrng1 + 2])
            fhigh[i] = np.interp(hafmax, SPL[idxrng2 - 1:idxrng2 + 1], Freq[idxrng2 - 1:idxrng2 + 1])

            # Calculate bandwidth
            BW[i] = fhigh[i] - flow[i]
            BW *= 2 # WARNING! soft fix

            # Handle edge cases for bandwidth
            if BW[i] == 0:
                BW[i] = 1  # If BW is zero, truncate to 1

        # Replace inf and NaN values in BW
        BW[np.isinf(BW) | np.isnan(BW)] = 1


        if len(ToneIdx) == 0:  # if ToneRef is empty, then there are no tones for this time-frame
            
            ## Outputs for this case

            w_tonal[iFrame] = 0  # Tonal weighting
            w_gr[iFrame] = 0  # Loudness weighting
            tonality[iFrame] = 0  # Tonality

        else:  # If tones are found

            idx = np.where(ToneL > 0)[0] # find idx of only positive levels
            ToneIdx = ToneIdx[idx] # idx of the tone
            ToneL = ToneL[idx] # SPL of the tones
            ToneF = ToneF[idx] #  central freq of the tone
            BW = BW[idx] # bandwidth

            ## Filtering out the tones from the signal
            y = insig[iFrame, :]  # get insig for each iFrames
            insigSpectrum = np.fft.fft(y)  # spectrum of insig for each iFrames
            SingleSidedinsigSpectrum = insigSpectrum[:int(np.ceil((len(insigSpectrum)+1)/ 2))] # single-sided spectrum of insig for each iFrames
            FreqSingleSidedinsigSpectrum = np.linspace(0, fs / 2, len(SingleSidedinsigSpectrum)) # freq vector of single-sided spectrum of insig for each iFrames

            for i in range(len(ToneF)):  # Loop across tones

                index_low = np.where(FreqSingleSidedinsigSpectrum >= (ToneF[i] - BW[i] / 2))[0][0] # find idx of i-th tone's lower freq
                index_up = np.where(FreqSingleSidedinsigSpectrum >= (ToneF[i] + BW[i] / 2))[0][0] # find idx of i-th tone's upper freq

                if index_low == 0: # may happen with low-freq tones with large bandwidth
                    magn = 0.5 * (np.abs(SingleSidedinsigSpectrum[index_low]) + # create a magnitude vector
                                np.abs(SingleSidedinsigSpectrum[index_up + 1]))
                else:
                    magn = 0.5 * (np.abs(SingleSidedinsigSpectrum[index_low - 1]) + # create a magnitude vector
                                np.abs(SingleSidedinsigSpectrum[index_up + 1]))

                phase = (np.random.rand(index_up - index_low + 1) - 0.5) * 2 * np.pi # create random phase vector
                SingleSidedinsigSpectrum[index_low:index_up + 1] = magn * np.exp(1j * phase) # replace tones

            doubleSideFilteredSpectrum = np.concatenate( # double-side the filtered spectrum
                [SingleSidedinsigSpectrum, np.conj(SingleSidedinsigSpectrum[-2:0:-1])]) # get filtered signal in time-domain
            filtered_signal = np.fft.ifft(doubleSideFilteredSpectrum).real

            # Compute w_gr (loudness weighting)
            L_total = Loudness_ISO532_1(y, fs,
                                        LoudnessField,
                                        1,
                                        time_resolution * 0.05,
                                        False)
            L_filtered = Loudness_ISO532_1(filtered_signal, fs,
                                           LoudnessField,
                                           1,
                                           time_resolution * 0.05,
                                           False)

            # loudness weighting per time frame
            w_gr[iFrame] = 1 - (L_filtered["Loudness"] / L_total["Loudness"])
            if w_gr[iFrame] < 0:
                w_gr[iFrame] = 0

            # Compute tonal weighting
            tone[iFrame] = {
                "Lcrop": SPLcrop, # SPL of the spectrum - SPLcrop = SPL(MinFrequencyindex:MaxFrequencyIndex);
                "freq": FreqCrop, # frequency vector - freq = freq_all(MinFrequencyindex:MaxFrequencyIndex);
                "ToneF": ToneF, # ToneF: central frequency of the tones
                "ToneL": ToneL, # ToneF: central frequency of the tones
                "BW": BW, # bandwidth of the tones
                "df": df, # freq discretization
            }
            tone[iFrame]["LX"] = il_SPL_excess(tone[iFrame]) # Sound pressure excess calculation (define aurally relevance of the tones)
            
            w_tonal[iFrame] = il_tonal_weighting(tone[iFrame]) # Tonal weighting

            ## TONALITY

            C = 1.125  # is a constant such that 1 kHz pure tone with a level of 60 dB would have a tonalness of 1, which for an ideal implementaiton should be =1.09
            tonality[iFrame] = abs(C * w_tonal[iFrame] ** 0.29 * w_gr[iFrame] ** 0.79)

    ## Output data
    OUT = {
        "InstantaneousTonality": tonality, # instantaneous tonality
        "TonalWeighting": w_tonal, # instantaneous tonal weighting
        "LoudnessWeighting": w_gr, # instantaneous loudness weighting
        "time": t, # time vector
    }

    # get statistics from Time-varying tonality
    idx = np.argmin(np.abs(OUT["time"] - time_skip)) # find idx of time_skip on time vector
    
    metric_statistics = "Tonality_Aures1985"
    OUT_statistics = get_statistics(tonality[idx:], metric_statistics) # get statistics
    OUT.update(OUT_statistics)

    # Plots
    if show:
        plt.figure("Aures tonality analysis", figsize=(16, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, tonality)
        plt.title("Instantaneous tonality")
        plt.ylabel("Aures tonality, K (t.u.)")
        plt.xlabel("Time, t (s)")
        plt.ylim([0, 1.1])

        plt.subplot(3, 1, 2)
        plt.plot(t, w_gr, "k")
        plt.title("Loudness weighting")
        plt.ylabel("Loudness weighting, W_Loudness")
        plt.xlabel("Time, t (s)")
        plt.ylim([0, 1.1])

        plt.subplot(3, 1, 3)
        plt.plot(t, w_tonal, "k")
        plt.title("Tonal weighting")
        plt.ylabel("Tonal weighting, W_Tonal")
        plt.xlabel("Time, t (s)")
        plt.ylim([0, 1.1])

        plt.tight_layout()
        plt.show()

    print("L_total:", L_total['Loudness'])
    print("L_filtered:", L_filtered['Loudness'])
    print(f"W_gr: {1-L_total['Loudness']/L_filtered['Loudness']}")
    return OUT

# Not finished. Will finish implementation in the future. Need to check ECMA helpers.
def Tonality_ECMA418_2(insig, fs, fieldtype='free-frontal', time_skip=0.304, show=False):
    """
    Returns tonality values and frequencies according to ECMA-418-2:2024.
    """

    # Input validation
    if insig.ndim > 2:
        raise ValueError("Input signal has more than 2 channels.")
    if insig.shape[1] > 2:
        insig = insig.T
        print("Warning: Input signal was transposed to match [Nx1] or [Nx2] format.")
    if insig.shape[0] < time_skip * fs:
        raise ValueError("Input signal is too short to calculate tonality (must be at least 304 ms).")
    if time_skip < 0.304:
        print("Warning: time_skip must be at least 304 ms. Setting time_skip to 304 ms.")
        time_skip = 0.304

    # Determine input channels
    inchans = insig.shape[1] if insig.ndim == 2 else 1
    binaural = inchans == 2
    chans = ["Stereo left", "Stereo right"] if binaural else ["Mono"]

    # Define constants

    sampleRate48k = 48000 # Signal sample rate prescribed to be 48kHz (to be used for resampling), Section 5.1.1 ECMA-418-2:2024 [r_s]
    deltaFreq0 = 81.9289 # defined in Section 5.1.4.1 ECMA-418-2:2024 [deltaf(f=0)]
    c = 0.1618 # Half-Bark band centre-frequency denominator constant defined in Section 5.1.4.1 ECMA-418-2:2024

    halfBark = np.arange(0.5, 26.5 + 0.5, 0.5) # half-critical band rate scale [z]
    bandCentreFreqs = (deltaFreq0 / c) * np.sinh(c * halfBark) # Section 5.1.4.1 Equation 9 ECMA-418-2:2024 [F(z)]
    dfz = np.sqrt(deltaFreq0**2 + (c * bandCentreFreqs)**2) # Section 5.1.4.1 Equation 10 ECMA-418-2:2024 [deltaf(z)]
    
    # Block and hop sizes Section 6.2.2 Table 4 ECMA-418-2:2024
    overlap = 0.75 # block overlap proportion
    # block sizes [s_b(z)]
    blockSize = np.concatenate([np.full(3, 8192), np.full(13, 4096), np.full(9, 2048), np.full(28, 1024)])
    # hop sizes (section 5.1.2 footnote 3 ECMA 418-2:2022) [s_h(z)]
    hopSize = (1 - overlap) * blockSize

    # Output sample rate based on hop sizes - Resampling to common time basis
    # Section 6.2.6 ECMA-418-2:2024 [r_sd]    
    sampleRate1875 = sampleRate48k / np.min(hopSize)

    # Number of bands that need averaging. Section 6.2.3 Table 5
    # ECMA-418-2:2024 [NB]
    NBandsAvg = np.array([
        np.concatenate(([0, 1], np.full(14, 2), np.full(9, 1), np.full(28, 0))),
        np.concatenate(([1, 1], np.full(14, 2), np.full(9, 1), np.full(28, 0)))
    ])

    # Critical band interpolation factors from Section 6.2.6 Table 6
    # ECMA-418-2:2024 [i]
    i_interp = blockSize / np.min(blockSize)

    # Noise reduction constants from Section 6.2.7 Table 7 ECMA-418-2:2024
    alpha = 20
    beta = 0.07

    # Sigmoid function factor parameters Section 6.2.7 Table 8 ECMA-418-2:2024
    # [c(s_b(z))]
    csz_b = np.concatenate([np.full(3, 18.21), np.full(13, 12.14),
                            np.full(9, 417.54), np.full(28, 962.68)])
    dsz_b = np.concatenate([np.full(3, 0.36), np.full(13, 0.36),
                            np.full(9, 0.71), np.full(28, 0.69)])
    
    # Scaling factor constants from Section 6.2.8 Table 9 ECMA-418-2:2024
    A = 35
    B = 0.003

    cal_T = 2.8758615 # calibration factor in Section 6.2.8 Equation 51 ECMA-418-2:2024 [c_T]
    cal_Tx = 1 / 0.9999043734252 # Adjustment to calibration factor (Footnote 22 ECMA-418-2:2024)

    ## Signal processing

   # Signal processing
    if fs != sampleRate48k: # Resample signal
        p_re, fs = shm_resample(insig, fs)
    else: # don't resample
        p_re = insig

    # get time vector of input signal
    timeInsig = np.arange(len(p_re)) / fs # TODO: check.

    # Section 5.1.2 ECMA-418-2:2024 Fade in weighting and zero-padding
    pn = shm_preproc(p_re, np.max(blockSize), np.max(hopSize))

    ## Apply outer & middle ear filter

    # Section 5.1.3.2 ECMA-418-2:2024 Outer and middle/inner ear signal filtering
    pn_om = shm_out_mid_ear_filter(pn, fieldtype)

    # Loop through channels in file

    basisLoudnessArray = {}

    for chan in range(inchans):

        ## Apply auditory filter bank

        # Filter equalised signal using 53 1/2Bark ERB filters according to Section 5.1.4.2 ECMA-418-2:2024
        pn_omz = shm_auditory_filt_bank(pn_om[:, chan])

        # Autocorrelation function analysis
        # Duplicate Banded Data for ACF
        # Averaging occurs over neighbouring bands, to do this the segmentation
        # needs to be duplicated for neigbouring bands. 'Dupe' has been added
        # to variables to indicate that the vectors/matrices have been modified
        # for duplicated neigbouring bands.

        pn_omzDupe = np.concatenate((pn_omz[0:5,:], pn_omz[1:18,:], pn_omz[15:26,:], pn_omz[25:53,:]))
        blockSizeDupe = np.concatenate((np.full(5, 8192), np.full(17, 4096), np.full(11, 2048), np.full(28, 1024)))
        bandCentreFreqsDupe = np.concatenate((bandCentreFreqs[0:5],bandCentreFreqs[1:18], bandCentreFreqs[15:26], bandCentreFreqs[25:53]))
        # (duplicated) indices corresponding with the NB bands around each z band
        i_NBandsAvgDupe = np.vstack((np.concatenate(([1],[1],[1],np.arange(6, 19), np.arange(23, 32), np.arange(34, 62))),
                                    np.concatenate(([2],[3],[5], np.arange(10,23), np.arange(25,34), np.arange(34, 62)))))
        
        for zBand in range(60, -1, -1):

            # Segmentation into blocks

            # Section 5.1.5 ECMA-418-2:2024
            i_start = blockSizeDupe[0] - blockSizeDupe[zBand]
            pn_lz, _ = shm_signal_segment(pn_omzDupe[zBand,:], 0,
                                          blockSizeDupe[zBand], overlap, i_start)
            
            pn_lz = pn_lz.squeeze(-1)
            
            # Transformation into Loudness
            # Sections 5.1.6 to 5.1.9 ECMA-418-2:2024 [N'_basis(z)]
            p_rlz, bandBasisLoudness, _ = shm_basis_loudness(pn_lz, bandCentreFreqsDupe[zBand])
            basisLoudnessArray[zBand] = bandBasisLoudness

            www = 0


    ## CHECKPOINT - TODO: Still check variables.

    return None

# -------------------------
#### ROUGHNESS METRICS ####
# -------------------------

def Roughness_Daniel1997(
    insig: FloatArray,
    fs: int,
    time_skip: float = 0.0,
    show: bool = False,
) -> Dict[str, Any]:
    """Compute time‑varying psycho‑acoustic roughness (Daniel & Weber 1997).

    Parameters
    ----------
    insig : ndarray, shape (N,) or (N, 1)
        Acoustic signal in Pascal.
    fs : int
        Sampling rate in Hz.
    time_skip : float, default 0 s
        Lead‑in to exclude from statistics.
    show : bool, default False
        If True produce the same diagnostic plots as the original MATLAB code.

    Returns
    -------
    OUT : dict
        Output structure with *exactly* the same field names as the MATLAB
        struct (`InstantaneousRoughness`, `Rmean`, …).
    """
    # ---------------------------------------------------------------------
    # Input handling & mono conversion
    # ---------------------------------------------------------------------
    audio = np.asarray(insig, dtype=float).squeeze()
    if audio.ndim != 1:
        raise ValueError("'insig' must be mono (1‑D array)")

    ## resampling input signal
    if fs not in (44_100, 40_960, 48_000):
        gcd_fs = np.gcd(48_000, fs)
        audio = resample_poly(audio, 48_000 // gcd_fs, fs // gcd_fs)
        fs = 48_000

    ## window settings
    time_resolution = 0.2  # time-step for the windowing
    N = int(round(fs * time_resolution)) # window length
    if N < 128:
        raise ValueError("FFT size too small – check sampling rate")

    hopsize = N // 2 # hopsize is the number of samples hop between successive windows (window length is 8192).
    window = blackman(N, sym=False)

    # Pre‑compute constants used every frame
    N2 = N // 2 + 1
    dFs = fs / N

    # ---------------------------------------------------------------------
    # Zwicker critical‑band table (exact copy of MATLAB literal)
    # ---------------------------------------------------------------------
    Bark = np.array([
        [ 0,     0,     50,   0.5],
        [ 1,   100,    150,   1.5],
        [ 2,   200,    250,   2.5],
        [ 3,   300,    350,   3.5],
        [ 4,   400,    450,   4.5],
        [ 5,   510,    570,   5.5],
        [ 6,   630,    700,   6.5],
        [ 7,   770,    840,   7.5],
        [ 8,   920,   1000,   8.5],
        [ 9,  1080,   1170,   9.5],
        [10,  1270,   1370,  10.5],
        [11,  1480,   1600,  11.5],
        [12,  1720,   1850,  12.5],
        [13,  2000,   2150,  13.5],
        [14,  2320,   2500,  14.5],
        [15,  2700,   2900,  15.5],
        [16,  3150,   3400,  16.5],
        [17,  3700,   4000,  17.5],
        [18,  4400,   4800,  18.5],
        [19,  5300,   5800,  19.5],
        [20,  6400,   7000,  20.5],
        [21,  7700,   8500,  21.5],
        [22,  9500,  10500,  22.5],
        [23, 12000,  13500,  23.5],
        [24, 15500,  20000,  24.5],
    ], dtype=float)

    # Bark2 is the concatenation used for interp
    Bark2 = np.column_stack(
        [np.sort(np.r_[Bark[:, 1], Bark[:, 2]]),
         np.sort(np.r_[Bark[:, 0], Bark[:, 3]])]
    )

    N0  = int(round(20 * N / fs)) # low frequency index @ 20 Hz
    N01 = N0
    Ntop = int(round(20_000 * N / fs))  # high frequency index @ 20 kHz

    # Make list with Barknumber of each frequency bin
    Barkno = np.zeros(N2)
    f = np.arange(N0, Ntop + 1, 1)
    Barkno[f] = np.interp(f * dFs, Bark2[:, 0], Bark2[:, 1])

    # Make list of frequency bins closest to Cf's
    Cf = np.empty((2, 24), dtype=int)
    Cf[0, :] = np.round(Bark[1:, 1] * N / fs).astype(int) - N0
    Cf[1, :] = Bark[1:, 1]

    # Make list of frequency bins closest to Critical Band Border frequencies
    Bf = np.empty((2, 25), dtype=int)
    Bf[0, 0] = int(round(Bark[0, 2] * N / fs))

    for a in range(24):
        Bf[0, a + 1] = int(round(Bark[a + 1, 2] * N / fs)) - N0
        Bf[1, a] = Bf[0, a] - 1

    Bf[1, 24] = int(round(Bark[24, 2] * N / fs)) - N0

    ## Make list of minimum excitation (Hearing Treshold)
    HTres = np.array([
        [ 0,   130],
        [ 0.01,  70],
        [ 0.17,  60],
        [ 0.8,  30],
        [ 1,    25],
        [ 1.5,  20],
        [ 2,    15],
        [ 3.3, 10],
        [ 4,   8.1],
        [ 5,   6.3],
        [ 6,    5],
        [ 8,   3.5],
        [10,   2.5],
        [12,   1.7],
        [13.3,  0],
        [15, -2.5],
        [16,  -4 ],
        [17, -3.7],
        [18, -1.5],
        [19,  1.4],
        [20,   3.8],
        [21,   5 ],
        [22,   7.5],
        [23,  15 ],
        [24,  48 ],
        [24.5, 60 ],
        [25, 130 ]
    ], dtype=float)

    k = np.arange(N0, Ntop + 1, 1)
    MinExcdB = np.interp(Barkno[k], HTres[:, 0], HTres[:, 1])

    ## Initialize constants and variables

    dz = 0.5 # Barks
    z = np.arange(0.5, 23.5 + dz, dz) # frequency in Barks

    gr = np.array([
        [ 0, 1, 2.5, 4.9, 6.5, 8, 9, 10, 11, 11.5, 13, 17.5, 21, 24 ],
        [ 0, 0.35, 0.7, 0.7, 1.1, 1.25, 1.26, 1.18, 1.08, 1, 0.66, 0.46, 0.38, 0.3 ]
    ])

    xi   = np.arange(1, 48) / 2
    cubic = interp1d(gr[0], gr[1], kind='cubic',   # same spline as MATLAB
                 bounds_error=False, fill_value=np.nan)
    gzi  = np.sqrt(cubic(xi))

    # calculate a0
    a0tab = np.array([
        [ 0,     0],
        [10,   0],
        [12,   1.15],
        [13,   2.31],
        [14,   3.85],
        [15,   5.62],
        [16,   6.92],
        [16.5, 7.38],
        [17, 6.92],
        [18, 4.23],
        [18.5, 2.31],
        [19,   0],
        [20, -1.43],
        [21, -2.59],
        [21.5, -3.57],
        [22, -5.19],
        [22.5, -7.41],
        [23, -11.3],
        [23.5, -20],
        [24, -40],
        [25, -130],
        [26, -999]
    ], dtype=float)

    a0 = np.ones(N, dtype=float)
    a0[k] = db2mag(np.interp(Barkno[k], a0tab[:, 0], a0tab[:, 1]))

    ## BEGIN Hweights

    # weights for freq. bins < N/2

    def _build_H(proto: NDArray[np.floating]) -> Tuple[int, NDArray[np.floating]]:
        """Return last usable bin and dense weight curve for a prototype table."""
        f_proto, w_proto = proto[:, 0], proto[:, 1]

        last = int(np.floor((f_proto[-1] / fs) * N))
        k = np.arange(2, last + 1, 1)
        f = (k) * fs / N
        H = np.zeros(N)
        H[k] = np.interp(f, f_proto, w_proto)
        return last, H

    H2_proto = np.array([
        [ 0, 0],
        [17, 0.8],
        [23, 0.95],
        [25, 0.975],
        [32, 1],
        [37, 0.975],
        [48, 0.9],
        [67, 0.8],
        [90, 0.7],
        [114, 0.6],
        [171, 0.4],
        [206, 0.3],
        [247, 0.2],
        [294, 0.1],
        [358, 0]
    ], dtype=float)

    H5_proto = np.array([
        [ 0, 0],
        [32, 0.8],
        [43, 0.95],
        [56, 1],
        [69, 0.975],
        [92, 0.9],
        [120, 0.8],
        [142, 0.7],
        [165, 0.6],
        [231, 0.4],
        [277, 0.3],
        [331, 0.2],
        [397, 0.1],
        [502, 0]
    ], dtype=float)

    H16_proto = np.array([
        [ 0, 0],
        [23.5, 0.4],
        [34, 0.6],
        [47, 0.8],
        [56, 0.9],
        [63, 0.95],
        [79, 1],
        [100, 0.975],
        [115, 0.95],
        [135, 0.9],
        [159, 0.85],
        [172, 0.8],
        [194, 0.7],
        [215, 0.6],
        [244, 0.5],
        [290, 0.4],
        [348, 0.3],
        [415, 0.2],
        [500, 0.1],
        [645, 0]
    ], dtype=float)

    H21_proto = np.array([
        [ 0, 0],
        [19, 0.4],
        [44, 0.8],
        [52.5, 0.9],
        [58, 0.95],
        [75, 1],
        [101.5, 0.95],
        [114.5, 0.9],
        [132.5, 0.85],
        [143.5, 0.8],
        [165.5, 0.7],
        [197.5, 0.6],
        [241, 0.5],
        [290, 0.4],
        [348, 0.3],
        [415, 0.2],
        [500, 0.1],
        [645, 0]
    ], dtype=float)

    H42_proto = np.array([
        [ 0, 0],
        [15, 0.4],
        [41, 0.8],
        [49, 0.9],
        [53, 0.965],
        [64, 0.99],
        [71, 1],
        [88, 0.95],
        [94, 0.9],
        [106, 0.85],
        [115, 0.8],
        [137, 0.7],
        [180, 0.6],
        [238, 0.5],
        [290, 0.4],
        [348, 0.3],
        [415, 0.2],
        [500, 0.1],
        [645, 0]
    ], dtype=float)

    _, H2  = _build_H(H2_proto)
    _, H5  = _build_H(H5_proto)
    _, H16 = _build_H(H16_proto)
    _, H21 = _build_H(H21_proto)
    _, H42 = _build_H(H42_proto)

    Hweight = np.zeros((47, N))

    Hweight[1, :] = Hweight[2, :] = Hweight[3, :] = Hweight[0, :] = H2 # H1-H4
    Hweight[4, :] = H5
    Hweight[5:15, :] = H5  # H6 – H15
    Hweight[15, :]  = H16
    Hweight[16:20, :] = H16  # H17 – H20
    Hweight[20, :]  = H21
    Hweight[21:41, :] = H21  # H22 – H41
    Hweight[41, :]  = H42
    Hweight[42:, :] = H42  # H43 – H47

    ## BEGIN process window

    AmpCal  = db2mag(91.2)*2/(N*blackman(N, sym=False).mean())

    # Calibration between wav-level and loudness-level (assuming blackman window and FFT will follow)
    Chno = 47  # number of channels
    Cal  = 0.50  # calibration factor, twice the old value (0.25)
    qb = np.arange(N0, Ntop + 1, 1)
    freqs = (qb+2) * fs / N # TODO: potential conflict?
    hBPi  = np.zeros((Chno, N))
    hBPrms = np.zeros(Chno)
    mdept  = np.zeros(Chno)
    ki     = np.zeros(Chno - 2)
    ri     = np.zeros(Chno)
    ei    = np.zeros((Chno, N))
    Fei   = np.zeros((Chno, N), dtype=complex)

    startIndex = 0
    endIndex   = N

    samples = audio.size
    n_frames = int(np.floor((samples - N) / hopsize))
    TimePoints = np.empty(n_frames)
    R_mat      = np.empty(n_frames)
    SPL_mat    = np.empty(n_frames)
    ri_mat     = np.empty((Chno, n_frames))

    # for each frame
    for windowNum in range(n_frames):

        dataIn = audio[startIndex:endIndex] * window
        currentTimePoint = startIndex / fs

        # Calculate Excitation Patterns
        TempIn = AmpCal * dataIn
        TempIn = a0 * fft(TempIn, axis=0)
        Lg     = np.abs(TempIn[qb]) # get absolute value of fourier transform for  indices in range of human hearing
        LdB    = mag2db(Lg)
        whichL = np.nonzero(LdB > MinExcdB)[0] # extract indices where FFT magnitudes exceed excitation threshold
        sizL   = whichL.size # get number of frequencies where this holds

        if sizL == 0:
            # Entire spectrum below threshold – roughness 0
            R_mat[windowNum] = 0.0
            ri_mat[:, windowNum] = 0.0
            TimePoints[windowNum] = currentTimePoint
            SPL_mat[windowNum] = -400.0
            startIndex += hopsize
            endIndex   += hopsize
            continue
        
        # steepness of slopes (Terhardt)
        S1 = -27.0
        S2 = np.zeros(sizL) # preallocate

        for w in range(sizL):

            # Steepness of upper slope [dB/Bark] in accordance with Terhardt
            steep = -24.0 - (230.0 / freqs[w]) + 0.2 * LdB[whichL[w]]

            if steep < 0:
                S2[w] = steep # set S2 with steepness value calculated earlier

        whichZ = np.empty((2, sizL), dtype=int) # preallocate
        whichZ[0, :] = np.floor(2 * Barkno[whichL] ).astype(int) # get bark band numbers
        whichZ[1, :] = np.ceil (2 * Barkno[whichL] ).astype(int)

        ExcAmp = np.zeros((sizL, Chno))
        Slopes = np.zeros((sizL, Chno))

        for k_l in range(sizL): # loop over freq indices above threshold
            Ltmp = LdB[whichL[k_l]] # copy FFT magnitude (in dB) above threshold
            Btmp = Barkno[whichL[k_l]] # and the bark number associated with it

            for l in range(whichZ[0, k_l]): # loop up to floored bark number of freq index k
                Stemp = (S1 * (Btmp - (l * 0.5))) + Ltmp
                if Stemp > MinExcdB[l]:
                    Slopes[k_l, l] = db2mag(Stemp)
            
            for l in range(whichZ[1, k_l], Chno): # loop up to ceil'd bark number
                Stemp = (S2[k_l] * ((l * 0.5) - Btmp)) + Ltmp
                if Stemp > MinExcdB[l]:
                    Slopes[k_l, l] = db2mag(Stemp) # critical filterbank upper side


        for k_ch in range(Chno): # loop over each channel
            etmp = np.zeros(N, dtype=complex)
            for l in range(sizL): # for each l index of fft bin in human hearing freq range
                N1tmp = whichL[l] # get freq index of bin
                if whichZ[0, l] == k_ch:
                    ExcAmp[l, k_ch] = 1.0
                elif whichZ[1, l] == k_ch:
                    ExcAmp[l, k_ch] = 1.0
                elif whichZ[1, l] > k_ch:
                    ExcAmp[l, k_ch] = Slopes[l, k_ch + 1] / Lg[N1tmp]
                else:
                    ExcAmp[l, k_ch] = Slopes[l, k_ch - 1] / Lg[N1tmp]
                etmp[N1tmp] = ExcAmp[l, k_ch] * TempIn[N1tmp]

            # this is the specific excitation time function
            ei[k_ch, :] = N * np.real(ifft(etmp)) # ifft to get time domain blocks of signal
            etmp_abs = np.abs(ei[k_ch, :])
            h0 = etmp_abs.mean()
            Fei[k_ch, :] = fft(etmp_abs - h0)
            hBPi[k_ch, :] = 2 * np.real(ifft(Fei[k_ch, :] * Hweight[k_ch, :]))
            hBPrms[k_ch] = float(np.sqrt(np.mean(np.square(hBPi[k_ch, :], dtype=float), dtype=float)))

            if h0 > 0:
                mdept[k_ch] = hBPrms[k_ch] / h0
                mdept[k_ch] = min(mdept[k_ch], 1.0)
            else:
                mdept[k_ch] = 0.0

        # find cross-correlation coefficients
        for k in range(45):
            cfac = np.cov(hBPi[k, :], hBPi[k + 2, :])
            den = np.diag(cfac)
            den = np.sqrt(np.outer(den, den)).squeeze()
            ki[k] = cfac[0, 1] / den[0, 1] if den[0, 1] > 0 else 0.0

        # Calculate specific roughness ri and total roughness R
        ri[0] = (gzi[0] * mdept[0] * ki[0]) ** 2
        ri[1] = (gzi[1] * mdept[1] * ki[1]) ** 2

        for k in range(2, 45):
            ri[k] = (gzi[k] * mdept[k] * ki[k - 2] * ki[k]) ** 2

        ri[45] = (gzi[45] * mdept[45] * ki[43]) ** 2
        ri[46] = (gzi[46] * mdept[46] * ki[44]) ** 2

        ri *= Cal # appropriately scaled specific roughness
        R = dz * ri.sum() # total R = integration of the specific R pattern

        spl_rms = np.mean(float(np.sqrt(np.mean(np.square(dataIn, dtype=float), dtype=float))))
        SPL = mag2db(spl_rms) + 83 if spl_rms > 0 else -400.0 # -20 dBFS <--> 60 dB SPL

        # matrices to return
        R_mat[windowNum] = R
        ri_mat[:, windowNum] = ri
        SPL_mat[windowNum] = SPL

        TimePoints[windowNum] = currentTimePoint
        startIndex += hopsize
        endIndex   += hopsize

    # ------------------------------------------------------------------
    # output struct
    # ------------------------------------------------------------------
    OUT: Dict[str, Any] = {
        "InstantaneousRoughness": R_mat, # instantaneous roughness
        "InstantaneousSpecificRoughness": ri_mat, # time-varying specific roughness
        "TimeAveragedSpecificRoughness": ri_mat.mean(axis=1), # mean specific roughness
        "time": TimePoints, # time
        "barkAxis": z, # critical band rate (for specific roughness)
        "dz": dz,
    }

    # Roughness statistics based on InstantaneousRoughness

    idx_skip = int(np.searchsorted(TimePoints, time_skip, side="left")) # find idx of time_skip on time vector
    OUT.update(get_statistics(R_mat[idx_skip:], "Roughness_Daniel1997")) # get statistics

    # plots
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8), num="Roughness analysis – Daniel 1997")

        ax1 = plt.subplot(2, 2, (1, 2))
        ax1.plot(TimePoints, R_mat, "r-")
        ax1.set(xlabel="Time (s)", ylabel="Roughness, R (asper)", title="Instantaneous roughness")
        ax1.grid(True)

        ax2 = plt.subplot(2, 2, 3)
        ax2.plot(np.arange(1, 48) / 2, ri_mat.mean(axis=1), "r-")
        ax2.set(xlabel="Critical band, z (Bark)", ylabel="Specific roughness, R′ (asper/Bark)",
                title="Time‑averaged specific roughness")
        ax2.grid(True)

        ax3 = plt.subplot(2, 2, 4)
        xx, yy = np.meshgrid(TimePoints, z)
        pcm = ax3.pcolormesh(xx, yy, ri_mat, shading="auto")
        plt.colorbar(pcm, ax=ax3, label="Specific roughness, R′ (asper/Bark)")
        ax3.set(xlabel="Time (s)", ylabel="Critical band, z (Bark)",
                title="Instantaneous specific roughness")

        plt.tight_layout()
        plt.show()

    return OUT


# ----------------------
#### HELPER FUNCTIONS ####
# ----------------------

def il_SPL_excess(input):
    """
    Compute sound pressure level excess for tonal components.
    """
    pref = 2e-5  # reference pressure, Pa
    Intensity = pref * 10 ** (input["Lcrop"] / 10)

    freq_Lx = input["freq"] # freq vector of the tone
    ToneF = input["ToneF"] # tone(s) central frequency
    ToneL = input["ToneL"] # tone(s) level
    NTones = len(ToneF) # number of tones

    toneBark = hz2bark(ToneF) # convert central freq of tones to Bark scale
    spectrumBark = hz2bark(freq_Lx) # convert freq vector to Bark scale

    LX = np.zeros(NTones) # initialize sound pressure level excess vector

    for i in range(NTones):

        # Intensity of noise for each tone paragraph after eq 7b in Ref. [3] (Terhard's papers)

        idx_cb = (spectrumBark >= round(toneBark[i] - 0.5)) & (spectrumBark <= round(toneBark[i] + 0.5)) # idx of the critical band around the tonal component
        
        idx_toneBark = np.argmin(np.abs(spectrumBark - toneBark[i])) # find idx of the tone on the Bark vector
        
        idx_cb[idx_toneBark - 2 : idx_toneBark + 3] = False # skip the five central samples around the tonal component

        EGR = np.sum(Intensity[idx_cb]) # Masking intensity of broadband noise

        # Secondary excitation level
        sumlo = 1e-99
        sumhi = 1e-99

        for j in range(NTones):

            if j < i:
                s = -24 - (230 / ToneF[j]) + (0.2 * ToneL[j]) # eq 7b from Ref. [3]
                Lji = ToneL[j] - s * (toneBark[j] - toneBark[i])
                sumlo += 10 ** (Lji / 20)

            elif j > i:
                s = 27
                Lji = ToneL[j] - s * (toneBark[j] - toneBark[i])
                sumhi += 10 ** (Lji / 20)

        AEK = sumlo + sumhi

        # Intensity at threshold of hearing
        EHS = il_Threshold(ToneF[i])
        EHS = 10 ** (EHS / 10)

        # % Sound pressure level excess
        if NTones == 1: # if there is only one tone
            LXi = ToneL[i] - 10 * np.log10(EGR + EHS) # eq 4 from Ref. [3]
        else:
            LXi = ToneL[i] - 10 * np.log10(AEK ** 2 + EGR + EHS) # eq 4 from Ref. [3]

        if LXi > 0:
            LX[i] = LXi

    return LX

def il_tonal_weighting(input):
    """
    Compute tonal weighting for tonal components.
    """
    bw = input["BW"] # bandwidth of the tones [Hz]
    fc = input["ToneF"] # central frequency of the tonal components
    delta_L = input["LX"] # SPL excess for each tonal component
    df = input["df"] # freq discretization

    ## w1 accounts for each tonal component bandwidth

    zup = hz2bark(fc + (bw / 2))
    zlow = hz2bark(fc - (bw / 2))
    dz = (zup - zlow) / df ** 2

    w1 = 0.13 / (dz + 0.13)

    ## w2 accounts for each tonal component's center frequency
    w2 = (1 / np.sqrt(1 + 0.2 * (fc / 700 + 700 / fc) ** 2)) ** 0.29

    # w3 accounts for each tonal component SPL excess
    w3 = (1 - np.exp(-delta_L / 15)) ** 0.29

    # prime weightings
    ww1 = w1 ** (1 / 0.29)
    ww2 = w2 ** (1 / 0.29)
    ww3 = w3 ** (1 / 0.29)

    # total tonal weighting
    w_tonal = np.sqrt(np.sum((ww1 * ww2 * ww3) ** 2))

    return w_tonal

def il_Threshold(f):
    """
    Compute hearing threshold in dB for a given frequency.
    """
    f = f / 1000
    L = 3.64 * f ** -0.8 - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * f ** 4
    return L

def wav2sig(insig, fs=None, dBFS=94):
    """
    Load a WAV file and return the signal and sampling frequency.
    If fs is provided, resample the signal to that frequency.
    """
    fs, insig_raw = wavfile.read(insig)

    # Convert to float if needed
    if insig_raw.dtype.kind in 'iu':
        max_val = np.iinfo(insig_raw.dtype).max
        insig = insig_raw.astype(np.float32) / max_val
    else:
        insig = insig_raw.astype(np.float32)

    # dBFS scaling (default: 94 dB SPL full scale = 1 Pa)
    gain_factor = 10 ** ((dBFS - 94) / 20)
    insig = gain_factor * insig

    # If stereo, convert to mono
    if insig.ndim > 1:
        insig = np.mean(insig, axis=1)

    insig = np.asarray(insig)

    return insig, fs

def db2mag(x: FloatArray | float) -> FloatArray | float:  # linear amplitude
    return from_db(x, 20.0)  # wrapper around utilities.from_db

def mag2db(x: FloatArray | float, *, eps: float = 1e-12) -> FloatArray | float:
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), eps))

if __name__ == "__main__":
    # # generate test signal for Tonality_ECMA418_2
    # fs = 48000
    # t = np.arange(0, 5, 1/fs)
    # insig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)  # Example signal with two tones
    # insig = insig.reshape(-1, 1)  # Reshape to [N, 1] for single channel
    # # Call the Tonality_ECMA418_2 function
    # OUT = Tonality_ECMA418_2(insig, fs, fieldtype='free-frontal', time_skip=0.304, show=True)

    fs = 48_000
    f_mod = 70
    f_carrier = 1_000.0

    p_rms = 20e-6 * 10**(60 / 20)
    A = p_rms * np.sqrt(2)
    t = np.arange(0.0, 2, 1 / fs)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * f_mod * t))
    signal = A * envelope * np.sin(2 * np.pi * f_carrier * t)
    insig = signal.astype(np.float32)

    OUT = Roughness_Daniel1997(insig, fs, time_skip=0.0, show=True)

    print(f"Reference-tone check (expected ≈ 1 asper)")
    print(f"  Mean roughness  : {OUT['Rmean']} asper")
    print(f"  Max  roughness  : {OUT['Rmax']} asper")
    print(f"  10 % exceedance : {OUT['R10']} asper")