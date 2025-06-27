from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.interpolate import interp1d 
from scipy.fft import fft, ifft
from scipy.signal.windows import hann, blackman
from matplotlib import pyplot as plt
from scipy.signal import resample
import warnings
import sys

from sound_metrics import *
from utilities import *

__all__ = ["Loudness_ISO532_1", "EPNL_FAR_Part36"]
FloatArray = NDArray[np.floating]

# ----------------------
#### LOUDNESS METRICS ####
# ----------------------

def Loudness_ISO532_1(insig, fs=None, field=0, method=2, time_skip=0, show=False, dBFS=94, export_excel=None):
    """
    ISO 532-1 (2017) **Zwicker loudness model** for stationary and
    time-varying sounds.

    The function accepts either a calibrated pressure waveform or third-
    octave band levels and returns overall, specific, and (optionally)
    time-resolved loudness quantities.  Three operating modes are offered:

    =============  =======================================================
    **Method**      **Description**
    -------------  -------------------------------------------------------
    ``0``          *Stationary* – supply ⅓-octave **SPL** in ``insig``  
                   (shape ``(1, 24)`` … ``(n, 24)``; 50 Hz–12.5 kHz).
    ``1``          *Stationary (from audio)* – supply a mono waveform; levels
                   are averaged after filtering.
    ``2``          *Time-varying (from audio)* – mono waveform; outputs instantaneous
                   loudness, loudness level, and specific loudness versus
                   time.
    =============  =======================================================

    Parameters
    ----------
    insig : ndarray | str
        *Mode 0* – matrix of ⅓-octave **SPL** values (dB re 20 µPa).  
        *Modes 1–2* – mono pressure signal in **pascals** or a WAV-file
        path.
    fs : int | float, optional
        Sampling frequency of ``insig`` (Hz).  Ignored when a filename is
        given or in *mode 0*.
    field : {0, 1}, default ``0``
        Acoustic field (0 = free field, 1 = diffuse) used for diffuse-field
        corrections.
    method : {0, 1, 2}, default ``2``
        Processing mode (see table above).
    time_skip : float, default ``0``
        Seconds to omit at the beginning when computing statistics or
        stationary levels.
    show : bool, default ``False``
        Plot the main intermediate and final results.
    dBFS : float, default ``94``
        SPL represented by a full-scale sine when reading a WAV file.
    export_excel : str, optional
        Path to save all returned fields as an **.xlsx** workbook.

    Returns
    -------
    dict
        Dictionary containing instantaneous data and summary statistics.

    Raises
    ------
    ValueError
        If *fs* is missing for waveform input, if an invalid *method* is
        selected, or when the input dimensions are inconsistent.
    RuntimeWarning
        When **method 0** receives an array with ≠ 24 columns.

    Notes
    -----
    * Waveforms are resampled to **48 kHz** because the standardised
      filter-bank is defined for this rate.
    * Time-varying mode follows the non-linear temporal integration of
      Zwicker (T<sub>short</sub> = 5 ms, T<sub>var</sub> = 75 ms,
      T<sub>long</sub> = 15 ms).
    * All outputs use the Bark scale with 0.1-Bark resolution
      (``barkAxis`` = 0.1 … 24.0).
    """


    # --- WAV file interface ---
    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)
    elif fs is None:
        raise ValueError("If insig is not a filename, fs must be provided.")

    # --- Input validation ---
    if fs is None or fs <= 0:
        raise ValueError("Sampling frequency (fs) must be a positive number.")

    # --- Input handling ---
    insig = np.atleast_2d(np.asarray(insig))

    # Transpose based on method and current shape
    if (method == 0 and insig.shape[0] != 1) or (method != 0 and insig.shape[1] != 1):
        insig = insig.T

    
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
    barkAxis = np.linspace(0.1, 24.0, 240)  # bark vector

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
        # STEP 1 - Resample to 48 kHz if necessary
        # **************************************************
        if fs != 48000:
            gcd_fs = np.gcd(48000, int(fs))  # greatest common denominator
            insig = resample_poly(insig.flatten(), 48000 // gcd_fs, int(fs) // gcd_fs)
            fs = 48000
            insig = insig[:, None]
        len_sig = insig.shape[0]

        # Assign values to global variables according to the selected method
        if method == 1:  # stationary from audio signal
            SampleRateLevel = 1
            NumSamplesLevel = 1
            DecFactorLoudness = 1
        elif method == 2:  # time_varying from audio signal
            SampleRateLevel = SR_LEVEL
            SampleRateLoudness = SR_LOUDNESS
            DecFactorLevel = fs / SampleRateLevel
            DecFactorLoudness = SampleRateLevel / SampleRateLoudness
            NumSamplesLevel = int(np.ceil(len_sig / DecFactorLevel))
            NumSamplesLoudness = int(np.ceil(NumSamplesLevel / DecFactorLoudness))

        # **************************************************
        # STEP 2 - Create filter bank and filter the signal
        # **************************************************
        filteredaudio, fc = ob13_iso532_1(insig.flatten(), fs)

        # **************************************************
        # STEP 3 - Squaring and smoothing by 3 1st order lowpass filters
        # **************************************************
        filteredaudio = filteredaudio ** 2

        N_bands = len(fc)
        ThirdOctaveLevel = np.zeros((NumSamplesLevel, N_bands))
        CentreFrequency = fc

        if method == 2:  # time-varying from audio signal
            # Pre-compute time constants and filter coefficients
            Tau = np.where(CentreFrequency <= 1000, 2 / (3 * CentreFrequency), 2 / (3 * 1000.))
            A1 = np.exp(-1 / (fs * Tau))
            B0 = 1 - A1
            
            # Pre-compute decimation indices
            decimation_indices = np.clip(np.arange(NumSamplesLevel) * DecFactorLevel, 0, len_sig - 1).astype(int)
            
            # Pre-compute filter coefficients for all bands (optimized for lfilter)
            # For cascaded first-order filters: H(z) = B0/(1-A1*z^-1) applied 3 times
            a_coeffs = np.column_stack([np.ones(N_bands), -A1])  # [1, -A1] for each band
            b_coeffs = np.column_stack([B0, np.zeros(N_bands)])   # [B0, 0] for each band
            
            # Process all bands with optimized filtering
            for i in range(N_bands):
                # Apply 3 cascaded 1st order low-pass filters using scipy's lfilter (more efficient)
                smoothed = filteredaudio[:, i]
                for k in range(3):
                    smoothed = lfilter(b_coeffs[i], a_coeffs[i], smoothed)
                
                # Decimate and convert to dB
                ThirdOctaveLevel[:, i] = 10 * np.log10((smoothed[decimation_indices] + TINY_VALUE) / I_REF)

        elif method == 1:  # stationary from audio signal
            NumSkip = int(np.floor(time_skip * fs))
            if NumSkip > len_sig / 2:
                raise ValueError('time signal too short')
            NumSkip = max(NumSkip, 1)
            
            # Vectorized computation for all bands
            smoothedaudio = filteredaudio[NumSkip:len_sig, :]
            ThirdOctaveLevel[NumSamplesLevel - 1, :] = 10 * np.log10((np.mean(smoothedaudio, axis=0) + TINY_VALUE) / I_REF)

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
    DDF = np.array([0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3, 2, 0, -1.4, -2, -1.9, -1, 0.5, 3, 4, 4.3, 4])

    # LEVEL CORRECTIONS TABLE A.6 (LTQ)
    # Critical band level at absolute threshold without taking into account the
    # transmission characteristics of the ear
    LTQ = np.array([30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])  # Threshold due to internal noise
    # Hearing thresholds for the excitation levels (each number corresponds to a critical band 12.5kHz is not included)

    # LEVEL CORRECTIONS TABLE A.7 DCB
    # Correction factor because using third octave band levels (rather than critical bands)
    DCB = np.array([-0.25, -0.6, -0.8, -0.8, -0.5, 0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 0.8, 0.5, 0, -0.5])

    # LEVEL CORRECTIONS TABLE A.4 (A0)
    # Attenuation due to transmission in the middle ear
    A0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6, -3.2, -5.4, -5.6, -4, -1.5, 2, 5, 12])
    # Moore et al disagrees with this being flat for low frequencies

    Le = np.zeros((NumSamplesLevel, 20))
    CoreL = np.zeros((NumSamplesLevel, 21))

    # Vectorized processing - extract relevant third octave bands (columns 8-26)
    Le[:, :19] = ThirdOctaveLevel[:, 8:27]
    Le[:, :3] = LCB[:, :3]
    Le[:, :19] -= A0[:19]

    if field == 1:
        Le[:, :19] += DDF[:19]
    mask = Le[:, :19] > LTQ[:19]
    S = 0.25
    MP1 = 0.0635 * 10 ** (0.025 * LTQ[:19])
    Le_corrected = Le[:, :19] - DCB[:19]
    MP2 = (((1 - S) + S * 10 ** (0.1 * (Le_corrected - LTQ[:19]))) ** 0.25) - 1
    CoreL_temp = MP1 * MP2
    CoreL[:, :19] = np.where(mask, np.maximum(CoreL_temp, 0), 0)

    # *************************************************************************
    # STEP 7 - Correction of specific loudness within the lowest critical band
    # *************************************************************************

    for j in range(NumSamplesLevel):
        CorrCL = min(1, 0.4 + 0.32 * CoreL[j, 0] ** 0.2)
        CoreL[j, 0] *= CorrCL

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

        # Pre-calculate decimation factor as integer for efficiency
        dec_factor_int = int(DecFactorLevel)
        
        # 1st order low-pass A (optimized)
        Tau = 3.5e-3
        A1 = np.exp(-1 / (SampleRateLevel * DecFactorLevel * Tau))
        B0 = 1 - A1
        Y1 = 0

        # Optimized filtering loop A
        for i in range(NumSamplesLevel):
            X0 = N_mat[i]
            Y1 = B0 * X0 + A1 * Y1
            Loudness_t1[i] = Y1

            if i < NumSamplesLevel - 1:
                Xd = (N_mat[i + 1] - X0) / DecFactorLevel  # Fixed: use next sample
                X_interp = X0
                for j in range(dec_factor_int):
                    X_interp += Xd
                    Y1 = B0 * X_interp + A1 * Y1

        # 1st order low-pass B (optimized)
        Tau = 70e-3
        A1 = np.exp(-1 / (SampleRateLevel * DecFactorLevel * Tau))
        B0 = 1 - A1
        Y1 = 0

        # Optimized filtering loop B
        for i in range(NumSamplesLevel):
            X0 = N_mat[i]
            Y1 = B0 * X0 + A1 * Y1
            Loudness_t2[i] = Y1
            
            if i < NumSamplesLevel - 1:
                Xd = (N_mat[i + 1] - X0) / DecFactorLevel  # Fixed: use next sample
                X_interp = X0
                for j in range(dec_factor_int):
                    X_interp += Xd
                    Y1 = B0 * X_interp + A1 * Y1

        # Combine the filters (vectorized)
        Loudness = 0.47 * Loudness_t1 + 0.53 * Loudness_t2

        # Decimate signal for decreased computation time by factor of 24 (fs = 2 Hz)
        # Optimized decimation using numpy indexing
        indices = np.arange(0, NumSamplesLevel, DecFactorLoudness, dtype=int)[:NumSamplesLoudness]
        Total_Loudness = Loudness[indices]

        # Optimized ns decimation
        ns_dec = ns[indices, :]

        # **********************************************************************
        # Compute loudness level - conversion from sone to phon (vectorized)
        # ***********************************************************************

        # Vectorized loudness level computation
        LN = 40 * Total_Loudness ** 0.35
        mask_ge_1 = Total_Loudness >= 1
        LN[mask_ge_1] = 40 + 10 * np.log2(Total_Loudness[mask_ge_1])
        LN = np.maximum(LN, 3)  # Efficiently handles both LN < 0 and LN < 3 conditions


        # **********************************************************************
        # output struct for time-varying signals
        # ***********************************************************************
        OUT = {}
        OUT['barkAxis'] = barkAxis  # Bark vector
        OUT['time'] = np.linspace(0, len(Total_Loudness) - 1, len(Total_Loudness)) * 2e-3  # time vector of the final loudness calculation, in seconds
        OUT['time_insig'] = np.linspace(0, len(insig) - 1, len(insig)) / fs  # time vector of the audio input, in seconds
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
            OUT['time_insig'] = np.linspace(0, len(insig) - 1, len(insig)) / fs # time vector of the audio input, in seconds
        
        OUT['barkAxis'] = np.linspace(0.1, 24.0, 240) # Bark vector
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

def EPNL_FAR_Part36(insig=None, fs=None, method=None, dt=None, threshold=None, show=None, dBFS=94, export_excel=None):

    """
    Calculate the **Effective Perceived Noise Level (EPNL)** in accordance
    with FAR Part 36 / ICAO Annex 16, Appendix 2.

    The routine supports two input modes:

    =============  =======================================================
    **Method**      **Description**
    -------------  -------------------------------------------------------
    ``0``          *Spectral Mode* – ``insig`` is a 2-D array of third-octave-band *SPL* values ``shape == (n_time, 24)``, covering 50 Hz – 10 kHz.
    ``1``          *Waveform Mode* – ``insig`` is a calibrated pressure signal in **pascals** (mono).
    =============  =======================================================

    A ⅓-octave filter-bank is applied internally.

    In both cases the time axis is segmented into blocks of length *dt*
    seconds, from which the algorithm derives:

    * Perceived noisiness ``PN`` (Noys)  
    * Perceived noise level ``PNL`` (PNdB)  
    * Tone-corrected noise level ``PNLT`` (TPNdB)  
    * Duration correction *D* (dB)

    Finally,

    ``EPNL = max(PNLT) + D``  [EPNdB]

    Parameters
    ----------
    insig : ndarray | str
        *Waveform mode* – mono pressure signal (Pa).  
        *Spectral mode* – 2-D matrix of SPLs (dB re 20 µPa).  
        If a string is supplied it is treated as a WAV-filename.
    fs : int | float
        Sampling rate of ``insig`` in hertz (ignored in spectral mode).
    method : {0, 1}, optional
        0 = spectral input, 1 = waveform input.  If omitted the mode is
        inferred (waveform assumed when ``insig`` is 1-D).
    dt : float, default ``0.5``
        Time step between successive analysis blocks (seconds).
    threshold : float, default ``10``
        Tone-correction threshold *PNLT<sub>M</sub> − threshold* (dB).
    show : bool, default ``False``
        Plot intermediate results and diagnostic figures.
    dBFS : float, default ``94``
        SPL represented by a full-scale sine (waveform mode only).
    export_excel : str, optional
        Path where results are written as an **.xlsx** workbook.

    Returns
    -------
    dict
        Dictionary containing instantaneous data and summary statistics.

    Notes
    -----
    * Waveforms are resampled to **48 kHz** for compatibility with the ISO
      532-1 filter-bank.
    * If ``method = 0`` the input must contain **exactly 24 columns**
      (50 Hz … 10 kHz).  Otherwise a :class:`RuntimeWarning` is issued and
      execution stops.
    * Duration correction *D* is computed between the points *t₁* and *t₂*
      where *PNLT* crosses ``PNLTM – threshold``.
    """

    # Handle input arguments similar to MATLAB's nargin
    num_args = sum(x is not None for x in [insig, fs, method, dt, threshold, show])

    # Set defaults for missing parameters in one consolidated block
    if dt is None:
        dt = 0.5
    if threshold is None:
        threshold = 10
    if show is None:
        show = False

    # Streamlined default assignment logic
    if num_args < 3:  # default situation where insig is a sound file
        method = 1
        dt = 0.5
        threshold = 10
        show = False
    elif method == 0 and num_args < 4:  # default situation for method == 0
        dt = 0.5
        threshold = 10
        show = False

    if method == 0:
        if insig.shape[1] != 24:  # insig matrix needs to have nFreq=24 columns
            warnings.warn('For method=0, the insig matrix should have nFreq=24 columns, which corresponds to 1/3 oct. bands from 50 Hz to 10 kHz. Please check the input matrix for the correct dimension!!!')
            return

    ##  insig pre-processing stage

    fc_TOB = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000])  # nominal center freq - preferred for freq. labeling (check Tabel E.1. of IEC 61260-1:2014)

    num_freqs = len(fc_TOB)  # number of freq bands = nFreq

    OUT = {}

    if method == 0:  # insig is a [nTime,nFreq] matrix containing nFreq=24 columns containing unweighted SPL values for each third octave band from 50 Hz to 10 kHz

        SPL_TOB_spectra = insig
        num_times = SPL_TOB_spectra.shape[0]  # number of time steps

        if num_freqs != SPL_TOB_spectra.shape[1]:  # insig matrix needs to have nFreq=24 columns
            warnings.warn('For method=0, the insig matrix should have nFreq=24 columns, which corresponds to 1/3 oct. bands from 50 Hz to 10 kHz. Please check the input matrix for the correct dimension!!!')
            return

        # Optimized SPL calculation with better numerical stability
        InstantaneousSPL = 10.0 * np.log10(np.sum(np.power(10.0, SPL_TOB_spectra * 0.1), axis=1))

        # Optimized time vector creation
        time = np.arange(num_times) * dt

        # OUTPUT
        OUT['InstantaneousSPL'] = InstantaneousSPL
        OUT['time'] = time
        OUT['TOB_freq'] = fc_TOB

    elif method == 1:  # insig is a [nTime,1] array corresponding to a calibrated audio signal (Pa)

        # Optimized dimension correction
        if insig.ndim == 2 and insig.shape[1] != 1:
            insig = insig.T
        elif insig.ndim == 1:
            insig = insig.reshape(-1, 1)

        # resample to 48 kHz if necessary
        if fs != 48000:
            insig_flat = insig.flatten()
            insig = resample(insig_flat, int(len(insig_flat) * 48000 / fs)).reshape(-1, 1)
            fs = 48000
            print(f'\n{sys._getframe().f_code.co_name}: The 1/3 octave band filter bank used in this script has only been validated at a sampling frequency fs=48 kHz, resampling to this fs value\n')

        len_insig = insig.shape[0]  # length of the (resample) input vector
        I_REF = 4e-10  # ref. pressure^2
        TINY_VALUE = 1e-12  # small value to avoid inf SPL values

        # filter insig to get 1/3-OB
        fmin = 50  # min freq of 1/3-OB is 50 Hz
        fmax = 10000  # max freq of 1/3-OB is 10 kHz

        insig_P_TOB, _ = ob13_iso532_1(insig, fs, fmin, fmax)  # get 1/3-OB spectra from insig - output is p [nTime,nFreq]

        # Optimized power calculation
        insig_Psquared_TOB = np.square(insig_P_TOB)

        # Optimized SPL calculation with vectorized operations
        InstantaneousSPL_insig = 10 * np.log10((np.sum(insig_Psquared_TOB, axis=1) + TINY_VALUE) / I_REF)
        time_insig = np.arange(1, len_insig + 1) / fs  # optimized time vector

        # calculate SPL in dt steps
        Nbins = round(fs * dt)  # define dt in N bins
        num_times = int(np.ceil(len_insig / Nbins))  # number of time steps of the signal in N blocks

        # Optimized buffering operation - vectorized instead of loop
        # Pad the signal to make it divisible by Nbins
        pad_length = num_times * Nbins - len_insig
        if pad_length > 0:
            insig_Psquared_TOB_padded = np.pad(insig_Psquared_TOB, ((0, pad_length), (0, 0)), mode='constant')
        else:
            insig_Psquared_TOB_padded = insig_Psquared_TOB[:num_times * Nbins]
        
        # Reshape and calculate mean in one vectorized operation
        buffered_data = insig_Psquared_TOB_padded.reshape(num_times, Nbins, num_freqs)
        Psquared_TOB = np.mean(buffered_data, axis=1)  # output is p^2[nTime*,nFreq]

        # Optimized SPL calculations
        SPL_TOB_spectra = 10 * np.log10((Psquared_TOB + TINY_VALUE) / I_REF)  # main SPL[nTime*,nFreq] matrix
        InstantaneousSPL = 10 * np.log10((np.sum(Psquared_TOB, axis=1) + TINY_VALUE) / I_REF)  # overall SPL vs. time

        # Optimized time vector calculation
        time_duration = time_insig[-1] - time_insig[0]
        time_steps = int(round(time_duration / dt)) + 1
        time = np.linspace(time_insig[0], time_insig[-1], min(time_steps, num_times))
        
        # Ensure time vector matches num_times
        if len(time) < num_times:
            time = np.linspace(time_insig[0], time_insig[-1], num_times)

        # OUTPUT - quantities from the original insig
        OUT['InstantaneousSPL_insig'] = InstantaneousSPL_insig
        OUT['time_insig'] = time_insig

        # OUTPUT - quantities averaged in dt time steps
        OUT['InstantaneousSPL'] = InstantaneousSPL
        OUT['time'] = time
        OUT['SPL_TOB_spectra'] = SPL_TOB_spectra
        OUT['TOB_freq'] = fc_TOB

    ## Calculate EPNL

    # Convert SPL to Perceived Noisiness (PN) and compute Perceived Noisiness Level (PNL)
    PN, PNL, PNLM, PNLM_idx = get_PNL(SPL_TOB_spectra)

    # Calculate tone-correction and Tone-Corrected Perceived Noise Level (PNLT)
    PNLT, PNLTM, PNLTM_idx, _ = get_PNLT(SPL_TOB_spectra, fc_TOB, PNL)

    # Calculate duration correction factor
    D, idx_t1, idx_t2 = get_Duration_Correction(PNLT, PNLTM, PNLTM_idx, dt, threshold)

    # Calculate Effective Perceived Noise Level, unit is EPNdB
    OUT['EPNL'] = PNLTM + D

    # Print calculated EPNL value
    print(f'\nThe calculated EPNL is {OUT["EPNL"]:.4g} (EPNdB)\n')

    # OUTPUTS
    OUT['PN'] = PN  # PERCEIVED NOISINESS, unit is Noys
    OUT['PNL'] = PNL  # PERCEIVED NOISE LEVEL, unit is PNdB
    OUT['PNLM'] = PNLM  # MAXIMUM PERCEIVED NOISE LEVEL, unit is PNdB
    OUT['PNLT'] = PNLT  # TONE-CORRECTED PERCEIVED NOISE LEVEL, unit is TPNdB
    OUT['PNLTM'] = PNLTM  # MAXIMUM TONE-CORRECTED PERCEIVED NOISE LEVEL (PNLTM)

    ##  Show plots

    if show == True:

        xmax = time[-1]  # used to define the x-axis on the plots

        if method == 0:

            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('EPNL calculation based on an input SPL matrix')

            # plot instantaneous sound pressure level (dBSPL) from original signal and time-averaged over a given dt value
            ax1 = plt.subplot(2, 6, (1, 2))
            plt.plot(time, InstantaneousSPL, linewidth=2)
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('SPL, $L_{\\mathrm{p}}$ (dB re 20~$\\mu$Pa)')
            plt.grid(True)
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3] * 1.1])
            plt.title('Instantaneous overall SPL (1/3 oct. bands)')

            # plot spectrogram (1/3 octave bands in dt time steps)
            ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=4)  # Spans columns 3 to 6 in the first row
            fnom = fc_TOB / 1000  # convert center freq to kHz to plot 
            xx, yy = np.meshgrid(time, fnom)
            pcm = plt.pcolormesh(xx, yy, SPL_TOB_spectra.T, shading='auto')
            plt.colorbar(pcm)
            plt.axis('tight')
            plt.set_cmap('jet')

            # freq labels
            ytick_vals = np.concatenate([fnom[:1], fnom[13:14], fnom[16:24]])
            plt.yticks(ytick_vals)
            plt.ylabel('Center frequency, $f$ (kHz)')
            ax2.set_yscale('linear')
            
            plt.xlabel('Time, $t$ (s)')
            plt.colorbar().set_label('SPL, $L_{\\mathrm{p}}$ (dB re 20~$\\mu$Pa)')
            plt.clim([0, np.max(SPL_TOB_spectra)])
            plt.title(f'Spectrogram (1/3 oct. bands, dt={dt:.4g} sec)')

            # plot perceived noisiness (noys vs. time)
            ax3 = plt.subplot(2, 6, (7, 8))
            plt.plot(time, PN)
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('PN (noys)')
            plt.grid(True)
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3] * 1.1])
            plt.title('Perceived noisiness')

            # plot perceived noise level (PNdB vs. time)
            ax4 = plt.subplot(2, 6, (9, 10))
            plt.plot(time, PNL)
            a = plt.plot(time[PNLM_idx], PNLM, 'ro', markersize=8)
            plt.legend([f'PNLM={PNLM:.4g} (PNdB)'])
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('PNL (PNdB)')
            plt.grid(True)
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3] * 1.1])
            plt.title('Perceived noise level')

            # plot tone-corrected perceived noise level (TPNdB vs. time)
            ax5 = plt.subplot(2, 6, (11, 12))
            plt.plot(time, PNLT)
            a = plt.plot(time[PNLTM_idx], PNLTM, 'ro', markersize=8)
            b = plt.axhline(y=PNLTM - threshold, color='r', linestyle='-')
            c = plt.plot(time[idx_t1], PNLT[idx_t1], 'r*', markersize=10)
            plt.plot(time[idx_t2], PNLT[idx_t2], 'r*', markersize=10)

            plt.legend([f'PNLTM={PNLM:.4g} (TPNdB)',
                       f'PNLTM-{threshold:.2g}={PNLM - threshold:.4g} (TPNdB)',
                       'PNLT(t1) and PNLT(t2)'], loc='lower left')

            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('PNLT (TPNdB)')
            plt.grid(True)
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3] * 1.05])
            plt.title(f'Tone-corrected perceived noise level - EPNL={OUT["EPNL"]:.4g} (EPNdB)')

            plt.tight_layout()

        elif method == 1:

            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('EPNL calculation based on an input sound file')

            # plot input signal
            ax1 = plt.subplot(2, 6, (1, 2))
            plt.plot(time_insig, insig.flatten())
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Sound pressure, $p$ (Pa)')
            max_insig = np.max(insig)
            plt.axis([0, xmax, max_insig * -2, max_insig * 2])
            plt.title('Input signal')

            # plot instantaneous sound pressure level (dBSPL) from original signal and time-averaged over a given dt value
            ax2 = plt.subplot(2, 6, (3, 4))
            plt.plot(time_insig, InstantaneousSPL_insig)
            plt.plot(time, InstantaneousSPL, linewidth=2)
            plt.legend([f'dt={1/fs:.4g} sec', f'dt={dt:.4g} sec'], loc='lower left')
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('SPL, $L_{\\mathrm{p}}$ (dB re 20~$\\mu$Pa)')
            plt.grid(True)
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3] * 1.1])
            plt.title('Instantaneous overall SPL (1/3 oct. bands)')

            # plot spectrogram (1/3 octave bands in dt time steps)
            ax3 = plt.subplot(2, 6, (5, 6))
            fnom = fc_TOB / 1000  # convert center freq to kHz to plot 
            xx, yy = np.meshgrid(time, fnom)
            pcm = plt.pcolormesh(xx, yy, SPL_TOB_spectra.T, shading='auto')
            plt.colorbar(pcm)
            plt.axis('tight')
            plt.set_cmap('jet')

            # freq labels
            ytick_vals = np.concatenate([fnom[:1], fnom[13:14], fnom[16:24]])
            plt.yticks(ytick_vals)
            plt.ylabel('Center frequency, $f$ (kHz)')
            
            plt.xlabel('Time, $t$ (s)')
            plt.colorbar().set_label('SPL, $L_{\\mathrm{p}}$ (dB re 20~$\\mu$Pa)')
            plt.clim([0, np.max(SPL_TOB_spectra)])
            plt.title(f'Spectrogram (1/3 oct. bands, dt={dt:.4g} sec)')

            # plot perceived noisiness (noys vs. time)
            ax4 = plt.subplot(2, 6, (7, 8))
            plt.plot(time, PN)
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('PN (noys)')
            plt.grid(True)
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3] * 1.1])
            plt.title('Perceived noisiness')

            # plot perceived noise level (PNdB vs. time)
            ax5 = plt.subplot(2, 6, (9, 10))
            plt.plot(time, PNL)
            a = plt.plot(time[PNLM_idx], PNLM, 'ro', markersize=8)
            plt.legend([f'PNLM={PNLM:.4g} (PNdB)'])
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('PNL (PNdB)')
            plt.grid(True)
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3] * 1.1])
            plt.title('Perceived noise level')

            # plot tone-corrected perceived noise level (TPNdB vs. time)
            ax6 = plt.subplot(2, 6, (11, 12))
            plt.plot(time, PNLT)
            a = plt.plot(time[PNLTM_idx], PNLTM, 'ro', markersize=8)
            b = plt.axhline(y=PNLTM - threshold, color='r', linestyle='-')
            c = plt.plot(time[idx_t1], PNLT[idx_t1], 'r*', markersize=10)
            plt.plot(time[idx_t2], PNLT[idx_t2], 'r*', markersize=10)

            plt.legend([f'PNLTM={PNLM:.4g} (TPNdB)',
                       f'PNLTM-{threshold:.2g}={PNLM - threshold:.4g} (TPNdB)',
                       'PNLT(t1) and PNLT(t2)'], loc='lower left')

            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('PNLT (TPNdB)')
            plt.grid(True)
            ax = plt.axis()
            plt.axis([0, xmax, ax[2], ax[3] * 1.05])
            plt.title(f'Tone-corrected perceived noise level - EPNL={OUT["EPNL"]:.4g} (EPNdB)')

            plt.tight_layout()

        plt.show()

    if export_excel is not None:
        export_dict_to_excel(OUT, filename=f"{export_excel}")

    return OUT

check_which = 1

if __name__ == "__main__":
    if check_which == 0: # NO TEST

        print("metrics_loudness.py")

    if check_which == 1: # Loudness_ISO532_1
        with_wavfile = 0

        """
        Validation clip for Loudness_ISO532_1
        -----------------------------------

        Generates a 5-second, 1 kHz sinusoid at 70 dB SPL, sampled at 48 kHz.
        """

        print("Running Loudness_ISO532_1 test...")

        fs = 48_000                      # sampling rate expected by the OB-filter bank
        duration = 5.0                   # seconds
        f_tone = 1_000                   # 1-kHz pure tone
        desired_spl = 40                                   # target acoustic level

        a_rms_pa = 2e-5 * 10**(desired_spl / 20)           # RMS pressure in pascals
        a_peak_pa = a_rms_pa * np.sqrt(2)                  # peak
        fullscale_pa = 2e-5 * 10**(94 / 20)                # 94 dB SPL corresponds to |x| = 1
        amplitude = a_peak_pa / fullscale_pa               # peak value in ±1 full-scale units

        t = np.linspace(0, duration, int(duration*fs), endpoint=False)  # time vector
        tone = amplitude * np.sin(2 * np.pi * f_tone * t)
        tone = tone.astype(np.float32)

        if with_wavfile == 1:
            wavfile.write("test_L1.wav", fs, tone)
            OUT = Loudness_ISO532_1(
                "test_L1.wav",
                fs,
                field=0,            # free-field
                method=2,           # time-varying
                time_skip=0,        # process whole signal
                show=True           # draw summary plots
            )
            os.remove("test_L1.wav") if os.path.exists("test_L1.wav") else None

        else:
            OUT = Loudness_ISO532_1(
                tone,
                fs,
                field=0,            # free-field
                method=2,           # time-varying
                time_skip=0,        # process whole signal
                show=True           # draw summary plots
            )

        print(f"Overall loudness (median of time-series): {np.median(OUT['InstantaneousLoudness']):.2f} sone")
        print(f"5-percentile loudness N5:  {OUT['N5'].item():.2f} sone")
        print(f"95-percentile loudness N95: {OUT['N95'].item():.2f} sone")
        print(f"Loudness level (median):   {np.median(OUT['InstantaneousLoudnessLevel']):.2f} phon")

    elif check_which == 2: # EPNL_FAR_Part36

        """
        Validation clip for EPNL_FAR_Part36
        -----------------------------------
        * broadband roar that rises, cruises, then decays (≈ aircraft fly-over)
        * a steady 800 Hz tone 20 dB above the surrounding band (forces tone
        correction logic)
        
        The whole signal lasts 20 s, is sampled at 48 kHz, and peaks at ≈90 dB
        overall SPL.  The script runs the function with its default parameters
        (method 1, dt = 0.5 s, threshold = 10 dB), prints the resulting EPNL
        and shows the built-in diagnostic plots.
        """
        
        print("Running EPNL_FAR_Part36 test...")

        fs          = 48_000          # Hz – the filter bank is validated at 48 kHz
        dur_total   = 20.0            # s   – total length
        tone_freq   = 800.0           # Hz  – a typical fan/blade tone
        spl_broad   = 90.0            # dB  – peak broadband SPL
        spl_tone    = spl_broad - 20  # dB  – tone 20 dB weaker than the overall
        dBFS        = 94.0            # Full-scale reference used by library

        pref        = 2e-5                          # Pa
        FS_pa       = pref * 10**(dBFS/20)         # 1.0 digital  ↔  94 dB SPL (rms)

        t           = np.linspace(0, dur_total, int(fs*dur_total), endpoint=False)
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
                show      = True  # let the function draw its figures
            )

        print(f"EPNL of validation clip: {OUT['EPNL']} EPNdB")
