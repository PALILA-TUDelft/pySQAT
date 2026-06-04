from __future__ import annotations
from typing import Dict, Any, Tuple
import math

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft
from scipy.signal.windows import blackman
from matplotlib import pyplot as plt
import csv
from fractions import Fraction
from scipy.signal import resample_poly

from sound_metrics import *
from utilities import *

__all__ = ["Roughness_Daniel1997"]
FloatArray = NDArray[np.floating]

# -------------------------
#### ROUGHNESS METRICS ####
# -------------------------

def Roughness_Daniel1997(insig: ArrayLike, fs: int, time_skip: float = 0, show: bool = False, dBFS = 94) -> Dict[str, NDArray]:
    """
    This function calculates time-varying roughness and time-averaged specific
    roughness using the roughness model by Daniel & Weber:
    Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation
    of an optimized model. Acustica(83), 113-123.

    Reference signal: 60 dB 1 kHz tone 100% modulated at 70 Hz should yield 1 asper.

    INPUT:
    insig : array [Nx1]
    acoustic signal, monophonic (Pa)

    fs : integer
    sampling frequency (Hz)

    time_skip : integer
    skip start of the signal in <time_skip> seconds for statistics calculations

    show : logical(boolean)
    optional parameter for figures (results) display
    'false' (disable, default value) or 'true' (enable).

    OUTPUT:
    OUT : dict containing the following fields

        * InstantaneousRoughness: instantaneous roughness (asper) as a 
          function of time
        * InstantaneousSpecificRoughness: specific roughness(asper/Bark) as
          a function of time and frequency (Bark scale)
        * TimeAveragedSpecificRoughness: time-averaged specific roughness 
          (asper/Bark) as a function of frequency (Bark scale)
        * barkAxis : vector of Bark band numbers used for the computation
          of specific roughness computation
        * time : time vector in seconds
        * Several statistics based on the InstantaneousRoughness
          ** Rmean : mean value of instantaneous roughness (asper)
          ** Rstd : standard deviation of instantaneous roughness (asper)
          ** Rmax : maximum of instantaneous roughness (asper)
          ** Rmin : minimum of instantaneous roughness (asper)
          ** Rx : roughness value exceeded during x percent of the time (asper)
    """
    
    # window settings
    time_resolution = 0.2  # time-step for the windowing
    N = int(fs * time_resolution)  # window length

    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)

    audio = np.asarray(insig).copy()

    if audio.ndim > 1 and audio.shape[1] != 1:  # if the insig is not a [Nx1] array
        audio = audio.T  # correct the dimension of the insig
    
    if audio.ndim > 1:
        audio = audio.flatten()

    hopsize = N // 2  # hopsize is the number of samples hop between successive windows (window length is 8192).
    window = blackman(N)

    # resampling input signal
    if not (fs == 44100 or fs == 40960 or fs == 48000):
        gcd_fs = math.gcd(48000, fs)
        audio = resample_poly(audio, 48000 // gcd_fs, fs // gcd_fs)
        fs = 48000

    samples = len(audio)
    n = int((samples - N) // hopsize)
    if n < 0: # Handle cases where signal is too short for even one window
        n = 0

    Bark = np.array([
        [0,     0,     50,    0.5],
        [1,   100,    150,    1.5],
        [2,   200,    250,    2.5],
        [3,   300,    350,    3.5],
        [4,   400,    450,    4.5],
        [5,   510,    570,    5.5],
        [6,   630,    700,    6.5],
        [7,   770,    840,    7.5],
        [8,   920,   1000,    8.5],
        [9,  1080,   1170,    9.5],
        [10, 1270,   1370,   10.5],
        [11, 1480,   1600,   11.5],
        [12, 1720,   1850,   12.5],
        [13, 2000,   2150,   13.5],
        [14, 2320,   2500,   14.5],
        [15, 2700,   2900,   15.5],
        [16, 3150,   3400,   16.5],
        [17, 3700,   4000,   17.5],
        [18, 4400,   4800,   18.5],
        [19, 5300,   5800,   19.5],
        [20, 6400,   7000,   20.5],
        [21, 7700,   8500,   21.5],
        [22, 9500,  10500,   22.5],
        [23, 12000, 13500,   23.5],
        [24, 15500, 20000,   24.5]
    ])

    N2 = N // 2 + 1
    dFs = fs / N
    
    # Create Bark2 by combining and sorting columns
    col1 = np.concatenate([Bark[:, 1], Bark[:, 2]])
    col2 = np.concatenate([Bark[:, 0], Bark[:, 3]])
    sort_indices = np.argsort(col1)
    Bark2 = np.column_stack([col1[sort_indices], col2[sort_indices]])

    N0 = int(np.round(20 * N / fs))  # low frequency index @ 20 Hz (0-based)
    N01 = N0
    Ntop = int(np.round(20000 * N / fs))  # high frequency index @ 20 kHz?


    # Make list with Barknumber of each frequency bin
    Barkno = np.zeros(N2)
    f = np.arange(N0, Ntop + 1)
    if len(f) > 0:
        Barkno[f] = np.interp(f * dFs, Bark2[:, 0], Bark2[:, 1])

    # Make list of frequency bins closest to Cf's
    Cf = np.ones((2, 24))
    for a in range(24):
        Cf[0, a] = int(np.round(Bark[a + 1, 1] * N / fs)) - N0
        Cf[1, a] = Bark[a + 1, 1]

    # Make list of frequency bins closest to Critical Band Border frequencies
    Bf = np.ones((2, 25))
    Bf[0, 0] = int(np.round(Bark[0, 2] * N / fs))

    for a in range(24):
        Bf[0, a + 1] = int(np.round(Bark[a + 1, 2] * N / fs)) - N0
        if a < 24: # This condition is always true for a in range(24)
            Bf[1, a] = Bf[0, a] - 1

    Bf[1, 24] = int(np.round(Bark[24, 2] * N / fs)) - N0

    # Make list of minimum excitation (Hearing Threshold)
    HTres = np.array([
        [0,      130],
        [0.01,    70],
        [0.17,    60],
        [0.8,     30],
        [1,       25],
        [1.5,     20],
        [2,       15],
        [3.3,     10],
        [4,      8.1],
        [5,      6.3],
        [6,        5],
        [8,      3.5],
        [10,     2.5],
        [12,     1.7],
        [13.3,     0],
        [15,    -2.5],
        [16,      -4],
        [17,    -3.7],
        [18,    -1.5],
        [19,     1.4],
        [20,     3.8],
        [21,       5],
        [22,     7.5],
        [23,      15],
        [24,      48],
        [24.5,    60],
        [25,     130]
    ])

    k_ht = np.arange(N0, Ntop + 1)
    if len(k_ht) > 0:
        MinExcdB = np.interp(Barkno[k_ht], HTres[:, 0], HTres[:, 1])
    else:
        MinExcdB = np.array([])

    # Initialize constants and variables
    dz = 0.5  # Barks
    z = np.arange(0.5, 24, dz)  # frequency in Barks
    
    zb = np.sort(np.concatenate([Bf[0, :], Cf[0, :]])).astype(int)
    
    # MATLAB indexes MinExcdB with zb directly after building the band table.
    if len(MinExcdB) > 0:
        zb_clamped = np.clip(zb, 0, len(MinExcdB) - 1)
        MinBf = MinExcdB[zb_clamped]
    else:
        MinBf = np.zeros_like(zb, dtype=float)

    ei = np.zeros((47, N))
    Fei = np.zeros((47, N), dtype=complex)

    gr = np.array([
        [0, 1, 2.5, 4.9, 6.5, 8, 9, 10, 11, 11.5, 13, 17.5, 21, 24],
        [0, 0.35, 0.7, 0.7, 1.1, 1.25, 1.26, 1.18, 1.08, 1, 0.66, 0.46, 0.38, 0.3]
    ])

    gzi = np.zeros(47)
    h0 = np.zeros(47)
    # MATLAB: k = 1:1:47; gzi(k) = sqrt(interp1(gr(1,:),gr(2,:),k/2,'spline'))
    # The curve must be sampled at k/2 = 0.5, 1.0, ..., 23.5 (1-based k), not
    # at 0, 0.5, ..., 23.0. Sampling half a Bark too low zeroes gzi[0] and
    # undervalues the steeply-rising low-frequency channels.
    k_range = np.arange(1, 48)

    # Use cubic spline interpolation
    interp_func = interp1d(gr[0, :], gr[1, :], kind='cubic', fill_value='extrapolate')
    gzi = np.sqrt(interp_func(k_range / 2))


    # calculate a0
    a0tab = np.array([
        [0,      0],
        [10,     0],
        [12,  1.15],
        [13,  2.31],
        [14,  3.85],
        [15,  5.62],
        [16,  6.92],
        [16.5, 7.38],
        [17,  6.92],
        [18,  4.23],
        [18.5, 2.31],
        [19,     0],
        [20, -1.43],
        [21, -2.59],
        [21.5, -3.57],
        [22, -5.19],
        [22.5, -7.41],
        [23, -11.3],
        [23.5,  -20],
        [24,    -40],
        [25,   -130],
        [26,   -999]
    ])

    a0 = np.ones(N)
    k_a0 = np.arange(N0, Ntop + 1)
    if len(k_a0) > 0:
        a0[k_a0] = from_db(np.interp(Barkno[k_a0], a0tab[:, 0], a0tab[:, 1]))

    # weights for freq. bins < N/2
    DCbins = 2

    H2 = np.array([
        [0,       0],
        [17,    0.8],
        [23,   0.95],
        [25,  0.975],
        [32,      1],
        [37,  0.975],
        [48,    0.9],
        [67,    0.8],
        [90,    0.7],
        [114,   0.6],
        [171,   0.4],
        [206,   0.3],
        [247,   0.2],
        [294,   0.1],
        [358,     0]
    ])

    H5 = np.array([
        [0,       0],
        [32,    0.8],
        [43,   0.95],
        [56,      1],
        [69,  0.975],
        [92,    0.9],
        [120,   0.8],
        [142,   0.7],
        [165,   0.6],
        [231,   0.4],
        [277,   0.3],
        [331,   0.2],
        [397,   0.1],
        [502,     0]
    ])

    H16 = np.array([
        [0,       0],
        [23.5,  0.4],
        [34,    0.6],
        [47,    0.8],
        [56,    0.9],
        [63,   0.95],
        [79,      1],
        [100, 0.975],
        [115,  0.95],
        [135,   0.9],
        [159,  0.85],
        [172,   0.8],
        [194,   0.7],
        [215,   0.6],
        [244,   0.5],
        [290,   0.4],
        [348,   0.3],
        [415,   0.2],
        [500,   0.1],
        [645,     0]
    ])

    H21 = np.array([
        [0,       0],
        [19,    0.4],
        [44,    0.8],
        [52.5,  0.9],
        [58,   0.95],
        [75,      1],
        [101.5, 0.95],
        [114.5,  0.9],
        [132.5, 0.85],
        [143.5,  0.8],
        [165.5,  0.7],
        [197.5,  0.6],
        [241,    0.5],
        [290,    0.4],
        [348,    0.3],
        [415,    0.2],
        [500,    0.1],
        [645,      0]
    ])

    H42 = np.array([
        [0,       0],
        [15,    0.4],
        [41,    0.8],
        [49,    0.9],
        [53,  0.965],
        [64,   0.99],
        [71,      1],
        [88,   0.95],
        [94,    0.9],
        [106,  0.85],
        [115,   0.8],
        [137,   0.7],
        [180,   0.6],
        [238,   0.5],
        [290,   0.4],
        [348,   0.3],
        [415,   0.2],
        [500,   0.1],
        [645,     0]
    ])

    Hweight = np.zeros((47, N))

    # weighting function H2
    last = int(np.floor((358 / fs) * N))
    k_vals = np.arange(DCbins, min(last + 1, N))
    if len(k_vals) > 0:
        f_vals = k_vals * fs / N
        # Clamp f_vals to the range of H2's x-coordinates
        f_vals_clamped = np.clip(f_vals, H2[:, 0].min(), H2[:, 0].max())
        Hweight[1, k_vals] = np.interp(f_vals_clamped, H2[:, 0], H2[:, 1])

    # weighting function H5
    last = int(np.floor((502 / fs) * N))
    k_vals = np.arange(DCbins, min(last + 1, N))
    if len(k_vals) > 0:
        f_vals = k_vals * fs / N
        # Clamp f_vals to the range of H5's x-coordinates
        f_vals_clamped = np.clip(f_vals, H5[:, 0].min(), H5[:, 0].max())
        Hweight[4, k_vals] = np.interp(f_vals_clamped, H5[:, 0], H5[:, 1])

    # weighting function H16
    last = int(np.floor((645 / fs) * N))
    k_vals = np.arange(DCbins, min(last + 1, N))
    if len(k_vals) > 0:
        f_vals = k_vals * fs / N
        # Clamp f_vals to the range of H16's x-coordinates
        f_vals_clamped = np.clip(f_vals, H16[:, 0].min(), H16[:, 0].max())
        Hweight[15, k_vals] = np.interp(f_vals_clamped, H16[:, 0], H16[:, 1])

    # weighting function H21
    # k_vals and f_vals here reuse the last calculated values from H16, as 'last' is the same.
    if len(k_vals) > 0:
        # Clamp f_vals to the range of H21's x-coordinates
        f_vals_clamped = np.clip(f_vals, H21[:, 0].min(), H21[:, 0].max())
        Hweight[20, k_vals] = np.interp(f_vals_clamped, H21[:, 0], H21[:, 1])

    # weighting function H42
    # k_vals and f_vals here reuse the last calculated values from H16, as 'last' is the same.
    if len(k_vals) > 0:
        # Clamp f_vals to the range of H42's x-coordinates
        f_vals_clamped = np.clip(f_vals, H42[:, 0].min(), H42[:, 0].max())
        Hweight[41, k_vals] = np.interp(f_vals_clamped, H42[:, 0], H42[:, 1])

    # H1-H4 (indices 0-3)
    Hweight[0, :] = Hweight[1, :]
    Hweight[2, :] = Hweight[1, :]
    Hweight[3, :] = Hweight[1, :]

    # H5-H15 (indices 4-14, but 5 is index 4)
    for l in range(5, 15):
        Hweight[l, :] = Hweight[4, :]

    # H17-H20 (indices 16-19)
    for l in range(16, 20):
        Hweight[l, :] = Hweight[15, :]

    # H22-H41 (indices 21-40)
    for l in range(21, 41):
        Hweight[l, :] = Hweight[20, :]

    # H43-H47 (indices 42-46)
    for l in range(42, 47):
        Hweight[l, :] = Hweight[41, :]

    AmpCal = from_db(91.2) * 2 / (N * np.mean(blackman(N, sym=False)))

    # Calibration between wav-level and loudness-level (assuming
    # blackman window and FFT will follow)
    Chno = 47  # number of channels
    Cal = 0.5
    qb = np.arange(N0, Ntop + 1)
    freqs = (qb + 2) * fs / N
    hBPi = np.zeros((Chno, N))
    hBPrms = np.zeros(Chno)
    mdept = np.zeros(Chno)
    ki = np.zeros(Chno - 2)
    ri = np.zeros(Chno)

    startIndex = 1
    endIndex = N + 1
    TimePoints = np.zeros(n)
    R_mat = np.zeros(n)
    SPL_mat = np.zeros(n)
    ri_mat = np.zeros((Chno, n))

    for windowNum in range(n):  # for each frame
        if endIndex > len(audio):
            break
            
        dataIn = audio[startIndex:endIndex] * window
        currentTimePoint = startIndex / fs

        # Calculate Excitation Patterns
        TempIn = dataIn * AmpCal

        TempIn = a0 * fft(TempIn)

        Lg = np.abs(TempIn[qb])  # get absolute value of fourier transform for indices in range of human hearing
        LdB = 20 * np.log10(np.maximum(Lg, np.finfo(float).eps))

        whichL = np.where(LdB > MinExcdB)[0]  # extract indices where FFT magnitudes exceed excitation threshold
        sizL = len(whichL)  # get number of frequencies where this holds

        # steepness of slopes (Terhardt)
        S1 = -27
        S2 = np.zeros(sizL)  # preallocate
        
        # Ensure freqs indexing is safe
        if sizL > 0:
            # MATLAB uses freqs(w) instead of freqs(whichL(w)); preserve that
            # behavior here for validation parity.
            current_freqs = freqs[:sizL]
            current_LdB = LdB[whichL]

            for w in range(sizL):
                # Steepness of upper slope [dB/Bark] in accordance with Terhardt
                steep = -24 - (230 / current_freqs[w]) + (0.2 * current_LdB[w])

                if steep < 0:
                    S2[w] = steep  # set S2 with steepness value calculated earlier

        whichZ = np.zeros((2, sizL), dtype=int)  # preallocate
        qd = np.arange(sizL)  # indices of frequencies above excitation threshold
        
        if sizL > 0:
            # Ensure Barkno indexing is safe
            barkno_indices = whichL[qd] + N01
            barkno_indices_clamped = np.clip(barkno_indices, 0, len(Barkno) - 1)
            # Convert MATLAB's 1-based Bark channel ids to Python's 0-based indices.
            whichZ[0, :] = np.floor(2 * Barkno[barkno_indices_clamped]).astype(int) - 1
            whichZ[1, :] = np.ceil(2 * Barkno[barkno_indices_clamped]).astype(int) - 1
            
            # Clamp whichZ values to valid channel range [0, Chno-1]
            whichZ = np.clip(whichZ, 0, Chno - 1)

        # Corrected initialization for ExcAmp
        # ExcAmp's first dimension must be large enough to hold max(N1tmp)
        # N1tmp is an index into Lg, which has length len(qb)
        ExcAmp = np.zeros((len(qb), Chno))
        Slopes = np.zeros((sizL, Chno))

        for k_idx in range(sizL):  # loop over freq indices above threshold
            Ltmp = LdB[whichL[k_idx]]  # copy FFT magnitude (in dB) above threshold
            Btmp = Barkno[whichL[k_idx] + N01]  # and the bark number associated

            for l in range(min(whichZ[0, k_idx] + 1, Chno)): # loop up to floored bark number of freq index k
                if l < len(MinBf):
                    Stemp = (S1 * (Btmp - ((l + 1) * 0.5))) + Ltmp
                    if Stemp > MinBf[l]:
                        Slopes[k_idx, l] = from_db(Stemp)

            for l in range(max(whichZ[1, k_idx], 0), Chno): # loop up to ceil'd bark number
                if l < len(MinBf):
                    Stemp = (S2[k_idx] * (((l + 1) * 0.5) - Btmp)) + Ltmp
                    if Stemp > MinBf[l]:
                        Slopes[k_idx, l] = from_db(Stemp)

        for k_idx in range(Chno):  # loop over each channel
            etmp = np.zeros(N, dtype=complex)
            for l in range(sizL):  # for each l index of fft bin in human hearing freq range
                N1tmp = whichL[l]  # get freq index of bin
                N2tmp = N1tmp + N01
                if N2tmp < N:
                    if whichZ[0, l] == k_idx:
                        ExcAmp[N1tmp, k_idx] = 1 
                    elif whichZ[1, l] == k_idx:
                        ExcAmp[N1tmp, k_idx] = 1
                    elif whichZ[1, l] > k_idx:
                        if k_idx + 1 < Chno:
                            ExcAmp[N1tmp, k_idx] = Slopes[l, k_idx + 1] / Lg[N1tmp]
                    else:
                        if k_idx - 1 >= 0:
                            ExcAmp[N1tmp, k_idx] = Slopes[l, k_idx - 1] / Lg[N1tmp]
                    etmp[N2tmp] = ExcAmp[N1tmp, k_idx] * TempIn[N2tmp]

            # ifft to get time domain blocks of signal
            ei[k_idx, :] = N * np.real(ifft(etmp))
            etmp_abs = np.abs(ei[k_idx, :])
            h0[k_idx] = np.mean(etmp_abs)
            Fei[k_idx, :] = fft(etmp_abs - h0[k_idx])
            hBPi[k_idx, :] = 2 * np.real(ifft(Fei[k_idx, :] * Hweight[k_idx, :]))
            hBPrms[k_idx] = np.sqrt(np.mean(hBPi[k_idx, :] ** 2))

            if h0[k_idx] > 0:
                mdept[k_idx] = hBPrms[k_idx] / h0[k_idx]

                if mdept[k_idx] > 1:
                    mdept[k_idx] = 1
            else:
                mdept[k_idx] = 0

        # find cross-correlation coefficients
        for k_idx in range(45):
            cfac = np.cov(hBPi[k_idx, :], hBPi[k_idx + 2, :])
            den = np.diag(cfac)
            den_prod = np.sqrt(den[0] * den[1])
            if den_prod > 0:
                ki[k_idx] = cfac[0, 1] / den_prod
            else:
                ki[k_idx] = 0

        # Calculate specific roughness ri and total roughness R
        ri[0] = (gzi[0] * mdept[0] * ki[0]) ** 2
        ri[1] = (gzi[1] * mdept[1] * ki[1]) ** 2

        for k_idx in range(2, 45):
            # Ensure ki indexing is safe
            ki_idx_minus_2 = k_idx - 2
            ki_idx = k_idx
            
            ki_val_minus_2 = ki[ki_idx_minus_2] if ki_idx_minus_2 >= 0 and ki_idx_minus_2 < len(ki) else 0
            ki_val = ki[ki_idx] if ki_idx >= 0 and ki_idx < len(ki) else 0

            ri[k_idx] = (gzi[k_idx] * mdept[k_idx] * ki_val_minus_2 * ki_val) ** 2

        ri[45] = (gzi[45] * mdept[45] * ki[43]) ** 2
        ri[46] = (gzi[46] * mdept[46] * ki[44]) ** 2

        ri = Cal * ri  # appropriately scaled specific roughness
        R = dz * np.sum(ri)  # total R = integration of the specific R pattern

        SPL = np.mean(np.sqrt(np.mean(dataIn ** 2)))
        if SPL > 0:
            SPL = 20 * np.log10(np.maximum(SPL, np.finfo(float).eps)) + 83  # -20 dBFS <--> 60 dB SPL
        else:
            SPL = -400

        # matrices to return
        R_mat[windowNum] = R
        ri_mat[:, windowNum] = ri
        SPL_mat[windowNum] = SPL

        startIndex = startIndex + hopsize
        endIndex = endIndex + hopsize
        TimePoints[windowNum] = currentTimePoint

    OUT = {}
    
    # main output results
    OUT['InstantaneousRoughness'] = R_mat                       # instantaneous roughness
    OUT['InstantaneousSpecificRoughness'] = ri_mat              # time-varying specific roughness
    OUT['TimeAveragedSpecificRoughness'] = np.mean(ri_mat, axis=1)       # mean specific roughness
    OUT['time'] = TimePoints                                    # time
    OUT['barkAxis'] = z                                         # critical band rate (for specific roughness)
    OUT['dz'] = dz

    # Roughness statistics based on InstantaneousRoughness
    if len(OUT['time']) > 0:
        idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector
    else:
        idx = 0 # If time is empty, start from beginning (which is also empty)

    metric_statistics = 'Roughness_Daniel1997'
    # Call the provided get_statistics function
    OUT_statistics = get_statistics(R_mat[idx:], metric_statistics)  # get statistics

    # copy fields of <OUT_statistics> dict into the <OUT> struct
    for fieldName, value in OUT_statistics.items():
        if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
            OUT[fieldName] = value

    # plots
    if show:
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Roughness analysis')

        # Time-varying roughness
        plt.subplot(2, 2, (1, 2))
        
        # Only plot if R_mat is not empty
        if len(TimePoints) > 0 and len(R_mat) > 0:
            plt.plot(TimePoints, R_mat, 'r-')
        else:
            plt.text(0.5, 0.5, 'No data to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.title('Instantaneous roughness')
        plt.xlabel('Time (s)')
        plt.ylabel('Roughness, R (asper)')

        # Time-averaged roughness as a function of critical band
        plt.subplot(2, 2, 3)
        
        # Only plot if ri_mat is not empty
        if ri_mat.shape[1] > 0:
            plt.plot(np.arange(1, 48) / 2, np.mean(ri_mat, axis=1), 'r-')
        else:
            plt.text(0.5, 0.5, 'No data to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.title('Time-averaged specific roughness')
        plt.xlabel('Critical band, z (Bark)')
        plt.ylabel('Specific roughness, R′ (asper/Bark)')

        # Specific roughness spectrogram
        plt.subplot(2, 2, 4)
        
        # Only plot if ri_mat is not empty
        if ri_mat.shape[1] > 0:
            xx, yy = np.meshgrid(TimePoints, OUT['barkAxis'])
            plt.pcolormesh(xx, yy, ri_mat, shading='auto')
            plt.colorbar(label='Specific roughness, R′ (asper/Bark)')
        else:
            plt.text(0.5, 0.5, 'No data to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.title('Instantaneous specific roughness')
        plt.xlabel('Time (s)')
        plt.ylabel('Critical band, z (Bark)')

        plt.tight_layout()
        plt.show()

    return OUT


check_which = 1

if __name__ == "__main__":
    if check_which == 0: # NO TEST

        print("metrics_roughness.py")
    
    elif check_which == 1: # Roughness_Daniel1997
        with_wavfile = 0

        """
        Validation clip for Roughness_Daniel1997
        -----------------------------------

        Generates a 5-second, 1 kHz sinusoid with 70 Hz amplitude modulation at 60 dB SPL, sampled at 48 kHz.
        """

        print("Running Roughness_Daniel1997 test...")

        fs          = 48_000            # sampling rate (Hz)
        dur         = 5.0               # seconds
        f_carrier   = 1_000.0           # Hz
        f_mod       = 70.0              # Hz
        L_spl       = 60.0              # dB SPL

        t = np.arange(0, dur, 1/fs)
        envelope = 0.5 * (1.0 + np.sin(2*np.pi*f_mod*t))      # 0…1
        p_rms    = 20e-6 * 10**(L_spl/20)                     # 60 dB SPL → Pa
        A        = p_rms / np.sqrt(np.mean(envelope**2))      # <<< key line
        signal   = A * envelope * np.sin(2*np.pi*f_carrier*t) # float64
        
        OUT = Roughness_Daniel1997(signal.astype(np.float32), fs,
                                time_skip=0.0, show=True)

        print(f"  Mean roughness  : {OUT['Rmean']} asper")
        print(f"  Max  roughness  : {OUT['Rmax']} asper")
        print(f"  10 % exceedance : {OUT['R10']} asper")
