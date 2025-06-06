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

from sound_metrics import *
from utilities import *

__all__ = ["Roughness_Daniel1997"]
FloatArray = NDArray[np.floating]

# -------------------------
#### ROUGHNESS METRICS ####
# -------------------------

def Roughness_Daniel1997(insig=None, fs=None, time_skip=None, show=None, dBFS=94):
    """
    Roughness calculation according to Daniel & Weber (1997)
    
    Parameters:
    insig: input signal
    fs: sampling frequency
    time_skip: time to skip for statistics calculation
    show: whether to show plots (1 or 0)
    
    Returns:
    OUT: dictionary containing roughness analysis results
    """
    
    # --- WAV file interface ---
    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)
    
    if show is None:
        # This simulates nargout check. In a real application, you might pass a flag
        # or check if the function is called in a context expecting a return value.
        # For simplicity, we'll assume if nargout is not 0, show is 0.
        # A more robust check might involve inspecting the call stack or function signature.
        try:
            # This line will raise an error if called without assignment,
            # simulating nargout == 0.
            _ = Roughness_Daniel1997.__code__.co_argcount
            show = 0 # If no error, assume output is expected
        except:
            show = 1 # If error (e.g., called directly without assignment), assume no output expected
    
    # window settings
    time_resolution = 0.2  # time-step for the windowing
    N = int(fs * time_resolution)  # window length
    
    audio = insig.copy()
    
    if audio.ndim > 1 and audio.shape[1] != 1:  # if the insig is not a [Nx1] array
        audio = audio.T  # correct the dimension of the insig
    
    if audio.ndim > 1:
        audio = audio.flatten()
    
    hopsize = N // 2  # hopsize is the number of samples hop between successive windows
    window = blackman(N)
    
    # resampling input signal
    if not (fs == 44100 or fs == 40960 or fs == 48000):
        gcd_fs = np.gcd(48_000, fs)
        audio = resample_poly(audio, 48_000 // gcd_fs, fs // gcd_fs)
        fs = 48_000
    
    samples = len(audio)
    n = int((samples - N) / hopsize)
    
    Bark = np.array([
        [0,     0,     50,   0.5],
        [1,   100,    150,   1.5],
        [2,   200,    250,   2.5],
        [3,   300,    350,   3.5],
        [4,   400,    450,   4.5],
        [5,   510,    570,   5.5],
        [6,   630,    700,   6.5],
        [7,   770,    840,   7.5],
        [8,   920,   1000,   8.5],
        [9,  1080,   1170,   9.5],
        [10, 1270,   1370,  10.5],
        [11, 1480,   1600,  11.5],
        [12, 1720,   1850,  12.5],
        [13, 2000,   2150,  13.5],
        [14, 2320,   2500,  14.5],
        [15, 2700,   2900,  15.5],
        [16, 3150,   3400,  16.5],
        [17, 3700,   4000,  17.5],
        [18, 4400,   4800,  18.5],
        [19, 5300,   5800,  19.5],
        [20, 6400,   7000,  20.5],
        [21, 7700,   8500,  21.5],
        [22, 9500,  10500,  22.5],
        [23, 12000, 13500,  23.5],
        [24, 15500, 20000,  24.5]
    ])
    
    N2 = N // 2 + 1
    dFs = fs / N
    Bark2 = np.column_stack([
        np.sort(np.concatenate([Bark[:, 1], Bark[:, 2]])),
        np.sort(np.concatenate([Bark[:, 0], Bark[:, 3]]))
    ])
    N0 = int(round(20 * N / fs)) + 1  # low frequency index @ 20 Hz (1-based)
    N01 = N0 - 1 # 0-based index for 20 Hz
    Ntop = int(round(20000 * N / fs)) + 1  # high frequency index @ 20 kHz (1-based)
    
    # Make list with Barknumber of each frequency bin
    Barkno = np.zeros(N2)
    f_indices_0based = np.arange(N0-1, Ntop)  # 0-based indices for relevant frequencies
    f_freq = f_indices_0based * dFs # Frequencies corresponding to these indices
    Barkno[f_indices_0based] = np.interp(f_freq, Bark2[:, 0], Bark2[:, 1])
    
    # Make list of frequency bins closest to Cf's
    Cf = np.ones((2, 24))
    for a in range(24):
        Cf[0, a] = round(Bark[a+1, 1] * N / fs) + 1 - N0 # 0-based index relative to N0
        Cf[1, a] = Bark[a+1, 1]
    
    # Make list of frequency bins closest to Critical Band Border frequencies
    Bf = np.ones((2, 25))
    Bf[0, 0] = round(Bark[0, 2] * N / fs) # 0-based index relative to N0
    
    for a in range(24):
        Bf[0, a+1] = round(Bark[a+1, 2] * N / fs) + 1 - N0 # 0-based index relative to N0
        Bf[1, a] = Bf[0, a] - 1 # This seems to be a relative index, not an absolute frequency
    
    Bf[1, 24] = round(Bark[24, 2] * N / fs) + 1 - N0 # 0-based index relative to N0
    
    # Make list of minimum excitation (Hearing Threshold)
    HTres = np.array([
        [0,      130], [0.01,    70], [0.17,    60], [0.8,     30], [1,       25],
        [1.5,     20], [2,       15], [3.3,     10], [4,      8.1], [5,      6.3],
        [6,        5], [8,      3.5], [10,     2.5], [12,     1.7], [13.3,     0],
        [15,    -2.5], [16,      -4], [17,    -3.7], [18,    -1.5], [19,     1.4],
        [20,     3.8], [21,       5], [22,     7.5], [23,      15], [24,      48],
        [24.5,    60], [25,     130]
    ])
    
    k_barkno_indices = np.arange(N0-1, Ntop)  # 0-based indices for Barkno
    MinExcdB = np.interp(Barkno[k_barkno_indices], HTres[:, 0], HTres[:, 1])
    
    # Initialize constants and variables
    dz = 0.5  # Barks
    z = np.arange(0.5, 24, dz)  # frequency in Barks
    
    # zb contains 0-based indices relative to N0
    zb = np.sort(np.concatenate([Bf[0, :], Cf[0, :]])).astype(int)
    MinBf = MinExcdB[zb] # This assumes zb are valid indices into MinExcdB
    
    # ExcAmp and Fei are sized for the full frequency spectrum (N2)
    ei = np.zeros((47, N))
    Fei = np.zeros((47, N), dtype=complex) # Fei is result of FFT, should be complex
    
    gr = np.array([
        [0, 1, 2.5, 4.9, 6.5, 8, 9, 10, 11, 11.5, 13, 17.5, 21, 24],
        [0, 0.35, 0.7, 0.7, 1.1, 1.25, 1.26, 1.18, 1.08, 1, 0.66, 0.46, 0.38, 0.3]
    ])
    
    gzi = np.zeros(47)
    h0 = np.zeros(47)
    k_gzi = np.arange(47)
    gzi = np.sqrt(np.interp(k_gzi/2, gr[0, :], gr[1, :], left=0, right=0)) # Add left/right for interp
    
    # calculate a0
    a0tab = np.array([
        [0,      0], [10,     0], [12,  1.15], [13,  2.31], [14,  3.85],
        [15,  5.62], [16,  6.92], [16.5, 7.38], [17,  6.92], [18,  4.23],
        [18.5, 2.31], [19,     0], [20, -1.43], [21, -2.59], [21.5, -3.57],
        [22, -5.19], [22.5, -7.41], [23, -11.3], [23.5,  -20], [24,    -40],
        [25,   -130], [26,   -999]
    ])
    
    a0 = np.ones(N)
    k_a0_indices = np.arange(N0-1, Ntop)  # 0-based indices for a0
    a0[k_a0_indices] = 10**(np.interp(Barkno[k_a0_indices], a0tab[:, 0], a0tab[:, 1], left=0, right=0)/20)
    
    DCbins = 2 # 0-based index for DC and first bin
    
    H2 = np.array([
        [0,      0], [17,   0.8], [23,  0.95], [25, 0.975], [32,     1],
        [37, 0.975], [48,   0.9], [67,   0.8], [90,   0.7], [114,  0.6],
        [171,  0.4], [206,  0.3], [247,  0.2], [294,  0.1], [358,    0]
    ])
    
    H5 = np.array([
        [0,      0], [32,   0.8], [43,  0.95], [56,     1], [69, 0.975],
        [92,   0.9], [120,  0.8], [142,  0.7], [165,  0.6], [231,  0.4],
        [277,  0.3], [331,  0.2], [397,  0.1], [502,    0]
    ])
    
    H16 = np.array([
        [0,      0], [23.5, 0.4], [34,   0.6], [47,   0.8], [56,   0.9],
        [63,  0.95], [79,     1], [100, 0.975], [115, 0.95], [135,  0.9],
        [159, 0.85], [172,  0.8], [194,  0.7], [215,  0.6], [244,  0.5],
        [290,  0.4], [348,  0.3], [415,  0.2], [500,  0.1], [645,    0]
    ])
    
    H21 = np.array([
        [0,      0], [19,   0.4], [44,   0.8], [52.5, 0.9], [58,  0.95],
        [75,     1], [101.5, 0.95], [114.5, 0.9], [132.5, 0.85], [143.5, 0.8],
        [165.5, 0.7], [197.5, 0.6], [241,  0.5], [290,  0.4], [348,  0.3],
        [415,  0.2], [500,  0.1], [645,    0]
    ])
    
    H42 = np.array([
        [0,      0], [15,   0.4], [41,   0.8], [49,   0.9], [53, 0.965],
        [64,  0.99], [71,     1], [88,  0.95], [94,   0.9], [106, 0.85],
        [115,  0.8], [137,  0.7], [180,  0.6], [238,  0.5], [290,  0.4],
        [348,  0.3], [415,  0.2], [500,  0.1], [645,    0]
    ])
    
    Hweight = np.zeros((47, N))
    
    # weighting function H2 (index 1 in Python)
    last = int(np.floor((358/fs)*N))
    k_freq_indices = np.arange(DCbins, last)
    f_values = k_freq_indices * fs / N
    Hweight[1, k_freq_indices] = np.interp(f_values, H2[:, 0], H2[:, 1], left=0, right=0)
    
    # weighting function H5 (index 4 in Python)
    last = int(np.floor((502/fs)*N))
    k_freq_indices = np.arange(DCbins, last)
    f_values = k_freq_indices * fs / N
    Hweight[4, k_freq_indices] = np.interp(f_values, H5[:, 0], H5[:, 1], left=0, right=0)
    
    # weighting function H16 (index 15 in Python)
    last = int(np.floor((645/fs)*N))
    k_freq_indices = np.arange(DCbins, last)
    f_values = k_freq_indices * fs / N
    Hweight[15, k_freq_indices] = np.interp(f_values, H16[:, 0], H16[:, 1], left=0, right=0)
    
    # weighting function H21 (index 20 in Python)
    Hweight[20, k_freq_indices] = np.interp(f_values, H21[:, 0], H21[:, 1], left=0, right=0)
    
    # weighting function H42 (index 41 in Python)
    Hweight[41, k_freq_indices] = np.interp(f_values, H42[:, 0], H42[:, 1], left=0, right=0)
    
    # H1-H4 (indices 0-3 in Python)
    Hweight[0, :] = Hweight[1, :]
    Hweight[2, :] = Hweight[1, :]
    Hweight[3, :] = Hweight[1, :]
    
    # H5-H15 (indices 4-14 in Python)
    for l_idx in range(5, 15):  # 6 to 15 in MATLAB = 5 to 14 in Python
        Hweight[l_idx, :] = Hweight[4, :]
    
    # H17-H20 (indices 16-19 in Python)
    for l_idx in range(16, 20):  # 17 to 20 in MATLAB = 16 to 19 in Python
        Hweight[l_idx, :] = Hweight[15, :]
    
    # H22-H41 (indices 21-40 in Python)
    for l_idx in range(21, 41):  # 22 to 41 in MATLAB = 21 to 40 in Python
        Hweight[l_idx, :] = Hweight[20, :]
    
    # H43-H47 (indices 42-46 in Python)
    for l_idx in range(42, 47):  # 43 to 47 in MATLAB = 42 to 46 in Python
        Hweight[l_idx, :] = Hweight[41, :]
    
    AmpCal = 10**(91.2/20) * 2 / (N * np.mean(blackman(N)))
    
    Chno = 47  # number of channels
    Cal = 0.50  # calibration factor, twice the old value (0.25)
    qb_indices_0based = np.arange(N0-1, Ntop)  # 0-based indices for relevant frequencies
    freqs = (qb_indices_0based + 1) * fs / N # Frequencies corresponding to these indices
    hBPi = np.zeros((Chno, N))
    hBPrms = np.zeros(Chno)
    mdept = np.zeros(Chno)
    ki = np.zeros(Chno-2)
    ri = np.zeros(Chno)
    
    startIndex = 0  # Convert to 0-based indexing
    endIndex = N
    TimePoints = np.zeros(n)
    R_mat = np.zeros(n)
    SPL_mat = np.zeros(n)
    ri_mat = np.zeros((Chno, n))
    
    for windowNum in range(n):  # for each frame
        
        dataIn = audio[startIndex:endIndex] * window
        currentTimePoint = startIndex / fs
        
        # Calculate Excitation Patterns
        TempIn = dataIn * AmpCal
        
        # Ensure TempIn is 1D for FFT
        if TempIn.ndim > 1:
            TempIn = TempIn.flatten()
        
        TempIn_fft = a0 * fft(TempIn) # Use a new variable for FFT result
        Lg = np.abs(TempIn_fft[qb_indices_0based])  # get absolute value of fourier transform for indices in range of human hearing
        LdB = 20 * np.log10(Lg)  # mag2db equivalent
        
        # whichL contains 0-based relative indices into LdB (and Lg)
        whichL = np.where(LdB > MinExcdB)[0]  
        sizL = len(whichL)  # get number of frequencies where this holds
        
        # steepness of slopes (Terhardt)
        S1 = -27
        S2 = np.zeros(sizL)  # preallocate
        
        for w in range(sizL):
            # Steepness of upper slope [dB/Bark] in accordance with Terhardt
            # freqs[whichL[w]] uses the relative index to get the corresponding frequency
            steep = -24 - (230/freqs[whichL[w]]) + (0.2*LdB[whichL[w]])
            
            if steep < 0:
                S2[w] = steep  # set S2 with steepness value calculated earlier
        
        # whichZ contains 0-based bark band numbers (0 to 46)
        whichZ = np.zeros((2, sizL), dtype=int)  # preallocate
        qd = np.arange(sizL)  # indices of frequencies above excitation threshold
        
        # Barkno is indexed by absolute frequency bins (0-based)
        # whichL[qd] are relative indices, so whichL[qd] + N01 gives absolute frequency bins
        whichZ[0, :] = np.floor(2 * Barkno[whichL[qd] + N01]).astype(int)
        whichZ[1, :] = np.ceil(2 * Barkno[whichL[qd] + N01]).astype(int)
        
        Slopes = np.zeros((sizL, Chno)) # Slopes is indexed by relative frequency index (l) and channel (k)
        
        for k_freq_idx in range(sizL):  # loop over freq indices above threshold (relative index)
            Ltmp = LdB[whichL[k_freq_idx]]  # copy FFT magnitude (in dB) above threshold
            Btmp = Barkno[whichL[k_freq_idx] + N01]  # and the bark number associated (absolute freq bin)
            
            for l_channel_idx in range(whichZ[0, k_freq_idx]):  # loop up to floored bark number of freq index k
                Stemp = (S1 * (Btmp - (l_channel_idx * 0.5))) + Ltmp
                if Stemp > MinBf[l_channel_idx]: # MinBf is indexed by bark band number
                    Slopes[k_freq_idx, l_channel_idx] = 10**(Stemp/20)  # db2mag equivalent
            
            for l_channel_idx in range(whichZ[1, k_freq_idx], Chno):  # loop up to ceil'd bark number
                Stemp = (S2[k_freq_idx] * ((l_channel_idx * 0.5) - Btmp)) + Ltmp
                if Stemp > MinBf[l_channel_idx]: # MinBf is indexed by bark band number
                    Slopes[k_freq_idx, l_channel_idx] = 10**(Stemp/20)  # critical filterbank upper side, db2mag equivalent
        
        # ExcAmp is indexed by absolute frequency bin and channel
        ExcAmp = np.zeros((N2, Chno)) # Re-initialize for each window, sized for absolute freq bins
        
        for k_channel_idx in range(Chno):  # loop over each channel (0-based)
            etmp = np.zeros(N, dtype=complex)
            for l_relative_freq_idx in range(sizL):  # for each l index of fft bin in human hearing freq range (relative index)
                
                # N1tmp_relative is the 0-based index into Lg/LdB/whichL
                N1tmp_relative = whichL[l_relative_freq_idx]
                
                # N1tmp_absolute is the 0-based absolute frequency bin index
                N1tmp_absolute = N1tmp_relative + N01 
                
                # Ensure N1tmp_absolute is within bounds of ExcAmp (N2) and TempIn_fft (N)
                if N1tmp_absolute >= N2 or N1tmp_absolute >= N:
                    continue # Skip if out of bounds, or handle as needed
                
                if (whichZ[0,l_relative_freq_idx] == k_channel_idx):
                    ExcAmp[N1tmp_absolute, k_channel_idx] = 1
                elif (whichZ[1,l_relative_freq_idx] == k_channel_idx):
                    ExcAmp[N1tmp_absolute, k_channel_idx] = 1
                elif (whichZ[1,l_relative_freq_idx] > k_channel_idx):
                    # Slopes is indexed by relative_freq_idx and channel_idx
                    ExcAmp[N1tmp_absolute, k_channel_idx] = Slopes[l_relative_freq_idx, k_channel_idx+1] / Lg[N1tmp_relative]
                else:
                    # Slopes is indexed by relative_freq_idx and channel_idx
                    ExcAmp[N1tmp_absolute, k_channel_idx] = Slopes[l_relative_freq_idx, k_channel_idx-1] / Lg[N1tmp_relative]
                
                # etmp and TempIn_fft are indexed by absolute frequency bin
                etmp[N1tmp_absolute] = ExcAmp[N1tmp_absolute, k_channel_idx] * TempIn_fft[N1tmp_absolute]
            
            # ifft to get time domain blocks of signal
            ei[k_channel_idx, :] = N * np.real(ifft(etmp))
            etmp_abs = np.abs(ei[k_channel_idx, :])
            h0[k_channel_idx] = np.mean(etmp_abs)
            Fei[k_channel_idx, :] = fft(etmp_abs - h0[k_channel_idx])
            hBPi[k_channel_idx, :] = 2 * np.real(ifft(Fei[k_channel_idx, :] * Hweight[k_channel_idx, :]))
            hBPrms[k_channel_idx] = np.sqrt(np.mean(hBPi[k_channel_idx, :]**2))  # rms equivalent
            
            if h0[k_channel_idx] > 0:
                mdept[k_channel_idx] = hBPrms[k_channel_idx] / h0[k_channel_idx]
                
                if mdept[k_channel_idx] > 1:
                    mdept[k_channel_idx] = 1
            else:
                mdept[k_channel_idx] = 0
        
        # find cross-correlation coefficients
        for k_corr in range(45):  # 1 to 45 in MATLAB = 0 to 44 in Python
            cfac = np.cov(hBPi[k_corr, :], hBPi[k_corr+2, :])
            den = np.diag(cfac)
            den = np.sqrt(den[0] * den[1])
            if den > 0:
                ki[k_corr] = cfac[0, 1] / den
            else:
                ki[k_corr] = 0
        
        # Calculate specific roughness ri and total roughness R
        ri[0] = (gzi[0] * mdept[0] * ki[0])**2  # Convert to 0-based indexing
        ri[1] = (gzi[1] * mdept[1] * ki[1])**2  # Convert to 0-based indexing
        
        for k_ri in range(2, 45):  # 3 to 45 in MATLAB = 2 to 44 in Python
            ri[k_ri] = (gzi[k_ri] * mdept[k_ri] * ki[k_ri-2] * ki[k_ri])**2
        
        ri[45] = (gzi[45] * mdept[45] * ki[43])**2  # Convert to 0-based indexing
        ri[46] = (gzi[46] * mdept[46] * ki[44])**2  # Convert to 0-based indexing
        
        ri = Cal * ri  # appropriately scaled specific roughness
        R = dz * np.sum(ri)  # total R = integration of the specific R pattern
        
        SPL = np.mean(np.sqrt(np.mean(dataIn**2)))  # rms equivalent
        if SPL > 0:
            SPL = 20 * np.log10(SPL) + 83  # -20 dBFS <--> 60 dB SPL, mag2db equivalent
        else:
            SPL = -400
        
        # matrices to return
        R_mat[windowNum] = R
        ri_mat[:, windowNum] = ri
        SPL_mat[windowNum] = SPL
        
        startIndex = startIndex + hopsize
        endIndex = endIndex + hopsize
        TimePoints[windowNum] = currentTimePoint
    
    # main output results
    OUT = {}
    OUT['InstantaneousRoughness'] = R_mat  # instantaneous roughness
    OUT['InstantaneousSpecificRoughness'] = ri_mat  # time-varying specific roughness
    OUT['TimeAveragedSpecificRoughness'] = np.mean(ri_mat, axis=1)  # mean specific roughness
    OUT['time'] = TimePoints  # time
    OUT['barkAxis'] = z  # critical band rate (for specific roughness)
    OUT['dz'] = dz
    
    idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector
    
    metric_statistics = 'Roughness_Daniel1997'
    OUT_statistics = get_statistics(R_mat[idx:], metric_statistics)  # get statistics
    
    # copy fields of <OUT_statistics> dict into the <OUT> dict
    for fieldName in OUT_statistics.keys():
        if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
            OUT[fieldName] = OUT_statistics[fieldName]
    
    if show == True:
        
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Roughness analysis')
        
        # Time-varying roughness
        plt.subplot(2, 2, (1, 2))
        plt.plot(TimePoints, R_mat, 'r-')
        plt.title('Instantaneous roughness')
        plt.xlabel('Time (s)')
        plt.ylabel('Roughness, $R$ (asper)')
        
        # Time-averaged roughness as a function of critical band
        plt.subplot(2, 2, 3)
        plt.plot(np.arange(1, 48)/2, np.mean(ri_mat, axis=1), 'r-')
        plt.title('Time-averaged specific roughness')
        plt.xlabel('Critical band, $z$ (Bark)')
        plt.ylabel('Specific roughness, $R^{\\prime}$ (asper/Bark)')
        
        # Specific roughness spectrogram
        plt.subplot(2, 2, 4)
        xx, yy = np.meshgrid(TimePoints, OUT['barkAxis'])
        im = plt.pcolormesh(xx, yy, ri_mat, shading='auto')
        plt.colorbar(im, label='Specific roughness, $R^{\\prime}$ ($\\mathrm{asper}/\\mathrm{Bark}$)')
        plt.title('Instantaneous specific roughness')
        plt.xlabel('Time (s)')
        plt.ylabel('Critical band, $z$ (Bark)')
        
        plt.tight_layout()
        plt.show()
    
    return OUT

check_which = 0

if __name__ == "__main__":
    if check_which == 0: # NO TEST

        print("metrics_roughness.py")
    
    elif check_which == 1: # Roughness_Daniel1997

        """
        Validation clip for Roughness_Daniel1997
        -----------------------------------

        Generates a 2-second, 1 kHz sinusoid with 70 Hz amplitude modulation at 60 dB SPL, sampled at 48 kHz.
        """

        print("Running Roughness_Daniel1997 test...")

        fs = 48_000
        f_mod = 70
        f_carrier = 1_000.0

        p_rms = 20e-6 * 10**(60 / 20)
        A = p_rms * np.sqrt(2)
        t = np.arange(0.0, 2, 1 / fs)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * f_mod * t))
        signal = A * envelope * np.sin(2 * np.pi * f_carrier * t)
        insig = signal.astype(np.float32)
        #wavfile.write("am_1kHz_70Hz_60dB.wav", fs, signal.astype(np.float32))

        OUT = Roughness_Daniel1997(insig, fs, time_skip=0.0, show=True)
        #OUT = Roughness_Daniel1997("am_1kHz_70Hz_60dB.wav", fs, time_skip=0.0, show=True)

        print(f"Reference-tone check (expected ≈ 1 asper)")
        print(f"  Mean roughness  : {OUT['Rmean']} asper")
        print(f"  Max  roughness  : {OUT['Rmax']} asper")
        print(f"  10 % exceedance : {OUT['R10']} asper")