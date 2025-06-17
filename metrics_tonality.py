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

from metrics_loudness import Loudness_ISO532_1

__all__ = ["Tonality_Aures1985"]
FloatArray = NDArray[np.floating]

# ----------------------
#### TONALITY METRICS ####
# ----------------------


def Tonality_Aures1985(insig, fs, LoudnessField, time_skip, show=None):

    """
    Estimate **Aures tonality** :math:`K` *(1985)* for a monaural waveform.

    Parameters
    ----------
    insig : array_like
        Input signal (mono).  If the sampling‐rate *fs* differs from
        44.1/48 kHz the waveform is polyphase-resampled.
    fs : int | float
        Original sampling frequency in hertz.
    LoudnessField : {0, 1}
        Acoustic field for the ISO 532-1 loudness calls  
        ``0 → free field``, ``1 → diffuse field``.
    time_skip : float
        Duration in seconds to discard at the beginning when computing
        summary statistics.
    show : bool, optional
        Display diagnostic plots (default: ``True`` if the caller does not
        capture the return value).

    Returns
    -------
    dict
        Dictionary containing instantaneous data and summary statistics.

    Notes
    -----
    *Tonal* and *loudness* weightings are **zero** when no eligible tones
    are found in a frame.

    Examples
    --------
    >>> K = Tonality_Aures1985(sig, 48000, LoudnessField=0,
    ...                         time_skip=0.2, show=False)
    >>> K['K50']      # median tonality
    0.12
    """


    if show is None:
        # If no return value is expected, show plots
        show = 1  # Assuming nargout equivalent is handled by caller
    
    ## resampling
    # resampling audio to 44.1 kHz or 48kHz
    if not (fs == 44100 or fs == 48000):
        gcd_fs = np.gcd(44100, fs)  # greatest common denominator
        insig = resample_poly(insig, 44100//gcd_fs, fs//gcd_fs)
        fs = 44100
    
    ## window parameters
    
    # time_resolution=80e-3;    # window length fixed in 80 ms (Terhard), gives a df=12.5 Hz
    time_resolution = 160e-3  # window length fixed in 160 ms, gives a df=6.25 Hz
    
    N = round(fs * time_resolution)  # define window length, N bins
    window = hann(N)
    
    fftgain = np.sqrt(2) / (N * np.mean(hann(N)))  # gain to be applied based on the FFT length
    
    ## freq vectors based on window input signals
    
    # from Terhardt [3]: Aurally relevant tonal information of any signal is
    #  confined in the frequency region of about 20 Hz to 5 kHz.
    
    MinFrequency = 20
    MinFrequencyindex = int(np.ceil(1 + (MinFrequency * (N / fs))))  # index corresponding to min frequency (20 Hz) for tone extraction
    
    MaxFrequency = 5000
    MaxFrequencyIndex = int(np.ceil(1 + (MaxFrequency * (N / fs))))  # index corresponding to max frequency (5 kHz) for tone extraction
    
    Freq = fs * (np.arange(1, round(N) + 1) - 1) / N  # freq vector
    FreqCrop = Freq[MinFrequencyindex-1:MaxFrequencyIndex]  # cropped freq vector from MinFrequencyindex till MaxFrequencyIndex
    df = FreqCrop[1] - FreqCrop[0]  # freq discretization
    
    ## initialize windowed vectors
    
    t_b = (np.arange(1, len(insig) + 1)) / fs  # time vector
    
    overlap = round(0.5 * N)  # overlap
    
    insig = buffer(insig, N, overlap, 'nodelay')
    t_b = buffer(t_b, N, overlap, 'nodelay')
    
    nFrames = insig.shape[1] - 1
    
    tone = [None] * nFrames  # Memory allocation: tone cell per time frame
    tonality = np.zeros(nFrames)  # Memory allocation for tonality computation
    t = np.zeros(nFrames)  # Memory allocation: time vector for iFrames
    w_gr = np.zeros(nFrames)  # Memory allocation: loudness weighting function per time frame
    w_tonal = np.zeros(nFrames)  # Memory allocation: tonal weighting function per time frame
    TINY_VALUE = 1e-99
    
    ## Here we go ...
    
    for iFrame in range(nFrames):
        
        ## windowed time-frame
        
        Winsig = insig[:, iFrame]  # cut insig for each iFrames
        
        t[iFrame] = t_b[0, iFrame]  # output time vector for iFrames
        
        Winsig = window * Winsig  # Apply window to frame
        
        ## compute SPL for each time-frame
        
        SpectralEnergy = np.abs(fft(Winsig * fftgain)) ** 2
        SPL = 10.0 * np.log10((SpectralEnergy + TINY_VALUE) / 4e-10)  # dBSPL
        
        ## Find peaks according to Terhard's criteria for each time-frame
        
        SPLcrop = SPL[MinFrequencyindex-1:MaxFrequencyIndex]  # crop SPL vector from MinFrequencyindex to MaxFrequencyIndex
        
        threshold = 7  # condition for tonal component, in dBSPL
        
        ToneIdx = np.zeros(len(SPLcrop), dtype=int)  # initialize vector, tonal components idx
        k = 0  # initialize counter (using 0-based indexing)
        
        # find tones...
        for i in range(3, len(SPLcrop) - 3):  # 4:(length(SPLcrop)-3) in Matlab (1-based) becomes 3:(len-3) in Python (0-based)
            
            if (SPLcrop[i] > SPLcrop[i-1] and
                SPLcrop[i] >= SPLcrop[i+1] and
                SPLcrop[i] - SPLcrop[i-3] >= threshold and
                SPLcrop[i] - SPLcrop[i-2] >= threshold and
                SPLcrop[i] - SPLcrop[i+2] >= threshold and
                SPLcrop[i] - SPLcrop[i+3] >= threshold):
                
                ToneIdx[k] = i  # get the idx of the tones on Lcrop
                k = k + 1
        
        # save tone information
        ToneIdx = ToneIdx[ToneIdx != 0]  # if no tones were found, ToneIdx shall remain empty
        ToneL = SPLcrop[ToneIdx]  # SPL of the tones
        NTones = np.where(ToneIdx)[0]  # number of tones
        ToneF = FreqCrop[ToneIdx]  # central freq of the tone
        
        # estimate bandwidth of the i-th tone using half-power (-3 dB decay) criteria (this analysis is made on the full SPL and freq vectors)
        flow = np.zeros(len(NTones))  # declare variable for memory allocation
        fhigh = np.zeros(len(NTones))
        BW = np.zeros(len(NTones))
        
        for i in range(len(NTones)):  # Source: https://de.mathworks.com/matlabcentral/answers/1441689-i-am-trying-to-find-the-full-width-at-half-max-value-and-plot-the-waveform-with-markers?s_tid=srchtitle
            
            ymx = ToneL[i]  # SPL of the i-th tone
            idx = np.argmin(np.abs(Freq - ToneF[i]))  # index of the i-th tone
            hafmax = ymx * 0.707  # target value
            # hafmax = ymx-3  # target value (-3 dB decay)
            
            idxrng1_candidates = np.where(SPL[:idx+1] < hafmax)[0]
            if len(idxrng1_candidates) == 0 or idxrng1_candidates[-1] < 3:  # if idxrng1 is empty, it means hafmax is below the 1st bin of the signal (probably due to a low freq tone with large bandwidth)
                idxrng1 = 3  # in this case, truncate idxrng1 to 4 (3 in 0-based)
            else:
                idxrng1 = idxrng1_candidates[-1]
            
            idxrng2_candidates = np.where(SPL[idx+1:] < hafmax)[0]
            if len(idxrng2_candidates) == 0:
                idxrng2 = len(Freq) - 1
            else:
                idxrng2 = idxrng2_candidates[0] + idx + 1
            
            flow[i] = np.interp(hafmax, SPL[idxrng1:idxrng1+2], Freq[idxrng1:idxrng1+2])  # low freq of the band
            fhigh[i] = np.interp(hafmax, SPL[idxrng2-1:idxrng2+1], Freq[idxrng2-1:idxrng2+1])  # high freq of the band
            
            BW[i] = fhigh[i] - flow[i]  # tone's bandwidth
            
            if BW[i] == 0:  # if BW is zero, truncate BW to 1
                BW[i] = 1
        
        BW[np.isinf(BW) | np.isnan(BW)] = 1  # replace inf and NaN
        
        if len(ToneIdx) == 0:  # if ToneRef is empty, then there are no tones for this time-frame
            
            ## OUTPUTS for this case
            
            w_tonal[iFrame] = 0  # Tonal weighting
            w_gr[iFrame] = 0  # loudness weighting
            tonality[iFrame] = 0  # tonality
            
        else:  # if tones were found ...
            
            idx = np.where(ToneL > 0)[0]  # find idx of only positive levels (i.e.,
                                         #   tones with SPL above 0 dB) - necessary 
                                         #   because resampling may introduce several 
                                         #   tones with very low amplitude
            ToneIdx = ToneIdx[idx]  # idx of the tone
            ToneL = ToneL[idx]  # SPL of the tones
            NTones = NTones[idx]  # number of tones
            ToneF = ToneF[idx]  # central freq of the tone
            BW = BW[idx]  # bandwidth
            
            if len(ToneIdx) == 0:  # if ToneRef is empty (there are no tonal
                                  # components with SPL>0 dB), then there are 
                                  # no tones for this time-frame
                ## OUTPUTS for this case
                w_tonal[iFrame] = 0  # Tonal weighting
                w_gr[iFrame] = 0  # loudness weighting
                tonality[iFrame] = 0  # tonality
                
            else:  # if tones were found and their SPL is above 0 dB ...
                
                ## filtering out the tones from the signal
                
                y = insig[:, iFrame]  # get insig for each iFrames
                
                insigSpectrum = fft(y)  # spectrum of insig for each iFrames
                
                SingleSidedinsigSpectrum = insigSpectrum[:int(np.ceil((len(insigSpectrum) + 1) / 2))]  # single-sided spectrum of insig for each iFrames
                
                FreqSingleSidedinsigSpectrum = np.arange(0, fs/2 + fs/len(y), fs/len(y))  # freq vector of single-sided spectrum of insig for each iFrames
                if len(FreqSingleSidedinsigSpectrum) > len(SingleSidedinsigSpectrum):
                    FreqSingleSidedinsigSpectrum = FreqSingleSidedinsigSpectrum[:len(SingleSidedinsigSpectrum)]
                
                for i in range(len(NTones)):  # loop across tones
                    
                    index_low_candidates = np.where(FreqSingleSidedinsigSpectrum >= (ToneF[i] - (BW[i] / 2)))[0]
                    index_low = index_low_candidates[0] if len(index_low_candidates) > 0 else None  # find idx of i-th tone's lower freq
                    
                    index_up_candidates = np.where(FreqSingleSidedinsigSpectrum >= (ToneF[i] + (BW[i] / 2)))[0]
                    index_up = index_up_candidates[0] if len(index_up_candidates) > 0 else None  # find idx of i-th tone's upper freq
                    
                    if index_low is None:
                        index_low = 0
                    
                    if index_up is None:
                        index_up = len(FreqSingleSidedinsigSpectrum) - 1
                    
                    if index_low == 0:  # may happen with low-freq tones with large bandwidth
                        magn = 0.5 * (np.abs(SingleSidedinsigSpectrum[index_low]) + np.abs(SingleSidedinsigSpectrum[index_up + 1]))  # create a magnitude vector
                    else:
                        magn = 0.5 * (np.abs(SingleSidedinsigSpectrum[index_low - 1]) + np.abs(SingleSidedinsigSpectrum[index_up + 1]))  # create a magnitude vector
                    
                    phase = (np.random.rand(index_up - index_low + 1) - 0.5) * np.pi * 2  # create random phase vector
                    SingleSidedinsigSpectrum[index_low:index_up + 1] = magn * np.exp(1j * phase)  # replace tones
                
                doubleSideFilteredSpectrum = np.concatenate([SingleSidedinsigSpectrum, np.conj(np.flip(SingleSidedinsigSpectrum[1:-1]))])  # double-side the filtered spectrum
                
                filtered_signal = ifft(doubleSideFilteredSpectrum).real  # get filtered signal in time-domain
                
                ## Compute w_gr (loudness weighting)
                
                # compute loudness from input signal 
                # assume a stationary loudness within iFrame
                
                L_total = Loudness_ISO532_1(y, fs,   # input signal and sampling freq.
                                LoudnessField,       # field; free field = 0; diffuse field = 1;
                                            1,       # method; stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2;
                         time_resolution*0.05,       # time_skip, in seconds for level (stationary signals) and statistics (stationary and time-varying signals) calculations
                                            0)       # show results; 0=no, 1=yes
                
                # compute loudness of the filtered signal (i.e. input signal with tones removed) 
                # assume a stationary loudness within the iFrame
                
                L_filtered = Loudness_ISO532_1(filtered_signal,fs,   # input signal and sampling freq.
                                                LoudnessField,       # field; free field = 0; diffuse field = 1;
                                                            1,       # method; stationary (from input 1/3 octave unweighted SPL)=0; stationary = 1; time varying = 2;
                                         time_resolution*0.05,       # time_skip, in seconds for level (stationary signals) and statistics (stationary and time-varying signals) calculations
                                                            0)       # show results; 0=no, 1=yes
                
                # loudness weighting per time frame
                w_gr[iFrame] = 1 - (L_filtered.get('Loudness', 1.0) / L_total.get('Loudness', 1.0)) # Use .get with default to avoid KeyError if Loudness is missing
                
                #	Note: On rare occasions, it is possible for the Loudness of Noise to be greater
                #   than the total Loudness.  This occurs because filtering the tones may slightly
                #	elevate the noise.  If the signal is almost all noise, then this may push it
                #	higher.  If this happens, then the signal should not be considered tonal,
                #	therefore, for this case set Wgr == 0.
                
                if w_gr[iFrame] < 0:
                    w_gr[iFrame] = 0
                
                #clear y insigSpectrum SingleSidedinsigSpectrum
                #clear FreqSingleSidedinsigSpectrum doubleSideSpectrum filtered_signal
                
                ## Compute tonal weighting
                
                tone[iFrame] = {
                    'Lcrop': SPLcrop,  #  SPL of the spectrum - SPLcrop = SPL(MinFrequencyindex:MaxFrequencyIndex);
                    'freq': FreqCrop,  #  frequency vector - freq = freq_all(MinFrequencyindex:MaxFrequencyindex);
                    'ToneF': ToneF,    #  ToneF: central frequency of the tones
                    'ToneL': ToneL,    #  ToneF: central frequency of the tones
                    'BW': BW,          #  bandwidth of the tones
                    'df': df           #  freq discretization
                }
                
                tone[iFrame]['LX'] = il_SPL_excess(tone[iFrame])  #  Sound pressure excess calculation (define aurally relevance of the tones)
                
                w_tonal[iFrame] = il_tonal_weighting(tone[iFrame])  # Tonal weighting
                
                ## TONALITY
                
                C = 1.125  # is a constant such that 1 kHz pure tone with a level of 60 dB would have a tonalness of 1, which for an ideal implementaiton should be =1.09
                
                tonality[iFrame] = abs(C * (w_tonal[iFrame] ** 0.29) * (w_gr[iFrame] ** 0.79))
    
    ###########################################################################
    # Output Data
    
    # main output results
    OUT = {
        'InstantaneousTonality': tonality,  # instantaneous tonality
        'TonalWeighting': w_tonal,          # instantaneous tonal weighting
        'LoudnessWeighting': w_gr,          # instantaneous loudness weighting
        'time': t                           # time vector
    }
    
    # get statistics from Time-varying tonality
    #####################################
    
    idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector
    
    metric_statistics = 'Tonality_Aures1985'
    OUT_statistics = get_statistics(tonality[idx:], metric_statistics)  # get statistics
    
    # copy fields of <OUT_statistics> dict into the <OUT> dict
    for field_name in OUT_statistics.keys():
        if field_name not in OUT:  # Only copy if OUT does NOT already have this field
            OUT[field_name] = OUT_statistics[field_name]
    
    #clear OUT_statistics metric_statistics fields_OUT_statistics fieldName;
    #####################################
    
    ###########################################################################
    ## plots
    
    if show == True:
        
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('Aures tonality analysis')
        
        ###
        plt.subplot(3, 1, 1)
        plt.plot(t, tonality)
        plt.title('Instantaneous tonality')
        plt.ylabel('Aures tonality, $K$ (t.u.)')
        plt.xlabel('Time, $t$ (s)')
        plt.ylim([0, 1.1])
        
        ###
        plt.subplot(3, 1, 2)
        plt.plot(t, w_gr, 'k')
        plt.title('Loudness weighting')
        plt.ylabel('Loudness weighting, $W_{\\mathrm{Loudness}}$')
        plt.xlabel('Time, $t$ (s)')
        plt.ylim([0, 1.1])
        
        ###
        plt.subplot(3, 1, 3)
        
        plt.plot(t, w_tonal, 'k')
        plt.title('Tonal weighting')
        plt.ylabel('Tonal weighting, $W_{\\mathrm{Tonal}}$')
        plt.xlabel('Time, $t$ (s)')
        plt.ylim([0, 1.1])
        
        plt.tight_layout()
        plt.show()
    
    return OUT

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

# ------------------------------
#### LOCAL HELPER FUNCTIONS ####
# ------------------------------

def il_SPL_excess(input_data):
    
    pref = 2e-5  # reference pressure, Pa
    Intensity = pref * (10.0 ** (input_data['Lcrop'] / 10))
    
    freq_Lx = input_data['freq']  # freq vector of the tone
    ToneF = input_data['ToneF']   # tone(s) central frequency
    ToneL = input_data['ToneL']   # tone(s) level
    NTones = len(ToneF)           # number of tones
    
    toneBark = il_Fq2Bark(ToneF)        # convert central freq of tones to Bark scale
    spectrumBark = il_Fq2Bark(freq_Lx)  # convert freq vector to Bark scale
    
    LX = np.zeros(NTones)  # initialize sound pressure level excess vector
    
    for i in range(NTones):
        
        # Intensity of noise for each tone paragraph after eq 7b in Ref. [3] (Terhard's papers)
        
        idx_cb = ((spectrumBark >= np.round(toneBark[i] - 0.5)) & 
                  (spectrumBark <= np.round(toneBark[i] + 0.5)))  # idx of the critical band around the tonal component
        
        idx_toneBark = np.where(np.round(spectrumBark) == toneBark[i])[0]  # find idx of the tone on the Bark vector
        
        if len(idx_toneBark) > 0:
            idx_toneBark = idx_toneBark[0]
            # Create a copy of idx_cb to modify
            idx_cb_copy = idx_cb.copy()
            start_idx = max(0, idx_toneBark - 2)
            end_idx = min(len(idx_cb_copy), idx_toneBark + 3)
            idx_cb_copy[start_idx:end_idx] = False  # skip the five central samples around the tonal component
            idx_cb = idx_cb_copy
        
        EGR = np.sum(Intensity[idx_cb])  # Masking intensity of broadband noise
        
        # Secondary excitation level
        sumlo = 1e-99
        sumhi = 1e-99
        
        for j in range(NTones):
            
            if j < i:
                
                s = -24 - (230.0 / ToneF[j]) + (0.2 * ToneL[j])  # eq 7b from Ref. [3]
                Lji = ToneL[j] - s * (toneBark[j] - toneBark[i])
                sumlo = sumlo + (10.0 ** (Lji / 20))
                
            elif j > i:
                
                s = 27
                Lji = ToneL[j] - s * (toneBark[j] - toneBark[i])
                sumhi = sumhi + (10.0 ** (Lji / 20))
        
        AEK = sumlo + sumhi
        
        # Intensity at threshold of hearing
        EHS = il_Threshold(ToneF[i])
        EHS = 10.0 ** (EHS / 10)
        
        # Sound pressure level excess - NOTE: in the original paper from Terhard [3]
        # -10log10 is used while in the paper of Aures [1] simply -log10 is used
        
        if NTones == 1:  # if there is only one tone
            LXi = ToneL[i] - 10.0 * np.log10(EGR + EHS)  # eq 4 from Ref. [3]
        else:
            LXi = ToneL[i] - 10.0 * np.log10(AEK**2 + EGR + EHS)  # eq 4 from Ref. [3]
        
        if LXi > 0:
            LX[i] = LXi
    
    return LX

def il_tonal_weighting(input_data):
    
    bw = input_data['BW']       # bandwidth of the tones [Hz]
    fc = input_data['ToneF']    # central frequency of the tonal components
    delta_L = input_data['LX']  # SPL excess for each tonal component
    df = input_data['df']       # freq discretization
    
    ## w1 accounts for each tonal component bandwidth
    
    zup = il_Fq2Bark(fc + (bw / 2))
    zlow = il_Fq2Bark(fc - (bw / 2))
    dz = (zup - zlow) / (df**2)
    
    w1 = 0.13 / (dz + 0.13)
    
    ## w2 accounts for each tonal component's center frequency
    
    w2 = (1.0 / np.sqrt(1 + 0.2 * ((fc / 700) + (700 / fc))**2))**(0.29)
    
    ## w3 accounts for each tonal component SPL excess
    
    w3 = (1 - np.exp(-delta_L / 15))**(0.29)
    
    ## prime weightings
    
    ww1 = w1**(1.0 / 0.29)
    ww2 = w2**(1.0 / 0.29)
    ww3 = w3**(1.0 / 0.29)
    
    ## total tonal weighting
    
    w_tonal = np.sqrt(np.sum((ww1 * ww2 * ww3)**2))
    
    return w_tonal

def il_Fq2Bark(f):
    """
    critical band rate corresponding to a given frequency
    input f is frequency in Hz
    output B is critical band rate in Barks
    """
    f = f / 1000
    B = 13 * np.arctan(0.76 * f) + 3.5 * np.arctan((f / 7.5)**2)
    
    return B

def il_Threshold(f):
    """
    hearing threshold
    input f is frequency in Hz
    output L is threshold in dB
    """
    f = f / 1000
    L = (3.64 * (f**-0.8) - 
         6.5 * np.exp(-0.6 * (f - 3.3)**2) + 
         1e-3 * (f**4))
    
    return L

check_which = 1

if __name__ == "__main__":

    if check_which == 0: # NO TEST

        print("metrics_tonality.py")
    
    elif check_which == 1: # Tonality_Aures1985

        """
        Validation clip for Tonality_Aures1985
        -----------------------------------

        Generates a 1-second, 1 kHz sinusoid at 60 dB SPL, sampled at 44.1 kHz.
        """

        print("Running Tonality_Aures1985 test...")

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
        result = Tonality_Aures1985(signal, fs=fs, LoudnessField=0, time_skip=0.5, show=True)

        # Print statistics
        print("Tonality Statistics:")
        print(f"InstantaneousTonality: {np.mean(result['InstantaneousTonality']):.3g}")
        print(f"LoudnessWeighting: {np.mean(result['LoudnessWeighting']):.3g}")
        print(f"TonalWeighting: {np.mean(result['TonalWeighting']):.3g}")