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

from sound_metrics import ob13_iso532_1
from metrics_loudness import Loudness_ISO532_1

from utilities import *

__all__ = ["Loudness_ISO532_1", "Tonality_Aures1985"]
FloatArray = NDArray[np.floating]

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

if __name__ == "__main__":
    # generate test signal for Tonality_ECMA418_2
    fs = 48000
    t = np.arange(0, 5, 1/fs)
    insig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)  # Example signal with two tones
    insig = insig.reshape(-1, 1)  # Reshape to [N, 1] for single channel
    # Call the Tonality_ECMA418_2 function
    OUT = Tonality_ECMA418_2(insig, fs, fieldtype='free-frontal', time_skip=0.304, show=True)