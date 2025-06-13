import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from typing import Union, Optional
from scipy.signal import lfilter, resample_poly, resample, filtfilt, butter, sosfilt
from scipy.special import comb
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import warnings
from scipy.interpolate import interp1d
from utilities import *
from sound_metrics import *

def shmAuditoryFiltBank(signal: np.ndarray, outplot: Union[bool, int, float] = False) -> np.ndarray:

    # Arguments validation
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a numpy array")
    
    if signal.ndim != 1 and not (signal.ndim == 2 and signal.shape[1] == 1):
        raise ValueError("signal must be a 1D array or 2D column vector")
    
    if signal.ndim == 2:
        signal = signal.flatten()
    
    if not np.isreal(signal).all():
        raise ValueError("signal must be real-valued")
    
    if not isinstance(outplot, (bool, int, float)):
        raise TypeError("outplot must be boolean or numeric")
    
    # Define constants
    sampleRate48k = 48e3  # Signal sample rate prescribed to be 48kHz (to be used for resampling), Section 5.1.1 ECMA-418-2:2024
    deltaFreq0 = 81.9289  # defined in Section 5.1.4.1 ECMA-418-2:2024
    c = 0.1618  # Half-Bark band centre-frequency demoninator constant defined in Section 5.1.4.1 ECMA-418-2:2024

    halfBark = np.arange(0.5, 27.0, 0.5)  # half-critical band rate scale
    bandCentreFreqs = (deltaFreq0/c)*np.sinh(c*halfBark)  # Section 5.1.4.1 Equation 9 ECMA-418-2:2024
    dfz = np.sqrt(deltaFreq0**2 + (c*bandCentreFreqs)**2)  # Section 5.1.4.1 Equation 10 ECMA-418-2:2024

    # Signal processing
    
    # Apply auditory filter bank
    # --------------------------
    # Filter equalised signal using 53 1/2Bark ERB filters according to 
    # Section 5.1.4.2 ECMA-418-2:2024

    k = 5  # filter order = 5, footnote 5 ECMA-418-2:2024
    e_i = np.array([0, 1, 11, 11, 1])  # filter coefficients for Section 5.1.4.2 Equation 15 ECMA-418-2:2024

    # Initialize output array
    signalFiltered = np.zeros((len(signal), 53))
    
    # Initialize plotting variables
    fig = None
    ax1 = None
    ax2 = None

    for zBand in range(52, -1, -1):  # 53:-1:1 in MATLAB (52 to 0 in Python)
        # Section 5.1.4.1 Equation 8 ECMA-418-2:2024
        tau = (1/(2**(2*k - 1))) * scipy.special.comb(2*k - 2, k - 1, exact=True) * (1/dfz[zBand])
        
        d = np.exp(-1/(sampleRate48k*tau))  # Section 5.1.4.1 ECMA-418-2:2024
        
        # Band-pass modifier Section 5.1.4.2 Equation 16/17 ECMA-418-2:2024
        bp = np.exp((1j*2*np.pi*bandCentreFreqs[zBand]*np.arange(k+2))/sampleRate48k)
        
        # Feed-backward coefficients, Section 5.1.4.2 Equation 14 ECMA-418-2:2024
        m = np.arange(1, k+1)
        a_m_terms = np.concatenate(([1], ((-d)**m) * np.array([scipy.special.comb(k, m_val, exact=True) for m_val in m])))
        a_m = a_m_terms * bp[:k+1]

        # Feed-forward coefficients, Section 5.1.4.2 Equation 15 ECMA-418-2:2024
        m = np.arange(0, k)
        i = np.arange(1, k)
        denominator_sum = np.sum(e_i[i] * (d**i))
        b_m = (((1-d)**k)/denominator_sum) * (d**m) * e_i[:k] * bp[:k]

        # Recursive filter Section 5.1.4.2 Equation 13 ECMA-418-2:2024
        # Note, the results are complex so 2x the real-valued band-pass signal
        # is required.
        signalFiltered[:, zBand] = 2*np.real(scipy.signal.lfilter(b_m, a_m, signal))

        # Plot figures
        if outplot:
            H, f = scipy.signal.freqz(b_m, a_m, worN=10000, whole=True, fs=48e3)
            phir = np.angle(H)
            phirUnwrap = np.unwrap(phir)
            phiUnwrap = phirUnwrap/np.pi*180
            
            # Plot frequency and phase response for filter
            if zBand == 52:  # 53 in MATLAB
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                # Move figure to center equivalent
                mngr = fig.canvas.manager
                if hasattr(mngr, 'window'):
                    try:
                        mngr.window.wm_geometry("+{}+{}".format(int(mngr.window.winfo_screenwidth()/2 - fig.get_figwidth()*fig.dpi/2), 
                                                              int(mngr.window.winfo_screenheight()/2 - fig.get_figheight()*fig.dpi/2)))
                    except:
                        pass
            
            ax1.semilogx(f, 20*np.log10(np.abs(H)))
            ax2.semilogx(f, phiUnwrap)

            if zBand == 0:  # 1 in MATLAB
                ax1.set_xlim([20, 20e3])
                ax1.set_xticks([31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
                ax1.tick_params(axis='x', which='minor', bottom=False)
                ax1.set_xticklabels(["31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
                ax1.set_xlabel("Frequency, Hz")
                ax1.set_ylabel(r"$\it{H}$, dB")
                ax1.set_prop_cycle(None)
                ax1.grid(True)
                
                ax2.set_xlim([20, 20e3])
                ax2.set_xticks([31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
                ax2.tick_params(axis='x', which='minor', bottom=False)
                ax2.set_xticklabels(["31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
                ax2.set_xlabel("Frequency, Hz")
                ax2.set_ylabel("Phase angle °")
                ax2.grid(True)
                
                # Set font properties
                for ax in [ax1, ax2]:
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontfamily('Arial')
                        item.set_fontsize(12)
                
                plt.tight_layout()
                plt.show()

    return signalFiltered

def shmBasisLoudness(signalSegmented, bandCentreFreq=None):
    
    # Arguments validation
    if not isinstance(signalSegmented, np.ndarray):
        raise TypeError("signalSegmented must be a numpy array")
    
    if not np.isrealobj(signalSegmented):
        raise ValueError("signalSegmented must be real")
    
    if bandCentreFreq is not None:
        if not isinstance(bandCentreFreq, (int, float, np.number)):
            raise TypeError("bandCentreFreq must be a number")
        if bandCentreFreq <= 0:
            raise ValueError("bandCentreFreq must be positive")
    
    # check if input is 2D and includes band centre frequency - otherwise raise
    # error
    if bandCentreFreq is None and len(signalSegmented.shape) == 2:
        raise ValueError("Band centre frequency must be specified for single band-limited input signal")
    
    # check if input band centre frequency is not a vector (arguments
    # validation does not allow empty default with specified size)
    if bandCentreFreq is not None:
        if isinstance(bandCentreFreq, np.ndarray):
            if bandCentreFreq.size != 1:
                raise ValueError("Band centre frequency input must be a single value")
        # Convert to scalar if it's a numpy array with single element
        if isinstance(bandCentreFreq, np.ndarray):
            bandCentreFreq = bandCentreFreq.item()
    
    # Define constants
    
    deltaFreq0 = 81.9289  # defined in Section 5.1.4.1 ECMA-418-2:2024
    c = 0.1618  # Half-Bark band centre-frequency denominator constant defined in Section 5.1.4.1 ECMA-418-2:2024
    
    halfBark = np.arange(0.5, 27.0, 0.5)  # half-critical band rate scale
    bandCentreFreqs = (deltaFreq0/c) * np.sinh(c * halfBark)  # Section 5.1.4.1 Equation 9 ECMA-418-2:2024
    
    cal_N = 0.0211668  # Calibration factor from Section 5.1.8 Equation 23 ECMA-418-2:2024
    cal_Nx = 1.00132  # Calibration multiplier (Footnote 8 ECMA-418-2:2024)
    
    a = 1.5  # Constant (alpha) from Section 5.1.8 Equation 23 ECMA-418-2:2024
    
    # Values from Section 5.1.8 Table 2 ECMA-418-2:2024
    p_threshold = 2e-5 * 10**(np.arange(15, 86, 10)/20)
    v = np.array([1, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434])
    
    # Loudness threshold in quiet Section 5.1.9 Table 3 ECMA-418-2:2024
    LTQz = np.array([0.3310, 0.1625, 0.1051, 0.0757, 0.0576, 0.0453, 0.0365, 0.0298,
                     0.0247, 0.0207, 0.0176, 0.0151, 0.0131, 0.0115, 0.0103, 0.0093,
                     0.0086, 0.0081, 0.0077, 0.0074, 0.0073, 0.0072, 0.0071, 0.0072,
                     0.0073, 0.0074, 0.0076, 0.0079, 0.0082, 0.0086, 0.0092, 0.0100,
                     0.0109, 0.0122, 0.0138, 0.0157, 0.0172, 0.0180, 0.0180, 0.0177,
                     0.0176, 0.0177, 0.0182, 0.0190, 0.0202, 0.0217, 0.0237, 0.0263,
                     0.0296, 0.0339, 0.0398, 0.0485, 0.0622])
    
    # Input check
    
    if bandCentreFreq is not None and not np.isin(bandCentreFreq, bandCentreFreqs):
        raise ValueError("Input half-Bark critical rate scale band centre frequency does not match ECMA-418-2:2024 values")
    
    # Signal processing
    
    # Half Wave Rectification
    # -----------------------
    # Section 5.1.6 Equation 21 ECMA-418-2:2020
    signalRectSeg = signalSegmented.copy()
    signalRectSeg[signalSegmented <= 0] = 0
    
    # Calculation of RMS
    # ------------------
    # Section 5.1.7 Equation 22 ECMA-418-2:2024
    blockRMS = np.sqrt((2/signalRectSeg.shape[0]) * np.sum(signalRectSeg**2, axis=0))
    
    # Transformation into Loudness
    # ----------------------------
    # Section 5.1.8 Equations 23 & 24 ECMA-418-2:2024

    factor_inside = np.divide(blockRMS, p_threshold)
    bandLoudness = cal_N * cal_Nx * (blockRMS/20e-6) * np.prod((1 + factor_inside**a)**(np.diff(v)/a), axis=0)
    
    D1 = np.max(bandLoudness)
    D2 = np.min(bandLoudness)
    
    # remove singleton dimension from block RMS output
    blockRMS = np.squeeze(blockRMS)
    
    # Section 5.1.9 Equation 25 ECMA-418-2:2024
    if bandCentreFreq is not None and len(signalSegmented.shape) == 2:
        # half-Bark critical band basis loudness
        D3 = LTQz[bandCentreFreq == bandCentreFreqs]
        basisLoudness = bandLoudness - LTQz[bandCentreFreq == bandCentreFreqs]
        basisLoudness[basisLoudness < 0] = 0
    else:
        # basis loudness for all bands
        if len(bandLoudness.shape) == 1:
            # Handle 1D case
            basisLoudness = bandLoudness - LTQz
        else:
            # Handle multi-dimensional case
            LTQz_reshaped = LTQz.reshape(1, 1, -1)
            basisLoudness = bandLoudness - np.tile(LTQz_reshaped, 
                                                   (1, bandLoudness.shape[1], 1))
        basisLoudness[basisLoudness < 0] = 0
    
    return signalRectSeg, basisLoudness, blockRMS

def shmNoiseRedLowPass(signal, sampleRatein):
    
    # Arguments validation
    signal_input = np.asarray(signal)
    if np.any(np.iscomplex(signal_input)):
        raise ValueError("signal must be real")
    
    signal = signal_input.astype(np.float64)
    sampleRatein = float(sampleRatein)
    
    # Validate sampleRatein is positive
    if sampleRatein <= 0:
        raise ValueError("sampleRatein must be positive")

    k = 3  # Footnote 21 ECMA-418-2:2024
    e_i = [0, 1, 1]  # Footnote 21 ECMA-418-2:2024

    # Footnote 20 ECMA-418-2:2024
    tau = 1/32*6/7

    d = np.exp(-1/(sampleRatein*tau))  # Section 5.1.4.2 ECMA-418-2:2024

    # Feed-backward coefficients, Equation 14 ECMA-418-2:2024
    m = np.arange(1, k+1)
    a = np.concatenate([[1], ((-d)**m) * np.array([comb(k, m_, exact=True) for m_ in m])])

    # Feed-forward coefficients, Equation 15 ECMA-418-2:2024
    m = np.arange(0, k)
    i = np.arange(1, k)
    e_i_array = np.array(e_i)
    b = (((1 - d)**k) / np.sum(e_i_array[i] * (d**i))) * (d**m) * e_i_array

    # Recursive filter Equation 13 ECMA-418-2:2024
    signalFiltered = lfilter(b, a, signal, axis=0)

    return signalFiltered

def shmOutMidEarFilter(signal, fieldtype='free-frontal', outplot=False):

    
    # Arguments validation
    signal = np.asarray(signal, dtype=float)
    if not np.isrealobj(signal):
        raise ValueError("signal must be real-valued")
    
    if not isinstance(fieldtype, str):
        raise TypeError("fieldtype must be a string")
    if fieldtype not in ['free-frontal', 'diffuse']:
        raise ValueError("fieldtype must be 'free-frontal' or 'diffuse'")
    
    if not isinstance(outplot, (bool, int, float)):
        raise TypeError("outplot must be boolean or numeric")
    outplot = bool(outplot)
    
    # Signal processing
    
    # Apply outer & middle ear filter bank
    # ------------------------------------

    b_0k = np.array([1.015896020255593, 0.958943219304445, 0.961371976333197,
                     2.225803503609735, 0.471735128494163, 0.115267139824401,
                     0.988029297230954, 1.952237687301361])
    b_1k = np.array([-1.925298877776079, -1.806088011849494, -1.763632154338248,
                     -1.434650484792157, -0.366091796830044, 0.0,
                     -1.91243380293387, 0.162319983017519])
    b_2k = np.array([0.922118060364679, 0.876438777856084, 0.821787991845146,
                     -0.498204282194628, 0.244144703885020, -0.115267139824401,
                     0.926131550180785, -0.667994113035186])
    a_0k = np.ones_like(b_0k)
    a_1k = np.array([-1.925298877776079, -1.806088011849494, -1.763632154338248,
                     -1.434650484792157, -0.366091796830044, -1.796002566692014,
                     -1.912433802933871, 0.162319983017519])
    a_2k = np.array([0.938014080620272, 0.835381997160530, 0.783159968178343,
                     0.727599221415107, -0.284120167620817, 0.805837815618546,
                     0.914160847411739, 0.284243574266175])
    
    if fieldtype == "free-frontal":
        sos = np.column_stack([b_0k, b_1k, b_2k, a_0k, a_1k, a_2k])
    elif fieldtype == "diffuse":
        # omit free field filter stages
        sos = np.column_stack([b_0k[2:], b_1k[2:], b_2k[2:],
                               a_0k[2:], a_1k[2:], a_2k[2:]])
    
    # Section 5.1.3.2 ECMA-418-2:2024 Outer and middle/inner ear signal filtering
    signalFiltered = sosfilt(sos, signal, axis=0)
    
    # Plot figures
    
    if outplot:
        H, f = signal.freqz_zpk(sos, worN=10000, whole=True, fs=48000)
        phir = np.angle(H)
        phirUnwrap = np.unwrap(phir, axis=0)
        phiUnwrap = phirUnwrap / np.pi * 180
        
        # Plot frequency and phase response for filter
        fig = plt.figure()
        fig.set_size_inches(8, 6)
        
        # Move figure to center (equivalent to movegui(fig, 'center'))
        mngr = fig.canvas.manager
        if hasattr(mngr, 'window'):
            try:
                mngr.window.wm_geometry("+%d+%d" % (100, 100))
            except:
                pass
        
        ax1 = plt.subplot(2, 1, 1)
        ax1.semilogx(f, 20*np.log10(np.abs(H)), color=[0.0, 0.2, 0.8])
        ax1.set_xlim([20, 20e3])
        ax1.set_xticks([31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
        ax1.tick_params(axis='x', which='minor', bottom=False)
        ax1.set_xticklabels(["31.5", "63", "125", "250", "500", "1k", "2k", "4k",
                            "8k", "16k"])
        ax1.set_xlabel("Frequency, Hz")
        ax1.set_ylabel(r"$\it{H}$, dB")
        ax1.set_title(fieldtype)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(12)
        ax1.grid(True)
        
        ax2 = plt.subplot(2, 1, 2)
        ax2.semilogx(f, phiUnwrap, color=[0.8, 0.1, 0.8])
        ax2.set_xlim([20, 20e3])
        ax2.set_xticks([31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
        ax2.tick_params(axis='x', which='minor', bottom=False)
        ax2.set_xticklabels(["31.5", "63", "125", "250", "500", "1k", "2k", "4k",
                            "8k", "16k"])
        ax2.set_xlabel("Frequency, Hz")
        ax2.set_ylabel(r"Phase angle $\circ$")
        ax2.grid(True)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(12)
        
        plt.tight_layout()
        plt.show()
    
    return signalFiltered

def shmPreProc(signal, blockSize, hopSize, padStart=True, padEnd=True):

    
    # Arguments validation
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    
    if not np.isreal(signal).all():
        raise ValueError("signal must be real")
    
    if not isinstance(blockSize, (int, np.integer)):
        blockSize = int(blockSize)
    if not isinstance(hopSize, (int, np.integer)):
        hopSize = int(hopSize)
    
    if not isinstance(padStart, (bool, int, float, np.number)):
        raise TypeError("padStart must be numeric or logical")
    if not isinstance(padEnd, (bool, int, float, np.number)):
        raise TypeError("padEnd must be numeric or logical")
    
    # Signal processing
    
    # Input pre-processing
    # --------------------
    #
    # Fade in weighting function Section 5.1.2 ECMA-418-2:2024
    fadeWeight = np.tile((0.5 - 0.5*np.cos(np.pi*np.arange(240)/240))[:, np.newaxis], (1, signal.shape[1]))
    # Apply fade in
    signalFade = np.vstack([fadeWeight*signal[0:240, :],
                           signal[240:, :]])
    
    # Zero-padding Section 5.1.2 ECMA-418-2:2024
    if padStart:
        n_zeross = blockSize  # start zero-padding
    else:
        n_zeross = 0  # end of if branch to zero pad start
    
    if padEnd:
        n_samples = signal.shape[0]
        n_new = hopSize*(np.ceil((n_samples + hopSize + n_zeross)/hopSize) - 1)
        n_zerose = int(n_new - n_samples)  # end zero-padding
    else:
        n_zerose = 0  # end of if branch to zero pad end
    
    # Apply zero-padding
    signalOut = np.vstack([np.zeros((n_zeross, signalFade.shape[1])),
                          signalFade,
                          np.zeros((n_zerose, signalFade.shape[1]))])
    
    return signalOut
    
def shmResample(signal, sampleRatein):
    
    # Arguments validation
    signal = np.asarray(signal, dtype=float)
    if not np.isrealobj(signal):
        raise ValueError("signal must be real")
    
    if not isinstance(sampleRatein, (int, float)) or sampleRatein <= 0 or sampleRatein != int(sampleRatein):
        raise ValueError("sampleRatein must be a positive integer")
    sampleRatein = int(sampleRatein)
    
    # Define constants
    
    # Section 5.1.1 ECMA-418-2:2024
    resampledRate = int(48e3)  # Signal sample rate prescribed to be 48 kHz
    
    # Signal processing
    
    # Input pre-processing
    # --------------------
    if sampleRatein != resampledRate:  # Resample signal
        up = resampledRate // np.gcd(resampledRate, sampleRatein)  # upsampling factor
        down = sampleRatein // np.gcd(resampledRate, sampleRatein)  # downsampling factor
        resampledSignal = resample_poly(signal, up, down, axis=0)  # apply resampling
    else:  # don't resample
        resampledSignal = signal
    
    return resampledSignal, resampledRate

def shmRoughLowPass(specRoughEstTform, sampleRate, riseTime, fallTime):
    
    # Arguments validation
    if not isinstance(specRoughEstTform, np.ndarray):
        specRoughEstTform = np.array(specRoughEstTform)
    
    if specRoughEstTform.ndim != 2:
        raise ValueError("specRoughEstTform must be a 2D array")
    
    if not np.isreal(specRoughEstTform).all():
        raise ValueError("specRoughEstTform must contain only real values")
    
    if not isinstance(sampleRate, (int, float)) or sampleRate <= 0:
        raise ValueError("sampleRate must be a positive scalar")
    
    if not isinstance(riseTime, (int, float)) or riseTime <= 0:
        raise ValueError("riseTime must be a positive scalar")
    
    if not isinstance(fallTime, (int, float)) or fallTime <= 0:
        raise ValueError("fallTime must be a positive scalar")

    riseExponent = np.exp(-1/(sampleRate*riseTime)) * np.ones((1, specRoughEstTform.shape[1]))
    fallExponent = np.exp(-1/(sampleRate*fallTime)) * np.ones((1, specRoughEstTform.shape[1]))

    specRoughness = specRoughEstTform.copy()

    for llBlock in range(1, specRoughEstTform.shape[0]):

        riseMask = specRoughEstTform[llBlock, :] >= specRoughness[llBlock - 1, :]
        fallMask = ~riseMask

        if np.any(riseMask):
            specRoughness[llBlock, riseMask] = (specRoughEstTform[llBlock, riseMask] * (1 - riseExponent[0, riseMask]) +
                                               specRoughness[llBlock - 1, riseMask] * riseExponent[0, riseMask])
        
        if np.any(fallMask):
            specRoughness[llBlock, fallMask] = (specRoughEstTform[llBlock, fallMask] * (1 - fallExponent[0, fallMask]) +
                                               specRoughness[llBlock - 1, fallMask] * fallExponent[0, fallMask])

    return specRoughness

def shmRoughWeight(modRate, modfreqMaxWeight, roughWeightParams):

    
    # Convert inputs to numpy arrays for proper array operations
    modRate = np.asarray(modRate)
    roughWeightParams = np.asarray(roughWeightParams)
    
    # Equation 85 [G_l,z,i(f_p,i(l,z))]
    roughWeight = 1.0 / (
        (1 + 
         ((modRate / modfreqMaxWeight 
           - modfreqMaxWeight / modRate) 
          * roughWeightParams[0, :, :]) ** 2) ** roughWeightParams[1, :, :]
    )
    
    return roughWeight

def shmSignalSegment(signal, axisn=1, blockSize=None, overlap=0, i_start=1, endShrink=False):
    
    # Arguments validation
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=float)
    
    if signal.ndim == 0:
        signal = signal.reshape(-1, 1)
    elif signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    elif signal.ndim > 2:
        raise ValueError("signal must be a vector or 2D matrix")
    
    if not np.isrealobj(signal):
        raise ValueError("signal must be real")
    
    if not isinstance(axisn, (int, np.integer)) or axisn not in [1, 2]:
        raise ValueError("axisn must be 1 or 2")
    
    if blockSize is None or blockSize is False:
        raise ValueError("blockSize must be specified")
    
    if not isinstance(blockSize, (int, np.integer)) or blockSize <= 0:
        raise ValueError("blockSize must be a positive integer")
    
    if not isinstance(overlap, (int, float, np.number)) or not (0 <= overlap < 1):
        raise ValueError("overlap must be >= 0 and < 1")
    
    if not isinstance(i_start, (int, np.integer)) or i_start <= 0:
        raise ValueError("i_start must be a positive integer")
    
    if not isinstance(endShrink, (bool, np.bool_, int, np.integer)):
        raise ValueError("endShrink must be numeric or logical")
    
    endShrink = bool(endShrink)
    
    # Signal pre-processing
    
    # Orient input
    if axisn == 2:
        signal = signal.T
        axisFlip = True
    else:
        axisFlip = False
    
    # Check sample index start will allow segmentation to proceed
    # Convert to 0-based indexing for Python
    i_start_0 = i_start - 1
    if signal[i_start_0:, :].shape[0] <= blockSize:
        raise ValueError("Signal is too short to apply segmentation using the selected parameters")
    
    # Assign number of channels
    nchans = signal.shape[1]
    
    # Hop size
    hopSize = int((1 - overlap) * blockSize)
    
    # Truncate the signal to start from i_start and to end at an index
    # corresponding with the truncated signal length that will fill an
    # integer number of overlapped blocks
    signalTrunc = signal[i_start_0:, :]
    n_blocks = int(np.floor((signalTrunc.shape[0] - overlap * blockSize) / hopSize))
    i_end = int(n_blocks * hopSize + overlap * blockSize)
    signalTrunc = signalTrunc[:i_end, :]
    
    # Signal segmentation
    
    # Initialize output array
    if endShrink and (signal[i_start_0:, :].shape[0] > signalTrunc.shape[0]):
        n_output_blocks = n_blocks + 1
    else:
        n_output_blocks = n_blocks
    
    signalSegmented = np.zeros((blockSize, n_output_blocks, nchans))
    
    # Arrange the signal into overlapped blocks - each block reads
    # along first axis, and each column is the succeeding overlapped
    # block. 3 columns of zeros are appended to the left side of the
    # matrix and the column shifted copies of this matrix are
    # concatenated. The first 6 columns are then discarded as these all
    # contain zeros from the appended zero columns.
    
    for chan in range(nchans - 1, -1, -1):  # nchans:-1:1 in MATLAB
        signalSegmentedChan = np.concatenate([
            np.zeros((hopSize, 3)),
            signalTrunc[:, chan].reshape(hopSize, -1, order='F')
        ], axis=1)
        
        # Create shifted versions using np.roll (equivalent to circshift)
        shifted_0 = np.roll(signalSegmentedChan, 0, axis=1)
        shifted_1 = np.roll(signalSegmentedChan, 1, axis=1)
        shifted_2 = np.roll(signalSegmentedChan, 2, axis=1)
        shifted_3 = np.roll(signalSegmentedChan, 3, axis=1)
        
        # Concatenate vertically (equivalent to cat(1, ...))
        signalSegmentedChan = np.concatenate([
            shifted_3,
            shifted_2,
            shifted_1,
            shifted_0
        ], axis=0)
        
        signalSegmentedChan = signalSegmentedChan[:, 6:]  # Remove first 6 columns
        
        # if branch to include block of end data with increased overlap
        if endShrink and (signal[i_start_0:, :].shape[0] > signalTrunc.shape[0]):
            end_block = signal[-blockSize:, chan].reshape(-1, 1)
            signalSegmentedChanOut = np.concatenate([signalSegmentedChan, end_block], axis=1)
            iBlocksOut = np.concatenate([
                np.arange(1, n_blocks * hopSize + 1, hopSize),
                [signal[i_start_0:, :].shape[0] - blockSize + 1]
            ])
        else:
            signalSegmentedChanOut = signalSegmentedChan
            iBlocksOut = np.arange(1, n_blocks * hopSize + 1, hopSize)
        
        signalSegmented[:, :, chan] = signalSegmentedChanOut
    
    # re-orient segmented signal to match input
    if axisFlip:
        signalSegmented = np.transpose(signalSegmented, (1, 0, 2))
    
    return signalSegmented, iBlocksOut

def Tonality_ECMA418_2(insig, fs, fieldtype='free-frontal', time_skip=304e-3, show=False):
    """
    Tonality analysis according to ECMA-418-2
    
    Parameters:
    insig : array_like
        Input signal (Nx1 or Nx2)
    fs : int
        Sample rate (must be positive integer)
    fieldtype : str
        Field type ('free-frontal' or 'diffuse'), default 'free-frontal'
    time_skip : float
        Time threshold in seconds, default 304e-3
    show : bool
        Whether to show plots, default False
        
    Returns:
    OUT : dict
        Dictionary containing all output parameters
    """
    
    # Arguments validation
    insig = np.asarray(insig, dtype=float)
    if not np.isreal(insig).all():
        raise ValueError("insig must be real")
    
    if not isinstance(fs, (int, float)) or fs <= 0 or fs != int(fs):
        raise ValueError("fs must be a positive integer")
    fs = int(fs)
    
    if fieldtype not in ['free-frontal', 'diffuse']:
        raise ValueError("fieldtype must be 'free-frontal' or 'diffuse'")
    
    if not isinstance(time_skip, (int, float)) or not np.isreal(time_skip):
        raise ValueError("time_skip must be real")
    
    if not isinstance(show, (bool, int, float)):
        raise ValueError("show must be numeric or logical")
    show = bool(show)
    
    # Input checks
    # define time threshold value from which all values before must be dropped.
    t_threshold = 304e-3
    
    # check insig dimension (only [Nx1] or [Nx2] are valid)
    if insig.ndim == 1:
        insig = insig.reshape(-1, 1)
    
    if insig.shape[0] > 2 and insig.shape[1] > 2:  # insig has more than 2 channels
        raise ValueError('Error: Input signal has more than 2 channels. ')
    elif insig.shape[1] > 2:  # insig is [1xN] or [2xN]
        insig = insig.T
        print('\nWarning: Input signal is not [Nx1] or [Nx2] and was transposed.\n')
    
    # Check the length of the input data (must be at least 304 ms)
    if insig.shape[0] < t_threshold * fs:
        raise ValueError('Error: Input signal is too short along the specified axis to calculate tonality (must be at least 304 ms)')
    
    # Check the channel number of the input data
    if insig.shape[1] > 2:
        raise ValueError('Error: Input signal comprises more than two channels')
    else:
        inchans = insig.shape[1]
        if inchans > 1:
            chans = ["Stereo left", "Stereo right"]
        else:
            chans = ["Mono"]
    
    if time_skip < t_threshold:
        warnings.warn("Time_skip must be at least 304 ms to avoid transient responses of the digital filters (see ECMA-418-2:2024 (Section 6.2.9)). Setting time_skip to 304 ms!!!")
        time_skip = t_threshold
    
    # Define constants
    sampleRate48k = 48e3  # Signal sample rate prescribed to be 48kHz (to be used for resampling), Section 5.1.1 ECMA-418-2:2024 [r_s]
    deltaFreq0 = 81.9289  # defined in Section 5.1.4.1 ECMA-418-2:2024 [deltaf(f=0)]
    c = 0.1618  # Half-Bark band centre-frequency denominator constant defined in Section 5.1.4.1 ECMA-418-2:2024
    
    halfBark = np.arange(0.5, 27, 0.5)  # half-critical band rate scale [z]
    bandCentreFreqs = (deltaFreq0/c) * np.sinh(c * halfBark)  # Section 5.1.4.1 Equation 9 ECMA-418-2:2024 [F(z)]
    dfz = np.sqrt(deltaFreq0**2 + (c * bandCentreFreqs)**2)  # Section 5.1.4.1 Equation 10 ECMA-418-2:2024 [deltaf(z)]
    
    # Block and hop sizes Section 6.2.2 Table 4 ECMA-418-2:2024
    overlap = 0.75  # block overlap proportion
    # block sizes [s_b(z)]
    blockSize = np.concatenate([8192 * np.ones(3, dtype=int), 4096 * np.ones(13, dtype=int), 
                               2048 * np.ones(9, dtype=int), 1024 * np.ones(28, dtype=int)])
    # hop sizes (section 5.1.2 footnote 3 ECMA 418-2:2022) [s_h(z)]
    hopSize = ((1 - overlap) * blockSize).astype(int)
    
    # Output sample rate based on hop sizes - Resampling to common time basis
    # Section 6.2.6 ECMA-418-2:2024 [r_sd]
    sampleRate1875 = sampleRate48k / np.min(hopSize)
    
    # Number of bands that need averaging. Section 6.2.3 Table 5
    # ECMA-418-2:2024 [NB]
    NBandsAvg = np.array([[0, 1] + [2]*14 + [1]*9 + [0]*28,
                         [1, 1] + [2]*14 + [1]*9 + [0]*28])
    
    # Critical band interpolation factors from Section 6.2.6 Table 6
    # ECMA-418-2:2024 [i]
    i_interp = (blockSize / np.min(blockSize)).astype(int)
    
    # Noise reduction constants from Section 6.2.7 Table 7 ECMA-418-2:2024
    alpha = 20
    beta = 0.07
    
    # Sigmoid function factor parameters Section 6.2.7 Table 8 ECMA-418-2:2024
    # [c(s_b(z))]
    csz_b = np.concatenate([18.21 * np.ones(3), 12.14 * np.ones(13), 
                           417.54 * np.ones(9), 962.68 * np.ones(28)])
    # [d(s_b(z))]
    dsz_b = np.concatenate([0.36 * np.ones(3), 0.36 * np.ones(13), 
                           0.71 * np.ones(9), 0.69 * np.ones(28)])
    
    # Scaling factor constants from Section 6.2.8 Table 9 ECMA-418-2:2024
    A = 35
    B = 0.003
    
    cal_T = 2.8758615  # calibration factor in Section 6.2.8 Equation 51 ECMA-418-2:2024 [c_T]
    cal_Tx = 1/0.9999043734252  # Adjustment to calibration factor (Footnote 22 ECMA-418-2:2024)
    
    # Signal processing
    # Input pre-processing
    # --------------------
    if fs != sampleRate48k:  # Resample signal
        p_re, _ = shmResample(insig, fs)
    else:  # don't resample
        p_re = insig
    
    # get time vector of input signal
    timeInsig = np.arange(len(p_re[:, 0])) / fs
    
    # Section 5.1.2 ECMA-418-2:2024 Fade in weighting and zero-padding
    pn = shmPreProc(p_re, np.max(blockSize), np.max(hopSize))
    
    # Apply outer & middle ear filter
    # -------------------------------
    #
    # Section 5.1.3.2 ECMA-418-2:2024 Outer and middle/inner ear signal filtering
    pn_om = shmOutMidEarFilter(pn, fieldtype)
    
    # Initialize output arrays
    num_time_points = int(np.ceil(p_re.shape[0] / sampleRate48k * sampleRate1875)) + 1
    specTonalLoudness = np.zeros((num_time_points, 53, pn_om.shape[1]))
    specNoiseLoudness = np.zeros((num_time_points, 53, pn_om.shape[1]))
    specTonalityFreqs = np.zeros((num_time_points, 53, pn_om.shape[1]))
    specTonality = np.zeros((num_time_points, 53, pn_om.shape[1]))
    tonalityTDep = np.zeros((num_time_points, pn_om.shape[1]))
    tonalityTDepFreqs = np.zeros((num_time_points, pn_om.shape[1]))
    tonalityAvg = np.zeros(pn_om.shape[1])
    specTonalityAvg = np.zeros((1, 53, pn_om.shape[1]))
    specTonalityAvgFreqs = np.zeros((1, 53, pn_om.shape[1]))
    
    # Loop through channels in file
    # -----------------------------
    for chan in range(pn_om.shape[1] - 1, -1, -1):
        
        # Apply auditory filter bank
        # --------------------------
        
        # Filter equalised signal using 53 1/2Bark ERB filters according to 
        # Section 5.1.4.2 ECMA-418-2:2024
        pn_omz = shmAuditoryFiltBank(pn_om[:, chan], False)
        
        # Autocorrelation function analysis
        # ---------------------------------
        # Duplicate Banded Data for ACF
        # Averaging occurs over neighbouring bands, to do this the segmentation
        # needs to be duplicated for neigbouring bands. 'Dupe' has been added
        # to variables to indicate that the vectors/matrices have been modified
        # for duplicated neigbouring bands.
        
        pn_omzDupe = np.concatenate([pn_omz[:, 0:5], pn_omz[:, 1:18], 
                                    pn_omz[:, 15:26], pn_omz[:, 25:53]], axis=1)
        blockSizeDupe = np.concatenate([8192 * np.ones(5, dtype=int), 4096 * np.ones(17, dtype=int),
                                       2048 * np.ones(11, dtype=int), 1024 * np.ones(28, dtype=int)])
        bandCentreFreqsDupe = np.concatenate([bandCentreFreqs[0:5], bandCentreFreqs[1:18],
                                            bandCentreFreqs[15:26], bandCentreFreqs[25:53]])
        
        # (duplicated) indices corresponding with the NB bands around each z band
        i_NBandsAvgDupe = np.array([[0, 0, 0] + list(range(5, 18)) + list(range(22, 31)) + list(range(33, 61)),
                                   [1, 2, 4] + list(range(9, 22)) + list(range(24, 33)) + list(range(33, 61))])
        
        unbiasedNormACFDupe = [None] * 61
        basisLoudnessArray = [None] * 61
        
        for zBand in range(60, -1, -1):
            
            # Segmentation into blocks
            # ------------------------
            # Section 5.1.5 ECMA-418-2:2024
            i_start = blockSizeDupe[0] - blockSizeDupe[zBand]
            pn_lz, _ = shmSignalSegment(pn_omzDupe[:, zBand], 1,
                                      blockSizeDupe[zBand], overlap, i_start)
            
            # Transformation into Loudness
            # ----------------------------
            # Sections 5.1.6 to 5.1.9 ECMA-418-2:2024 [N'_basis(z)]
            pn_rlz, bandBasisLoudness, _ = shmBasisLoudness(pn_lz, bandCentreFreqsDupe[zBand])
            basisLoudnessArray[zBand] = bandBasisLoudness
            
            # Apply ACF
            # ACF implementation using DFT
            # Section 6.2.2 Equations 27 & 28 ECMA-418-2:2024
            # [phi_unscaled,l,z(m)]
            unscaledACF = np.fft.ifft(np.abs(np.fft.fft(pn_rlz, 2*blockSizeDupe[zBand], axis=0))**2,
                                     2*blockSizeDupe[zBand], axis=0)
            
            # Section 6.2.2 Equation 29 ECMA-418-2:2024 [phi_l,z(m)]
            cumsum_forward = np.cumsum(pn_rlz**2, axis=0)
            cumsum_reverse = np.cumsum(pn_rlz**2[::-1, :], axis=0)[::-1, :]
            denom = np.sqrt(cumsum_reverse * np.flip(cumsum_forward, axis=0)) + 1e-12
            
            # note that the block length is used here, rather than the 2*s_b,
            # for compatability with the remaining code - beyond 0.75*s_b is
            # assigned (unused) zeros in the next line
            unbiasedNormACF = unscaledACF[:blockSizeDupe[zBand], :] / denom
            unbiasedNormACF[int(0.75*blockSizeDupe[zBand]):blockSizeDupe[zBand], :] = 0
            
            # Section 6.2.2 Equation 30 ECMA-418-2:2024 [phi_z'(m)
            unbiasedNormACFDupe[zBand] = basisLoudnessArray[zBand] * unbiasedNormACF
        
        # Average the ACF over nB bands - Section 6.2.3 ECMA-418-2:2024        
        for zBand in range(52, -1, -1):  # Loop through 53 critical band filtered signals
            
            NBZ = NBandsAvg[0, zBand] + NBandsAvg[1, zBand] + 1  # Total number of bands to average over
            
            # Averaging of frequency bands
            band_start = i_NBandsAvgDupe[0, zBand]
            band_end = i_NBandsAvgDupe[1, zBand] + 1
            
            # Concatenate the ACFs for averaging
            acf_data = []
            for b in range(band_start, band_end):
                acf_data.append(unbiasedNormACFDupe[b])
            
            acf_combined = np.concatenate(acf_data, axis=1)
            acf_reshaped = acf_combined.reshape(blockSize[zBand], -1, NBZ)
            meanScaledACF = np.mean(acf_reshaped, axis=2)
            
            # Average the ACF over adjacent time blocks [phibar_z'(m)]
            if zBand <= 15:  # 0-indexed, so 15 corresponds to 16 in MATLAB
                if meanScaledACF.shape[1] >= 3:
                    # Apply moving mean with window size 3, discarding endpoints
                    temp = np.zeros_like(meanScaledACF)
                    temp[:, 1:-1] = (meanScaledACF[:, :-2] + meanScaledACF[:, 1:-1] + meanScaledACF[:, 2:]) / 3
                    temp[:, 0] = meanScaledACF[:, 0]
                    temp[:, -1] = meanScaledACF[:, -1]
                    meanScaledACF = temp
            
            # Application of ACF lag window Section 6.2.4 ECMA-418-2:2024
            tauz_start = max(0.5/dfz[zBand], 2e-3)  # Equation 31 ECMA-418-2:2024 [tau_start(z)]
            tauz_end = max(4/dfz[zBand], tauz_start + 1e-3)  # Equation 32 ECMA-418-2:2024 [tau_end(z)]
            # Equations 33 & 34 ECMA-418-2:2024
            mz_start = int(np.ceil(tauz_start * sampleRate48k))  # Starting lag window index [m_start(z)]
            mz_end = int(np.floor(tauz_end * sampleRate48k))  # Ending lag window index [m_end(z)]
            M = mz_end - mz_start + 1
            
            # Equation 35 ECMA-418-2:2024
            # lag-windowed, detrended ACF [phi'_z,tau(m)]
            lagWindowACF = np.zeros_like(meanScaledACF)
            if mz_end < meanScaledACF.shape[0]:
                lagWindowACF[mz_start:mz_end+1, :] = (meanScaledACF[mz_start:mz_end+1, :] - 
                                                     np.mean(meanScaledACF[mz_start:mz_end+1, :], axis=0))
            
            # Estimation of tonal loudness
            # ----------------------------
            # Section 6.2.5 Equation 36 ECMA-418-2:2024
            # ACF spectrum in the lag window [Phi'_z,tau(k)]
            magFFTlagWindowACF = np.abs(np.fft.fft(lagWindowACF, 2*np.max(blockSize), axis=0))
            magFFTlagWindowACF[np.isnan(magFFTlagWindowACF)] = 0
            
            # Section 6.2.5 Equation 37 ECMA-418-2:2024 [Nhat'_tonal(z)]
            # first estimation of specific loudness of tonal component in critical band
            bandTonalLoudness = meanScaledACF[0, :].copy()
            max_fft_vals = 2 * np.max(magFFTlagWindowACF, axis=0) / (M/2)
            mask = max_fft_vals <= meanScaledACF[0, :]
            bandTonalLoudness[~mask] = max_fft_vals[~mask]
            
            # Section 6.2.5 Equation 38 & 39 ECMA-418-2:2024
            # [k_max(z)]
            kz_max = np.argmax(magFFTlagWindowACF, axis=0)
            # frequency of maximum tonal component in critical band [f_ton(z)]
            bandTonalFreqs = kz_max * (sampleRate48k / (2 * np.max(blockSize)))
            
            # Section 6.2.7 Equation 41 ECMA-418-2:2024 [N'_signal(l,z)]
            # specific loudness of complete band-pass signal in critical band
            bandLoudness = meanScaledACF[0, :].copy()
            
            # Resampling to common time basis Section 6.2.6 ECMA-418-2:2024
            if i_interp[zBand] > 1:
                # Note: use of interpolation function avoids rippling caused by
                # resample function, which otherwise affects specific loudness 
                # calculation for tonal and noise components
                l_n = meanScaledACF.shape[1]
                x = np.linspace(0, l_n-1, l_n)
                xq = np.linspace(0, l_n-1, i_interp[zBand] * (l_n - 1) + 1)
                
                f_tonal = interp1d(x, bandTonalLoudness, kind='linear', bounds_error=False, fill_value='extrapolate')
                bandTonalLoudness = f_tonal(xq)
                
                f_loud = interp1d(x, bandLoudness, kind='linear', bounds_error=False, fill_value='extrapolate')
                bandLoudness = f_loud(xq)
                
                f_freq = interp1d(x, bandTonalFreqs, kind='linear', bounds_error=False, fill_value='extrapolate')
                bandTonalFreqs = f_freq(xq)
            
            # Remove end zero-padded samples Section 6.2.6 ECMA-418-2:2024
            l_end = int(np.ceil(p_re.shape[0] / sampleRate48k * sampleRate1875)) + 1  # Equation 40 ECMA-418-2:2024
            
            bandTonalLoudness = bandTonalLoudness[:l_end]
            bandLoudness = bandLoudness[:l_end]
            bandTonalFreqs = bandTonalFreqs[:l_end]
            
            # Noise reduction Section 6.2.7 ECMA-418-2:2020
            # ---------------------------------------------
            # Equation 42 ECMA-418-2:2024 signal-noise-ratio first approximation
            # (ratio of tonal component loudness to non-tonal component loudness in critical band)
            # [SNRhat(l,z)]
            SNRlz1 = bandTonalLoudness / ((bandLoudness - bandTonalLoudness) + 1e-12)
            
            # Equation 43 ECMA-418-2:2024 low pass filtered specific loudness
            # of non-tonal component in critical band [Ntilde'_tonal(l,z)]
            bandTonalLoudness = shmNoiseRedLowPass(bandTonalLoudness, sampleRate1875)
            
            # Equation 44 ECMA-418-2:2024 lowpass filtered SNR (improved estimation)
            # [SNRtilde(l,z)]
            SNRlz = shmNoiseRedLowPass(SNRlz1, sampleRate1875)
            
            # Equation 46 ECMA-418-2:2024 [g(z)]
            gz = csz_b[zBand] / (bandCentreFreqs[zBand]**dsz_b[zBand])
            
            # Equation 45 ECMA-418-2:2024 [nr(l,z)]
            crit = np.exp(-alpha * ((SNRlz/gz) - beta))
            nrlz = 1 - crit  # sigmoidal weighting function
            nrlz[crit >= 1] = 0
            
            # Equation 47 ECMA-418-2:2024 [N'_tonal(l,z)]
            bandTonalLoudness = nrlz * bandTonalLoudness
            
            # Section 6.2.8 Equation 48 ECMA-418-2:2024 [N'_noise(l,z)]
            bandNoiseLoudness = shmNoiseRedLowPass(bandLoudness, sampleRate1875) - bandTonalLoudness  # specific loudness of non-tonal component in critical band
            
            # Store critical band results
            # ---------------------------
            # specific time-dependent loudness of tonal component in each critical band  [N'_tonal(l,z)]
            specTonalLoudness[:len(bandTonalLoudness), zBand, chan] = bandTonalLoudness
            
            # specific time-dependent loudness of non-tonal component in each critical band [N'_noise(l,z)]
            specNoiseLoudness[:len(bandNoiseLoudness), zBand, chan] = bandNoiseLoudness
            
            # time-dependent frequency of tonal component in each critical band [f_ton(z)]
            specTonalityFreqs[:len(bandTonalFreqs), zBand, chan] = bandTonalFreqs
        
        # Calculation of specific tonality
        # --------------------------------
        # Section 6.2.8 Equation 49 ECMA-418-2:2024 [SNR(l)]
        overallSNR = np.max(specTonalLoudness[:, :, chan], axis=1) / (1e-12 + np.sum(specNoiseLoudness[:, :, chan], axis=1))  # loudness signal-noise-ratio
        
        # Section 6.2.8 Equation 50 ECMA-418-2:2024 [q(l)]
        crit = np.exp(-A * (overallSNR - B))
        ql = 1 - crit  # sigmoidal scaling factor
        ql[crit >= 1] = 0
        
        # Section 6.2.8 Equation 51 ECMA-418-2:2024 [T'(l,z)]
        specTonality[:, :, chan] = cal_T * cal_Tx * ql[:, np.newaxis] * specTonalLoudness[:, :, chan]  # time-dependent specific tonality
        
        # Section 6.2.8 Equation 52 ECMA-418-2:2024
        # time (s) corresponding with results output [t]
        timeOut = np.arange(specTonality.shape[0]) / sampleRate1875
        
        time_skip_idx = np.argmin(np.abs(timeOut - time_skip))  # find idx of time_skip on timeOut
        idx_insig = np.argmin(np.abs(timeInsig - time_skip))  # find idx of time_skip on timeInsig
        
        # Calculation of time-averaged specific tonality Section 6.2.9
        # ECMA-418-2:2024 [T'(z)]
        for zBand in range(52, -1, -1):
            mask = specTonality[:, zBand, chan] > 0.02  # criterion Section 6.2.9 point 2
            mask[:time_skip_idx] = False  # criterion Section 6.2.9 point 1 - time index takes <time_skip> into consideration
            
            # Section 6.2.9 Equation 53 ECMA-418-2:2024  
            if np.sum(mask) > 0:
                specTonalityAvg[0, zBand, chan] = np.sum(specTonality[mask, zBand, chan]) / (np.sum(mask) + 1e-12)
                specTonalityAvgFreqs[0, zBand, chan] = np.sum(specTonalityFreqs[mask, zBand, chan]) / (np.sum(mask) + 1e-12)
            else:
                specTonalityAvg[0, zBand, chan] = 0
                specTonalityAvgFreqs[0, zBand, chan] = 0
        
        # Calculation of total (non-specific) tonality Section 6.2.10
        # -----------------------------------------------------------
        # Further update can add the user input frequency range to determine
        # total tonality - not yet incorporated
        
        # Section 6.2.10 Equation 61 ECMA-418-2:2024
        # Time-dependent total tonality [T(l)]
        tonalityTDep[:, chan] = np.max(specTonality[:, :, chan], axis=1)
        zmax = np.argmax(specTonality[:, :, chan], axis=1)
        
        for ll in range(specTonalityFreqs.shape[0]-1, -1, -1):
            tonalityTDepFreqs[ll, chan] = specTonalityFreqs[ll, zmax[ll], chan]
        
        # Calculation of representative values Section 6.2.11 ECMA-418-2:2024
        # Time-averaged total tonality
        mask = tonalityTDep[:, chan] > 0.02  # criterion Section 6.2.9 point 2
        mask[:time_skip_idx] = False    # criterion Section 6.2.9 point 1 - time index takes <time_skip> into consideration
        
        # Section 6.2.11 Equation 63 ECMA-418-2:2024
        # Time-averaged total tonality [T]
        tonalityAvg[chan] = np.sum(tonalityTDep[mask, chan]) / (np.sum(mask) + np.finfo(float).eps)
        
        # Output plotting
        if show:
            # colormap
            try:
                cmap_plasma = np.loadtxt('cmap_plasma.txt')
                cmap_plasma = ListedColormap(cmap_plasma)
            except:
                cmap_plasma = plt.cm.plasma
            
            # generate A-weighting filter for LAeq calculation
            b, a = Gen_weighting_filters(fs, 'A')
            insig_A = filtfilt(b, a, insig, axis=0)  # filter signal
            LAeq_all = 20 * np.log10(np.sqrt(np.mean(insig_A[idx_insig:, :]**2, axis=0)) / 2e-5)  # calculate LAeq
            
            # Plot results
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            fig.canvas.manager.set_window_title(f'Tonality analysis - ECMA-418-2 ({chans[chan]} signal)')
            
            # Surface plot
            X, Y = np.meshgrid(timeOut, bandCentreFreqs)
            Z = specTonality[:, :, chan].T
            im = ax1.pcolormesh(X, Y, Z, shading='auto', cmap=cmap_plasma)
            ax1.set_xlim([timeOut[0], timeOut[-1] + (timeOut[1] - timeOut[0])])
            ax1.set_ylim([bandCentreFreqs[0], bandCentreFreqs[-1]])
            ax1.set_clim([0, np.ceil(np.max(tonalityTDep[:, chan]) * 10) / 10])
            ax1.set_yticks([63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3])
            ax1.set_yticklabels(["63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"])
            ax1.set_yscale('log')
            ax1.set_ylabel("Frequency (Hz)")
            ax1.set_xlabel("Time (s)")
            ax1.set_title(f'{chans[chan]} signal, $L_{{\\mathrm{{Aeq}}}} =$ {LAeq_all[chan]:.3g} (dB SPL)')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax1)
            cbar.set_label('Specific Tonality,\n(tu$_{HMS}$/Bark$_{HMS}$)')
            
            # Time series plot
            ax2.plot(timeOut, tonalityTDep[:, chan], color=cmap_plasma(166/255), 
                    linewidth=0.75, label="Time-\ndependent")
            ax2.plot(timeOut, tonalityAvg[chan] * np.ones_like(timeOut), '--', 
                    color=cmap_plasma(34/255), linewidth=1, label="Time-\naverage")
            
            ax2.set_xlim([timeOut[0], timeOut[-1] + (timeOut[1] - timeOut[0])])
            ax2.set_ylim([0, 1.1 * np.ceil(np.max(tonalityTDep[:, chan]) * 10) / 10])
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Tonality (tu$_{HMS}$)")
            ax2.grid(True, alpha=0.075, linestyle='--', linewidth=0.25)
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            
            plt.tight_layout()
            plt.show()
    
    # Output assignment
    # Discard singleton dimensions
    if inchans > 1:
        specTonalityAvg = np.squeeze(specTonalityAvg)
        specTonalityAvgFreqs = np.squeeze(specTonalityAvgFreqs)
    else:
        specTonalityAvg = specTonalityAvg.T
        specTonalityAvgFreqs = specTonalityAvgFreqs.T
    
    OUT = {}
    OUT['specTonality'] = specTonality
    OUT['specTonalityAvg'] = specTonalityAvg
    OUT['specTonalityFreqs'] = specTonalityFreqs
    OUT['specTonalityAvgFreqs'] = specTonalityAvgFreqs
    
    OUT['specTonalLoudness'] = specTonalLoudness
    OUT['specNoiseLoudness'] = specNoiseLoudness
    
    OUT['tonalityTDep'] = tonalityTDep
    OUT['tonalityAvg'] = tonalityAvg
    OUT['tonalityTDepFreqs'] = tonalityTDepFreqs
    OUT['bandCentreFreqs'] = bandCentreFreqs
    
    OUT['timeOut'] = timeOut
    OUT['timeInsig'] = timeInsig
    OUT['soundField'] = fieldtype
    
    ##########################################################################
    # Tonality statistics based on tonalityTDep ["Stereo left"; "Stereo right"]; for stereo case
    ##########################################################################
    
    metric_statistics = 'Tonality_ECMA418_2'
    OUT_statistics = get_statistics(tonalityTDep[time_skip_idx:, :tonalityTDep.shape[1]], metric_statistics)  # get statistics
    
    # copy fields of <OUT_statistics> dict into the <OUT> dict
    fields_OUT_statistics = list(OUT_statistics.keys())  # Get all field names in OUT_statistics
    
    for i in range(len(fields_OUT_statistics)):
        fieldName = fields_OUT_statistics[i]
        if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
            OUT[fieldName] = OUT_statistics[fieldName]
    
    return OUT

if __name__ == "__main__":


    fc = 1000  # 1 kHz
    Lp = 60    # 60 dB SPL
    fs = 48000 # Sample rate in Hz (consistent with ECMA-418-2 internal rate)
    duration = 5 # seconds
    p_ref = 20e-6

    p_rms = p_ref * (10**(Lp / 20))
    amplitude = np.sqrt(2) * p_rms

    t = np.arange(0, duration, 1/fs)
    signal = amplitude * np.sin(2 * np.pi * fc * t)

    pure_tone_signal = signal.reshape(-1, 1)

    print(f"Generated pure tone signal with shape: {pure_tone_signal.shape}")
    print(f"Signal RMS (Pa): {np.sqrt(np.mean(pure_tone_signal**2)):.2e}")
    print(f"Signal dB SPL: {20 * np.log10(np.sqrt(np.mean(pure_tone_signal**2)) / 20e-6):.2f} dB SPL")


    results = Tonality_ECMA418_2(pure_tone_signal, fs, show=True)

    print(results.keys())