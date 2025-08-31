from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import resample_poly, lfilter, sosfilt, iirfilter
from scipy.interpolate import interp1d 
from scipy.fft import fft, ifft
from scipy.signal.windows import hann, blackman
from matplotlib import pyplot as plt
import warnings
import sys

from sound_metrics import *
from utilities import *

__all__ = ["FluctuationStrength_Osses2016"]
FloatArray = NDArray[np.floating]

def FluctuationStrength_Osses2016(insig, fs, method, time_skip=None, show=None, struct_opt=None, dBFS=94, export_excel=None):
    """
    Calculate psycho-acoustic **fluctuation strength** following the model
    of Osses *et al.* (2016).
    
    The routine supports two analysis modes:

    =============  ==============================================================
    **Method**     **Description**
    -------------  --------------------------------------------------------------
    ``0``          *Stationary mode* – the **entire waveform** is processed as a single window (length = signal duration).
    ``1``          *Time-varying mode* – successive **2-second windows** with 90 % overlap are analysed, producing an instantaneous trace *FS(t)*.
    =============  ==============================================================

    Parameters
    ----------
    insig : str | numpy.ndarray
        Input waveform.  A string is interpreted as a WAV filename and
        loaded via :pyfunc:`wav2sig`; otherwise a 1-D NumPy array of
        samples is expected.
    fs : int | float
        Sampling frequency of *insig* (Hz).  Ignored when *insig* is a
        filename; the file’s native rate is used instead.
    method : {0, 1}
        ``0`` – stationary analysis (single window covering the whole
        signal).  
        ``1`` – time-varying analysis with 90 %-overlapped 2-s windows.
    time_skip : float, optional
        Seconds to omit from the start when statistical descriptors are
        computed.  Default is 0.
    show : bool, optional
        If ``True`` and *method* == 1, plot the instantaneous and specific
        fluctuation-strength traces.  Default ``False``.
    struct_opt : dict, optional
        Additional model options.  Only the key ``'a0_type'`` is currently
        recognised (default ``'fluctuationstrength_osses2016'``).
    dBFS : float, default 94
        Full-scale calibration: a full-scale sine is assumed to equal
        *dBFS* dB SPL.
    export_excel : str, optional
        Path to an ``.xlsx`` file in which all entries of the returned
        dictionary are written sheet-by-sheet.

    Returns
    -------
    dict
        Dictionary containing instantaneous data and summary statistics.

    Raises
    ------
    ValueError
        If *fs* is omitted when *insig* is given as a NumPy array.

    Notes
    -----
    * The signal is resampled to 44.1 kHz (or 48 kHz if already at that
      rate) because the model parameters are tuned for those rates.
    * Units – overall fluctuation strength in **vacil**; specific
      fluctuation strength in **vacil · Bark⁻¹**.
    """

    # --- WAV file interface ---
    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)

    elif fs is None:
        raise ValueError("If insig is not a filename, fs must be provided.")
    
    # Handle default arguments (equivalent to nargin checks)
    if insig is None:
        help(FluctuationStrength_Osses2016)
        return
    
    if show is None:
        # Default for show - equivalent to nargout check
        show = 0  # Assuming we typically don't want plots by default
    
    if struct_opt is None:
        struct_opt = {}
    
    if insig.ndim > 1 and insig.shape[1] != 1:  # if the insig is not a [Nx1] array
        insig = insig.T  # correct the dimension of the insig
    
    if insig.ndim > 1:
        insig = insig.flatten()
    
    ## Resampling audio to 44.1 kHz or 48 kHz
    if not (fs == 44100 or fs == 48000):
        gcd_fs = np.gcd(44100, int(fs))  # greatest common denominator
        insig = resample_poly(insig, 44100//gcd_fs, int(fs)//gcd_fs)
        fs = 44100
    
    if 'a0_type' not in struct_opt:
        struct_opt['a0_type'] = 'fluctuationstrength_osses2016'  # this is the default of this model
    
    ## Checking which method
    if method == 1:  # 'time_varying'
        
        # This is the default from the original authors.
        time_resolution = 2  # window length fixed in 2s (Osses et al., 2016)
        N = round(fs * time_resolution)
        
        if N >= len(insig):  # if the signal's length is smaller than the window size, force method==0
            warnings.warn('The signal is shorter than 2 seconds. The analysis will be automatically changed to \'stationary\', i.e. method=0 and window size=length(insig). This analysis window may lead to inaccurate fluctuation-strength estimates, especially if the modulation components are low (below 10 Hz).')
            method = 0
    
    if method == 0:  # 'stationary'
        N = len(insig)  # window size (N)=length(signal) kind of an rms value
    
    ###########################################################################
    
    model_par = il_Get_fluctuation_strength_params(N, fs)
    model_par['debug'] = 'none'
    
    # model_par = ef(model_par,'window_type','cosine');
    
    t_b = np.arange(1, len(insig) + 1) / fs
    
    overlap = round(0.9 * N)
    insig = buffer(insig, N, overlap, 'nodelay')
    t_b = buffer(t_b, N, overlap, 'nodelay')
    nFrames = insig.shape[1]
    fluct = np.zeros(nFrames)  # Memory allocation
    
    ## ei = peripheral_stage(insig,fs,N);
    # 1. Cosine window:
    
    window = np.ones(N)
    attackrelease = 50
    
    window = il_Do_cos_ramp(window, fs, attackrelease, attackrelease)
    
    # Initialize output arrays
    kp_fr = np.zeros((nFrames, model_par['Chno']))
    gzi_fr = np.zeros((nFrames, model_par['Chno']))
    md_fr = np.zeros((nFrames, model_par['Chno']))
    fi = np.zeros((nFrames, model_par['Chno']))
    t = np.zeros(nFrames)
    
    for iFrame in range(nFrames):
        
        signal = insig[:, iFrame]
        t[iFrame] = t_b[0, iFrame]
        
        # Apply window to frame
        signal = (window * signal).T
        
        ## 2. Peripheral stages
        # 2.1 Peripheral hearing system (transmission factor a0)
        #     (see 'model_par.a0_in_time' == 1, in _debug version):
        #
        # 4096th order FIR filter:
        signal = il_PeripheralHearingSystem_t(signal, fs, struct_opt)
        
        # 2.2 Excitation patterns
        #     (see model_par.filterbank == 'terhardt', in _debug version):
        
        dBFS = 94  # corresponds to 1 Pa (new default in SQAT)
        # dBFS = 100; # unit amplitude corresponds to 100 dB (AMT Toolbox 
                      # convention, default by the original authors)
        ei = TerhardtExcitationPatterns_v3(signal, fs, dBFS)
        dz = 0.5  # Barks, frequency step
        z = np.arange(0.5, 24, dz)  # Bark
        fc = bark2hz(z)
        flow = bark2hz(z - 0.5)
        flow[0] = 0.01
        fup = bark2hz(z + 0.5)
        BWHz = fup - flow
        
        ## 3. Modulation depth (estimation)
        mdept, hBPi = il_modulation_depths(ei, model_par['Hweight'])
        
        ## 4. Cross-correlation coefficient:
        #     (see model_par.dataset == 0, in _debug version)
        
        # # here cross-correlation is computed before band-pass filtering:
        # Ki = il_cross_correlation(inoutsig); # with hBPi Ki goes down but not as much as 'it should'
        Ki = il_cross_correlation(hBPi)
        fi_, mdept, kp, gzi = il_specific_fluctuation(mdept, Ki, model_par)
        
        kp_fr[iFrame, :] = kp
        gzi_fr[iFrame, :] = gzi
        md_fr[iFrame, :] = mdept
        fi[iFrame, :] = model_par['cal'] * fi_ #not completely accurate (TODO: check)
        fluct[iFrame] = dz * np.sum(fi[iFrame, :])  # total fluct = integration of the specific fluct. strength pattern
    
    ## ************************************************************************
    # output struct
    # *************************************************************************
    
    # main output results
    OUT = {}
    OUT['InstantaneousFluctuationStrength'] = fluct  # instantaneous fluctuation strength
    OUT['InstantaneousSpecificFluctuationStrength'] = fi  # time-varying specific fluctuation strength
    OUT['TimeAveragedSpecificFluctuationStrength'] = np.mean(fi, axis=0)  # mean specific fluctuation strength
    OUT['time'] = t  # time
    OUT['barkAxis'] = z.T  # critical band rate (for specific fluctuation strength)
    OUT['dz'] = dz
    
    # get statistics from time-varying fluctuation Strength
    #############################
    
    if time_skip is not None:
        idx = np.argmin(np.abs(OUT['time'] - time_skip))  # find idx of time_skip on time vector
    else:
        idx = 0
    
    metric_statistics = 'FluctuationStrength_Osses2016'
    OUT_statistics = get_statistics(fluct[idx:], metric_statistics)  # get statistics
    
    # Get all field names in OUT_statistics
    for fieldName in OUT_statistics.keys():
        if fieldName not in OUT:  # Only copy if OUT does NOT already have this field
            OUT[fieldName] = OUT_statistics[fieldName]
    
    #############################
    
    if show == True and method == 1:
        
        plt.figure(figsize=(15, 10))
        plt.suptitle('Fluctuation strength analysis')
        
        # Time-varying Fluctuation Strength
        plt.subplot(2, 2, (1, 2))
        
        plt.plot(t, fluct, 'r-')
        
        plt.title('Instantaneous fluctuation strength')
        plt.xlabel('Time (s)')
        plt.ylabel('Fluctuation strength, $\\mathrm{FS}$ (vacil)')
        
        # Time-averaged Fluctuation strength as a function of critical band
        plt.subplot(2, 2, 3)
        
        plt.plot(z.T, np.mean(fi, axis=0), 'r-')
        
        plt.title('Time-averaged specific fluctuation strength')
        plt.xlabel('Critical band, $z$ (Bark)')
        plt.ylabel('Specific fluctuation strength, $\\mathrm{FS}^{\\prime}$ (vacil/Bark)')
        
        # Specific fluctuation strength spectrogram
        plt.subplot(2, 2, 4)
        
        xx, yy = np.meshgrid(t, z)
        plt.pcolormesh(xx, yy, fi.T, shading='gouraud')
        plt.colorbar()
        
        plt.title('Instantaneous specific fluctuation strength')
        plt.xlabel('Time (s)')
        plt.ylabel('Critical band, $z$ (Bark)')
        cbar = plt.colorbar()
        cbar.set_label('Specific fluctuation strength, $\\mathrm{FS}^{\\prime}$ ($\\mathrm{vacil}/\\mathrm{Bark}$)')
        
        plt.tight_layout()
        plt.show()
    
    print('')

    if export_excel is not None:
        export_dict_to_excel(OUT, filename=f"{export_excel}")
    
    return OUT

# ------------------------------
#### LOCAL HELPER FUNCTIONS ####
# ------------------------------

def il_modulation_depths(ei, Hweight):
    
    Chno, Nc = ei.shape
    mdept = np.zeros(Chno)
    
    ei = np.abs(ei).T
    h0 = np.mean(ei, axis=0)
    ei = ei - np.tile(h0, (Nc, 1))
    
    if not isinstance(Hweight, np.ndarray):
        # older versions of MATLAB equivalent
        hBPi = lfilter(Hweight, 1, ei, axis=0)  # getting the envelopes
    else:
        hBPi = sosfilt(Hweight, ei, axis=0)
    
    try:
        # In case LTFAT toolbox is installed (overloads rms from signal processing toolbox)
        hBPrms = np.sqrt(np.mean(hBPi**2, axis=0))
    except:
        # uses the default rms calculation from the signal processing toolbox
        hBPrms = np.sqrt(np.mean(hBPi**2, axis=0))
    
    hBPi = hBPi.T
    
    idx = np.where(h0 > 0)[0]
    mdept[idx] = hBPrms[idx] / h0[idx]
    
    idx = np.where(h0 == 0)[0]
    mdept[idx] = 0
    
    idx = np.where(h0 < 0)[0]
    if len(idx) != 0:
        raise ValueError('There is an error in the algorithm')
    
    return mdept, hBPi

def il_cross_correlation(hBPi):
    Chno, _ = hBPi.shape
    
    ki = np.zeros((2, Chno))
    for k in range(Chno - 2):
        try:
            cfac = np.cov(hBPi[k, :], hBPi[k + 2, :])
        except:
            raise ValueError('You do not have the function cov (stats toolbox). Contact me at ale.a.osses@gmail.com to solve this problem')
        
        den = np.diag(cfac)
        den = np.sqrt(den[0] * den[1])
        
        if den > 0:  # Pearson correlation
            ki[0, k] = cfac[0, 1] / den
        elif den == 0:
            ki[0, k] = 0
        else:
            warnings.warn('Cross correlation factor less than 1')
            ki[0, k] = 0
    
    try:
        # Interpolation for last elements
        f_interp = interp1d([0, 0.5], ki[0, Chno-4:Chno-2], kind='linear', fill_value='extrapolate')
        ki[0, Chno-2] = f_interp(1)
        ki[0, Chno-1] = f_interp(1)
        
        f_interp2 = interp1d([0.5, 1], ki[0, 2:4], kind='linear', fill_value='extrapolate')
        ki[1, 1] = f_interp2(0)
        ki[1, 0] = f_interp2(0)
    except:
        ki[0, Chno-2] = ki[0, Chno-3]
        ki[0, Chno-1] = ki[0, Chno-3]
        ki[1, 0] = ki[0, 2]
        ki[1, 1] = ki[0, 2]
    
    ki[1, 2:Chno] = ki[0, 0:Chno-2]
    
    return ki

def il_specific_fluctuation(mdept, Ki, model_par, dataset=0):
    
    gzi = model_par['gzi']
    p_g = model_par['p_g']
    p_m = model_par['p_m']
    p_k = model_par['p_k']
    
    Chno = len(gzi)
    
    fi = np.zeros(Chno)
    
    if dataset in [0, 90, 99]:
        
        # Version 3: # Improves approximation for FM tones
        thres = 0.7
        idx = np.where(mdept > thres)[0]
        exceed = mdept[idx] - thres
        mdept[idx] = thres + (1 - thres) * exceed
        md = np.minimum(mdept, np.ones(mdept.shape))
        
    elif dataset == 1:
        md = np.minimum(mdept, np.ones(mdept.shape))
        md = mdept - 0.1 * np.ones(mdept.shape)
        md = np.maximum(mdept, np.zeros(mdept.shape))
    
    kp = Ki[0, :] * Ki[1, :]
    kpsign = np.sign(kp)
    kp = np.abs(kp)
    
    if dataset in [0, 90, 99]:
        fi = (gzi ** p_g) * (md ** p_m) * (kp ** p_k) * kpsign
    elif dataset == 1:
        fi = (gzi ** p_g) * (md ** p_m) * (kp ** p_k)
    else:
        raise ValueError('Dataset does not include the calculation of fi')
    
    mdept = md
    
    return fi, mdept, kp, gzi

def il_Get_fluctuation_strength_params(N, fs):
    # function params = il_Get_fluctuation_strength_params(N,fs)
    
    params = {}
    params['fs'] = fs
    params['N'] = N
    params['Chno'] = 47
    params['debug'] = 'none'
    
    # dataset = 0; # 0 = Approved version
    params['window_type'] = 'cosine'
    params['filterbank'] = 'terhardt'
    params['p_g'] = 1
    params['p_m'] = 1.7
    params['p_k'] = 1.7  # warning('Temporal value')
    params['a0_in_time'] = 1
    params['a0_in_freq'] = not params['a0_in_time']
    
    params['cal'] = 0.4980  # this value is twice 0.2490 on 15/06/2016
    params['bIdle'] = 1  # v5
    
    params['Hweight'] = Get_Hweight_fluctuation(fs)
    params['gzi'] = il_Get_gzi_fluctuation(params['Chno'])
    
    return params

def il_Do_cos_ramp(insig, fs, attack_ms, release_ms):
    # Applies a cosine ramp with attack and release times given in [ms]
    
    sig_len = len(insig)
    r = cos_ramp(sig_len, fs, attack_ms, release_ms)
    try:
        outsig = r.T * insig
    except:
        outsig = r * insig
    
    return outsig

def il_PeripheralHearingSystem_t(insig, fs, struct_opt):
    # Applies the effect of transmission from free field to the cochlea to a
    
    K = 2**12  # FIR filter order
    
    if struct_opt['a0_type'] == 'fluctuationstrength_osses2016':
        B, _, _ = calculate_a0(fs, K, 'fluctuationstrength_osses2016')
    elif struct_opt['a0_type'] == 'fastl2007':
        B, _, _ = calculate_a0(fs, K, 'fastl2007')
    else:
        # Choosing the default:
        B, _, _ = calculate_a0(fs, K, 'fluctuationstrength_osses2016')
    
    outsig = lfilter(B, 1, np.concatenate([insig, np.zeros(K//2)]))
    outsig = outsig[K//2:]
    
    return outsig

def il_Get_gzi_fluctuation(Chno):
    # Returns gzi parameters using the specified number of channels.
    
    Chstep = 0.5
    
    # Hz:   100 250   519   717 926 1084 1255 1465 1571   1972 2730 4189   15550
    g0 = np.array([[0,  1,  2.5,  4.9, 6.5,  8,   9,  10,  11,  11.5,  13,  15,  17.5,   24],
                   [1,  1,  1  ,  1  , 1  ,  1,   1,   1,   1,   1  ,   1, 0.9,   0.7, 0.5]]).T
    
    x_vals = np.arange(1, Chno + 1) * Chstep
    f_interp = interp1d(g0[:, 0], g0[:, 1], kind='linear', fill_value=g0[-1, 1], bounds_error=False)
    gzi = f_interp(x_vals)
    gzi[np.isnan(gzi)] = g0[-1, 1]  # 0
    
    return gzi

def TerhardtExcitationPatterns_v3(insig, fs, dBFS=100):

    corr = dBFS + 3
    
    dB2calibrate = rmsdb(insig) + dBFS
    
    # General parameters
    params = il_calculate_params(insig, fs)
    N01 = params['N01']
    freqs = params['freqs']
    
    dfreq = fs / params['N']
    freq = dfreq * np.arange(1, params['N'] + 1)  # freqs and freq are the same array, but freqs starts at bin N0
    
    # Transforms input signal to frequency domain
    corr_factor = 10**(corr/20)  # il_From_dB(corr)
    insig_fft = corr_factor * np.fft.fft(insig) / params['N']  # 3 dB added to adjust the SPL values to be put into slope equations
    
    # Use only samples that fall into the audible range
    Lg = np.abs(insig_fft[params['qb']])
    LdB = 20 * np.log10(Lg)
    
    # Use only components that are above the hearing threshold
    whichL = np.where(LdB > params['MinExcdB'])[0]
    nL = len(whichL)
    
    # Steepness of slopes
    S1 = -27
    S2 = np.zeros(nL)
    for w in range(nL):
        idx = whichL[w]
        steep = -24 - (230 / freqs[idx]) + (0.2 * LdB[idx])
        if steep < 0:
            S2[w] = steep
    
    whichZ = np.zeros((2, nL), dtype=int)
    whichZ[0, :] = np.floor(2 * params['Barkno'][whichL + N01]).astype(int)
    whichZ[1, :] = np.ceil(2 * params['Barkno'][whichL + N01]).astype(int)
    
    # Calculate slopes from steep values
    Slopes = np.zeros((nL, params['Chno']))
    Slopes_dB = np.full((nL, params['Chno']), np.nan)
    
    for l in range(nL):
        Li = LdB[whichL[l]]
        zi = params['Barkno'][whichL[l] + N01]
        
        for k in range(1, whichZ[0,l]+1):
            k_idx = k-1
            zk = k * 0.5
            delta_z = zi - zk
            Stemp = (S1 * delta_z) + Li
            if Stemp > params['MinBf'][k_idx]:
                Slopes[l, k_idx]    = 10**(Stemp/20)
                Slopes_dB[l,k_idx]  = Stemp
        
        for k in range(whichZ[1, l], params['Chno']+1):
            k_idx = k-1
            zk = k * 0.5
            delta_z = zi - zk
            delta_z = zk - zi
            Stemp = S2[l] * delta_z + Li
            if Stemp > params['MinBf'][k]:
                Slopes[l, k_idx] = 10**(Stemp/20)
                Slopes_dB[l, k_idx] = Stemp
    
    # Excitation patterns:
    #   Each frequency having a level above the absolute threshold is looked at.
    #   The contribution of that level (and frequency) onto the other critical
    #   band levels is computed and then assigned.
    ei = np.zeros((params['Chno'], params['N']))
    ei_f = np.zeros((params['Chno'], params['N']))
    
    togetshape = []
    for l in range(nL):
        togetshape.append(whichL[l])
    ExcAmp = np.zeros((np.max(togetshape)+1, params['Chno']))

    for i in range(params['Chno']):
        i_band = i + 1                                   # MATLAB band number
        etmp = np.zeros(params['N'], dtype=complex)

        for l in range(nL):
            N1tmp = whichL[l]
            N2tmp = N1tmp + N01

            if whichZ[0,l] == i_band or whichZ[1,l] == i_band:
                ExcAmp[N1tmp, i] = 1
            elif whichZ[1,l] > i_band:
                ExcAmp[N1tmp, i] = Slopes[l, i+1] / Lg[N1tmp]
            else:
                ExcAmp[N1tmp, i] = Slopes[l, i-1] / Lg[N1tmp]

            etmp[N2tmp] = ExcAmp[N1tmp, i] * insig_fft[N2tmp]

        mag = np.maximum(np.abs(etmp), 1e-12)
        ei_f[i,:] = 20*np.log10(mag)
        ei[i,:]  = 2*params['N']*np.real(np.fft.ifft(etmp))
    
    outsig = np.sum(ei, axis=0) # not exactly same (TODO: check)
    gain = dB2calibrate - (rmsdb(outsig) + dBFS)
    gain_factor = 10**(gain/20)
    ei = gain_factor * ei
    
    return ei

def il_calculate_params(x, fs):
    
    params = {}
    params['N'] = len(x)
    params['Chno'] = 47
    
    # Defines audible range indexes and frequencies
    df = fs / params['N']
    N0 = round(20 / df) + 1  # start at 20 Hz
    Ntop = round(20e3 / df) + 1  # start at 20 kHz
    params['N01'] = N0 - 1
    params['qb'] = np.arange(N0, Ntop + 1) - 1  # Convert to 0-based indexing
    params['freqs'] = params['qb'] * df
    
    params['Barkno'], Bark_raw = get_bark(params['N'], params['qb'], params['freqs'])
    # Loudness threshold related parameters
    params['MinExcdB'] = il_calculate_MinExcdB(params['N01'], params['qb'], params['Barkno'])
    params['MinBf'] = il_calculate_MinBf(params['N01'], df, Bark_raw, params['MinExcdB'])
    
    return params

def il_calculate_MinExcdB(N01, qb, Barkno):
    
    HTres = np.array([
        [0,     130],
        [0.01,   70],
        [0.17,   60],
        [0.8,    30],
        [1,      25],
        [1.5,    20],
        [2,      15],
        [3.3,    10],
        [4,      8.1],
        [5,      6.3],
        [6,      5],
        [8,      3.5],
        [10,     2.5],
        [12,     1.7],
        [13.3,   0],
        [15,    -2.5],
        [16,    -4],
        [17,    -3.7],
        [18,    -1.5],
        [19,     1.4],
        [20,     3.8],
        [21,     5],
        [22,     7.5],
        [23,     15],
        [24,     48],
        [24.5,   60],
        [25,     130]
    ])
    
    MinExcdB = np.zeros(len(qb))
    f_interp = interp1d(HTres[:, 0], HTres[:, 1], kind='linear', fill_value='extrapolate')
    MinExcdB = f_interp(Barkno[qb])
    
    return MinExcdB

def il_calculate_MinBf(N01, df, Bark, MinExcdB):
    
    Cf = np.round(Bark[1:25, 1] / df) - N01
    Bf = np.round(Bark[0:25, 2] / df) - N01
    
    zb = np.sort(np.concatenate([Bf.astype(int), Cf.astype(int)]))
    MinBf = MinExcdB[zb]
    
    return MinBf

def Get_gzi_fluctuation(Chno):
    
    Chstep = 0.5
    
    # Hz:   100 250   519   717 926 1084 1255 1465 1571   1972 2730 4189   15550
    g0 = np.array([[0,  1,  2.5,  4.9, 6.5,  8,   9,  10,  11,  11.5,  13,  15,  17.5,   24],
                   [1,  1,  1  ,  1  , 1  ,  1,   1,   1,   1,   1  ,   1, 0.9,   0.7, 0.5]]).T
    
    x_vals = np.arange(1, Chno + 1) * Chstep
    f_interp = interp1d(g0[:, 0], g0[:, 1], kind='linear', fill_value=g0[-1, 1], bounds_error=False)
    gzi = f_interp(x_vals)
    gzi[np.isnan(gzi)] = g0[-1, 1]  # 0
    
    return gzi

def Get_Hweight_fluctuation(fs):

    if fs == 44100:

        Hweight_HP = [[0.9998626848836257, -1.9997253697672515, 0.9998626848836257, 1.0, -1.9997253509118103, 0.9997253886226927]]
        
        Hweight_LP= [[9.1532542002435e-07, 1.8306508400487e-06, 9.1532542002435e-07, 1.0, -1.9985323820728271, 0.9985360433745073],
                     [9.143788790959651e-07, 1.8287577581919303e-06, 9.143788790959651e-07, 1.0, -1.9964656933794398, 0.9964693508949561]]
        
        Hweight = np.vstack([Hweight_HP, Hweight_LP])
        
        return Hweight

    elif fs == 48000:

        Hweight_HP = [[0.9998738410329782, -1.9997476820659563, 0.9998738410329782, 1.0, -1.9997476661498712, 0.9997476979820413]]

        Hweight_LP = [[7.726735179422239e-07, 1.5453470358844478e-06, 7.726735179422239e-07, 1.0, -1.9986518191092326, 0.9986549098033047],
                            [7.719393091742981e-07, 1.5438786183485963e-06, 7.719393091742981e-07, 1.0, -1.9967526629254815, 0.9967557506827183]]
        
        Hweight = np.vstack([Hweight_HP, Hweight_LP])

        return Hweight
        
    else:
        
        # Design parameters of band-pass filter
        sf1 = 0.5  # 0.5
        pf1 = 3.1  # 2
        pf2 = 12  # Hz # 8
        sf2 = 20
        passAtt1 = 17.5
        passAtt2 = 14
        
        # Design lowpass filter
        sos_lp = iirfilter(6, pf2, btype='low', ftype='butter', fs=fs, output='sos')
        
        # Design highpass filter  
        sos_hp = iirfilter(6, pf1, btype='high', ftype='butter', fs=fs, output='sos')
        
        Hweight_HP = sos_hp  # second-order sections
        Hweight_LP = sos_lp  # second-order sections
        
        # Save filters
        try:
            np.save(f'Hweight-{fs:.0f}-Hz-LP.npy', Hweight_LP)
            np.save(f'Hweight-{fs:.0f}-Hz-HP.npy', Hweight_HP)
        except:
            pass
        
        Hweight = np.vstack([Hweight_HP, Hweight_LP])
    
    return Hweight

def buffer(x, n, p, opt='nodelay'):
    """
    Buffer signal into matrix of data frames
    
    Parameters:
    x: input signal
    n: frame length  
    p: overlap length
    opt: 'nodelay' option
    
    Returns:
    y: buffered signal matrix
    """
    if opt == 'nodelay':
        # Calculate step size
        step = n - p
        
        # Calculate number of frames
        num_frames = int(np.ceil((len(x) - p) / step))
        
        # Initialize output matrix
        y = np.zeros((n, num_frames))
        
        # Fill the buffer
        for i in range(num_frames):
            start_idx = i * step
            end_idx = start_idx + n
            
            if end_idx <= len(x):
                y[:, i] = x[start_idx:end_idx]
            else:
                # Zero-pad if necessary
                remaining = len(x) - start_idx
                y[:remaining, i] = x[start_idx:]
                
    return y

check_which = 1

if __name__ == "__main__":
    if check_which == 0: # NO TEST
        print("metrics_fluctuation.py")
    
    elif check_which == 1: # FluctuationStrength_Osses2016 (original test case)
        with_wavfile = 0

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

        if with_wavfile == 1:
            wavfile.write('test_F1.wav', fs, signal_py)
            OUT_py = FluctuationStrength_Osses2016('test_F1.wav', fs, method=1, time_skip=2.0, show=True)
        
        else:
            os.remove("test_F1.wav") if os.path.exists("test_F1.wav") else None
            OUT_py = FluctuationStrength_Osses2016(signal_py, fs, method=1, time_skip=2.0, show=True)

        print('\nInstantaneous Fluctuation Strength (vacil):')
        print(f'  First 5 values: {OUT_py["InstantaneousFluctuationStrength"][0]:.6f}, {OUT_py["InstantaneousFluctuationStrength"][1]:.6f}, {OUT_py["InstantaneousFluctuationStrength"][2]:.6f}, {OUT_py["InstantaneousFluctuationStrength"][3]:.6f}, {OUT_py["InstantaneousFluctuationStrength"][4]:.6f}')
        print(f'  Mean: {OUT_py["FSmean"][0]:.6f}')
        print(f'  Std:  {OUT_py["FSstd"][0]:.6f}')
        print(f'  Max:  {OUT_py["FSmax"][0]:.6f}')
        print(f'  Min:  {OUT_py["FSmin"][0]:.6f}')

        print('\nTime-Averaged Specific Fluctuation Strength (vacil/Bark):')
        print(f'  First 5 values: {OUT_py["TimeAveragedSpecificFluctuationStrength"][0]:.6f}, {OUT_py["TimeAveragedSpecificFluctuationStrength"][1]:.6f}, {OUT_py["TimeAveragedSpecificFluctuationStrength"][2]:.6f}, {OUT_py["TimeAveragedSpecificFluctuationStrength"][3]:.6f}, {OUT_py["TimeAveragedSpecificFluctuationStrength"][4]:.6f}')
        print(f'  Mean: {np.mean(OUT_py["TimeAveragedSpecificFluctuationStrength"]):.6f}')
        print(f'  Std:  {np.std(OUT_py["TimeAveragedSpecificFluctuationStrength"]):.6f}')
        print(f'  Max:  {np.max(OUT_py["TimeAveragedSpecificFluctuationStrength"]):.6f}')
        print(f'  Min:  {np.min(OUT_py["TimeAveragedSpecificFluctuationStrength"]):.6f}')

        print('\nTime Vector (s):')
        print(f'  First 5 values: {OUT_py["time"][0]:.6f}, {OUT_py["time"][1]:.6f}, {OUT_py["time"][2]:.6f}, {OUT_py["time"][3]:.6f}, {OUT_py["time"][4]:.6f}')
        print(f'  Length: {len(OUT_py["time"])}')

        print('\nBark Axis:')
        print(f'  First 5 values: {OUT_py["barkAxis"][0]:.6f}, {OUT_py["barkAxis"][1]:.6f}, {OUT_py["barkAxis"][2]:.6f}, {OUT_py["barkAxis"][3]:.6f}, {OUT_py["barkAxis"][4]:.6f}')
        print(f'  Length: {len(OUT_py["barkAxis"])}')