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

    if num_args == 0 or insig is None or fs is None:
        print(EPNL_FAR_Part36.__doc__)
        return

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
            ax2 = plt.subplot(2, 6, (3, 4, 5, 6))
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

check_which = 2

if __name__ == "__main__":
    if check_which == 0: # NO TEST

        print("metrics_loudness.py")

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
