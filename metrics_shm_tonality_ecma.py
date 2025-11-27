from __future__ import annotations
from typing import Dict, Any


import numpy as np
from numpy.typing import NDArray

from utilities import wav2sig
from sottek_hearing_model import shm_tonality_ecma

# Add matplotlib imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm


__all__ = ["shm_tonality_ecma_wrapper"]

FloatArray = NDArray[np.floating]

def shm_tonality_ecma_wrapper(insig, fs=None, field=0, method=1,
                              time_skip=0, show=False, dBFS=94,
                              export_excel=None):
    """
    Wrapper for ECMA-418-2 (2022) Tonality using the Sottek Hearing Model.

    Parameters
    ----------
    insig : str or np.ndarray
        Input signal. Can be a file path (str) or a numpy array.
    fs : int, optional
        Sampling frequency. Required if insig is an array.
    field : int, default 0
        Sound field. 0 for free field (free_frontal), 1 for diffuse field.
    method : int, default 1
        Method for statistics.
    time_skip : float, default 0
        Time to skip at the beginning for statistics.
    show : bool, default False
        If True, shows plots.
    dBFS : float, default 94
        Calibration value for WAV files.
    export_excel : str, optional
        Path to export results to Excel.

    Returns
    -------
    dict
        Dictionary containing tonality metrics.
    """

    # --- WAV file interface ---
    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)
    elif fs is None:
        raise ValueError("If insig is not a filename, fs must be provided.")

    # --- Input validation ---
    if fs is None or fs <= 0:
        raise ValueError("Sampling frequency (fs) must be a positive number.")

    # Map field parameter
    sound_field = 'free_frontal' if field == 0 else 'diffuse'

    # Calculate Tonality
    tonality_out = shm_tonality_ecma(insig, fs, soundfield=sound_field, wait_bar=True, out_plot=False)

    if not isinstance(tonality_out, dict):
         raise RuntimeError("Unexpected return type from tonality function; expected dict.")

    # --- Output Mapping ---
    OUT: Dict[str, Any] = {}

    def get_val(d, keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    # Direct mappings 
    OUT['specTonality'] = get_val(tonality_out, ['spec_tonality', 'specTonality'])
    OUT['specTonalityAvg'] = get_val(tonality_out, ['spec_tonality_avg', 'specTonalityAvg'])
    OUT['specTonalityFreqs'] = get_val(tonality_out, ['spec_tonality_freqs', 'specTonalityFreqs'])
    OUT['specTonalityAvgFreqs'] = get_val(tonality_out, ['spec_tonality_avg_freqs', 'specTonalityAvgFreqs'])
    
    OUT['specTonalLoudness'] = get_val(tonality_out, ['spec_tonal_loudness', 'specTonalLoudness'])
    OUT['specNoiseLoudness'] = get_val(tonality_out, ['spec_noise_loudness', 'specNoiseLoudness'])
    
    OUT['tonalityTDep'] = get_val(tonality_out, ['tonality_t', 'tonalityTDep'])
    if OUT['tonalityTDep'] is not None:
        OUT['tonalityTDep'] = np.asarray(OUT['tonalityTDep'])

    OUT['tonalityAvg'] = get_val(tonality_out, ['tonality_avg', 'tonalityAvg'])
    OUT['tonalityTDepFreqs'] = get_val(tonality_out, ['tonality_t_freqs', 'tonalityTDepFreqs'])
    
    OUT['bandCentreFreqs'] = get_val(tonality_out, ['band_centre_freqs', 'bandCentreFreqs'])
    
    OUT['timeOut'] = get_val(tonality_out, ['time_out', 'timeOut'])
    OUT['timeInsig'] = get_val(tonality_out, ['time_insig', 'timeInsig'])
    OUT['soundField'] = sound_field

    # Statistics
    if OUT['tonalityTDep'] is not None:
        try:
            from utilities import get_statistics, export_dict_to_excel
            stats = get_statistics(OUT['tonalityTDep'], 'Tonality_ECMA418_2')
            for k, v in stats.items():
                if k not in OUT:
                    OUT[k] = v
        except Exception:
            # Optional stats/export; failure should not break wrapper
            pass

    if export_excel:
        try:
            from utilities import export_dict_to_excel
            export_dict_to_excel(OUT, filename=export_excel)
        except Exception:
            pass

    if OUT.get('tonalityAvg') is not None:
        print(f"SHM ECMA Tonality: {OUT['tonalityAvg']} t.u.")

    # Plot results if requested; plotting is optional and must not raise errors
    if show:
        try:
            # Gather available arrays (safe access)
            time_out = OUT.get('timeOut')                     # 1D time axis
            band_freqs = OUT.get('bandCentreFreqs')           # 1D band centre freqs
            spec_tonality = OUT.get('specTonality')           # time x bands x (channels?)
            spec_tonal_loudness = OUT.get('specTonalLoudness')
            spec_noise_loudness = OUT.get('specNoiseLoudness')
            tonality_t = OUT.get('tonalityTDep')              # time x (channels?)
            tonality_avg = OUT.get('tonalityAvg')             # (channels,)

            # Convert everything to numpy arrays if present
            if time_out is not None:
                time_out = np.asarray(time_out)
            if band_freqs is not None:
                band_freqs = np.asarray(band_freqs)

            # Determine how many channels there are (assume last axis is channels)
            def _ensure_3d(arr):
                # Ensure arr has shape [time, bands, channels]
                if arr is None:
                    return None
                a = np.asarray(arr)
                if a.ndim == 1:
                    # If 1D, we can't make a 3D array; return as-is
                    return a
                if a.ndim == 2:
                    # Cases: time x bands  OR bands x time  -> prefer time x bands
                    return a.reshape((a.shape[0], a.shape[1], 1))
                if a.ndim == 3:
                    return a
                # Unexpected shape, try to squeeze then re-handle
                a = np.squeeze(a)
                if a.ndim == 2:
                    return a.reshape((a.shape[0], a.shape[1], 1))
                return a

            spec_tonality_3d = _ensure_3d(spec_tonality)
            spec_tonal_3d = _ensure_3d(spec_tonal_loudness)
            spec_noise_3d = _ensure_3d(spec_noise_loudness)

            tonality_t_arr = None
            if tonality_t is not None:
                tonality_t_arr = np.asarray(tonality_t)
                if tonality_t_arr.ndim == 1:
                    tonality_t_arr = tonality_t_arr.reshape((tonality_t_arr.shape[0], 1))

            tonality_avg_arr = None
            if tonality_avg is not None:
                tonality_avg_arr = np.asarray(tonality_avg)
                if tonality_avg_arr.ndim == 0:
                    tonality_avg_arr = tonality_avg_arr.reshape((1,))

            # Guess channel count based on available arrays
            ch_count = 1
            if spec_tonality_3d is not None and spec_tonality_3d.ndim == 3:
                ch_count = spec_tonality_3d.shape[2]
            elif spec_tonal_3d is not None and spec_tonal_3d.ndim == 3:
                ch_count = spec_tonal_3d.shape[2]
            elif tonality_t_arr is not None and tonality_t_arr.ndim == 2:
                ch_count = tonality_t_arr.shape[1]

            # Use a plasma colormap
            cmap = cm.get_cmap('plasma')

            # Iterate channels and plot
            for ch in range(ch_count):
                fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

                # Heatmap of specific tonality (time vs band)
                if spec_tonality_3d is not None:
                    # Extract channel map (time x bands)
                    spec_matrix = spec_tonality_3d if spec_tonality_3d.ndim == 3 else spec_tonality_3d
                    if spec_matrix.ndim == 3:
                        data = spec_matrix[:, :, ch]
                    elif spec_matrix.ndim == 2:
                        data = spec_matrix
                    else:
                        data = None

                    if data is not None and time_out is not None and band_freqs is not None:
                        pcm = axs[0].pcolormesh(time_out, band_freqs, data.T,
                                                shading='gouraud', cmap=cmap, vmin=0)
                        axs[0].set(yscale='log',
                                   ylabel='Frequency (Hz)',
                                   xlabel='Time (s)')
                        # Frequencies to show on y ticks
                        ys = [63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3]
                        axs[0].set_yticks([y for y in ys if y >= band_freqs.min() and y <= band_freqs.max()])
                        axs[0].set_title(f"Channel {ch} specific tonality")
                        cbar = fig.colorbar(pcm, ax=axs[0], label='Specific tonality (tu per Bark)')
                    else:
                        axs[0].set_visible(False)
                else:
                    axs[0].set_visible(False)

                # Time-dependent total tonality
                if tonality_t_arr is not None and time_out is not None:
                    tonality_line = tonality_t_arr[:, ch] if tonality_t_arr.ndim == 2 else tonality_t_arr
                    axs[1].plot(time_out[:tonality_line.shape[0]], tonality_line,
                                lw=1.25, color=cmap(0.6), label='Time-dependent tonality')
                    if tonality_avg_arr is not None and ch < tonality_avg_arr.size:
                        axs[1].axhline(tonality_avg_arr[ch], color='k', lw=1.0, linestyle='--',
                                       label='Time-average')
                    axs[1].set(xlabel='Time (s)', ylabel='Tonality (tu)', title=f'Channel {ch} total tonality')
                    axs[1].legend()
                else:
                    axs[1].set_visible(False)

                # Time-averaged tonal & noise loudness vs band frequency
                if spec_tonal_3d is not None and spec_noise_3d is not None and band_freqs is not None:
                    tonal = spec_tonal_3d if spec_tonal_3d.ndim == 3 else spec_tonal_3d
                    noise = spec_noise_3d if spec_noise_3d.ndim == 3 else spec_noise_3d
                    if tonal.ndim == 3:
                        tonal_mean = np.mean(tonal[:, :, ch], axis=0)
                        noise_mean = np.mean(noise[:, :, ch], axis=0)
                    elif tonal.ndim == 2:
                        tonal_mean = np.mean(tonal, axis=0)
                        noise_mean = np.mean(noise, axis=0)
                    else:
                        tonal_mean = noise_mean = None

                    if tonal_mean is not None:
                        axs[2].semilogx(band_freqs, tonal_mean, '-o', lw=1.0, ms=3, label='Tonal loudness')
                        axs[2].semilogx(band_freqs, noise_mean, '-o', lw=1.0, ms=3, label='Noise loudness')
                        axs[2].set(xlabel='Frequency (Hz)', ylabel='Loudness (tu per Bark)',
                                   title=f'Channel {ch} time-averaged tonal/noise loudness')
                        axs[2].legend()
                    else:
                        axs[2].set_visible(False)
                else:
                    axs[2].set_visible(False)

                # Only show plotted figures
                plt.show()

        except Exception as e:
            # Do not raise exceptions because of plotting; report and proceed
            print("Plotting failed:", str(e))
            import traceback
            traceback.print_exc()

    return OUT

if __name__ == "__main__":
    print("metrics_shm_tonality_ecma.py: Test run")
    try:
        fs = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(fs*duration), endpoint=False)
        # 1 kHz tone at 60 dB SPL
        freq = 1000
        pref = 2e-5
        desired_spl = 60
        a_rms_pa = pref * 10**(desired_spl / 20)
        a_peak_pa = a_rms_pa * np.sqrt(2)
        fullscale_pa = pref * 10**(94 / 20)
        amplitude = a_peak_pa / fullscale_pa
        
        sig = amplitude * np.sin(2*np.pi*freq*t)
        
        print(f"Running tonality analysis on {desired_spl} dB SPL, {freq} Hz tone...")
        OUT = shm_tonality_ecma_wrapper(sig, fs=fs, show=True)
        
        print("\nReturn keys:", list(OUT.keys()))
        if 'tonalityAvg' in OUT:
            print(f"Tonality: {OUT['tonalityAvg']} t.u.")
            
    except Exception as e:
        print("Test failed:", e)
        import traceback
        traceback.print_exc()


