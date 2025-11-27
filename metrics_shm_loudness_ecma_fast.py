from __future__ import annotations
from typing import Dict, Any


import warnings
import numpy as np
from numpy.typing import NDArray
from sottek_hearing_model import shm_tonality_ecma, shm_loudness_ecma_from_comp

# Add matplotlib imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm


__all__ = ["shm_loudness_ecma_fast_wrapper"]

FloatArray = NDArray[np.floating]


def shm_loudness_ecma_fast_wrapper(insig, fs=None, field=0, method=1,
                                   time_skip=0, show=False, dBFS=94,
                                   export_excel=None):
    """Fast wrapper using Sottek's `shm_loudness_ecma_from_comp`.

    This wrapper computes tonality components via `shm_tonality_ecma`, then
    calls `shm_loudness_ecma_from_comp` which is much faster when the
    tonal/noise components are already available or when repeated loudness
    computations are needed.

    Behaviour mirrors the older wrapper: accepts filename or numpy array
    (with `fs` required for arrays), applies the same dBFS scaling, optionally
    shows plots, and returns a dict with mapped keys and simple statistics.
    """
    # --- WAV file interface ---
    if isinstance(insig, str):
        # Prefer soundfile for robust multi-channel reads
        try:
            import soundfile as _sf
            insig_raw, fs = _sf.read(insig, always_2d=False)
        except Exception:
            from scipy.io import wavfile as _wavfile
            fs, insig_raw = _wavfile.read(insig)

        # Convert integer types to float in [-1, 1] while preserving channels
        if hasattr(insig_raw, 'dtype') and insig_raw.dtype.kind in 'iu':
            max_val = np.iinfo(insig_raw.dtype).max
            insig = insig_raw.astype(np.float32) / float(max_val)
        else:
            insig = insig_raw.astype(np.float32)

        # Apply dBFS scaling (same convention used elsewhere in the repo)
        gain_factor = 10 ** ((dBFS - 94) / 20)
        insig = gain_factor * insig
    elif fs is None:
        raise ValueError("If insig is not a filename, fs must be provided.")

    # Call tonality to obtain spec components.
   
    tonality = shm_tonality_ecma(insig, fs, axis=0, soundfield='free_frontal', wait_bar=False, out_plot=False)


    if not isinstance(tonality, dict):
        try:
            tonality = {k: getattr(tonality, k) for k in dir(tonality) if not k.startswith("_")}
        except Exception:
            raise RuntimeError("Unexpected return type from tonality function; expected dict-like.")

    # Extract required components
    if 'spec_tonal_loudness' not in tonality or 'spec_noise_loudness' not in tonality:
        raise RuntimeError("Tonality output did not contain 'spec_tonal_loudness' and 'spec_noise_loudness'.")

    spec_tonal = tonality['spec_tonal_loudness']
    spec_noise = tonality['spec_noise_loudness']

    loudness = shm_loudness_ecma_from_comp(spec_tonal, spec_noise, out_plot=False, binaural=True)

    # Map outputs to common schema used in repository
    OUT: Dict[str, Any] = {}
    # band centres
    if 'band_centre_freqs' in loudness:
        OUT['barkAxis'] = loudness['band_centre_freqs']
    # time vectors
    if 'time_out' in loudness:
        OUT['time'] = loudness['time_out']

    # Instantaneous loudness (sone)
    if 'loudness_t' in loudness:
        OUT['InstantaneousLoudness'] = np.asarray(loudness['loudness_t'])

    # Overall loudness (sone)
    if 'loudness_powavg' in loudness:
        lp = np.asarray(loudness['loudness_powavg'])
        # handle binaural (array length 3) or stereo (2) or mono
        if lp.ndim == 0:
            OUT['Loudness'] = float(lp)
        else:
            if lp.size == 3:
                OUT['Loudness'] = lp[:2]
                OUT['LoudnessBin'] = float(lp[2])
            else:
                OUT['Loudness'] = lp

    # Specific loudness mapping
    if 'spec_loudness' in loudness:
        OUT['SpecificLoudness'] = loudness['spec_loudness']
    if 'spec_loudness_powavg' in loudness:
        OUT['SpecificLoudness_powavg'] = loudness['spec_loudness_powavg']

    # compute simple statistics similar to original wrapper
    try:
        from utilities import get_statistics, export_dict_to_excel
        if 'InstantaneousLoudness' in OUT:
            inst = np.asarray(OUT['InstantaneousLoudness'])
            if inst.ndim == 1:
                stats = get_statistics(inst, 'shm_loudness_ecma')
            else:
                stats = get_statistics(inst, 'shm_loudness_ecma')
            for k, v in stats.items():
                if k not in OUT:
                    OUT[k] = v
    except Exception:
        warnings.warn('Could not compute extra statistics; missing utilities.get_statistics')

    if export_excel is not None:
        try:
            export_dict_to_excel(OUT, filename=f"{export_excel}")
        except Exception:
            warnings.warn('Failed to export excel; check export function availability')

    # Optional simple plot if show=True but Sottek plotting was not used
    if show:
        try:
           
            existing = plt.get_fignums()
            if len(existing) == 0 and 'InstantaneousLoudness' in OUT:
                inst = np.asarray(OUT['InstantaneousLoudness'])
                t = np.asarray(OUT.get('time', np.arange(inst.shape[0])))

                band_freqs = np.asarray(OUT.get('barkAxis')) if OUT.get('barkAxis') is not None else None
                spec_loud = OUT.get('SpecificLoudness')
                spec_loud_avg = OUT.get('SpecificLoudness_powavg')

                # Helper: ensure time x bands x channels
                def _ensure_3d(arr):
                    if arr is None:
                        return None
                    a = np.asarray(arr)
                    if a.ndim == 1:
                        return a  # time only -> keep 1D
                    if a.ndim == 2:
                        # assume time x bands OR bands x channels
                        # prefer time x bands (time axis length equals t)
                        if a.shape[0] == t.size:
                            return a.reshape((a.shape[0], a.shape[1], 1))
                        else:
                            # bands x channels -> convert to time x bands x channels not possible,
                            # return as bands x channels for averaged plots
                            return a
                    if a.ndim == 3:
                        return a
                    a = np.squeeze(a)
                    if a.ndim == 2:
                        return a.reshape((a.shape[0], a.shape[1], 1))
                    return a

                spec_loud_3d = _ensure_3d(spec_loud)
                spec_loud_avg_arr = np.asarray(spec_loud_avg) if spec_loud_avg is not None else None

                # Convert inst loudness to 2D time x channels
                if inst.ndim == 1:
                    inst_arr = inst.reshape((inst.shape[0], 1))
                else:
                    inst_arr = inst

                # Number of channels guess
                ch_count = 1
                if spec_loud_3d is not None and hasattr(spec_loud_3d, 'ndim') and spec_loud_3d.ndim == 3:
                    ch_count = spec_loud_3d.shape[2]
                elif inst_arr is not None and inst_arr.ndim == 2:
                    ch_count = inst_arr.shape[1]
                elif spec_loud_avg_arr is not None:
                    if spec_loud_avg_arr.ndim == 1:
                        ch_count = 1
                    elif spec_loud_avg_arr.ndim == 2:
                        ch_count = spec_loud_avg_arr.shape[1]

                # Labels
                if ch_count == 1:
                    labels = ["Mono"]
                elif ch_count == 2:
                    labels = ["Left", "Right"]
                else:
                    labels = ["Left", "Right", "Binaural"] + [f"Ch{i}" for i in range(4, ch_count+1)]

                cmap = cm.get_cmap("viridis")

                for ch in range(ch_count):
                    fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

                    # Heatmap: specific loudness (time x bands)
                    if spec_loud_3d is not None:
                        if spec_loud_3d.ndim == 3:
                            data = spec_loud_3d[:, :, ch]
                        else:
                            data = spec_loud_3d
                        if data is not None and band_freqs is not None and t is not None:
                            pcm = axs[0].pcolormesh(t, band_freqs, data.T, shading="gouraud", cmap=cmap)
                            axs[0].set(xlabel="Time (s)", yscale="log", ylabel="Frequency (Hz)")
                            ys = [63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3]
                            axs[0].set_yticks([y for y in ys if y >= band_freqs.min() and y <= band_freqs.max()])
                            axs[0].set_title(f"{labels[ch]} specific loudness")
                            fig.colorbar(pcm, ax=axs[0], label="Specific loudness (sone/Bark)")
                        else:
                            axs[0].set_visible(False)
                    else:
                        axs[0].set_visible(False)

                    # Instantaneous loudness plot
                    if inst_arr is not None and t is not None:
                        inst_line = inst_arr[:, ch] if inst_arr.ndim == 2 else inst_arr
                        axs[1].plot(t[:inst_line.shape[0]], inst_line, lw=1.25, color=cmap(0.6), label="Instantaneous loudness")
                        # plot powavg or overall if available
                        if 'Loudness' in OUT:
                            L = np.asarray(OUT['Loudness'])
                            if L.ndim == 0:
                                axs[1].axhline(float(L), color='k', lw=1.0, linestyle='--', label="PowAvg")
                            else:
                                if ch < L.size:
                                    axs[1].axhline(float(L[ch]), color='k', lw=0.8, linestyle='--', label="PowAvg")
                        if 'LoudnessBin' in OUT and ch >= 2:
                            try:
                                axs[1].axhline(float(OUT['LoudnessBin']), color='r', lw=0.8, linestyle=':', label="Binaural")
                            except Exception:
                                pass
                        axs[1].set(xlabel="Time (s)", ylabel="Loudness (sone)", title=f"{labels[ch]} instantaneous loudness")
                        axs[1].legend()
                    else:
                        axs[1].set_visible(False)

                    # Time-averaged specific loudness vs frequency
                    if spec_loud_avg_arr is not None and band_freqs is not None:
                        if spec_loud_avg_arr.ndim == 2:
                            avg_line = spec_loud_avg_arr[:, ch]
                        else:
                            avg_line = spec_loud_avg_arr
                        axs[2].semilogx(band_freqs, avg_line, "-o", lw=1.0, ms=3, label="Specific loudness (time-avg)")
                        axs[2].set(xlabel="Frequency (Hz)", ylabel="Loudness (sone/Bark)", title=f"{labels[ch]} time-averaged specific loudness")
                        axs[2].legend()
                    else:
                        axs[2].set_visible(False)

                    plt.show()

        except Exception:
            warnings.warn('Could not produce plots; matplotlib may be unavailable or data shapes unsupported.')

    # Print concise summary similar to requested format
    try:
        header = 'Loudness (ECMA-418-2:2024 - Hearing Model of Sottek):'
        print(header)
        # If a filename was passed, show it
        if isinstance(insig, str):
            print(f"    - Stereo signal: {insig}")
        # print per-channel and binaural values if present
        if 'Loudness' in OUT:
            L = np.asarray(OUT['Loudness'])
            if L.ndim == 0:
                print(f"    - Overall loudness value: {float(L):.5f} (sone).")
            else:
                if L.size >= 1:
                    print(f"    - Overall loudness value (channel 1):  {float(L[0]):.5f} (sone).")
                if L.size >= 2:
                    print(f"    - Overall loudness value (channel 2):  {float(L[1]):.5f} (sone).")
        if 'LoudnessBin' in OUT:
            print(f"    - Overall loudness value (combined binaural):  {float(OUT['LoudnessBin']):.5f} (sone).")
    except Exception:
        # printing should never break function
        pass

    return OUT


if __name__ == "__main__":
    print("metrics_shm_loudness_ecma_fast.py: Test run")
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
        # SQAT convention: 94 dB SPL = 1.0 amplitude
        fullscale_pa = pref * 10**(94 / 20)
        amplitude = a_peak_pa / fullscale_pa
        
        sig = amplitude * np.sin(2*np.pi*freq*t)
        # Make it stereo
        sig_stereo = np.column_stack((sig, sig))
        
        print(f"Running loudness analysis on {desired_spl} dB SPL, {freq} Hz tone...")
        OUT = shm_loudness_ecma_fast_wrapper(sig_stereo, fs=fs, show=True)
        
        print("\nReturn keys:", list(OUT.keys()))
        if 'Loudness' in OUT:
            print(f"Loudness: {OUT['Loudness']}")
        if 'LoudnessBin' in OUT:
            print(f"Binaural Loudness: {OUT['LoudnessBin']}")
            
    except Exception as e:
        print("Test failed:", e)
        import traceback
        traceback.print_exc()
