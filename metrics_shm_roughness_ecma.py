from __future__ import annotations
from typing import Dict, Any

import importlib
import numpy as np
from numpy.typing import NDArray

from utilities import wav2sig
from sottek_hearing_model import shm_roughness_ecma

# Add matplotlib imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm

__all__ = ["shm_roughness_ecma_wrapper"]

FloatArray = NDArray[np.floating]

def shm_roughness_ecma_wrapper(insig, fs=None, field=0, method=1,
                               time_skip=0, show=False, dBFS=94,
                               export_excel=None):
    """
    Wrapper for ECMA-418-2 (2022) Roughness using the Sottek Hearing Model.

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
        Dictionary containing roughness metrics.
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

    # Calculate Roughness
    # Pass show into out_plot so the library doesn't plot unless requested
    roughness_out = shm_roughness_ecma(p=insig, samp_rate_in=fs, axis=0,
                                      soundfield=sound_field, wait_bar=True,
                                      out_plot=False)

    if not isinstance(roughness_out, dict):
         raise RuntimeError("Unexpected return type from roughness function; expected dict.")

    # --- Output Mapping ---
    OUT: Dict[str, Any] = {}

    def get_val(d, keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    OUT['barkAxis'] = get_val(roughness_out, ['band_centre_freqs', 'bandCentreFreqs'])
    OUT['time'] = get_val(roughness_out, ['time_out', 'timeOut'])

    # Instantaneous Roughness
    inst_roughness = get_val(roughness_out, ['roughness_t', 'roughnessTDep'])
    if inst_roughness is not None:
        OUT['InstantaneousRoughness'] = np.asarray(inst_roughness)

    # Overall Roughness (90th percentile)
    roughness_90 = get_val(roughness_out, ['roughness90pc', 'roughness_90pc'])
    if roughness_90 is not None:
        OUT['Roughness'] = roughness_90
    else:
        if 'InstantaneousRoughness' in OUT:
            OUT['Roughness'] = np.percentile(OUT['InstantaneousRoughness'], 90)

    # Specific Roughness
    spec_roughness = get_val(roughness_out, ['spec_roughness', 'specRoughness'])
    if spec_roughness is not None:
        OUT['InstantaneousSpecificRoughness'] = spec_roughness

    spec_roughness_avg = get_val(roughness_out, ['spec_roughness_avg', 'specRoughnessAvg'])
    if spec_roughness_avg is not None:
        OUT['SpecificRoughness'] = spec_roughness_avg

    # Map binaural values if present
    OUT['InstantaneousSpecificRoughnessBin'] = get_val(roughness_out, ['spec_roughness_bin', 'specRoughnessBin'])
    OUT['SpecificRoughnessBin'] = get_val(roughness_out, ['spec_roughness_avg_bin', 'specRoughnessAvgBin'])
    OUT['InstantaneousRoughnessBin'] = get_val(roughness_out, ['roughness_t_bin', 'roughnessTDepBin'])
    OUT['RoughnessBin'] = get_val(roughness_out, ['roughness90pc_bin', 'roughness90pcBin'])

    # Statistics: import get_statistics and export utility lazily like other wrappers
    if 'InstantaneousRoughness' in OUT:
        try:
            from utilities import get_statistics
            stats = get_statistics(OUT['InstantaneousRoughness'], 'shm_roughness_ecma')
            for k, v in stats.items():
                if k not in OUT:
                    OUT[k] = v
        except Exception:
            # Optional statistics; do not fail the wrapper if utilities aren't available
            pass

    if export_excel:
        try:
            from utilities import export_dict_to_excel
            export_dict_to_excel(OUT, filename=export_excel)
        except Exception:
            pass

    if 'Roughness' in OUT:
        print(f"SHM ECMA Roughness: {OUT['Roughness']} asper")

    # Plot results if requested; plotting is optional and must not raise errors
    if show:
        try:
            time_out = OUT.get('time')                      # time axis
            band_freqs = OUT.get('barkAxis')                # band centre freqs
            spec_rough = OUT.get('InstantaneousSpecificRoughness')
            spec_rough_bin = OUT.get('InstantaneousSpecificRoughnessBin')
            inst_rough = OUT.get('InstantaneousRoughness')
            roughness_avg = OUT.get('SpecificRoughness')
            roughness_overall = OUT.get('Roughness')

            # Convert to numpy arrays where needed
            if time_out is not None:
                time_out = np.asarray(time_out)
            if band_freqs is not None:
                band_freqs = np.asarray(band_freqs)

            # Helper to ensure arr is time x bands x channels
            def _ensure_3d(arr):
                if arr is None:
                    return None
                a = np.asarray(arr)
                if a.ndim == 1:
                    return a  # cannot be converted to 3D meaningfully
                if a.ndim == 2:
                    # time x bands -> time x bands x 1
                    return a.reshape((a.shape[0], a.shape[1], 1))
                if a.ndim == 3:
                    return a
                a = np.squeeze(a)
                if a.ndim == 2:
                    return a.reshape((a.shape[0], a.shape[1], 1))
                return a

            spec_rough_3d = _ensure_3d(spec_rough)
            spec_rough_bin_3d = _ensure_3d(spec_rough_bin)

            # If binaural data present, combine channels (L, R, Bin) where available
            if spec_rough_bin_3d is not None and spec_rough_3d is not None:
                # spec_rough_3d may be (time, bands, 2) - concatenate binaural as 3rd channel
                if spec_rough_3d.ndim == 3 and spec_rough_bin_3d.ndim == 3:
                    spec_rough_3d = np.concatenate((spec_rough_3d, spec_rough_bin_3d), axis=2)

            # Convert inst_rough to 2D time x channels
            inst_rough_arr = None
            if inst_rough is not None:
                inst_rough_arr = np.asarray(inst_rough)
                if inst_rough_arr.ndim == 1:
                    inst_rough_arr = inst_rough_arr.reshape((inst_rough_arr.shape[0], 1))

            # Convert roughness_avg to bands x channels
            roughness_avg_arr = None
            if roughness_avg is not None:
                roughness_avg_arr = np.asarray(roughness_avg)
                if roughness_avg_arr.ndim == 1:
                    roughness_avg_arr = roughness_avg_arr.reshape((roughness_avg_arr.shape[0], 1))

            # Option: if BINARY overall present, create roughness_overall array
            roughness_overall_arr = None
            if isinstance(roughness_overall, (list, np.ndarray)):
                roughness_overall_arr = np.asarray(roughness_overall)
                if roughness_overall_arr.ndim == 0:
                    roughness_overall_arr = roughness_overall_arr.reshape((1,))

            # Determine channel count
            ch_count = 1
            if spec_rough_3d is not None and spec_rough_3d.ndim == 3:
                ch_count = spec_rough_3d.shape[2]
            elif inst_rough_arr is not None and inst_rough_arr.ndim == 2:
                ch_count = inst_rough_arr.shape[1]
            elif roughness_avg_arr is not None and roughness_avg_arr.ndim == 2:
                ch_count = roughness_avg_arr.shape[1]

            # Create channel labels
            if ch_count == 1:
                labels = ["Mono"]
            elif ch_count == 2:
                labels = ["Left", "Right"]
            else:
                labels = ["Left", "Right", "Binaural"] + [f"Ch{i}" for i in range(4, ch_count+1)]

            cmap = cm.get_cmap("inferno")
            for ch in range(ch_count):
                fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

                # Heatmap: specific roughness time vs frequency
                if spec_rough_3d is not None:
                    if spec_rough_3d.ndim == 3:
                        data = spec_rough_3d[:, :, ch]
                    else:
                        data = spec_rough_3d
                    if data is not None and time_out is not None and band_freqs is not None:
                        pcm = axs[0].pcolormesh(time_out, band_freqs, data.T,
                                                shading="gouraud", cmap=cmap, vmin=0)
                        axs[0].set(xlabel="Time (s)", yscale="log", ylabel="Frequency (Hz)")
                        ys = [63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3]
                        axs[0].set_yticks([y for y in ys if y >= band_freqs.min() and y <= band_freqs.max()])
                        axs[0].set_title(f"{labels[ch]} specific roughness")
                        fig.colorbar(pcm, ax=axs[0], label="Specific roughness (asper per Bark)")
                    else:
                        axs[0].set_visible(False)
                else:
                    axs[0].set_visible(False)

                # Time-dependent overall roughness
                if inst_rough_arr is not None and time_out is not None:
                    r_line = inst_rough_arr[:, ch] if inst_rough_arr.ndim == 2 else inst_rough_arr
                    axs[1].plot(time_out[:r_line.shape[0]], r_line,
                                lw=1.25, color=cmap(0.6), label="Time-dependent roughness")
                    # plot overall 90th percentile if available
                    if roughness_overall_arr is not None and ch < roughness_overall_arr.size:
                        axs[1].axhline(roughness_overall_arr[ch], color="k", lw=1.0, linestyle="--",
                                       label="90th percentile")
                    axs[1].set(xlabel="Time (s)", ylabel="Roughness (asper)", title=f"{labels[ch]} overall roughness")
                    axs[1].legend()
                else:
                    axs[1].set_visible(False)

                # Time-averaged specific roughness vs frequency
                if roughness_avg_arr is not None and band_freqs is not None:
                    if roughness_avg_arr.ndim == 2:
                        avg_line = roughness_avg_arr[:, ch]
                    else:
                        avg_line = roughness_avg_arr
                    axs[2].semilogx(band_freqs, avg_line, "-o", lw=1.0, ms=3, label="Specific roughness")
                    axs[2].set(xlabel="Frequency (Hz)", ylabel="Roughness (asper per Bark)",
                               title=f"{labels[ch]} time-averaged specific roughness")
                    axs[2].legend()
                else:
                    axs[2].set_visible(False)

                plt.show()

        except Exception as e:
            print("Plotting failed:", e)
            import traceback
            traceback.print_exc()

    return OUT

if __name__ == "__main__":
    print("metrics_shm_roughness_ecma.py: Test run")
    try:
        fs = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(fs*duration), endpoint=False)
        # 1 kHz tone AM modulated at 70 Hz (roughness)
        fc = 1000
        fmod = 70
        pref = 2e-5
        desired_spl = 60
        a_rms_pa = pref * 10**(desired_spl / 20)
        a_peak_pa = a_rms_pa * np.sqrt(2)
        fullscale_pa = pref * 10**(94 / 20)
        amplitude = a_peak_pa / fullscale_pa
        
        # 100% AM modulation
        sig = amplitude * (1 + 1.0 * np.sin(2*np.pi*fmod*t)) * np.sin(2*np.pi*fc*t)
        
        print(f"Running roughness analysis on {desired_spl} dB SPL, {fc} Hz tone AM at {fmod} Hz...")
        OUT = shm_roughness_ecma_wrapper(sig, fs=fs, show=True)
        
        print("\nReturn keys:", list(OUT.keys()))
        if 'Roughness' in OUT:
            print(f"Roughness: {OUT['Roughness']} asper")
            
    except Exception as e:
        print("Test failed:", e)
        import traceback
        traceback.print_exc()
