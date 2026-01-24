from __future__ import annotations
from typing import Dict, Any

import importlib
import warnings
import numpy as np
from numpy.typing import NDArray

from utilities import get_statistics, export_dict_to_excel, wav2sig

__all__ = ["shm_loudness_ecma_wrapper"]

FloatArray = NDArray[np.floating]


def shm_loudness_ecma_wrapper(insig, fs=None, field=0, method=1,
                              time_skip=0, show=False, dBFS=94,
                              export_excel=None):
    """
    Wrapper for ECMA-418-2 (2022) Loudness using the Sottek Hearing Model.
    
    This implementation uses the optimized approach: it first calculates tonality
    to obtain tonal and noise components, and then uses `shm_loudness_ecma_from_comp`
    for faster loudness calculation.

    Parameters
    ----------
    insig : str or np.ndarray
        Input signal. Can be a file path (str) or a numpy array.
    fs : int, optional
        Sampling frequency. Required if insig is an array.
    field : int, default 0
        Sound field. 0 for free field (free_frontal), 1 for diffuse field.
    method : int, default 1
        Method for statistics (similar to other metrics, though ECMA is inherently time-varying).
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
        Dictionary containing loudness metrics.
    """
    
    # --- WAV file interface ---
    if isinstance(insig, str):
        insig, fs = wav2sig(insig, fs, dBFS)
    elif fs is None:
        raise ValueError("If insig is not a filename, fs must be provided.")

    # --- Input validation ---
    if fs is None or fs <= 0:
        raise ValueError("Sampling frequency (fs) must be a positive number.")

    # Import Sottek package
    try:
        from sottek_hearing_model import shm_tonality_ecma, shm_loudness_ecma_from_comp
    except ImportError:
        raise ImportError(
            "Could not import required functions from 'sottek_hearing_model'.\n"
            "Install it, e.g.: pip install sottek-hearing-model"
        )

    # Map field parameter
    sound_field = 'free_frontal' if field == 0 else 'diffuse'

    # 1. Calculate Tonality to get components
    try:
        # Note: wait_bar=True to show progress
        tonality_out = shm_tonality_ecma(p=insig, samp_rate_in=fs, axis=0, 
                                  soundfield=sound_field, wait_bar=True, out_plot=False)
    except TypeError:
        # Try with different argument names if the API differs slightly
        try:
            tonality_out = shm_tonality_ecma(insig, fs, soundField=sound_field, waitBar=False, outPlot=False)
        except Exception as e:
             raise RuntimeError(f"Calling Sottek tonality function failed: {e}")

    if not isinstance(tonality_out, dict):
         raise RuntimeError("Unexpected return type from tonality function; expected dict.")

    # Check for required components
    if 'spec_tonal_loudness' not in tonality_out or 'spec_noise_loudness' not in tonality_out:
        # Try camelCase keys if snake_case fails (based on refmap code)
        if 'specTonalLoudness' in tonality_out and 'specNoiseLoudness' in tonality_out:
             spec_tonal = tonality_out['specTonalLoudness']
             spec_noise = tonality_out['specNoiseLoudness']
        else:
            raise RuntimeError("Tonality output missing 'spec_tonal_loudness'/'spec_noise_loudness' components.")
    else:
        spec_tonal = tonality_out['spec_tonal_loudness']
        spec_noise = tonality_out['spec_noise_loudness']

    # 2. Calculate Loudness using components
    try:
        loudness_out = shm_loudness_ecma_from_comp(spec_tonal, spec_noise, out_plot=show, binaural=True)
    except TypeError:
        try:
             loudness_out = shm_loudness_ecma_from_comp(spec_tonal, spec_noise, outPlot=show, binaural=True)
        except Exception as e:
            raise RuntimeError(f"Calling Sottek fast loudness function failed: {e}")

    # --- Output Mapping ---
    OUT: Dict[str, Any] = {}
    
    # Helper to get value with multiple possible keys
    def get_val(d, keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    # Map keys
    OUT['barkAxis'] = get_val(loudness_out, ['band_centre_freqs', 'bandCentreFreqs'])
    OUT['time'] = get_val(loudness_out, ['time_out', 'timeOut'])
    
    # Instantaneous Loudness
    inst_loudness = get_val(loudness_out, ['loudness_t', 'loudnessTDep'])
    if inst_loudness is not None:
        OUT['InstantaneousLoudness'] = np.asarray(inst_loudness)

    # Overall Loudness (Time Averaged)
    # Note: loudness_powavg might be per channel or binaural
    loudness_avg = get_val(loudness_out, ['loudness_powavg', 'loudnessPowAvg'])
    if loudness_avg is not None:
        lp = np.asarray(loudness_avg)
        if lp.ndim == 0:
             OUT['Loudness'] = float(lp)
        else:
            # If multiple values, it might be [Left, Right, Binaural] or just [Left, Right]
            # We'll store the array but also try to extract a single representative value if needed
            OUT['Loudness'] = lp
            if lp.size >= 3:
                OUT['LoudnessBin'] = float(lp[2]) # Assuming 3rd is binaural

    # Specific Loudness
    spec_loudness = get_val(loudness_out, ['spec_loudness', 'specLoudness']) # Time dependent specific?
    if spec_loudness is not None:
        OUT['InstantaneousSpecificLoudness'] = spec_loudness

    spec_loudness_avg = get_val(loudness_out, ['spec_loudness_powavg', 'specLoudnessPowAvg'])
    if spec_loudness_avg is not None:
        OUT['SpecificLoudness'] = spec_loudness_avg

    # Statistics
    if 'InstantaneousLoudness' in OUT:
        inst = OUT['InstantaneousLoudness']
        # Handle multi-channel statistics if necessary, for now just pass to get_statistics
        # which usually expects 1D. If 2D, we might need to decide what to do.
        # SQAT usually handles mono or specific channel. 
        # For now, let's calculate stats for the first channel or combined if available.
        
        stats_input = inst
        if inst.ndim > 1:
             # If we have binaural combined (often last column or separate key), use that?
             # But here inst is likely [samples, channels] or [channels, samples]
             # Let's assume we want stats for each channel or just the first one for the dict
             pass 

        try:
            stats = get_statistics(inst, 'shm_loudness_ecma')
            for k, v in stats.items():
                if k not in OUT:
                    OUT[k] = v
        except Exception:
            pass

    if export_excel:
        export_dict_to_excel(OUT, filename=export_excel)

    # Print summary
    if 'Loudness' in OUT:
        print(f"SHM ECMA Loudness: {OUT['Loudness']} sone")

    return OUT
