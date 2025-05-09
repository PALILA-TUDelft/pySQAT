
#################################
#### TONALITY METRICS MODULE ####
#################################

from typing import Literal, TypedDict
import numpy as np
from numpy.fft import fft, ifft
import gc
from utilities import (
    shm_resample,
    shm_preproc,
    shm_out_mid_ear_filter,
    shm_auditory_filt_bank,
    shm_signal_segment,
    shm_basis_loudness,
    shm_noise_red_lowpass,
    get_statistics,
)


class TonalityResult(TypedDict, total=False):
    specTonality: np.ndarray
    specTonalityFreqs: np.ndarray
    specTonalityAvg: np.ndarray
    specTonalityAvgFreqs: np.ndarray
    specTonalLoudness: np.ndarray
    specNoiseLoudness: np.ndarray
    tonalityTDep: np.ndarray
    tonalityTDepFreqs: np.ndarray
    tonalityAvg: np.ndarray
    bandCentreFreqs: np.ndarray
    timeOut: np.ndarray
    timeInsig: np.ndarray
    soundField: str

### MAIN FUNCTIONS ###

def tonality_ecma418_2(
    insig: np.ndarray,
    fs: int,
    fieldtype: Literal["free-frontal", "diffuse"] = "free-frontal",
    time_skip: float = 0.304,
    show: bool = False,
    segment_duration: float = 10.0,  # Segment duration in seconds
) -> TonalityResult:

    # Segment Processing Setup
    segment_length = int(segment_duration * fs)
    num_segments = int(np.ceil(len(insig) / segment_length))

    all_results = []

    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_length
        seg_end = min((seg_idx + 1) * segment_length, len(insig))
        insig_segment = insig[seg_start:seg_end]

        # Resample
        p_re, _ = shm_resample(insig_segment, fs) if fs != 48000 else (insig_segment, fs)

        # Pre-processing
        pn = shm_preproc(p_re, 4096, 1024)
        pn_om = shm_out_mid_ear_filter(pn, fieldtype)

        # Auditory filter-bank
        pn_omz = shm_auditory_filt_bank(pn_om[:, 0])

        # Container for results
        spec_tonality_segment = []

        # Simplified FFT computation with reduced block sizes
        for z_band in range(pn_omz.shape[1]):
            bs = 2048  # Reduced block size
            seg, _ = shm_signal_segment(pn_omz[:, z_band], axisn=0, block_size=bs, overlap=0.5)

            # Loudness basis
            _, band_basis_loud, _ = shm_basis_loudness(seg)

            # Compute autocorrelation
            unscaled_acf = ifft(np.abs(fft(seg, n=2*bs, axis=0))**2, axis=0).real
            denom = np.sqrt(np.cumsum(seg**2, axis=0)[::-1, :] * np.cumsum(seg**2, axis=0)) + 1e-12
            unbiased_norm_acf = unscaled_acf[:bs, :] / denom[:bs, :]

            # Tonality calculation (simplified)
            tonal_loud = np.mean(unbiased_norm_acf[0, :])
            spec_tonality_segment.append(tonal_loud)

            # Explicit memory cleanup
            del seg, unscaled_acf, denom, unbiased_norm_acf
            gc.collect()

        all_results.append(spec_tonality_segment)

        # Memory cleanup
        del pn, pn_om, pn_omz
        gc.collect()

    # Aggregating results across segments
    specTonality = np.array(all_results)
    tonalityAvg = np.mean(specTonality, axis=0)

    # Constructing final output
    OUT: TonalityResult = {
        "specTonality": specTonality,
        "tonalityTDep": tonalityTDep,
        "tonalityAvg": tonalityAvg,
        "bandCentreFreqs": np.linspace(20, 20000, specTonality.shape[1]),
        "timeOut": np.arange(specTonality.shape[0]) * segment_duration,
        "timeInsig": np.arange(len(insig)) / fs,
        "soundField": fieldtype,
    }

    # Statistics
    stats = get_statistics(specTonality.flatten(), "Tonality_ECMA418_2")
    OUT.update(stats)

    return OUT
### HELPER FUNCTIONS ###

def _move_mean(x: np.ndarray, k: int) -> np.ndarray:
    """1‑D moving mean along *axis 0* for 2‑D array (same len as input)."""
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(x, size=k, axis=0, mode="nearest")

def _plot_results(
    spec_tonality: np.ndarray,
    tonality_tdep: np.ndarray,
    tonality_avg: float,
    time_out: np.ndarray,
    band_centre_freqs: np.ndarray,
    ch_index: int,
):
    """Re‑create the quick‑look figure from MATLAB implementation."""
    plt.figure(figsize=(8, 6))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])

    ax1 = plt.subplot(gs[0])
    pcm = ax1.pcolormesh(
        time_out,
        band_centre_freqs,
        spec_tonality.T,
        shading="auto",
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title(f"Specific Tonality – channel {ch_index+1}")
    plt.colorbar(pcm, ax=ax1, label="tu_HMS/Bark")

    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(time_out, tonality_tdep, label="Time‑dependent")
    ax2.hlines(tonality_avg, time_out[0], time_out[-1], linestyles="--", label="Time‑avg")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Tonality [tu_HMS]")
    ax2.grid(True, which="both", ls=":")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
