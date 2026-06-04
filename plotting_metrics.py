"""
Plotting utilities for psychoacoustic metrics.
Generates publication-quality plots matching the style of metrics_loudness.py
"""

import numpy as np
from matplotlib.figure import Figure
from typing import Dict, Any, Optional


def plot_loudness_analysis(out: Dict[str, Any], metric_name: str = "Loudness (ISO 532-1)") -> Figure:
    """
    Generate comprehensive loudness analysis plot similar to metrics_loudness.py
    
    Parameters
    ----------
    out : dict
        Output dictionary from Loudness metric function
    metric_name : str
        Name of the loudness metric (ISO 532-1 or ECMA 418-2)
    
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Check if time-varying or stationary
    is_time_varying = "InstantaneousLoudness" in out and "time" in out
    
    if is_time_varying:
        return _plot_loudness_time_varying(out, metric_name)
    else:
        return _plot_loudness_stationary(out, metric_name)


def _plot_loudness_time_varying(out: Dict[str, Any], metric_name: str) -> Figure:
    """Time-varying loudness plot (6 subplots)"""
    fig = Figure(figsize=(16, 8), dpi=100)
    
    # Get data
    time = np.asarray(out.get("time", [])).ravel()
    time_insig = np.asarray(out.get("time_insig", [])).ravel()
    insig = np.asarray(out.get("InstantaneousSPL", [])).ravel()
    inst_loudness_level = np.asarray(out.get("InstantaneousLoudnessLevel", [])).ravel()
    inst_loudness = np.asarray(out.get("InstantaneousLoudness", [])).ravel()
    spec_loudness = np.asarray(out.get("SpecificLoudness", [])).ravel()
    inst_spec_loudness = np.asarray(out.get("InstantaneousSpecificLoudness", []))
    bark_axis = np.asarray(out.get("barkAxis", [])).ravel()
    
    xmax = time[-1] if len(time) > 0 else 1
    
    # Plot 1: Input signal SPL
    ax1 = fig.add_subplot(2, 6, (1, 2))
    if len(time_insig) > 0 and len(insig) > 0:
        ax1.plot(np.linspace(0, time_insig[-1], len(insig)), insig, 'b-', linewidth=1.5)
        ax1.set_title('Instantaneous overall SPL (1/3 octave)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax1.set_ylabel('SPL, $L_p$ (dB re 20 μPa)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, xmax])
    
    # Plot 2: Instantaneous loudness level (phon)
    ax2 = fig.add_subplot(2, 6, (3, 4))
    if len(time) > 0 and len(inst_loudness_level) > 0:
        ax2.plot(time, np.abs(inst_loudness_level), 'g-', linewidth=1.5)
        ax2.set_title('Instantaneous loudness level', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax2.set_ylabel('Loudness level, $L_N$ (phon)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, xmax])
    
    # Plot 3: Instantaneous loudness (sone)
    ax3 = fig.add_subplot(2, 6, (5, 6))
    if len(time) > 0 and len(inst_loudness) > 0:
        ax3.plot(time, inst_loudness, 'r-', linewidth=2)
        ax3.set_title('Instantaneous loudness', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax3.set_ylabel('Loudness, $N$ (sone)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, xmax])
        
        # Add summary statistics
        if "Nmean" in out:
            nmean = np.asarray(out["Nmean"]).ravel()[0]
            ax3.axhline(y=nmean, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mean: {nmean:.3f}')
        if "N5" in out:
            n5 = np.asarray(out["N5"]).ravel()[0]
            ax3.axhline(y=n5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'5th: {n5:.3f}')
        ax3.legend(loc='best', fontsize=9)
    
    # Plot 4: Specific loudness (sone/bark)
    ax4 = fig.add_subplot(2, 6, (7, 8))
    if len(bark_axis) > 0 and len(spec_loudness) > 0:
        ax4.plot(bark_axis, spec_loudness, 'purple', linewidth=2)
        ax4.set_title('Time-averaged specific loudness', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Critical band, $z$ (Bark)', fontsize=10)
        ax4.set_ylabel("Specific loudness, $N'$ (sone/Bark)", fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 24])
    
    # Plot 5: Instantaneous specific loudness (heatmap)
    ax5 = fig.add_subplot(2, 6, (9, 10))
    if inst_spec_loudness.ndim == 2 and len(time) > 0:
        xx, yy = np.meshgrid(time, bark_axis)
        pcm = ax5.pcolormesh(xx, yy, inst_spec_loudness.T, shading='auto', cmap='viridis')
        ax5.set_title('Instantaneous specific loudness', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax5.set_ylabel('Critical band, $z$ (Bark)', fontsize=10)
        ax5.set_ylim([0, 24])
        cbar = fig.colorbar(pcm, ax=ax5)
        cbar.set_label("$N'$ (sone/Bark)", fontsize=9)
    
    # Plot 6: Summary statistics
    ax6 = fig.add_subplot(2, 6, (11, 12))
    ax6.axis('off')
    stats_text = f"Metric: {metric_name}\n\n"
    if "Nmean" in out:
        stats_text += f"Mean Loudness: {np.asarray(out['Nmean']).ravel()[0]:.3f} sone\n"
    if "N5" in out:
        stats_text += f"5th Percentile: {np.asarray(out['N5']).ravel()[0]:.3f} sone\n"
    if "N95" in out:
        stats_text += f"95th Percentile: {np.asarray(out['N95']).ravel()[0]:.3f} sone\n"
    if "Nmax" in out:
        stats_text += f"Maximum: {np.asarray(out['Nmax']).ravel()[0]:.3f} sone\n"
    if "Nmin" in out:
        stats_text += f"Minimum: {np.asarray(out['Nmin']).ravel()[0]:.3f} sone\n"
    if "LoudnessLevel" in out:
        ll = np.asarray(out["LoudnessLevel"]).ravel()[0]
        stats_text += f"\nLoudness Level: {ll:.1f} phon"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    return fig


def _plot_loudness_stationary(out: Dict[str, Any], metric_name: str) -> Figure:
    """Stationary loudness plot (2 subplots)"""
    fig = Figure(figsize=(10, 8), dpi=100)
    
    bark_axis = np.asarray(out.get("barkAxis", [])).ravel()
    spec_loudness = np.asarray(out.get("SpecificLoudness", [])).ravel()
    loudness = out.get("Loudness", 0)
    loudness_level = out.get("LoudnessLevel", 0)
    
    # Plot 1: Specific loudness
    ax1 = fig.add_subplot(2, 1, 1)
    if len(bark_axis) > 0 and len(spec_loudness) > 0:
        ax1.plot(bark_axis, spec_loudness, 'b-', linewidth=2)
        ax1.set_title('Specific loudness', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Critical band, $z$ (Bark)', fontsize=11)
        ax1.set_ylabel("Specific loudness, $N'$ (sone/Bark)", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 24])
        
        # Add summary box
        stats_text = f"Loudness, $N$={loudness:.3f} (sone)\nLoudness level, $L_N$={loudness_level:.1f} (phon)"
        ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Summary statistics
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axis('off')
    stats_text = f"Metric: {metric_name}\n\n"
    stats_text += f"Overall Loudness: {loudness:.3f} sone\n"
    stats_text += f"Loudness Level: {loudness_level:.1f} phon\n"
    if "N5" in out:
        stats_text += f"\n5th Percentile: {np.asarray(out['N5']).ravel()[0]:.3f} sone\n"
    if "Nmean" in out:
        stats_text += f"Mean: {np.asarray(out['Nmean']).ravel()[0]:.3f} sone\n"
    if "N95" in out:
        stats_text += f"95th Percentile: {np.asarray(out['N95']).ravel()[0]:.3f} sone\n"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    fig.tight_layout()
    return fig


def plot_sharpness_analysis(out: Dict[str, Any]) -> Figure:
    """Generate comprehensive sharpness analysis plot matching metrics_sharpness.py"""
    fig = Figure(figsize=(15, 8), dpi=100)
    
    time = np.asarray(out.get("time", [])).ravel()
    inst_sharpness = np.asarray(out.get("InstantaneousSharpness", [])).ravel()
    
    xmax = time[-1] if len(time) > 0 else 1
    
    # Plot 1: Instantaneous sharpness
    ax1 = fig.add_subplot(2, 2, 1)
    if len(time) > 0 and len(inst_sharpness) > 0:
        ax1.plot(time, inst_sharpness, 'b-', linewidth=2)
        ax1.set_title('Instantaneous Sharpness (DIN 45692)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax1.set_ylabel('Sharpness, $S$ (acum)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, xmax])
        
        if "Smean" in out:
            smean = np.asarray(out["Smean"]).ravel()[0]
            ax1.axhline(y=smean, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mean: {smean:.3f}')
        if "S5" in out:
            s5 = np.asarray(out["S5"]).ravel()[0]
            ax1.axhline(y=s5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'5th: {s5:.3f}')
        ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: Loudness component (from out dict)
    ax2 = fig.add_subplot(2, 2, 2)
    if "loudness" in out:
        loudness = out["loudness"]
        if isinstance(loudness, dict) and "InstantaneousLoudness" in loudness:
            loud_inst = np.asarray(loudness["InstantaneousLoudness"]).ravel()
            if len(time) > 0 and len(loud_inst) > 0:
                ax2.plot(time, loud_inst, 'purple', linewidth=2)
                ax2.set_title('Instantaneous Loudness (Component)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Time, $t$ (s)', fontsize=11)
                ax2.set_ylabel('Loudness, $N$ (sone)', fontsize=11)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim([0, xmax])
    
    # Plot 3: Psychoacoustic metrics comparison
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    # Show some additional metrics if available
    info_text = "Sharpness Components\n\n"
    if "SharpnessContribution" in out:
        info_text += "Method: Psychoacoustic model\n"
        info_text += "(Based on specific loudness pattern)\n"
    info_text += f"\nMetric: DIN 45692"
    
    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot 4: Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = "Sharpness (DIN 45692) - Statistics\n\n"
    
    stat_keys = [
        ("Smean", "Mean sharpness", "acum"),
        ("S5", "5th percentile", "acum"),
        ("S10", "10th percentile", "acum"),
        ("S95", "95th percentile", "acum"),
        ("Smax", "Maximum", "acum"),
        ("Smin", "Minimum", "acum"),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<25} {val:>10.4f} {unit}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    fig.tight_layout()
    return fig
    
    fig.tight_layout()
    return fig


def plot_roughness_analysis(out: Dict[str, Any]) -> Figure:
    """Generate comprehensive roughness analysis plot with 4 subplots"""
    fig = Figure(figsize=(15, 8), dpi=100)
    
    time = np.asarray(out.get("time", [])).ravel()
    inst_roughness = np.asarray(out.get("InstantaneousRoughness", [])).ravel()
    bark_axis = np.asarray(out.get("barkAxis", [])).ravel()
    spec_roughness = np.asarray(out.get("TimeAveragedSpecificRoughness", [])).ravel()
    inst_spec_roughness = np.asarray(out.get("InstantaneousSpecificRoughness", []))
    
    xmax = time[-1] if len(time) > 0 else 1
    
    # Plot 1: Instantaneous roughness
    ax1 = fig.add_subplot(2, 2, 1)
    if len(time) > 0 and len(inst_roughness) > 0:
        ax1.plot(time, inst_roughness, 'g-', linewidth=2)
        ax1.set_title('Instantaneous Roughness', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax1.set_ylabel('Roughness, $R$ (asper)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, xmax])
        
        if "Rmean" in out:
            rmean = np.asarray(out["Rmean"]).ravel()[0]
            ax1.axhline(y=rmean, color='g', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mean: {rmean:.3f}')
        ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: Time-averaged specific roughness (Bark scale)
    ax2 = fig.add_subplot(2, 2, 2)
    if len(bark_axis) > 0 and len(spec_roughness) > 0:
        ax2.plot(bark_axis, spec_roughness, 'darkgreen', linewidth=2)
        ax2.set_title('Time-averaged Specific Roughness', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Critical band, $z$ (Bark)', fontsize=11)
        ax2.set_ylabel("Specific Roughness, $R'$ (asper/Bark)", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 24])
    
    # Plot 3: Instantaneous specific roughness heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    if inst_spec_roughness.ndim == 2 and len(time) > 0:
        im = ax3.imshow(inst_spec_roughness.T, aspect='auto', cmap='viridis', origin='lower')
        ax3.set_title('Instantaneous Specific Roughness (Heatmap)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel('Critical band, $z$ (Bark)', fontsize=11)
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label("$R'$ (asper/Bark)", fontsize=11)
    
    # Plot 4: Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = "Roughness (Daniel 1997) - Statistics\n\n"
    
    stat_keys = [
        ("Rmean", "Mean roughness", "asper"),
        ("R5", "5th percentile", "asper"),
        ("R95", "95th percentile", "asper"),
        ("Rmax", "Maximum", "asper"),
        ("Rmin", "Minimum", "asper"),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<25} {val:>10.4f} {unit}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    fig.tight_layout()
    return fig


def plot_fluctuation_analysis(out: Dict[str, Any]) -> Figure:
    """Generate comprehensive fluctuation strength analysis plot matching metrics_fluctuation.py"""
    fig = Figure(figsize=(15, 8), dpi=100)
    
    time = np.asarray(out.get("time", [])).ravel()
    inst_fs = np.asarray(out.get("InstantaneousFluctuationStrength", [])).ravel()
    
    xmax = time[-1] if len(time) > 0 else 1
    
    # Plot 1: Instantaneous fluctuation strength
    ax1 = fig.add_subplot(2, 2, 1)
    if len(time) > 0 and len(inst_fs) > 0:
        ax1.plot(time, inst_fs, 'c-', linewidth=2)
        ax1.set_title('Instantaneous Fluctuation Strength', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax1.set_ylabel('Fluctuation Strength, $FS$ (vacil)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, xmax])
        
        if "FSmean" in out:
            fsmean = np.asarray(out["FSmean"]).ravel()[0]
            ax1.axhline(y=fsmean, color='c', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mean: {fsmean:.3f}')
        ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: Temporal envelope (modulation depth)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis('off')
    info_text = "Fluctuation Strength Components\n\n"
    info_text += "Method: Osses & Melcher\n"
    info_text += "Based on temporal modulation\n"
    info_text += "at psychoacoustic bands\n\n"
    info_text += "Optimal modulation: 4 Hz\n"
    info_text += "Units: vacil"
    
    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    
    # Plot 3: Time-averaged specific fluctuation strength (Bark scale)
    ax3 = fig.add_subplot(2, 2, 3)
    bark = np.asarray(out.get("barkAxis", [])).ravel()
    spec_fs = np.asarray(out.get("TimeAveragedSpecificFluctuationStrength", [])).ravel()
    if len(bark) > 0 and len(spec_fs) > 0:
        n = min(len(bark), len(spec_fs))
        ax3.plot(bark[:n], spec_fs[:n], color='teal', linewidth=2)
        ax3.set_title('Time-averaged specific fluctuation strength', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Critical band, $z$ (Bark)', fontsize=11)
        ax3.set_ylabel("Specific FS, $FS'$ (vacil/Bark)", fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 24])
    else:
        ax3.axis('off')

    # Plot 4: Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = "Fluctuation Strength - Statistics\n\n"
    
    stat_keys = [
        ("FSmean", "Mean fluctuation", "vacil"),
        ("FS5", "5th percentile", "vacil"),
        ("FS95", "95th percentile", "vacil"),
        ("FSmax", "Maximum", "vacil"),
        ("FSmin", "Minimum", "vacil"),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<25} {val:>10.4f} {unit}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    fig.tight_layout()
    return fig


def plot_tonality_analysis(out: Dict[str, Any]) -> Figure:
    """Generate comprehensive tonality analysis plot matching metrics_tonality.py"""
    fig = Figure(figsize=(15, 8), dpi=100)
    
    time = np.asarray(out.get("time", [])).ravel()
    inst_tonality = np.asarray(out.get("InstantaneousTonality", [])).ravel()
    
    xmax = time[-1] if len(time) > 0 else 1
    
    # Plot 1: Instantaneous tonality
    ax1 = fig.add_subplot(2, 2, 1)
    if len(time) > 0 and len(inst_tonality) > 0:
        ax1.plot(time, inst_tonality, 'm-', linewidth=2)
        ax1.set_title('Instantaneous Tonality (Aures 1985)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax1.set_ylabel('Tonality, $K$ (t.u.)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, xmax])
        
        if "Kmean" in out:
            kmean = np.asarray(out["Kmean"]).ravel()[0]
            ax1.axhline(y=kmean, color='m', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mean: {kmean:.3f}')
        ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: Tonal components analysis
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis('off')
    info_text = "Tonality Components\n\n"
    info_text += "Method: Aures (1985)\n"
    info_text += "Measures tonal content\n"
    info_text += "of acoustic signal\n\n"
    info_text += "Detection at:\n"
    info_text += "Narrow spectral peaks"
    
    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    # Plot 3: Frequency content
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    
    # Plot 4: Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = "Tonality (Aures 1985) - Statistics\n\n"
    
    stat_keys = [
        ("Kmean", "Mean tonality", "t.u."),
        ("K5", "5th percentile", "t.u."),
        ("K95", "95th percentile", "t.u."),
        ("Kmax", "Maximum", "t.u."),
        ("Kmin", "Minimum", "t.u."),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<25} {val:>10.4f} {unit}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.7))
    
    fig.tight_layout()
    return fig


def plot_annoyance_analysis(out: Dict[str, Any]) -> Figure:
    """Generate comprehensive annoyance analysis plot with all components"""
    fig = Figure(figsize=(18, 10), dpi=100)
    
    time = np.asarray(out.get("time", [])).ravel()
    inst_annoyance = np.asarray(out.get("InstantaneousPA", [])).ravel()
    
    # Component signals (from sub-metric calculations)
    inst_loudness = np.asarray(out.get("L", {}).get("InstantaneousLoudness", [])).ravel()
    inst_sharpness = np.asarray(out.get("S", {}).get("InstantaneousSharpness", [])).ravel()
    inst_roughness = np.asarray(out.get("R", {}).get("InstantaneousRoughness", [])).ravel()
    inst_fluctuation = np.asarray(out.get("FS", {}).get("InstantaneousFluctuationStrength", [])).ravel()
    inst_tonality = np.asarray(out.get("K", {}).get("InstantaneousTonality", [])).ravel()
    
    # Component weights
    ws = np.asarray(out.get("ws", [])).ravel()
    wfr = np.asarray(out.get("wfr", [])).ravel()
    wt = np.asarray(out.get("wt", [])).ravel()
    
    xmax = time[-1] if len(time) > 0 else 1
    
    # Plot 1: Instantaneous annoyance
    ax1 = fig.add_subplot(3, 3, 1)
    if len(time) > 0 and len(inst_annoyance) > 0:
        ax1.plot(time, inst_annoyance, 'k-', linewidth=2.5)
        ax1.set_title('Instantaneous Annoyance (PA)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax1.set_ylabel('Annoyance, $PA$', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, xmax])
        
        if "PAmean" in out:
            pamean = np.asarray(out["PAmean"]).ravel()[0]
            ax1.axhline(y=pamean, color='r', linestyle='--', alpha=0.6, linewidth=1.5, label=f'Mean: {pamean:.2f}')
            ax1.legend(loc='best', fontsize=8)
    
    # Plot 2: Loudness component
    ax2 = fig.add_subplot(3, 3, 2)
    if len(time) > 0 and len(inst_loudness) > 0:
        ax2.plot(time, inst_loudness, 'b-', linewidth=2)
        ax2.set_title('Loudness (Primary)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax2.set_ylabel('Loudness, $N$ (sone)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, xmax])
        
        if "L" in out and "N5" in out["L"]:
            n5 = np.asarray(out["L"]["N5"]).ravel()[0]
            ax2.axhline(y=n5, color='r', linestyle='--', alpha=0.6, linewidth=1, label=f'5th %ile: {n5:.2f}')
            ax2.legend(loc='best', fontsize=8)
    
    # Plot 3: Sharpness component
    ax3 = fig.add_subplot(3, 3, 3)
    if len(time) > 0 and len(inst_sharpness) > 0:
        ax3.plot(time, inst_sharpness, 'g-', linewidth=2)
        ax3.set_title('Sharpness (Secondary)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax3.set_ylabel('Sharpness, $S$ (acum)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, xmax])
        
        if "S" in out and "S5" in out["S"]:
            s5 = np.asarray(out["S"]["S5"]).ravel()[0]
            ax3.axhline(y=s5, color='r', linestyle='--', alpha=0.6, linewidth=1, label=f'5th %ile: {s5:.2f}')
            ax3.legend(loc='best', fontsize=8)
    
    # Plot 4: Roughness component
    ax4 = fig.add_subplot(3, 3, 4)
    if len(time) > 0 and len(inst_roughness) > 0:
        ax4.plot(time, inst_roughness, 'orange', linewidth=2)
        ax4.set_title('Roughness', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax4.set_ylabel('Roughness, $R$ (asper)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, xmax])
        
        if "R" in out and "R5" in out["R"]:
            r5 = np.asarray(out["R"]["R5"]).ravel()[0]
            ax4.axhline(y=r5, color='r', linestyle='--', alpha=0.6, linewidth=1, label=f'5th %ile: {r5:.2f}')
            ax4.legend(loc='best', fontsize=8)
    
    # Plot 5: Fluctuation strength component
    ax5 = fig.add_subplot(3, 3, 5)
    if len(time) > 0 and len(inst_fluctuation) > 0:
        ax5.plot(time, inst_fluctuation, color='purple', linewidth=2)
        ax5.set_title('Fluctuation Strength', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax5.set_ylabel('Fluctuation, $FS$ (vacil)', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([0, xmax])
        
        if "FS" in out and "FS5" in out["FS"]:
            fs5 = np.asarray(out["FS"]["FS5"]).ravel()[0]
            ax5.axhline(y=fs5, color='r', linestyle='--', alpha=0.6, linewidth=1, label=f'5th %ile: {fs5:.2f}')
            ax5.legend(loc='best', fontsize=8)
    
    # Plot 6: Tonality component
    ax6 = fig.add_subplot(3, 3, 6)
    if len(time) > 0 and len(inst_tonality) > 0:
        ax6.plot(time, inst_tonality, color='brown', linewidth=2)
        ax6.set_title('Tonality', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Time, $t$ (s)', fontsize=10)
        ax6.set_ylabel('Tonality, $K$ (t.u.)', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim([0, xmax])
        
        if "K" in out and "K5" in out["K"]:
            k5 = np.asarray(out["K"]["K5"]).ravel()[0]
            ax6.axhline(y=k5, color='r', linestyle='--', alpha=0.6, linewidth=1, label=f'5th %ile: {k5:.2f}')
            ax6.legend(loc='best', fontsize=8)
    
    # Plot 7: Weights heatmap
    ax7 = fig.add_subplot(3, 3, 7)
    if len(time) > 0 and len(ws) > 0:
        weights_matrix = np.column_stack([ws, wfr, wt])
        im = ax7.imshow(weights_matrix.T, aspect='auto', cmap='YlOrRd', origin='lower')
        ax7.set_title('Component Weights', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Time step', fontsize=10)
        ax7.set_ylabel('Weight Type', fontsize=10)
        ax7.set_yticks([0, 1, 2])
        ax7.set_yticklabels(['$w_S$ (Sharp)', '$w_{FR}$ (Rough/Fluct)', '$w_T$ (Tonality)'], fontsize=9)
        fig.colorbar(im, ax=ax7, label='Weight magnitude')
    
    # Plot 8: Summary info
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.axis('off')
    info_text = "Annoyance Model Components\n\n"
    info_text += "Primary:   Loudness (N)\n"
    info_text += "Secondary: Sharpness (S)\n"
    info_text += "Secondary: Roughness (R)\n"
    info_text += "Secondary: Fluctuation (FS)\n"
    info_text += "Tertiary:  Tonality (K)\n\n"
    info_text += "PA = N × (1 + √(w_s² + w_fr² + w_t²))"
    
    ax8.text(0.1, 0.9, info_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Plot 9: Statistics
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    stats_text = "Annoyance Statistics\n\n"
    
    stat_keys = [
        ("PA5", "5th percentile", ""),
        ("PAmean", "Mean", ""),
        ("PA95", "95th percentile", ""),
        ("PAmax", "Maximum", ""),
        ("PAmin", "Minimum", ""),
        ("PAstd", "Std Dev", ""),
        ("ScalarPA", "Scalar (overall)", ""),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            if unit:
                stats_text += f"{label:.<20} {val:>9.2f} {unit}\n"
            else:
                stats_text += f"{label:.<20} {val:>9.2f}\n"
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.7))
    
    fig.tight_layout()
    return fig


def plot_epnl_analysis(out: Dict[str, Any]) -> Figure:
    """Generate comprehensive EPNL analysis plot matching metrics_epnl.py"""
    fig = Figure(figsize=(15, 8), dpi=100)
    
    time = np.asarray(out.get("time", [])).ravel()
    pnl = np.asarray(out.get("PNL", [])).ravel()
    pnlt = np.asarray(out.get("PNLT", [])).ravel()
    
    xmax = time[-1] if len(time) > 0 else 1
    
    # Plot 1: PNL and PNLT curves
    ax1 = fig.add_subplot(2, 2, 1)
    if len(time) > 0:
        if len(pnl) > 0:
            ax1.plot(time, pnl, 'b-', linewidth=2, label='PNL')
        if len(pnlt) > 0:
            ax1.plot(time, pnlt, 'r-', linewidth=2.5, label='PNLT (Tone-corrected)', alpha=0.8)
        
        ax1.set_title('EPNL (FAR Part 36)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax1.set_ylabel('Level, $L$ (PNdB)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, xmax])
        ax1.legend(loc='best', fontsize=10)
    
    # Plot 2: Tone correction effect
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis('off')
    info_text = "EPNL Certification\n\n"
    info_text += "Standard: FAR Part 36\n"
    info_text += "ICAO Annex 16\n\n"
    info_text += "Time window: ~10s\n"
    info_text += "around PNLTM peak"
    
    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot 3: Time-averaged 1/3-octave band SPL spectrum
    ax3 = fig.add_subplot(2, 2, 3)
    tob_f   = np.asarray(out.get("TOB_freq", [])).ravel()
    spl_tob = np.asarray(out.get("SPL_TOB_avg", [])).ravel()
    if len(tob_f) > 0 and len(spl_tob) > 0:
        n = min(len(tob_f), len(spl_tob))
        ax3.semilogx(tob_f[:n], spl_tob[:n], color='steelblue', marker='o',
                     markersize=3, linewidth=1.8)
        ax3.set_title('Time-averaged 1/3-octave band SPL', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Frequency, $f$ (Hz)', fontsize=11)
        ax3.set_ylabel('SPL, $L_p$ (dB)', fontsize=11)
        ax3.grid(True, alpha=0.3, which='both')
    else:
        ax3.axis('off')

    # Plot 4: Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = "EPNL - Statistics (FAR Part 36)\n\n"
    
    stat_keys = [
        ("EPNL", "Effective PNL", "EPNdB"),
        ("PNLM", "Max PNL", "PNdB"),
        ("PNLTM", "Max Tone-corrected", "PNdB"),
        ("ToneCorrectionMax", "Tone correction peak", "dB"),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<25} {val:>10.4f} {unit}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    fig.tight_layout()
    return fig


def create_metric_plot(metric: str, out: Dict[str, Any]) -> Optional[Figure]:
    """
    Dispatcher function to create appropriate plot for metric type
    
    Parameters
    ----------
    metric : str
        Metric name (e.g., "Loudness (ISO 532-1)")
    out : dict
        Output dictionary from metric function
    
    Returns
    -------
    Figure or None
        Matplotlib figure object, or None if no suitable plot
    """
    try:
        if "Loudness" in metric:
            return plot_loudness_analysis(out, metric)
        elif "Sharpness" in metric:
            return plot_sharpness_analysis(out)
        elif "Roughness" in metric:
            return plot_roughness_analysis(out)
        elif "Fluctuation" in metric:
            return plot_fluctuation_analysis(out)
        elif "Tonality" in metric:
            return plot_tonality_analysis(out)
        elif "Annoyance" in metric:
            return plot_annoyance_analysis(out)
        elif "EPNL" in metric:
            return plot_epnl_analysis(out)
    except Exception as e:
        print(f"Error creating plot for {metric}: {e}")

    return None


# ── "plot together" overlay configuration ─────────────────────────────────────
# Per metric, the list of 1-D curves that are *directly comparable* across sound
# files (they share a common, meaningful x-axis), so several files can be
# overlaid on the same axes. Each panel becomes one subplot in which every file
# is drawn as one curve.
#
# Deliberately EXCLUDED (overlaying multiple files is not meaningful / not
# directly comparable):
#   * 2-D spectrograms / heat-maps  (instantaneous specific loudness/roughness/…)
#   * raw input-signal waveforms and per-frame instantaneous spectra
# Each panel dict: y (data key), x ("time" or a data key), xlabel, ylabel,
# title, xlog.
def _panel(y, x, xlabel, ylabel, title, xlog=False):
    return dict(y=y, x=x, xlabel=xlabel, ylabel=ylabel, title=title, xlog=xlog)


_T = "Time, $t$ (s)"
_Z = "Critical band, $z$ (Bark)"

_OVERLAY_PANELS: Dict[str, list] = {
    "Loudness (ISO 532-1)": [
        _panel("InstantaneousLoudness", "time", _T, "Loudness, $N$ (sone)", "Instantaneous loudness"),
        _panel("SpecificLoudness", "barkAxis", _Z, "Specific loudness, $N'$ (sone/Bark)", "Time-avg specific loudness"),
    ],
    "Loudness (ECMA 418-2)": [
        _panel("InstantaneousLoudness", "time", _T, "Loudness, $N$ (sone)", "Instantaneous loudness"),
        _panel("SpecificLoudness", "barkAxis", _Z, "Specific loudness", "Time-avg specific loudness"),
    ],
    "Sharpness (DIN 45692)": [
        _panel("InstantaneousSharpness", "time", _T, "Sharpness, $S$ (acum)", "Instantaneous sharpness"),
    ],
    "Roughness (Daniel 1997)": [
        _panel("InstantaneousRoughness", "time", _T, "Roughness, $R$ (asper)", "Instantaneous roughness"),
        _panel("TimeAveragedSpecificRoughness", "barkAxis", _Z, "Specific roughness, $R'$ (asper/Bark)", "Time-avg specific roughness"),
    ],
    "Roughness (ECMA 418-2)": [
        _panel("InstantaneousRoughness", "time", _T, "Roughness, $R$ (asper)", "Instantaneous roughness"),
    ],
    "Fluctuation Strength (Osses 2016)": [
        _panel("InstantaneousFluctuationStrength", "time", _T, "Fluctuation strength, $FS$ (vacil)", "Instantaneous fluctuation strength"),
        _panel("TimeAveragedSpecificFluctuationStrength", "barkAxis", _Z, "Specific FS, $FS'$ (vacil/Bark)", "Time-avg specific fluctuation strength"),
    ],
    "Tonality (Aures 1985)": [
        _panel("InstantaneousTonality", "time", _T, "Tonality, $K$ (t.u.)", "Instantaneous tonality"),
        _panel("TonalWeighting", "time", _T, "Tonal weighting, $w_T$", "Tonal weighting"),
        _panel("LoudnessWeighting", "time", _T, "Loudness weighting, $w_{Gr}$", "Loudness weighting"),
    ],
    "Tonality (ECMA 418-2)": [
        _panel("InstantaneousTonality", "time", _T, "Tonality, $T$ (t.u.)", "Instantaneous tonality"),
    ],
    "EPNL (FAR Part 36)": [
        _panel("PNLT", "time", _T, "PNLT (TPNdB)", "Tone-corrected perceived noise level"),
        _panel("PNL", "time", _T, "PNL (PNdB)", "Perceived noise level"),
        _panel("SPL_TOB_avg", "TOB_freq", "Frequency, $f$ (Hz)", "SPL, $L_p$ (dB)", "Time-avg 1/3-octave SPL", xlog=True),
    ],
}
for _pa in ("Annoyance (Di 2016)", "Annoyance (Zwicker 1999)", "Annoyance (More 2010)"):
    _OVERLAY_PANELS[_pa] = [
        _panel("InstantaneousPA", "time", _T, "Annoyance, $PA$", "Instantaneous annoyance"),
    ]


def _overlay_xy(out: Dict[str, Any], panel: dict):
    """Return (x, y) arrays for one panel of one file, or (None, None)."""
    y = np.asarray(out.get(panel["y"], [])).ravel()
    xk = panel["x"]
    x = (np.asarray(out.get("time", [])).ravel() if xk == "time"
         else np.asarray(out.get(xk, [])).ravel())
    if x.size == 0 or y.size == 0:
        return None, None
    n = min(len(x), len(y))
    return x[:n], y[:n]


def create_metric_overlay_plot(metric: str,
                               results_by_label: Dict[str, Dict[str, Any]]) -> Optional[Figure]:
    """
    Overlay the results of several sound files for one metric on the same figure,
    for *every* output that is physically/mathematically comparable across files
    (time-series and time-averaged Bark / 1/3-octave profiles), with a
    descriptive legend (one entry per file).

    2-D spectrograms / heat-maps and raw waveforms are excluded, since overlaying
    multiple files for those is not meaningful.

    Parameters
    ----------
    metric : str
        Metric name (e.g. "Loudness (ISO 532-1)").
    results_by_label : dict
        Mapping ``{file_label: output_dict}`` produced by analysing each file.

    Returns
    -------
    Figure or None
        A figure with one overlaid subplot per comparable output, or ``None``
        when nothing comparable is available.
    """
    panels = None
    for name, pl in _OVERLAY_PANELS.items():
        if name == metric or name.split(" (")[0] in metric:
            panels = pl
            break
    if not panels:
        return None

    # Keep only panels with data in at least one file.
    usable = []
    for panel in panels:
        if any(isinstance(out, dict) and _overlay_xy(out, panel)[0] is not None
               for out in results_by_label.values()):
            usable.append(panel)
    if not usable:
        return None

    n      = len(usable)
    ncols  = 1 if n == 1 else 2
    nrows  = int(np.ceil(n / ncols))
    fig    = Figure(figsize=(7.0 * ncols, 4.2 * nrows + 0.3), dpi=100)
    cmap   = plt_get_cmap(len(results_by_label))

    for pi, panel in enumerate(usable):
        ax = fig.add_subplot(nrows, ncols, pi + 1)
        for i, (label, out) in enumerate(results_by_label.items()):
            if not isinstance(out, dict):
                continue
            x, y = _overlay_xy(out, panel)
            if x is None:
                continue
            ax.plot(x, y, linewidth=1.6, color=cmap[i % len(cmap)], label=label)
        if panel.get("xlog"):
            try:
                ax.set_xscale("log")
            except Exception:
                pass
        ax.set_title(panel["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel(panel["xlabel"], fontsize=10)
        ax.set_ylabel(panel["ylabel"], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.9, title="Sound file")

    fig.suptitle(f"{metric} — all sound files", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plt_get_cmap(n: int):
    """Return a list of ``n`` distinct colours for overlay curves."""
    import matplotlib.cm as cm
    base = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
            "#0891b2", "#be185d", "#65a30d", "#475569", "#ea580c"]
    if n <= len(base):
        return base[:max(n, 1)]
    colours = cm.get_cmap("tab20", n)
    return [colours(i) for i in range(n)]


def create_metric_plot_split(metric: str, out: Dict[str, Any]) -> Optional[Dict[str, Figure]]:
    """
    Create split component plots for a metric (each subplot as separate figure)
    
    Parameters
    ----------
    metric : str
        Metric name (e.g., "Loudness (ISO 532-1)")
    out : dict
        Output dictionary from metric function
    
    Returns
    -------
    dict or None
        Dictionary mapping component names to Figure objects, or None if splitting not available
    
    Examples
    --------
    >>> figures = create_metric_plot_split("Loudness (ISO 532-1)", loudness_output)
    >>> for name, fig in figures.items():
    ...     fig.savefig(f"{name}.png")
    """
    try:
        if "Loudness" in metric:
            return _split_loudness_components(out, metric)
        elif "Sharpness" in metric:
            return _split_sharpness_components(out)
        elif "Roughness" in metric:
            return _split_roughness_components(out)
        elif "Fluctuation" in metric:
            return _split_fluctuation_components(out)
        elif "Tonality" in metric:
            return _split_tonality_components(out)
        elif "Annoyance" in metric:
            return _split_annoyance_components(out)
        elif "EPNL" in metric:
            return _split_epnl_components(out)
    except Exception as e:
        print(f"Error creating split plots for {metric}: {e}")
    
    return None


def _split_loudness_components(out: Dict[str, Any], metric_name: str) -> Dict[str, Figure]:
    """Split loudness plot into individual component figures"""
    is_time_varying = "InstantaneousLoudness" in out and "time" in out
    components = {}
    
    if is_time_varying:
        # Time-varying loudness: create 6 separate figures
        
        # 1. SPL
        fig1 = Figure(figsize=(10, 5), dpi=100)
        ax = fig1.add_subplot(1, 1, 1)
        time_insig = np.asarray(out.get("time_insig", [])).ravel()
        insig = np.asarray(out.get("InstantaneousSPL", [])).ravel()
        if len(time_insig) > 0 and len(insig) > 0:
            ax.plot(np.linspace(0, time_insig[-1], len(insig)), insig, 'b-', linewidth=1.5)
            ax.set_title('Instantaneous overall SPL (1/3 octave)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time, $t$ (s)', fontsize=11)
            ax.set_ylabel('SPL, $L_p$ (dB re 20 μPa)', fontsize=11)
            ax.grid(True, alpha=0.3)
        fig1.tight_layout()
        components['SPL'] = fig1
        
        # 2. Loudness Level
        fig2 = Figure(figsize=(10, 5), dpi=100)
        ax = fig2.add_subplot(1, 1, 1)
        time = np.asarray(out.get("time", [])).ravel()
        inst_loudness_level = np.asarray(out.get("InstantaneousLoudnessLevel", [])).ravel()
        if len(time) > 0 and len(inst_loudness_level) > 0:
            ax.plot(time, np.abs(inst_loudness_level), 'g-', linewidth=1.5)
            ax.set_title('Instantaneous loudness level', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time, $t$ (s)', fontsize=11)
            ax.set_ylabel('Loudness level, $L_N$ (phon)', fontsize=11)
            ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        components['Level'] = fig2
        
        # 3. Loudness (sone)
        fig3 = Figure(figsize=(10, 5), dpi=100)
        ax = fig3.add_subplot(1, 1, 1)
        inst_loudness = np.asarray(out.get("InstantaneousLoudness", [])).ravel()
        if len(time) > 0 and len(inst_loudness) > 0:
            ax.plot(time, inst_loudness, 'r-', linewidth=2)
            ax.set_title('Instantaneous loudness', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time, $t$ (s)', fontsize=11)
            ax.set_ylabel('Loudness, $N$ (sone)', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            if "Nmean" in out:
                nmean = np.asarray(out["Nmean"]).ravel()[0]
                ax.axhline(y=nmean, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mean: {nmean:.3f}')
            if "N5" in out:
                n5 = np.asarray(out["N5"]).ravel()[0]
                ax.axhline(y=n5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'5th: {n5:.3f}')
            ax.legend(loc='best', fontsize=10)
        fig3.tight_layout()
        components['Loudness'] = fig3
        
        # 4. Specific loudness profile
        fig4 = Figure(figsize=(10, 5), dpi=100)
        ax = fig4.add_subplot(1, 1, 1)
        spec_loudness = np.asarray(out.get("SpecificLoudness", [])).ravel()
        bark_axis = np.asarray(out.get("barkAxis", [])).ravel()
        if len(bark_axis) > 0 and len(spec_loudness) > 0:
            ax.plot(bark_axis, spec_loudness, 'purple', linewidth=2)
            ax.set_title('Time-averaged specific loudness', fontsize=12, fontweight='bold')
            ax.set_xlabel('Critical band, $z$ (Bark)', fontsize=11)
            ax.set_ylabel("Specific loudness, $N'$ (sone/Bark)", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 24])
        fig4.tight_layout()
        components['SpecificLoudness'] = fig4
        
        # 5. Specific loudness heatmap
        fig5 = Figure(figsize=(12, 5), dpi=100)
        ax = fig5.add_subplot(1, 1, 1)
        inst_spec_loudness = np.asarray(out.get("InstantaneousSpecificLoudness", []))
        if inst_spec_loudness.size > 0:
            im = ax.imshow(inst_spec_loudness, aspect='auto', cmap='viridis', origin='lower')
            ax.set_title('Instantaneous specific loudness (heatmap)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=11)
            ax.set_ylabel('Critical band, $z$ (Bark)', fontsize=11)
            cbar = fig5.colorbar(im, ax=ax)
            cbar.set_label("$N'$ (sone/Bark)", fontsize=11)
        fig5.tight_layout()
        components['Heatmap'] = fig5
        
        # 6. Statistics
        fig6 = Figure(figsize=(10, 5), dpi=100)
        ax = fig6.add_subplot(1, 1, 1)
        ax.axis('off')
        stats_text = f"{metric_name} - Statistics\n\n"
        
        stat_keys = [
            ("Nmean", "Mean loudness", "sone"),
            ("N5", "5th percentile", "sone"),
            ("N10", "10th percentile", "sone"),
            ("N95", "95th percentile", "sone"),
            ("Nmax", "Maximum", "sone"),
        ]
        
        for key, label, unit in stat_keys:
            if key in out:
                val = np.asarray(out[key]).ravel()[0]
                stats_text += f"{label:.<30} {val:>10.4f} {unit}\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        fig6.tight_layout()
        components['Statistics'] = fig6
    else:
        # Stationary loudness: simpler splitting
        fig1 = Figure(figsize=(10, 5), dpi=100)
        ax = fig1.add_subplot(1, 1, 1)
        spec_loudness = np.asarray(out.get("SpecificLoudness", [])).ravel()
        bark_axis = np.asarray(out.get("barkAxis", [])).ravel()
        if len(bark_axis) > 0 and len(spec_loudness) > 0:
            ax.plot(bark_axis, spec_loudness, 'purple', linewidth=2)
            ax.set_title('Specific loudness', fontsize=12, fontweight='bold')
            ax.set_xlabel('Critical band, $z$ (Bark)', fontsize=11)
            ax.set_ylabel("Specific loudness, $N'$ (sone/Bark)", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 24])
        fig1.tight_layout()
        components['SpecificLoudness'] = fig1
        
        fig2 = Figure(figsize=(10, 5), dpi=100)
        ax = fig2.add_subplot(1, 1, 1)
        ax.axis('off')
        stats_text = f"{metric_name} - Statistics\n\n"
        
        stat_keys = [
            ("Nmean", "Loudness", "sone"),
            ("N5", "5th percentile", "sone"),
            ("N10", "10th percentile", "sone"),
            ("N95", "95th percentile", "sone"),
            ("Nmax", "Maximum", "sone"),
        ]
        
        for key, label, unit in stat_keys:
            if key in out:
                val = np.asarray(out[key]).ravel()[0]
                stats_text += f"{label:.<30} {val:>10.4f} {unit}\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        fig2.tight_layout()
        components['Statistics'] = fig2
    
    return components


def _split_sharpness_components(out: Dict[str, Any]) -> Dict[str, Figure]:
    """Split sharpness plot into component figures"""
    components = {}
    time = np.asarray(out.get("time", [])).ravel()
    inst_sharpness = np.asarray(out.get("InstantaneousSharpness", [])).ravel()
    
    # Sharpness over time
    fig1 = Figure(figsize=(10, 5), dpi=100)
    ax = fig1.add_subplot(1, 1, 1)
    if len(time) > 0 and len(inst_sharpness) > 0:
        ax.plot(time, inst_sharpness, 'b-', linewidth=2)
        ax.set_title('Instantaneous Sharpness (DIN 45692)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax.set_ylabel('Sharpness, $S$ (acum)', fontsize=11)
        ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    components['Sharpness'] = fig1
    
    # Statistics
    fig2 = Figure(figsize=(10, 5), dpi=100)
    ax = fig2.add_subplot(1, 1, 1)
    ax.axis('off')
    stats_text = "Sharpness (DIN 45692) - Statistics\n\n"
    
    stat_keys = [
        ("Smean", "Mean sharpness", "acum"),
        ("Smax", "Maximum", "acum"),
        ("S5", "5th percentile", "acum"),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<30} {val:>10.4f} {unit}\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    fig2.tight_layout()
    components['Statistics'] = fig2
    
    return components


def _split_roughness_components(out: Dict[str, Any]) -> Dict[str, Figure]:
    """Split roughness plot into component figures"""
    components = {}
    time = np.asarray(out.get("time", [])).ravel()
    inst_roughness = np.asarray(out.get("InstantaneousRoughness", [])).ravel()
    
    # Roughness over time
    fig1 = Figure(figsize=(10, 5), dpi=100)
    ax = fig1.add_subplot(1, 1, 1)
    if len(time) > 0 and len(inst_roughness) > 0:
        ax.plot(time, inst_roughness, 'b-', linewidth=2)
        ax.set_title('Instantaneous Roughness', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax.set_ylabel('Roughness, $R$ (asper)', fontsize=11)
        ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    components['Roughness'] = fig1
    
    # Statistics
    fig2 = Figure(figsize=(10, 5), dpi=100)
    ax = fig2.add_subplot(1, 1, 1)
    ax.axis('off')
    stats_text = "Roughness - Statistics\n\n"
    
    stat_keys = [
        ("Rmean", "Mean roughness", "asper"),
        ("Rmax", "Maximum", "asper"),
        ("R5", "5th percentile", "asper"),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<30} {val:>10.4f} {unit}\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    fig2.tight_layout()
    components['Statistics'] = fig2
    
    return components


def _split_fluctuation_components(out: Dict[str, Any]) -> Dict[str, Figure]:
    """Split fluctuation strength plot into component figures"""
    components = {}
    time = np.asarray(out.get("time", [])).ravel()
    inst_fluctuation = np.asarray(out.get("InstantaneousFluctuation", [])).ravel()
    
    # Fluctuation strength over time
    fig1 = Figure(figsize=(10, 5), dpi=100)
    ax = fig1.add_subplot(1, 1, 1)
    if len(time) > 0 and len(inst_fluctuation) > 0:
        ax.plot(time, inst_fluctuation, 'b-', linewidth=2)
        ax.set_title('Instantaneous Fluctuation Strength', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax.set_ylabel('Fluctuation Strength, $F$ (vacil)', fontsize=11)
        ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    components['Fluctuation'] = fig1
    
    # Statistics
    fig2 = Figure(figsize=(10, 5), dpi=100)
    ax = fig2.add_subplot(1, 1, 1)
    ax.axis('off')
    stats_text = "Fluctuation Strength - Statistics\n\n"
    
    stat_keys = [
        ("Fmean", "Mean fluctuation", "vacil"),
        ("Fmax", "Maximum", "vacil"),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<30} {val:>10.4f} {unit}\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    fig2.tight_layout()
    components['Statistics'] = fig2
    
    return components


def _split_tonality_components(out: Dict[str, Any]) -> Dict[str, Figure]:
    """Split tonality plot into component figures"""
    components = {}
    time = np.asarray(out.get("time", [])).ravel()
    inst_tonality = np.asarray(out.get("InstantaneousTonality", [])).ravel()
    
    # Tonality over time
    fig1 = Figure(figsize=(10, 5), dpi=100)
    ax = fig1.add_subplot(1, 1, 1)
    if len(time) > 0 and len(inst_tonality) > 0:
        ax.plot(time, inst_tonality, 'b-', linewidth=2)
        ax.set_title('Instantaneous Tonality', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax.set_ylabel('Tonality, $T$ (t.u.)', fontsize=11)
        ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    components['Tonality'] = fig1
    
    # Statistics
    fig2 = Figure(figsize=(10, 5), dpi=100)
    ax = fig2.add_subplot(1, 1, 1)
    ax.axis('off')
    stats_text = "Tonality - Statistics\n\n"
    
    stat_keys = [
        ("Tmean", "Mean tonality", "t.u."),
        ("Tmax", "Maximum", "t.u."),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<30} {val:>10.4f} {unit}\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.7))
    fig2.tight_layout()
    components['Statistics'] = fig2
    
    return components


def _split_annoyance_components(out: Dict[str, Any]) -> Dict[str, Figure]:
    """Split annoyance plot into detailed component figures with all metrics"""
    components = {}
    time = np.asarray(out.get("time", [])).ravel()
    inst_annoyance = np.asarray(out.get("InstantaneousPA", [])).ravel()
    
    # Component signals
    inst_loudness = np.asarray(out.get("L", {}).get("InstantaneousLoudness", [])).ravel()
    inst_sharpness = np.asarray(out.get("S", {}).get("InstantaneousSharpness", [])).ravel()
    inst_roughness = np.asarray(out.get("R", {}).get("InstantaneousRoughness", [])).ravel()
    inst_fluctuation = np.asarray(out.get("FS", {}).get("InstantaneousFluctuationStrength", [])).ravel()
    inst_tonality = np.asarray(out.get("K", {}).get("InstantaneousTonality", [])).ravel()
    
    xmax = time[-1] if len(time) > 0 else 1
    
    # Figure 1: Instantaneous annoyance over time
    fig1 = Figure(figsize=(12, 5), dpi=100)
    ax = fig1.add_subplot(1, 1, 1)
    if len(time) > 0 and len(inst_annoyance) > 0:
        ax.plot(time, inst_annoyance, 'k-', linewidth=2.5)
        ax.fill_between(time, inst_annoyance, alpha=0.3)
        ax.set_title('Instantaneous Psychoacoustic Annoyance (PA)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax.set_ylabel('Annoyance, $PA$', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, xmax])
        
        if "PAmean" in out:
            pmean = np.asarray(out["PAmean"]).ravel()[0]
            ax.axhline(y=pmean, color='r', linestyle='--', alpha=0.6, linewidth=2, label=f'Mean: {pmean:.2f}')
            ax.legend(loc='best', fontsize=10)
    fig1.tight_layout()
    components['01_Annoyance_Timeseries'] = fig1
    
    # Figure 2: All component metrics on time axis
    fig2 = Figure(figsize=(14, 8), dpi=100)
    
    ax1 = fig2.add_subplot(3, 2, 1)
    if len(time) > 0 and len(inst_loudness) > 0:
        ax1.plot(time, inst_loudness, 'b-', linewidth=2)
        ax1.set_title('Loudness (Primary)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('N (sone)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, xmax])
    
    ax2 = fig2.add_subplot(3, 2, 2)
    if len(time) > 0 and len(inst_sharpness) > 0:
        ax2.plot(time, inst_sharpness, 'g-', linewidth=2)
        ax2.set_title('Sharpness (Secondary)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('S (acum)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, xmax])
    
    ax3 = fig2.add_subplot(3, 2, 3)
    if len(time) > 0 and len(inst_roughness) > 0:
        ax3.plot(time, inst_roughness, color='orange', linewidth=2)
        ax3.set_title('Roughness', fontsize=11, fontweight='bold')
        ax3.set_ylabel('R (asper)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, xmax])
    
    ax4 = fig2.add_subplot(3, 2, 4)
    if len(time) > 0 and len(inst_fluctuation) > 0:
        ax4.plot(time, inst_fluctuation, color='purple', linewidth=2)
        ax4.set_title('Fluctuation Strength', fontsize=11, fontweight='bold')
        ax4.set_ylabel('FS (vacil)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, xmax])
    
    ax5 = fig2.add_subplot(3, 2, 5)
    if len(time) > 0 and len(inst_tonality) > 0:
        ax5.plot(time, inst_tonality, color='brown', linewidth=2)
        ax5.set_title('Tonality', fontsize=11, fontweight='bold')
        ax5.set_ylabel('K (t.u.)', fontsize=10)
        ax5.set_xlabel('Time (s)', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([0, xmax])
    
    ax6 = fig2.add_subplot(3, 2, 6)
    if len(time) > 0 and len(inst_annoyance) > 0:
        ax6.plot(time, inst_annoyance, 'k-', linewidth=2)
        ax6.fill_between(time, inst_annoyance, alpha=0.2)
        ax6.set_title('Resulting Annoyance', fontsize=11, fontweight='bold')
        ax6.set_ylabel('PA', fontsize=10)
        ax6.set_xlabel('Time (s)', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim([0, xmax])
    
    fig2.tight_layout()
    components['02_Components_Timeseries'] = fig2
    
    # Figure 3: Component weights heatmap
    fig3 = Figure(figsize=(12, 6), dpi=100)
    ax = fig3.add_subplot(1, 1, 1)
    
    ws = np.asarray(out.get("ws", [])).ravel()
    wfr = np.asarray(out.get("wfr", [])).ravel()
    wt = np.asarray(out.get("wt", [])).ravel()
    
    if len(time) > 0 and len(ws) > 0:
        weights_matrix = np.column_stack([ws, wfr, wt])
        im = ax.imshow(weights_matrix.T, aspect='auto', cmap='YlOrRd', origin='lower', extent=[0, xmax, -0.5, 2.5])
        ax.set_title('Component Weights Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax.set_ylabel('Weight Type', fontsize=11)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['$w_S$ (Sharpness)', '$w_{FR}$ (Roughness/Fluctuation)', '$w_T$ (Tonality)'], fontsize=10)
        cbar = fig3.colorbar(im, ax=ax, label='Weight magnitude')
        cbar.ax.tick_params(labelsize=9)
    
    fig3.tight_layout()
    components['03_Weights_Heatmap'] = fig3
    
    # Figure 4: Statistics summary
    fig4 = Figure(figsize=(12, 6), dpi=100)
    ax = fig4.add_subplot(1, 1, 1)
    ax.axis('off')
    
    stats_text = "Annoyance Statistics Summary\n"
    stats_text += "="*50 + "\n\n"
    
    stats_text += "PRIMARY ANNOYANCE METRICS:\n"
    stats_text += "-"*50 + "\n"
    stat_keys_main = [
        ("PA5", "5th percentile", ""),
        ("PAmean", "Mean", ""),
        ("PA95", "95th percentile", ""),
        ("PAmax", "Maximum", ""),
        ("PAmin", "Minimum", ""),
        ("PAstd", "Std Dev", ""),
        ("ScalarPA", "Scalar (overall)", ""),
    ]
    
    for key, label, unit in stat_keys_main:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            if unit:
                stats_text += f"  {label:.<30} {val:>10.3f} {unit}\n"
            else:
                stats_text += f"  {label:.<30} {val:>10.3f}\n"
    
    stats_text += "\nCOMPONENT PERCENTILES (5th):\n"
    stats_text += "-"*50 + "\n"
    component_stats = [
        ("L", "N5", "Loudness", "sone"),
        ("S", "S5", "Sharpness", "acum"),
        ("R", "R5", "Roughness", "asper"),
        ("FS", "FS5", "Fluctuation", "vacil"),
        ("K", "K5", "Tonality", "t.u."),
    ]
    
    for subdict_key, stat_key, label, unit in component_stats:
        if subdict_key in out and stat_key in out[subdict_key]:
            val = np.asarray(out[subdict_key][stat_key]).ravel()[0]
            stats_text += f"  {label:.<30} {val:>10.3f} {unit}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    fig4.tight_layout()
    components['04_Statistics'] = fig4
    
    # Figure 5: Comparison of percentiles across components
    fig5 = Figure(figsize=(12, 6), dpi=100)
    ax = fig5.add_subplot(1, 1, 1)
    
    components_list = ['Loudness', 'Sharpness', 'Roughness', 'Fluctuation', 'Tonality', 'Annoyance']
    percentiles_5 = []
    percentiles_95 = []
    labels_display = ['N', 'S', 'R', 'FS', 'K', 'PA']
    units_list = ['sone', 'acum', 'asper', 'vacil', 't.u.', '-']
    
    # Collect data
    if "L" in out and "N5" in out["L"]:
        percentiles_5.append(np.asarray(out["L"]["N5"]).ravel()[0])
        percentiles_95.append(np.asarray(out["L"]["N95"]).ravel()[0])
    
    if "S" in out and "S5" in out["S"]:
        percentiles_5.append(np.asarray(out["S"]["S5"]).ravel()[0])
        percentiles_95.append(np.asarray(out["S"]["S95"]).ravel()[0])
    
    if "R" in out and "R5" in out["R"]:
        percentiles_5.append(np.asarray(out["R"]["R5"]).ravel()[0])
        percentiles_95.append(np.asarray(out["R"]["R95"]).ravel()[0])
    
    if "FS" in out and "FS5" in out["FS"]:
        percentiles_5.append(np.asarray(out["FS"]["FS5"]).ravel()[0])
        percentiles_95.append(np.asarray(out["FS"]["FS95"]).ravel()[0])
    
    if "K" in out and "K5" in out["K"]:
        percentiles_5.append(np.asarray(out["K"]["K5"]).ravel()[0])
        percentiles_95.append(np.asarray(out["K"]["K95"]).ravel()[0])
    
    if "PA5" in out:
        percentiles_5.append(np.asarray(out["PA5"]).ravel()[0])
        percentiles_95.append(np.asarray(out["PA95"]).ravel()[0])
    
    if len(percentiles_5) > 0:
        x_pos = np.arange(len(percentiles_5))
        width = 0.35
        
        ax.bar(x_pos - width/2, percentiles_5, width, label='5th percentile', color='skyblue')
        ax.bar(x_pos + width/2, percentiles_95, width, label='95th percentile', color='salmon')
        
        ax.set_xlabel('Component Metric', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('Percentile Distribution Across All Components', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_display, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig5.tight_layout()
    components['05_Percentiles_Comparison'] = fig5
    
    return components


def _split_epnl_components(out: Dict[str, Any]) -> Dict[str, Figure]:
    """Split EPNL plot into component figures"""
    components = {}
    time = np.asarray(out.get("time", [])).ravel()
    pnl = np.asarray(out.get("PNL", [])).ravel()
    pnlt = np.asarray(out.get("PNLT", [])).ravel()
    
    # PNL and PNLT curves
    fig1 = Figure(figsize=(10, 5), dpi=100)
    ax = fig1.add_subplot(1, 1, 1)
    if len(time) > 0:
        if len(pnl) > 0:
            ax.plot(time, pnl, 'b-', linewidth=2, label='PNL')
        if len(pnlt) > 0:
            ax.plot(time, pnlt, 'r-', linewidth=2, label='PNLT (Tone-corrected)')
        
        ax.set_title('EPNL (FAR Part 36)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time, $t$ (s)', fontsize=11)
        ax.set_ylabel('Level, $L$ (PNdB)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    fig1.tight_layout()
    components['PNL'] = fig1
    
    # Statistics
    fig2 = Figure(figsize=(10, 5), dpi=100)
    ax = fig2.add_subplot(1, 1, 1)
    ax.axis('off')
    stats_text = "EPNL - Statistics\n\n"
    
    stat_keys = [
        ("EPNL", "EPNL", "EPNdB"),
        ("PNLM", "Max PNL", "PNdB"),
        ("PNLTM", "Max PNLT (Tone-corrected)", "PNdB"),
    ]
    
    for key, label, unit in stat_keys:
        if key in out:
            val = np.asarray(out[key]).ravel()[0]
            stats_text += f"{label:.<30} {val:>10.4f} {unit}\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    fig2.tight_layout()
    components['Statistics'] = fig2
    
    return components
