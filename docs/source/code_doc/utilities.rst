Utilities
==================

This module provides foundational utilities for psychoacoustic signal processing,
analysis, and modeling. It supports perceptual metric evaluation, signal calibration,
and data handling tasks compatible with ISO and DIN standards. The functions
are modular and reusable across psychoacoustic research, audio model development,
and noise analysis.

----------
Overview
----------
Summary of functions and their purposes.

* Audio Visualization (``see``):
   Plots waveform and spectrogram of a WAV file, converting stereo to mono and applying a logarithmic frequency scale.

* Frequency Scale Conversions (``hz2bark``, ``bark2hz``):
   Converts between Hertz and the Bark psychoacoustic scale using analytical and interpolated methods.

* Loudness Scale Conversions (``phon2sone``, ``sone2phon``):
   Translates perceived loudness between phon and sone scales using Stevens’ power law and its inverse.

* Percentile Statistics (``get_exceeded_value``):
   Computes the value in a dataset that is exceeded by a specified percentage of samples.

* Psychoacoustic Metric Statistics (``get_statistics``):
   Generates summary statistics (mean, std, percentiles) for common psychoacoustic metrics using a configurable naming scheme.

* Bark Band Mapping (``get_bark``):
   Maps frequency bins to Bark scale values using standard critical-band definitions and interpolation.

* Decibel Conversion (``from_db``):
   Converts dB values to linear scale (voltage or power), with configurable divisor.

* FIR Filter Design (``create_a0_FIR``):
   Designs a custom FIR filter from frequency–gain pairs for psychoacoustic weighting, with optional plotting.

* a0 Weighting Filter Generator (``calculate_a0``):
   Interpolates psychoacoustic a₀ weighting curves (Fastl 2007 or Osses 2016) and returns the corresponding FIR filter.

* Signal Calibration (``calibrate``):
   Calibrates an input signal using a reference signal and dB SPL level.

* Model Configuration Defaults (``get_defaults``):
   Provides default parameter dictionaries for supported psychoacoustic models.

* Excel Export Utility (``export_dict_to_excel``):
   Exports a dictionary of arrays, scalars, or lists to a multi-sheet Excel file.

* WAV-to-Signal Converter (``wav2sig``):
   Reads a WAV file, converts to mono, applies dBFS scaling, and returns the signal and sampling rate.

* Signal Buffering (``buffer``):
   Segments a signal into overlapping or non-overlapping frames for windowed analysis.

* Cosine Envelope Generator (``cos_ramp``):
   Creates a cosine ramp (attack/release envelope) for signal windowing, with optional plotting.

* RMS Level Calculator (``rmsdb``):
   Computes the RMS level of a signal segment in decibels, supporting full-array, partial, or file input.

----------
Functions
----------
Detailed decsription of the functions, including I/O parameters and examples of usage.

.. automodule:: utilities
   :members:
   :undoc-members:
   :show-inheritance: