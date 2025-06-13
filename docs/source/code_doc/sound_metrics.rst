Sound Metrics
=============

This module implements core audio signal processing routines for objective sound analysis,
compatible with audio research, psychoacoustics, and environmental noise studies.
It contains standalone, high-level functions for filtering, level computation,
and metrics based on international standards (such as ISO 532-1 and IEC 61672).

----------
Overview
----------
Summary of functions and their purposes.

* Auditory Filterbank (``ob13_iso532_1``):
   Implements the ISO 532-1 standard for loudness by filtering audio through a bank of cascaded IIR filters (third-octave bands, 25 Hz–12.5 kHz). Automatically resamples input audio to 48 kHz if needed and outputs filtered signals per frequency band and corresponding center frequencies.

* Weighting Filter Design (``gen_weighting_filters``):
   Generates digital filter coefficients for standard audio frequency weightings (A, B, C, D, R, Z), based on IEC 61672-1:2013. Supports plotting frequency responses for verification.

* Sound Level Meter (``do_slm``):
   Simulates a compliant sound level meter: applies frequency weighting, time integration (fast, slow, impulse), calibrates to physical SPL units, and outputs a dB SPL time series. Optionally plots the resulting level trace.

* Equivalent Sound Level Calculation (``get_leq``):
   Computes Leq (equivalent continuous sound level) either for a whole signal or as a running (windowed) time series, supporting flexible window/hop durations.

----------
Functions
----------
Detailed decsription of the functions, including I/O parameters and examples of usage.

.. automodule:: sound_metrics
   :members:
   :undoc-members:
   :show-inheritance: