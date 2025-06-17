Roughness Metrics
=================

This module implements psycho-acoustic **roughness** following Daniel &
Weber (1997).  
Roughness captures the sensation of rapid amplitude fluctuations
(modulation frequencies ≈ 15–300 Hz) and is expressed in
*asper* (1 asper ≙ the roughness of a 1 kHz, 60 dB SPL tone that is 100 %
amplitude-modulated at 70 Hz).

----------
Overview
----------

Summary of the public functions and their purposes.

* **Roughness_Daniel1997** (``Roughness_Daniel1997``)  
  Computes stationary or time-varying roughness from an audio signal
  (WAV file or NumPy array).  
  The algorithm builds critical-band excitation patterns, applies
  modulation depth and inter-channel correlation weightings, and
  returns the instantaneous as well as statistical roughness measures.

----------
Functions
----------

Individual functions are documented in detail below.  Their full
signatures and parameters are pulled automatically from the source
code.

.. automodule:: metrics_roughness
   :members:
   :undoc-members:
   :show-inheritance:
