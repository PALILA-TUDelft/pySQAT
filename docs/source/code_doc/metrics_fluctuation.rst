Fluctuation Strength Metrics
===========================

This module provides an implementation of the psycho-acoustic
**fluctuation strength** model described by Osses et al. (2016).  
Fluctuation strength quantifies slow amplitude modulations
(*≈ 0.5 – 20 Hz*) that give sounds a “wobbling” character; its unit is
the *vacil* (1 vacil ≈ the sensation produced by a 1 kHz, 60 dB SPL tone
modulated 100 % at 4 Hz).

----------
Overview
----------

Summary of the public functions and their purposes.

* **FluctuationStrength_Osses2016** (``FluctuationStrength_Osses2016``)  
  Computes stationary or time-varying fluctuation strength from a raw
  audio signal or returns values from a pre-computed specific-loudness
  matrix.  Intermediate stages (specific fluctuation, modulation depth,
  cross-channel correlation) and statistical descriptors are also
  provided.

----------
Functions
----------

Individual functions are documented in detail below.  Their full
signatures and parameters are pulled automatically from the source
code.

.. automodule:: metrics_fluctuation
   :members:
   :undoc-members:
   :show-inheritance:
