Loudness Metrics
================

This module bundles two industry-standard metrics used to quantify
the perceived loudness of airborne sound:

* **ISO 532-1 Zwicker loudness** for general psycho-acoustics.
* **Effective Perceived Noise Level (EPNL)** for aircraft-noise
  certification according to FAR Part 36 and ICAO Annex 16.

----------
Overview
----------

Summary of the public functions and their purposes.

* **ISO532-1 Loudness** (``Loudness_ISO532_1``)  
  Calculates stationary or time-varying loudness, loudness level,
  and specific loudness following ISO 532-1 (2017).

* **EPNL FAR Part36** (``EPNL_FAR_Part36``)  
  Derives the Effective Perceived Noise Level of an aircraft pass-by,
  including tone-correction, duration-correction, and all intermediate
  metrics (PN, PNL, PNLT).

----------
Functions
----------

Individual functions are documented in detail below.  Their full
signatures and parameters are pulled automatically from the source
code.

.. automodule:: metrics_loudness
   :members:
   :undoc-members:
   :show-inheritance: