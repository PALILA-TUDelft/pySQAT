Tonality Metrics
================

This module implements psycho-acoustic **tonality** measures.  
At present it contains an implementation of the classic Aures (1985)
algorithm, which quantifies the tonal prominence of stationary or
time-varying sounds in “tonality units” (t.u.).

----------
Overview
----------

Summary of the public functions and their purposes.

* **Tonality_Aures1985** (``Tonality_Aures1985``)  
  Extracts tonal components with Terhardt’s peak-picking rules,
  applies loudness and tonal prominence weightings, and returns the
  instantaneous as well as summary statistics of tonality.

----------
Functions
----------

Individual functions are documented in detail below.  Their full
signatures and parameters are pulled automatically from the source
code.

.. automodule:: metrics_tonality
   :members:
   :undoc-members:
   :show-inheritance:
