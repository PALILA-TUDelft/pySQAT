Sharpness Metrics
=================

This module implements psycho-acoustic **sharpness** measures in
accordance with DIN 45692.  
Sharpness reflects the spectral balance of a sound: high-frequency
components contribute more than low-frequency ones, so whistles and
squeaks yield larger sharpness values (in *acum*) than rumbles of the
same loudness.

----------
Overview
----------

Summary of the public functions and their purposes.

* **Sharpness_DIN45692** (``Sharpness_DIN45692``)  
  Calculates stationary or time-varying sharpness using the DIN weighting
  curve, or the alternative Aures and von Bismarck weightings.  
  The routine can either accept a raw audio signal (internally calling
  the ISO 532-1 loudness model) **or** operate directly on a
  pre-computed specific-loudness matrix.

----------
Functions
----------

Individual functions are documented in detail below.  Their full
signatures and parameters are pulled automatically from the source
code.

.. automodule:: metrics_sharpness
   :members:
   :undoc-members:
   :show-inheritance: