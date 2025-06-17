Psychoacoustic Annoyance Metrics
================================

This module gathers three implementations of *psychoacoustic annoyance*
(PA) models that combine several elementary metrics (loudness, sharpness,
roughness, fluctuation strength and, when relevant, tonality) to yield a
single descriptor of perceived annoyance.

----------
Overview
----------

Summary of the public functions and their purposes.

* **Di et al. (2016) model** – ``PsychoacousticAnnoyance_Di2016``  
  Five-factor PA formulation that adds a dedicated tonality term to
  Zwicker’s original framework.

* **Zwicker & Fastl (1999) model** – ``PsychoacousticAnnoyance_Zwicker1999``  
  Canonical four-factor PA model (loudness + sharpness + roughness +
  fluctuation strength).

* **More (2010) aircraft-noise model** – ``PsychoacousticAnnoyance_More2010``  
  Aircraft-tailored revision of Zwicker’s model including an additional
  tonality weighting and modified coefficient set.

----------
Functions
----------

Individual functions are documented in detail below.  Their full
signatures and parameters are pulled automatically from the source
code.

.. automodule:: metrics_annoyance
   :members:
   :undoc-members:
   :show-inheritance:
