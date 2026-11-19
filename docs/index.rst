DimSpin documentation
=====================

DimSpin is an EPR/ESR spectrum simulator for spin Hamiltonians and
spin-electric coupling (SEC) tensors, with magnetic-field, electric-field and
parameter modulation, single-crystal rotation, powder averaging and
space-group expansion.

Installation
------------

.. code-block:: bash

   pip install DimSpin

For plotting support install the ``plot`` extra:

.. code-block:: bash

   pip install "DimSpin[plot]"

Quick start
-----------

.. code-block:: python

   from dimspin import Spec, Spin, autosim

   spin = Spin(1/2, 0)
   spin.g = [2.009, 2.006, 2.002]

   spec = Spec()
   spec.mwfreq = 9560.0
   spec.fieldrange = [3300.0, 3600.0]
   spec.specpoints = 800
   spec.lw_lorentzian = 0.5
   spec.Bmod = 1.0

   spectrum = autosim(spin, spec)

Citation
--------

If you use DimSpin in your work, please cite the associated article
(see ``CITATION.cff`` in the repository). The article is currently in
preparation; until it is published, cite the archived software release.
The concept DOI for all versions is doi:10.5281/zenodo.24358712; the 0.1.0
release has doi:10.5281/zenodo.24358713.

Contents
--------

.. toctree::
   :maxdepth: 2

   api
   theory
   examples
