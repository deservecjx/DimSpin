Theory and conventions
======================

Spin Hamiltonian
----------------

The simulated Hamiltonian is

.. math::

   H = \mu_B\, \mathbf{B}\cdot\mathbf{g}\cdot\mathbf{S}
     + \mathbf{S}\cdot\mathbf{D}\cdot\mathbf{S}
     + \mathbf{S}\cdot\mathbf{A}\cdot\mathbf{I}
     + \sum_{k,q} B_k^q O_k^q(\mathbf{S})
     + H_\mathrm{mod},

where :math:`\mu_B` is expressed in frequency units, :math:`O_k^q` are the
extended Stevens operators for k = 2, 4, 6 and :math:`H_\mathrm{mod}` collects
the modulation terms:

- magnetic-field modulation: the field is shifted by the modulation amplitude
  with a positive or negative sign;
- electric-field modulation: the SEC tensors T, R and F couple the electric
  field to the Zeeman, zero-field splitting and hyperfine terms;
- parameter modulation: the modulation term is built directly from the
  amplitudes :math:`\Delta g`, :math:`\Delta D` and :math:`\Delta A`. Our
  apparatus keeps the electric field at a fixed orientation relative to the
  crystal, so the modulated term is then constant in the crystal frame and
  can be given directly, without going through the SEC tensors. The same
  construction could also describe other perturbations, such as pressure.

Units and conventions
---------------------

- Magnetic field in gauss (G), energy and frequency in MHz, temperature in K.
- Euler angles and rotation matrices use radians; crystal-rotation ranges are
  in degrees.
- The electron and nuclear spin operators use the standard
  :math:`S_z = \mathrm{diag}(S, S-1, \ldots, -S)` convention.
- g, D, A and the SEC tensors are defined in the tensor frame and rotated
  through the molecular and crystal frames to the laboratory frame.

Frames
------

Orientations are composed as

.. math::

   R_{\mathrm{tf2lf}} = R_{\mathrm{sf2lf}}\, R_{\mathrm{mf2sf}}\, R_{\mathrm{tf2mf}},

where the crystal orientation maps the crystal frame to the laboratory frame,
the molecular orientation maps the molecular frame to the crystal frame, and
the tensor orientation maps the tensor frame to the molecular frame. The
molecular orientation is not a user-set frame: it is the identity in an
ordinary calculation, and the space-group expansion fills it with the rotation
of each inequivalent site. Euler angles use the zyz convention. Second-order
tensors transform as :math:`R T R^{T}`; improper operations with determinant
-1 are accepted.

Resonance search
----------------

Two methods are available:

- ``High_Field_Method`` selects candidate transitions from the microwave
  transition matrix at the high-field end and solves each resonance field with
  a Brent root search.
- ``Adaptive_Grid_Method`` refines the energy-level diagram adaptively and
  solves the roots of Hermite cubic spline segments analytically, without an
  iterative refinement step.

``Eigenfield_Method`` solves the generalized eigenvalue problem of the
commutator superoperator and is available as an independent reference; its
cost grows as the square of the level count.

The adaptive-grid approach follows ideas from EasySpin [Stoll2006]_; DimSpin
is an independent implementation and is not a port of EasySpin code.

.. [Stoll2006] S. Stoll and A. Schweiger, *EasySpin, a comprehensive software
   package for spectral simulation and analysis in EPR*, J. Magn. Reson.
   178 (2006) 42-55, doi:10.1016/j.jmr.2005.08.013.

Line broadening
---------------

Line shapes are area-normalized. A Lorentzian, Gaussian or Voigt profile is
used depending on which widths are nonzero. Broadening can be evaluated by
direct summation or by FFT convolution; the FFT path adds transitions outside
the field grid by direct evaluation so their tails are retained.

Limitations
-----------

- A further resonance-search method, aimed at faster and more accurate root
  finding, is under development; it is still being tested and is not part of
  this release.
- The space-group expansion rotates the second-order tensors g, D and A; the
  high-order zero-field splitting coefficients and the third-order T, R and F
  tensors are not rotated there.
- An electric-field powder average sums both signs of the electric field; for
  strictly symmetric cases the first-order modulated signal can cancel.
