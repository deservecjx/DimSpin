Theory and conventions
======================

Spin Hamiltonian
----------------

The simulated Hamiltonian is

.. math::

   H = \mu_B\, \mathbf{B}^{T}\mathbf{g}\mathbf{S}
     + \mathbf{S}^{T}\mathbf{D}\mathbf{S}
     + \mathbf{I}^{T}\mathbf{A}\mathbf{S}
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

Electric-field modulation follows the same construction. With
:math:`H_\mathrm{SH}` the spin Hamiltonian before modulation, the SEC term is
its first derivative with respect to the electric field,

.. math::

   H_\mathrm{SEC} = \mathbf{E}\cdot\frac{\partial H_\mathrm{SH}}
     {\partial \mathbf{E}} = \mathbf{E}\cdot\left(
     \mu_B\,\mathbf{B}^{T}\mathbf{T}\mathbf{S}
     + \mathbf{S}^{T}\mathbf{R}\mathbf{S}
     + \mathbf{I}^{T}\mathbf{F}\mathbf{S} + \cdots\right),

with the electric-field-response tensors

.. math::

   \mathbf{T} \equiv \frac{\partial \mathbf{g}}{\partial \mathbf{E}},\qquad
   \mathbf{R} \equiv \frac{\partial \mathbf{D}}{\partial \mathbf{E}},\qquad
   \mathbf{F} \equiv \frac{\partial \mathbf{A}}{\partial \mathbf{E}}.

Each term in the parentheses is a vector, so its scalar product with the
electric field is well defined. The tensors are third order (3x3x3); a 3x6
input stores, for the electric-field directions x, y and z, the symmetric 3x3
matrix of the corresponding response. The SEC term is treated as a first-order
perturbation of the spin Hamiltonian and enters the simulation with the
modulation sign (+E or -E).

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

The resonance fields are found in one of two ways. The high-field approach
selects candidate transitions from the microwave transition matrix at the
high-field end and solves each resonance field with a Brent root search. The
adaptive-grid approach refines the energy-level diagram adaptively and solves
the roots of Hermite cubic spline segments analytically, without an iterative
refinement step.

``Eigenfield_Method`` solves the generalized eigenvalue problem of the
commutator superoperator and is available as an independent reference; its
cost grows as the square of the level count.

The high-field method follows the levels sorted by energy from the high-field
end; at strong level crossings that ordering can be followed incorrectly,
which is a known limitation of the method. The adaptive-grid method does not
depend on this ordering.

The adaptive-grid approach follows ideas from EasySpin [Stoll2006]_; DimSpin
is an independent implementation and is not a port of EasySpin code.

.. [Stoll2006] S. Stoll and A. Schweiger, *EasySpin, a comprehensive software
   package for spectral simulation and analysis in EPR*, J. Magn. Reson.
   178 (2006) 42-55, doi:10.1016/j.jmr.2005.08.013.

Transition intensities
----------------------

A transition between levels j and k at resonance field B carries the intensity

.. math::

   I_{jk} \propto |\langle k|H_1|j\rangle|^2\,(p_j - p_k),

where :math:`H_1 = \sum_j g_{xj} S_j` is the microwave coupling for a
microwave field along the laboratory x axis and :math:`p` are the Boltzmann
populations at that field, normalized to unit sum.

The search is run twice, once for each modulation sign: magnetic-field
modulation shifts the swept field by :math:`\pm B_\mathrm{mod}`, electric-field
modulation changes the sign of the T, R and F contributions, and parameter
modulation flips :math:`\Delta g`, :math:`\Delta D` and :math:`\Delta A`. The
two transition sets enter the broadening with opposite signs, so their sum is
the first-derivative spectrum that field modulation produces experimentally.

The search returns every transition it finds. ``truncate_observable`` keeps
the ``2S(2I+1)`` strongest ones, corresponding to the allowed transitions,
and then further transitions while their intensity stays above
:math:`10^{-3}` of the maximum; the rest are dropped.

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
- The space-group expansion rotates the second-order tensors g, D and A only.
  The high-order zero-field splitting coefficients, the third-order T, R and F
  tensors and the modulation amplitudes dg, dD and dA are not rotated, so the
  expansion is only guaranteed to be correct for magnetic-field modulation of
  systems described by g, D and A; electric-field and parameter modulation,
  and systems with high-order zero-field terms, are not covered by it yet.
- An electric-field powder average sums both signs of the electric field; for
  strictly symmetric cases the first-order modulated signal can cancel.
