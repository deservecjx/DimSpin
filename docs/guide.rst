User guide
==========

This page lists the parameters of :class:`~dimspin.Class_Spin.Spin` and
:class:`~dimspin.Class_Spec.Spec`, the values each one accepts and what those
choices cost. The physics behind them is in :doc:`theory`, and a complete
runnable script for every feature is in :doc:`examples`.

Data model
----------

A simulation has three pieces:

- a :class:`~dimspin.Class_Spin.Spin` holds the spin quantum numbers and the
  tensors of one centre;
- a :class:`~dimspin.Class_Spec.Spec` holds the experimental settings;
- :func:`~dimspin.AutoSim.autosim` takes one or more spins followed by the
  spec, checks the settings, expands the spins into their magnetically
  inequivalent orientations when a space group is given, and returns the
  spectrum.

.. code-block:: python

   from dimspin import Spec, Spin, autosim

   spin = Spin(1/2, 0)
   spin.g = [2.009, 2.006, 2.002]

   spec = Spec()
   spec.modtype = "Bmod"
   spec.Bmod = 1.0
   spec.mwfreq = 9560.0
   spec.fieldrange = [3300.0, 3600.0]
   spec.specpoints = 800
   spec.lw_lorentzian = 0.5
   spec.lw_gaussian = 0.2

   spectrum = autosim(spin, spec)

Magnetic fields are in gauss (G), energies and frequencies in MHz and
temperatures in kelvin. Euler angles and rotation matrices use radians;
crystal-rotation ranges are in degrees.

Spin parameters
---------------

Spin quantum numbers
~~~~~~~~~~~~~~~~~~~~

``Spin(S, I)`` takes the electron spin ``S`` (a positive number) and the
nuclear spin ``I`` (zero or a positive number). Half-integer values may be
written as fractions or as decimals: ``Spin(5/2, 3/2)`` and
``Spin(2.5, 1.5)`` are equivalent. This documentation writes them as
fractions; a decimal value must be an exact half-integer.

``S``
   Electron spin. Determines the Hilbert-space size together with ``I``.
``I``
   Nuclear spin. ``I = 0`` means no hyperfine coupling.

Magnetic tensors
~~~~~~~~~~~~~~~~

``g``
   Electron Zeeman tensor. Accepts a scalar (isotropic), a 3-vector (the
   diagonal) or a full 3x3 matrix.
``D``
   Zero-field splitting tensor. Accepts a scalar (axial ``D`` with ``E = 0``),
   a ``(D, E)`` pair, a 3-vector (the diagonal) or a full 3x3 matrix. It is
   ignored for ``S = 1/2``, where zero-field splitting is not defined.
``A``
   Hyperfine tensor. Accepts a scalar (isotropic), a 3-vector (the
   diagonal) or a full 3x3 matrix. It is ignored for ``I = 0``.

Parameter-modulation amplitudes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dg``, ``dD``, ``dA``
   Changes of ``g``, ``D`` and ``A`` used by parameter modulation, in the same
   input forms as the tensors themselves. They only contribute when
   ``Spec.modtype`` is ``"param"``, used, for example, when the perturbation
   keeps a fixed orientation relative to the crystal.

Spin-electric coupling tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``T``, ``R``, ``F``
   Coupling tensors of the electric field to the Zeeman, zero-field splitting
   and hyperfine terms. Each accepts a 3x6 array whose columns are
   ``(xx, yy, zz, yz, xz, xy)`` for the electric field along x, y and z, or a
   full 3x3x3 array. ``R`` also accepts a 3x5 array whose columns are
   ``(D, E, yz, xz, xy)``, with the traceless diagonal block
   ``(-D/3 + E, -D/3 - E, 2D/3)``. They only contribute when
   ``Spec.modtype`` is ``"Emod"``. ``R`` is ignored for ``S = 1/2`` and ``F``
   for ``I = 0``.

High-order zero-field splitting coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``B2``, ``B4``, ``B6``
   Coefficients :math:`B_k^q` of the extended Stevens operators, in MHz,
   with 5, 9 and 13 entries ordered by :math:`q = -k, \ldots, k`. The entries
   ``B2[2]`` and ``B2[4]`` correspond to :math:`B_2^0` and :math:`B_2^2`,
   which duplicate the ``D`` tensor and cannot be given together with a
   nonzero ``D``. Terms with :math:`k > 2S` are zero.

Tensor orientations
~~~~~~~~~~~~~~~~~~~

``tensor_orientation``
   Orientation of the tensor frame of g, D and A in the molecular frame: a
   3x3 matrix or three Euler angles in the zyz convention (radians). ``None``
   means the tensor frame coincides with the molecular frame.
``Etensor_orientation``
   Orientation of the tensor frame of the spin-electric coupling tensors
   T, R and F, also in the molecular frame. It is independent of
   ``tensor_orientation``: the two are applied separately, never composed, so
   tilting the E tensors relative to g, D and A requires passing the composed
   3x3 matrix. ``None`` makes it equal to ``tensor_orientation``.

Spec parameters
---------------

Resonance search
~~~~~~~~~~~~~~~~

``method``
   ``"High_Field_Method"`` (default) or ``"Adaptive_Grid_Method"``. The first
   selects candidate transitions at the high-field end and solves each root
   with Brent iterations; the second refines the energy-level diagram
   adaptively and solves spline segments analytically. See :doc:`theory` for
   how they differ and when the high-field tracking can fail.

Sample mode
~~~~~~~~~~~

``sample``
   ``"Crystal_Oneside"`` (default), ``"Crystal_Rotation"`` or ``"Powder"``.
   A one-side or powder run returns an array of shape ``(specpoints,)``; a
   rotation run returns ``(angnum, specpoints)``, one row per angle.

Modulation
~~~~~~~~~~

``modtype``
   ``"Bmod"``, ``"Emod"`` or ``"param"``. Unless ``"param"`` is set
   explicitly it is detected from ``Bmod`` and ``Emod``, and giving both at
   once is an error.
``Bmod``, ``Emod``
   Modulation amplitudes. Use one of them for magnetic- or electric-field
   modulation, and leave both zero for parameter modulation, which takes its
   amplitudes from ``Spin.dg``, ``Spin.dD`` and ``Spin.dA`` (see
   :doc:`theory`).
``Edirection``
   Direction of the electric field, a 3-vector in the laboratory frame.
   Required when ``Emod`` is nonzero.

Field grid and line shape
~~~~~~~~~~~~~~~~~~~~~~~~~

``mwfreq``
   Microwave frequency in MHz.
``fieldrange``
   Either ``[left, right]``, expanded to ``specpoints`` points, or an
   already expanded axis, which must be uniformly spaced.
``specpoints``
   Number of field points. If ``fieldrange`` is already expanded, the two
   must agree, and a zero value takes the length of the axis.
``lw_lorentzian``, ``lw_gaussian``
   Lorentzian half width and Gaussian standard deviation, in the same unit as
   the field grid. Giving both gives a Voigt profile; at least one must be
   nonzero. Line shapes are area-normalized.

Rotation settings
~~~~~~~~~~~~~~~~~

``anglerange``
   ``[start, end]`` or ``[start, end, step]`` in degrees; the step must be
   positive and ``end`` must not be smaller than ``start``. The crystal is
   turned about the laboratory x axis.
``crystal_orientation``
   Initial orientation of the crystal in the laboratory frame: a 3x3 matrix
   or three Euler angles (zyz, radians). ``None`` means the two frames coincide
   at the first angle. The rotation axis is fixed to the laboratory x axis.

Powder settings
~~~~~~~~~~~~~~~

``npowder``
   Number of sampled crystallite orientations; the default is 2000. Each orientation
   costs one resonance search, so this is the main cost of a powder run. For an
   electric-field modulated spectrum every orientation is evaluated together with
   its orientation-reversed partner.(the expected signal is zero)

Space-group expansion
~~~~~~~~~~~~~~~~~~~~~

``group_number``
   Space-group number from 1 to 230. When it is set, each spin is expanded
   into its magnetically inequivalent orientations and the spectra are summed
   with their multiplicities. The expansion rotates g, D and A only, so it is
   guaranteed to be correct for magnetic-field modulation of systems described
   by g, D and A; electric-field and parameter modulation, and high-order
   zero-field terms, are not covered yet (see :doc:`theory`).
``frac_pos``
   Fractional coordinate of the spin centre in the unit cell, a 3-vector in
   the standard setting of the group (unique axis b for monoclinic groups,
   hexagonal axes for rhombohedral groups). It is optional: without it every
   symmetry operation is treated as a distinct site (general position), and
   with it the operations are grouped by the position they produce, so
   special positions are counted correctly. For a single spin centre this only
   rescales the overall intensity, because every site of the orbit is repeated
   equally; the relative weights change when several inequivalent centres are
   combined. Tensors that do not satisfy the site symmetry give extra
   orientations without ``frac_pos``, and a warning with it.

Parallel settings
~~~~~~~~~~~~~~~~~

``max_workers``
   Number of worker processes; ``None`` uses a built-in default of at most
   eight. Passing ``max_workers`` to :func:`~dimspin.AutoSim.autosim`
   overrides this value. A count above the logical core count is allowed with
   a warning.

Running a simulation
~~~~~~~~~~~~~~~~~~~~

:func:`~dimspin.AutoSim.autosim` accepts one or more spins followed by the
spec:

.. code-block:: python

   spectrum = autosim(spin_a, spin_b, spec)      # two centres, summed
   spectrum = autosim(spin, spec, show=True)      # draw the result
   spectrum = autosim(spin, spec, parallel=False) # run in this process
   spectrum = autosim(spin, spec, max_workers=4)  # four worker processes

Settings are validated before the calculation, so a wrong field range,
modulation or linewidth stops the run with a ``ValueError`` instead of a
wrong spectrum. The checks are idempotent; calling
``Spec.check_simpara`` yourself is safe.

Parallel evaluation is on by default and splits the independent tasks: one
per angle in rotation mode and one per (spin, orientation) pair in powder
mode. On Windows, scripts must guard the entry point with
``if __name__ == "__main__":`` so that worker processes do not run the whole
script again; see ``examples/06_powder_parallel.py``.

Post-processing
~~~~~~~~~~~~~~~

``show=True`` draws the result: a line for one-side and powder spectra, and a
2D angle-dependent spectra for rotation. The helpers
:func:`~dimspin.Show_Spectra.show_spectrum` and
:func:`~dimspin.Show_Spectra.show_rotation` draw on an existing axes when one
is given, which is convenient for multi-panel figures.
:func:`~dimspin.Levels_Plot.plot_energy_levels` draws the energy levels of one
crystal orientation with the resonance fields marked; it only supports
``sample = "Crystal_Oneside"``.

Common errors
~~~~~~~~~~~~~

- ``Both Bmod and Emod are nonzero``: set one modulation amplitude, or
  ``modtype = "param"`` when neither field is modulated.
- ``An expanded 'fieldrange' must be uniformly spaced``: pass a two-value
  interval or a uniform axis; a non-uniform axis is not accepted.
- ``The length of 'fieldrange' does not match 'specpoints'``: an expanded axis
  fixes the number of points.
- ``Lorentzian and Gaussian linewidths cannot be zero at the same time``: give
  at least one width.
- ``B2^0/B2^2 are equivalent to the D tensor``: use ``D`` or the ``B2``
  entries, not both.
- ``Electric field modulation is selected, but modulating electric field is
  not given``: ``Emod`` needs a nonzero ``Edirection``.
- ``Parameter modulation is selected, but no ... tensor``: ``modtype =
  "param"`` needs a nonzero ``dg``, ``dD`` or ``dA``.
