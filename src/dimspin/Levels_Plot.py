"""Energy-level diagrams for a single crystal orientation.

Levels are drawn as E_i(B) versus B, with a vertical double arrow at every
resonance field connecting the two levels involved. The module is separate
from the spectrum calculation; it only supports ``spec.sample ==
'Crystal_Oneside'`` because a rotation or powder sweep has no single
energy-level diagram.

The same H(B) = H0 + B * H1 construction as the root finders is used, and the
resonance arrows come from the selected method, so they match the spectrum.
"""

import numpy as np


def _prepare(spin, spec):
    """
    Inputs: Spin and Spec objects.

    Returns: a prepared copy of the spin.

    Notes: the Spec parameters are checked and the tensor orientations,
    field gradients and field-independent Hamiltonian are prepared.
    """
    from copy import deepcopy
    spec.check_simpara()
    prepared = deepcopy(spin)
    prepared.identity_tensor(spec.crystal_orientation, np.eye(3))
    prepared.update_Btensor()
    prepared.update_H_initial()
    return prepared


def _level_curves(H0, H1, fieldrange, npoints):
    """
    Inputs: Hamiltonian constants; field range; number of points.

    Returns: (field grid, E) with E[i, :] the i-th level along the grid.
    """
    B = np.linspace(fieldrange[0], fieldrange[-1], npoints)
    E = np.array([np.linalg.eigvalsh(H0 + b * H1) for b in B]).T
    return B, E


def _resonance_pairs(spin, spec, modulation_symbol, method, truncate):
    """
    Inputs: prepared Spin; Spec object; modulation_symbol; method name;
    whether to apply the spectrum truncation.

    Returns: a list of (field, u, v) resonance pairs, with the intensity
    dropped.

    Notes: truncate=True applies the same observable-transition selection as
    Spec_Oneside, so the arrows match the spectrum; False keeps every
    transition returned by the method.
    """
    num_allowed = int(2 * spin.S * (2 * spin.I + 1))
    if method == 'High_Field_Method':
        from .High_Field_Method import high_field_method
        transitions = high_field_method(spin, spec, modulation_symbol)
    elif method == 'Adaptive_Grid_Method':
        from .Adaptive_Grid_Method import adaptive_grid_method
        transitions = adaptive_grid_method(spin, spec, modulation_symbol)
    else:
        raise ValueError(f"unknown method {method!r}")
    if truncate:
        from .Truncate_Observable import truncate_observable
        transitions = truncate_observable(transitions, num_allowed)
    return [(t[0], t[1], t[2]) for t in transitions]


def plot_energy_levels(spin, spec, modulation_symbol='positive', method=None,
                       npoints=2000, show_resonances=True, truncate=True,
                       ax=None, **kwargs):
    """
    Inputs: Spin and Spec objects; modulation_symbol ("positive" or
    "negative"); method name (defaults to ``spec.method``); number of points
    on the level curves; whether to draw resonance arrows; whether to apply
    the spectrum truncation; optional axes; extra line keywords.

    Returns: the axes used.

    Notes: only ``spec.sample == 'Crystal_Oneside'`` is supported. The arrows
    use the same method and modulation sign as the spectrum, so they stay
    consistent with the simulated peaks. With High_Field_Method the sorted
    level indices can be followed incorrectly at strong level crossings; this
    is a known limitation of the method itself.
    """
    import matplotlib.pyplot as plt

    if spec.sample != 'Crystal_Oneside':
        raise ValueError(
            "plot_energy_levels only supports a single orientation "
            f"(spec.sample == 'Crystal_Oneside'); got {spec.sample!r}. "
            "A rotation/powder sweep has no single energy-level diagram.")

    if method is None:
        method = spec.method

    prepared = _prepare(spin, spec)
    from .Eigenfield_Method import _build_H0_H1
    H0, H1 = _build_H0_H1(prepared, spec, modulation_symbol)

    B, E = _level_curves(H0, H1, spec.fieldrange, npoints)

    if ax is None:
        ax = plt.gca()
    for i in range(E.shape[0]):
        ax.plot(B, E[i], lw=0.8, **kwargs)

    if show_resonances:
        for Bk, u, v in _resonance_pairs(prepared, spec, modulation_symbol, method, truncate):
            ev = np.linalg.eigvalsh(H0 + Bk * H1)
            ax.annotate('', xy=(Bk, ev[v]), xytext=(Bk, ev[u]),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=0.8))

    ax.set_xlabel('Magnetic field (G)')
    ax.set_ylabel('Energy (MHz)')
    return ax
