"""Transitions and broadened spectra for one crystal orientation."""

import numpy as np

from .Truncate_Observable import truncate_observable


def spec_transitions(spin, spec):
    """
    Inputs: Spin and Spec objects.

    Returns: (positions, intensities) of all transitions for a single
    orientation, without line broadening.

    Notes: intensities carry the modulation sign, so one broadening step on
    the combined arrays gives the difference spectrum (positive minus
    negative modulation). Powder mode accumulates these transitions over all
    orientations and broadens once. ``num_allowed = 2 S (2 I + 1)`` is the
    number of allowed transitions (2 S spin steps times 2 I + 1 nuclear
    states), not the number of level pairs; truncate_observable always keeps
    these and adds the forbidden transitions above the relative threshold.
    """
    spin.update_Btensor()
    spin.update_H_initial()

    num_allowed = int(2 * spin.S * (2 * spin.I + 1))
    positions = []
    intensities = []

    def _collect(transitions, sign):
        positions.extend(float(t[0]) for t in transitions)
        intensities.extend(sign * float(t[3]) for t in transitions)

    if spec.method == 'High_Field_Method':
        from .High_Field_Method import high_field_method
        _collect(truncate_observable(high_field_method(spin, spec, 'positive'), num_allowed), +1.0)
        _collect(truncate_observable(high_field_method(spin, spec, 'negative'), num_allowed), -1.0)
    elif spec.method == 'Adaptive_Grid_Method':
        from .Adaptive_Grid_Method import adaptive_grid_method
        _collect(truncate_observable(adaptive_grid_method(spin, spec, 'positive'), num_allowed), +1.0)
        _collect(truncate_observable(adaptive_grid_method(spin, spec, 'negative'), num_allowed), -1.0)

    return (np.asarray(positions, dtype=float),
            np.asarray(intensities, dtype=float))


def broaden_transitions(spec, positions, intensities):
    """
    Inputs: Spec object with the field grid and linewidths; transition
    positions and intensities.

    Returns: the broadened spectrum on ``spec.fieldrange``.
    """
    if positions.size == 0:
        return np.zeros(spec.specpoints)
    from .Line_Broadening import broaden
    return broaden(spec.fieldrange, positions, intensities,
                   spec.lw_lorentzian, spec.lw_gaussian)


def spec_oneside(spin, spec):
    """
    Inputs: Spin and Spec objects.

    Returns: the broadened spectrum for one crystal orientation.
    """
    positions, intensities = spec_transitions(spin, spec)
    return broaden_transitions(spec, positions, intensities)
