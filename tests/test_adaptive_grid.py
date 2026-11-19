"""Tests for the adaptive-grid method and the eigenfield reference."""

import numpy as np
from toy_spin import G_ISO, ToySpin, make_spec

from dimspin.Adaptive_Grid_Method import _cubic_roots_t, _dedup_roots, adaptive_grid_method
from dimspin.Eigenfield_Method import _build_H0_H1, eigenfield_method
from dimspin.High_Field_Method import high_field_method
from dimspin.Team_Zeeman import MU_B_H
from dimspin.Truncate_Observable import truncate_observable

ISO = G_ISO * np.eye(3)


def fields_of(transitions):
    return np.sort([transition[0] for transition in transitions])


def test_free_electron_resonance_field() -> None:
    spin = ToySpin(1/2, 0, ISO)
    spec = make_spec('Bmod')
    transitions = adaptive_grid_method(spin, spec, 'positive')

    expected = spec.mwfreq / (MU_B_H * G_ISO) - spec.Bmod
    assert len(transitions) == 1
    assert np.isclose(transitions[0][0], expected, atol=1e-6)
    assert transitions[0][3] > 0.0


def test_hyperfine_doublet_spacing() -> None:
    spin = ToySpin(1/2, 1/2, ISO, A=100.0 * np.eye(3))
    spec = make_spec('Bmod')
    fields = fields_of(adaptive_grid_method(spin, spec, 'positive'))

    assert len(fields) == 2
    assert np.isclose(fields[1] - fields[0], 100.0 / (MU_B_H * G_ISO), atol=1e-3)


def test_axial_zfs_matches_eigenfield_reference() -> None:
    D, E = 500.0, 50.0
    tensor = np.diag([-D / 3 + E, -D / 3 - E, 2 * D / 3])
    spin = ToySpin(1, 0, ISO, D=tensor)
    spec = make_spec('Bmod')

    adaptive = fields_of(adaptive_grid_method(spin, spec, 'positive'))
    reference = fields_of(eigenfield_method(spin, spec, 'positive'))
    assert len(adaptive) == len(reference) == 2
    assert np.allclose(adaptive, reference, atol=1e-2)


def test_eigenfield_matches_high_field_for_simple_systems() -> None:
    spin = ToySpin(1/2, 1/2, ISO, A=100.0 * np.eye(3))
    spec = make_spec('Bmod')
    high = fields_of(high_field_method(spin, spec, 'positive'))
    reference = fields_of(eigenfield_method(spin, spec, 'positive'))
    assert np.allclose(high, reference, atol=1e-2)


def test_electric_field_modulation_shifts_resonance() -> None:
    T_zz = 0.2
    Tz = np.zeros((3, 3))
    Tz[2, 2] = T_zz
    spin = ToySpin(1/2, 0, ISO, T=(np.zeros((3, 3)), np.zeros((3, 3)), Tz))
    spec = make_spec('Emod', E1=np.array([0.0, 0.0, 1.0]))
    transitions = adaptive_grid_method(spin, spec, 'positive')

    expected = spec.mwfreq / (MU_B_H * G_ISO + T_zz)
    assert len(transitions) == 1
    assert np.isclose(transitions[0][0], expected, atol=1e-6)


def test_parameter_modulation_shifts_resonance() -> None:
    delta_g = -0.05
    spin = ToySpin(1/2, 0, ISO, dg=delta_g * np.eye(3))
    spec = make_spec('param')
    transitions = adaptive_grid_method(spin, spec, 'positive')

    expected = spec.mwfreq / (MU_B_H * (G_ISO + delta_g))
    assert len(transitions) == 1
    assert np.isclose(transitions[0][0], expected, atol=1e-6)


def test_build_H0_H1_is_linear_in_field() -> None:
    spin = ToySpin(1/2, 1/2, ISO, A=100.0 * np.eye(3))
    spec = make_spec('Bmod')
    H0, H1 = _build_H0_H1(spin, spec, 'positive')

    field = 3400.0
    spin.modulate_B(field, spec.Bmod, 'positive')
    assert np.allclose(H0 + field * H1, spin.H_total)

    H0_neg, H1_neg = _build_H0_H1(spin, spec, 'negative')
    spin.modulate_B(field, spec.Bmod, 'negative')
    assert np.allclose(H0_neg + field * H1_neg, spin.H_total)


def test_cubic_roots_for_a_linear_gap() -> None:
    ts = _cubic_roots_t(-1.0, 1.0, 2.0, 2.0)
    assert len(ts) == 1
    assert np.isclose(ts[0], 0.5)


def test_dedup_roots_merges_only_the_same_pair() -> None:
    roots = [(10.0, 0, 1), (10.02, 0, 1), (10.5, 0, 1), (10.0, 0, 2)]
    deduped = _dedup_roots(roots)
    assert deduped == [(10.0, 0, 1), (10.5, 0, 1), (10.0, 0, 2)]


def test_truncate_observable_keeps_allowed_and_strong_forbidden() -> None:
    transitions = [(1.0, 0, 1, 1.0), (2.0, 0, 1, 0.5),
                   (3.0, 1, 2, 0.01), (4.0, 1, 2, 1e-4)]
    kept = truncate_observable(transitions, num_allowed=1, rel_thresh=1e-3)
    assert [transition[0] for transition in kept] == [1.0, 2.0, 3.0]


def test_truncate_observable_handles_empty_and_zero_intensity() -> None:
    assert truncate_observable([], 1) == []
    assert truncate_observable([(1.0, 0, 1, 0.0)], 1) == []
