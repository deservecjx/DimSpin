"""Tests for the high-field root finder and transition selection."""

import numpy as np
import pytest
from scipy.optimize import brentq
from toy_spin import G_ISO, ToySpin, make_spec

from dimspin.High_Field_Method import _select_transition_pairs, high_field_method
from dimspin.Root_Finder import _brentq, population_difference, transition_probability
from dimspin.Team_Zeeman import MU_B_H


def test_free_electron_resonance_field() -> None:
    spin = ToySpin(1/2, 0, G_ISO * np.eye(3))
    spec = make_spec('Bmod')
    transitions = high_field_method(spin, spec, 'positive')

    expected = spec.mwfreq / (MU_B_H * G_ISO) - spec.Bmod
    assert len(transitions) == 1
    field, jj, kk, intensity = transitions[0]
    assert (jj, kk) == (0, 1)
    assert np.isclose(field, expected, atol=1e-3)
    assert intensity > 0.0


def test_hyperfine_doublet_spacing() -> None:
    spin = ToySpin(1/2, 1/2, G_ISO * np.eye(3), A=100.0 * np.eye(3))
    spec = make_spec('Bmod')
    transitions = high_field_method(spin, spec, 'positive')

    fields = np.sort([transition[0] for transition in transitions])
    assert len(fields) == 2
    assert np.isclose(fields[1] - fields[0], 100.0 / (MU_B_H * G_ISO), atol=1e-3)


def test_axial_zfs_gives_two_transitions() -> None:
    D, E = 500.0, 50.0
    tensor = np.diag([-D / 3 + E, -D / 3 - E, 2 * D / 3])
    spin = ToySpin(1, 0, G_ISO * np.eye(3), D=tensor)
    spec = make_spec('Bmod')
    transitions = high_field_method(spin, spec, 'positive')

    assert len(transitions) == 2
    fields = [transition[0] for transition in transitions]
    assert all(3000.0 <= field <= 4000.0 for field in fields)


def test_transition_probability_for_spin_half() -> None:
    spin = ToySpin(1/2, 0, G_ISO * np.eye(3))
    spin.modulate_B(3500.0, 0.0)
    _, eigvecs = np.linalg.eigh(spin.H_total)
    expected = (G_ISO / 2) ** 2
    assert np.isclose(transition_probability(1/2, 0, spin.g1, eigvecs, 0, 1), expected)
    assert np.isclose(transition_probability(1/2, 0, spin.g1, eigvecs, 0, 0), 0.0)


def test_population_difference_is_antisymmetric_and_small() -> None:
    eigvals = np.array([0.0, 2.0, 5.0, 9.0])
    ground = population_difference(eigvals, 0, 1, 300.0)
    reverse = population_difference(eigvals, 1, 0, 300.0)
    assert np.isclose(ground, -reverse)
    assert 0.0 < ground < 1e-5


def test_population_difference_matches_boltzmann_formula() -> None:
    eigvals = np.array([0.0, 100.0])
    temperature = 300.0
    beta = 6.626e-28 / (1.380649e-23 * temperature)
    factors = np.exp(-eigvals * beta)
    expected = (factors[0] - factors[1]) / factors.sum()
    assert np.isclose(population_difference(eigvals, 0, 1, temperature), expected)


def test_brentq_matches_scipy() -> None:
    cases = [
        (lambda x: x ** 3 - 2 * x - 5, 1.0, 3.0),
        (lambda x: np.cos(x) - x, 0.0, 1.0),
        (lambda x: np.exp(x) - 3 * x, 0.0, 1.0),
    ]
    for func, a, b in cases:
        ours = _brentq(func, a, b, 1e-12, 1e-14)
        theirs = brentq(func, a, b, xtol=1e-12, rtol=1e-14)
        assert np.isclose(ours, theirs, rtol=0, atol=1e-12)


def test_brentq_rejects_equal_signs() -> None:
    with pytest.raises(ValueError):
        _brentq(lambda x: x * x + 1.0, -1.0, 1.0, 1e-9, 1e-9)


def test_electric_field_modulation_shifts_resonance() -> None:
    T_zz = 0.2
    Tz = np.zeros((3, 3))
    Tz[2, 2] = T_zz
    spin = ToySpin(1/2, 0, G_ISO * np.eye(3), T=(np.zeros((3, 3)), np.zeros((3, 3)), Tz))
    spec = make_spec('Emod', E1=np.array([0.0, 0.0, 1.0]))
    transitions = high_field_method(spin, spec, 'positive')

    expected = spec.mwfreq / (MU_B_H * G_ISO + T_zz)
    assert len(transitions) == 1
    assert np.isclose(transitions[0][0], expected, atol=1e-3)


def test_parameter_modulation_shifts_resonance() -> None:
    delta_g = -0.05
    spin = ToySpin(1/2, 0, G_ISO * np.eye(3), dg=delta_g * np.eye(3))
    spec = make_spec('param')
    transitions = high_field_method(spin, spec, 'positive')

    expected = spec.mwfreq / (MU_B_H * (G_ISO + delta_g))
    assert len(transitions) == 1
    assert np.isclose(transitions[0][0], expected, atol=1e-3)


def test_parameter_modulation_requires_delta_tensors() -> None:
    spin = ToySpin(1/2, 0, G_ISO * np.eye(3))
    with pytest.raises(ValueError):
        high_field_method(spin, make_spec('param'), 'positive')


def test_transition_pairs_exclude_forbidden_transitions() -> None:
    tensor = np.diag([-500 / 3, -500 / 3, 1000 / 3])
    spin = ToySpin(1, 0, G_ISO * np.eye(3), D=tensor)
    spin.modulate_B(4000.0, 0.0)
    _, eigvecs = np.linalg.eigh(spin.H_total)
    pairs = _select_transition_pairs(spin, eigvecs)

    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (0, 2) not in pairs
