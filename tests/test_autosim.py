"""Smoke and contract tests for the simulation entry point."""

import logging

import numpy as np
import pytest

from dimspin.AutoSim import autosim
from dimspin.Class_Spec import Spec
from dimspin.Class_Spin import Spin
from dimspin.Show_Spectra import show_rotation, show_spectrum
from dimspin.Space_Group_Expand import build_space_group_orientations
from dimspin.Spec_Oneside import spec_oneside

logging.disable(logging.WARNING)


def make_spin(g=2.0023, A=None, dg=None):
    spin = Spin(1/2, 1/2 if A is not None else 0)
    spin.g = g
    if A is not None:
        spin.A = A
    if dg is not None:
        spin.dg = dg
    return spin


def make_spec(method='High_Field_Method', sample='Crystal_Oneside', **overrides):
    spec = Spec()
    spec.method = method
    spec.sample = sample
    spec.mwfreq = 9800.0
    spec.fieldrange = [3400.0, 3600.0]
    spec.specpoints = 150
    spec.lw_lorentzian = 5.0
    spec.lw_gaussian = 0.0
    spec.Bmod = 5.0
    spec.Temperature = 300.0
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


@pytest.mark.parametrize('method', ['High_Field_Method', 'Adaptive_Grid_Method'])
def test_oneside_returns_one_dimensional_spectrum(method) -> None:
    spectrum = autosim(make_spin(), make_spec(method), parallel=False)
    assert spectrum.shape == (150,)
    assert np.all(np.isfinite(spectrum))
    assert np.max(spectrum) > 0.0


@pytest.mark.parametrize('method', ['High_Field_Method', 'Adaptive_Grid_Method'])
def test_rotation_returns_angle_field_map(method) -> None:
    spec = make_spec(method, 'Crystal_Rotation', anglerange=[0, 90, 30], rotaxis=[0, 0, 1])
    spectra = autosim(make_spin(), spec, parallel=False)
    assert spectra.shape == (4, 150)
    assert np.all(np.isfinite(spectra))


@pytest.mark.parametrize('method', ['High_Field_Method', 'Adaptive_Grid_Method'])
def test_powder_returns_one_dimensional_spectrum(method) -> None:
    spec = make_spec(method, 'Powder', npowder=20)
    spectrum = autosim(make_spin(), spec, parallel=False)
    assert spectrum.shape == (150,)
    assert np.all(np.isfinite(spectrum))
    assert np.max(spectrum) > 0.0


def test_both_methods_agree() -> None:
    a = autosim(make_spin(), make_spec('High_Field_Method'), parallel=False)
    b = autosim(make_spin(), make_spec('Adaptive_Grid_Method'), parallel=False)
    assert np.allclose(a, b)


def test_hyperfine_splits_the_line() -> None:
    spec = make_spec('High_Field_Method')
    single = autosim(make_spin(), spec, parallel=False)
    doublet = autosim(make_spin(A=100.0 * np.eye(3)), spec, parallel=False)

    def peak_width(spectrum):
        return np.count_nonzero(np.abs(spectrum) > 0.5 * np.max(np.abs(spectrum)))

    assert peak_width(doublet) > peak_width(single)


def test_multiple_spins_are_weighted_sum() -> None:
    spin_a = make_spin(g=2.0)
    spin_b = make_spin(g=2.1)
    spec = make_spec()
    combined = autosim(spin_a, spin_b, spec, parallel=False)
    expected = autosim(spin_a, spec, parallel=False) + autosim(spin_b, spec, parallel=False)
    assert np.allclose(combined, expected)


def test_space_group_expansion_weights_by_multiplicity() -> None:
    g = np.array([[2.0, 0.005, 0.0],
                  [0.005, 2.0, 0.0],
                  [0.0, 0.0, 2.0]])
    spec = make_spec()
    spec.group_number = 3
    spectrum = autosim(make_spin(g=g), spec, parallel=False)

    reference = Spin(1/2, 0)
    reference.g = g
    orientations, multiplicities = build_space_group_orientations(reference, 3)
    expected = np.zeros(spec.specpoints)
    for R_cart, m in zip(orientations, multiplicities):
        group_spin = reference.clone()
        group_spin.identity_tensor(None, R_cart)
        expected = expected + m * spec_oneside(group_spin, spec)

    assert np.allclose(spectrum, expected)


def test_parameter_modulation_runs() -> None:
    spin = make_spin(dg=0.01 * np.eye(3))
    spec = make_spec(modtype='param', Bmod=0.0)
    spectrum = autosim(spin, spec, parallel=False)
    assert np.all(np.isfinite(spectrum))
    assert np.max(np.abs(spectrum)) > 0.0


def test_electric_field_modulation_requires_a_direction() -> None:
    spec = make_spec(Emod=1.0, Edirection=np.zeros(3), Bmod=0.0)
    with pytest.raises(ValueError):
        spec.check_simpara()


def test_unknown_method_is_rejected() -> None:
    spec = make_spec(method='Spline_Method')
    with pytest.raises(ValueError):
        spec.check_simpara()


def test_unknown_sample_is_rejected() -> None:
    spec = make_spec(sample='Single_Crystal')
    with pytest.raises(ValueError):
        spec.check_simpara()


def test_wrong_argument_order_is_rejected() -> None:
    with pytest.raises(ValueError):
        autosim(make_spec(), make_spin())
    with pytest.raises(ValueError):
        autosim(make_spin())


def test_physical_scalars_are_validated() -> None:
    for field, value in (('mwfreq', 0.0), ('mwfreq', -1.0), ('Temperature', 0.0), ('Temperature', -10.0)):
        spec = make_spec(**{field: value})
        with pytest.raises(ValueError):
            spec.check_simpara()


def test_both_modulation_fields_are_rejected() -> None:
    spec = make_spec(Bmod=5.0, Emod=1.0, Edirection=[0, 0, 1])
    with pytest.raises(ValueError):
        spec.check_simpara()


def test_electric_field_direction_shape_is_validated() -> None:
    spec = make_spec(Emod=1.0, Bmod=0.0, Edirection=[0, 1])
    with pytest.raises(ValueError):
        spec.check_simpara()


def test_specpoints_must_be_a_positive_integer() -> None:
    for value in (-10, 0.5):
        spec = make_spec(specpoints=value)
        with pytest.raises(ValueError):
            spec.check_simpara()


def test_expanded_fieldrange_must_be_uniform_and_sets_specpoints() -> None:
    spec = make_spec(fieldrange=np.array([3400.0, 3450.0, 3600.0]), specpoints=0)
    with pytest.raises(ValueError):
        spec.check_simpara()

    grid = np.linspace(3400.0, 3600.0, 50)
    spec = make_spec(fieldrange=grid, specpoints=0)
    spec.check_simpara()
    assert spec.specpoints == 50
    assert np.allclose(spec.fieldrange, grid)


def test_rotation_angles_are_validated() -> None:
    with pytest.raises(ValueError):
        make_spec('High_Field_Method', 'Crystal_Rotation', anglerange=[0, 90, 0]).check_simpara()
    with pytest.raises(ValueError):
        make_spec('High_Field_Method', 'Crystal_Rotation', anglerange=[90, 0, 1]).check_simpara()


def test_powder_count_is_validated() -> None:
    for value in (0, -5, 1.5):
        spec = make_spec('High_Field_Method', 'Powder', npowder=value)
        with pytest.raises(ValueError):
            spec.check_simpara()


def test_worker_count_is_validated() -> None:
    for value in (0, -1, 1.5):
        spec = make_spec(max_workers=value)
        with pytest.raises(ValueError):
            spec.check_simpara()


def test_autosim_sets_the_worker_count() -> None:
    spec = make_spec()
    spectrum = autosim(make_spin(), spec, show=False, parallel=False, max_workers=3)
    assert spec.max_workers == 3
    assert spectrum.shape == (spec.specpoints,)


def test_space_group_parameters_are_validated() -> None:
    spec = make_spec(group_number=0)
    with pytest.raises(ValueError):
        spec.check_simpara()
    spec = make_spec(group_number=3, frac_pos=[0.1, 0.2])
    with pytest.raises(ValueError):
        spec.check_simpara()


def test_check_simpara_is_idempotent() -> None:
    spec = make_spec()
    spec.check_simpara()
    fieldrange = spec.fieldrange.copy()
    spec.check_simpara()
    assert np.allclose(spec.fieldrange, fieldrange)
    assert spec.specpoints == 150


def test_show_spectrum_draws_one_line() -> None:
    plt = pytest.importorskip('matplotlib.pyplot')
    spec = make_spec()
    spectrum = autosim(make_spin(), spec, parallel=False)
    ax = show_spectrum(spec, spectrum)
    assert len(ax.lines) == 1
    plt.close(ax.figure)


def test_show_rotation_draws_a_field_angle_map() -> None:
    plt = pytest.importorskip('matplotlib.pyplot')
    spec = make_spec('High_Field_Method', 'Crystal_Rotation', anglerange=[0, 90, 30], rotaxis=[0, 0, 1])
    spectra = autosim(make_spin(), spec, parallel=False)
    ax = show_rotation(spec, spectra)
    assert len(ax.collections) >= 1
    plt.close(ax.figure)


def test_autosim_show_runs_for_all_samples() -> None:
    plt = pytest.importorskip('matplotlib.pyplot')
    settings = {
        'Crystal_Oneside': {},
        'Crystal_Rotation': {'anglerange': [0, 90, 30], 'rotaxis': [0, 0, 1]},
        'Powder': {'npowder': 10},
    }
    for sample, extra in settings.items():
        spec = make_spec('High_Field_Method', sample, **extra)
        spectrum = autosim(make_spin(), spec, show=True, parallel=False)
        assert np.all(np.isfinite(spectrum))
        plt.close('all')
