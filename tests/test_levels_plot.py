"""Tests for the energy-level plot."""

import logging

import pytest

from dimspin.Class_Spec import Spec
from dimspin.Class_Spin import Spin
from dimspin.Levels_Plot import plot_energy_levels

logging.disable(logging.WARNING)


def make_spin():
    spin = Spin(1/2, 0)
    spin.g = 2.0023
    return spin


def make_spec(method='High_Field_Method', sample='Crystal_Oneside'):
    spec = Spec()
    spec.method = method
    spec.sample = sample
    spec.mwfreq = 9800.0
    spec.fieldrange = [3400.0, 3600.0]
    spec.specpoints = 100
    spec.lw_lorentzian = 5.0
    spec.Bmod = 5.0
    spec.Temperature = 300.0
    return spec


def test_level_curves_without_resonances() -> None:
    plt = pytest.importorskip('matplotlib.pyplot')
    _, ax = plt.subplots()
    result = plot_energy_levels(make_spin(), make_spec(), show_resonances=False,
                                npoints=200, ax=ax)
    assert result is ax
    assert len(ax.lines) == 2
    assert len(ax.texts) == 0
    plt.close(ax.figure)


@pytest.mark.parametrize('method', ['High_Field_Method', 'Adaptive_Grid_Method'])
def test_resonance_arrow_is_drawn(method) -> None:
    plt = pytest.importorskip('matplotlib.pyplot')
    _, ax = plt.subplots()
    plot_energy_levels(make_spin(), make_spec(method), npoints=200, ax=ax)
    assert len(ax.lines) == 2
    assert len(ax.texts) == 1
    plt.close(ax.figure)


def test_untruncated_transitions_can_be_shown() -> None:
    plt = pytest.importorskip('matplotlib.pyplot')
    _, ax = plt.subplots()
    plot_energy_levels(make_spin(), make_spec(), truncate=False, npoints=200, ax=ax)
    assert len(ax.texts) >= 1
    plt.close(ax.figure)


def test_rotation_sample_is_rejected() -> None:
    spec = make_spec(sample='Crystal_Rotation')
    with pytest.raises(ValueError):
        plot_energy_levels(make_spin(), spec)


def test_unknown_method_is_rejected() -> None:
    plt = pytest.importorskip('matplotlib.pyplot')
    with pytest.raises(ValueError):
        plot_energy_levels(make_spin(), make_spec(), method='Spline_Method')
    plt.close('all')
