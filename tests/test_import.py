"""Tests for the public package API."""

import logging
import re

import numpy as np

import dimspin

logging.disable(logging.WARNING)


def test_version_is_exposed() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", dimspin.__version__)


def test_public_api_lists_expected_names() -> None:
    assert set(dimspin.__all__) == {"Spin", "Spec", "autosim", "plot_energy_levels", "__version__"}


def test_public_objects_are_available() -> None:
    assert isinstance(dimspin.Spin, type)
    assert isinstance(dimspin.Spec, type)
    assert callable(dimspin.autosim)
    assert callable(dimspin.plot_energy_levels)


def test_minimal_simulation_through_the_public_api() -> None:
    spin = dimspin.Spin(0.5, 0)
    spin.g = 2.0023

    spec = dimspin.Spec()
    spec.mwfreq = 9800.0
    spec.fieldrange = [3400.0, 3600.0]
    spec.specpoints = 100
    spec.lw_lorentzian = 5.0
    spec.Bmod = 5.0

    spectrum = dimspin.autosim(spin, spec, parallel=False)
    assert spectrum.shape == (100,)
    assert np.all(np.isfinite(spectrum))
    assert np.max(spectrum) > 0.0
