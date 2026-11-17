"""Tests for line-shape broadening."""

import numpy as np
import pytest

from dimspin.Line_Broadening import (
    broaden,
    broaden_direct,
    broaden_fft,
    line_shape,
)


def _area(y: np.ndarray, spacing: float) -> float:
    return float(np.sum(y) * spacing)


def test_lorentzian_is_area_normalized() -> None:
    field = np.linspace(-5000.0, 5000.0, 200001)
    y = line_shape(field, 0.0, 2.0, 1.0, 0.0)
    assert np.isclose(_area(y, field[1] - field[0]), 2.0, rtol=1e-3)


def test_gaussian_is_area_normalized() -> None:
    field = np.linspace(-50.0, 50.0, 20001)
    y = line_shape(field, 0.0, 1.5, 0.0, 1.0)
    assert np.isclose(_area(y, field[1] - field[0]), 1.5, rtol=1e-3)


def test_direct_matches_sum_of_single_lines() -> None:
    field = np.linspace(-20.0, 20.0, 4001)
    positions = [-1.0, 0.0, 1.0]
    intensities = [1.0, 2.0, 0.5]
    direct = broaden_direct(field, positions, intensities, 1.0, 0.5)
    manual = sum(
        line_shape(field, position, intensity, 1.0, 0.5)
        for position, intensity in zip(positions, intensities)
    )
    assert np.allclose(direct, manual)


def test_fft_matches_direct_convolution() -> None:
    field = np.linspace(-20.0, 20.0, 4001)
    positions = np.linspace(-5.0, 5.0, 9)
    intensities = np.linspace(0.5, 1.5, 9)
    direct = broaden_direct(field, positions, intensities, 1.0, 0.5)
    fft = broaden_fft(field, positions, intensities, 1.0, 0.5)
    assert np.max(np.abs(fft - direct)) < 0.02 * np.max(np.abs(direct))


def test_fft_includes_boundary_and_outside_tails() -> None:
    field = np.linspace(0.0, 10.0, 1001)
    positions = np.array([-0.5, 10.0, 10.5])
    intensities = np.array([1.0, 2.0, 0.5])
    direct = broaden_direct(field, positions, intensities, 1.0, 0.0)
    fft = broaden_fft(field, positions, intensities, 1.0, 0.0)
    assert np.all(fft > 0.0)
    assert np.max(np.abs(fft - direct)) < 0.02 * np.max(np.abs(direct))


def test_auto_uses_direct_for_few_transitions() -> None:
    field = np.linspace(-20.0, 20.0, 4001)
    positions = [0.0, 1.0]
    intensities = [1.0, 0.5]
    assert np.allclose(
        broaden(field, positions, intensities, 1.0, 0.5),
        broaden_direct(field, positions, intensities, 1.0, 0.5),
    )


def test_broaden_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        broaden([0.0, 1.0], [0.5], [1.0], 1.0, 0.0, method="unknown")


def test_broaden_fft_requires_uniform_grid() -> None:
    field = np.array([0.0, 1.0, 3.0])
    with pytest.raises(ValueError):
        broaden_fft(field, [1.0], [1.0], 1.0, 0.0)
