"""Tests for point-count estimation."""

from dimspin.Pointsnum_Optimise import get_points_spectrum


def test_minimum_points() -> None:
    assert get_points_spectrum([0.0, 100.0], 10.0, 0.0) == 100


def test_five_samples_per_linewidth() -> None:
    assert get_points_spectrum([0.0, 1000.0], 10.0, 0.0) == 500


def test_uses_smaller_nonzero_linewidth() -> None:
    assert get_points_spectrum([0.0, 1000.0], 20.0, 10.0) == 500


def test_zero_linewidth_falls_back_to_one() -> None:
    assert get_points_spectrum([0.0, 100.0], 0.0, 0.0) == 500
