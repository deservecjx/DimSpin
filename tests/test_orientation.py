"""Tests for orientation sampling and rotation between orientations."""

import numpy as np

from dimspin.Fibonacci_Orientation import fibonacci_orientation
from dimspin.Orientation_Switch import orientation_switch


def test_fibonacci_orientation_is_unit_length() -> None:
    data = fibonacci_orientation(100)
    assert data.shape == (100, 3)
    assert np.allclose(np.linalg.norm(data, axis=1), 1.0)
    assert np.isclose(data[0, 2], -0.99)
    assert np.isclose(data[-1, 2], 0.99)


def test_orientation_switch_rotates_one_direction_to_another() -> None:
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    matrix = orientation_switch(target, source)
    assert np.allclose(matrix @ source, target)
    assert np.isclose(np.linalg.det(matrix), 1.0)


def test_orientation_switch_identity() -> None:
    vector = np.array([1.0, 0.0, 0.0])
    assert np.allclose(orientation_switch(vector, vector), np.eye(3))


def test_orientation_switch_antiparallel() -> None:
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([-1.0, 0.0, 0.0])
    matrix = orientation_switch(target, source)
    assert np.allclose(matrix @ source, target)
    assert np.isclose(np.linalg.det(matrix), 1.0)
