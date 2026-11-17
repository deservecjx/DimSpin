"""Tests for rotation and orientation helpers."""

import numpy as np
import pytest

from dimspin.Eular2Rot import eular2rotzyz, resolve_orientation
from dimspin.Rodrigues_Rotation import rotation_matrix
from dimspin.Vec_normalize import vec_normalize


def test_vec_normalize() -> None:
    assert np.allclose(vec_normalize(np.array([3.0, 0.0, 4.0])), [0.6, 0.0, 0.8])


def test_rotation_matrix_rotates_x_to_y() -> None:
    matrix = rotation_matrix([0.0, 0.0, 1.0], np.pi / 2)
    assert np.allclose(matrix @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])


def test_rotation_matrix_is_orthogonal() -> None:
    matrix = rotation_matrix([1.0, 2.0, 3.0], 0.7)
    assert np.allclose(matrix @ matrix.T, np.eye(3))
    assert np.isclose(np.linalg.det(matrix), 1.0)


def test_identity_euler_angles() -> None:
    assert np.allclose(eular2rotzyz(0.0, 0.0, 0.0), np.eye(3))


def test_zyz_rotation_is_orthogonal() -> None:
    matrix = eular2rotzyz(0.3, 0.4, 0.5)
    assert np.allclose(matrix @ matrix.T, np.eye(3))
    assert np.isclose(np.linalg.det(matrix), 1.0)


def test_resolve_orientation_accepts_matrix_and_euler_angles() -> None:
    matrix = rotation_matrix([0.0, 1.0, 0.0], 0.4)
    assert np.allclose(resolve_orientation(matrix), matrix)
    assert np.allclose(resolve_orientation(None), np.eye(3))

    from_euler = resolve_orientation([0.1, 0.2, 0.3])
    assert from_euler.shape == (3, 3)
    assert np.allclose(from_euler @ from_euler.T, np.eye(3))


def test_resolve_orientation_rejects_non_orthogonal_matrix() -> None:
    with pytest.raises(ValueError):
        resolve_orientation(np.ones((3, 3)))
