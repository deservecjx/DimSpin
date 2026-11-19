"""Tests for Spin input validation and tensor normalization."""

import logging

import numpy as np
import pytest

from dimspin.Class_Spin import Spin
from dimspin.Tensor import Dtensor


def quiet_spin(S=0.5, I=0.5):
    logging.disable(logging.WARNING)
    return Spin(S, I)


def test_scalar_isotropic_tensors_are_expanded() -> None:
    spin = quiet_spin()
    spin.g = 2.0
    spin.A = 100.0
    spin.dg = 0.01
    spin.dA = 2.0
    spin.identity_tensor()

    assert np.allclose(spin.g, 2.0 * np.eye(3))
    assert np.allclose(spin.A, 100.0 * np.eye(3))
    assert np.allclose(spin.dg, 0.01 * np.eye(3))
    assert np.allclose(spin.dA, 2.0 * np.eye(3))


def test_scalar_d_is_treated_as_axial() -> None:
    spin = quiet_spin(S=2, I=0)
    spin.D = 30.0
    spin.identity_tensor()
    assert np.allclose(spin.D, Dtensor(30.0, 0.0))


def test_d_and_e_pair_matches_dtensor() -> None:
    spin = quiet_spin(S=2, I=0)
    spin.D = (30.0, 5.0)
    spin.identity_tensor()
    assert np.allclose(spin.D, Dtensor(30.0, 5.0))


def test_vectors_are_expanded_to_diagonal_matrices() -> None:
    spin = quiet_spin(S=2, I=1)
    spin.g = [2.0, 2.1, 2.2]
    spin.D = [1.0, 2.0, 3.0]
    spin.A = [10.0, 20.0, 30.0]
    spin.identity_tensor()

    assert np.allclose(spin.g, np.diag([2.0, 2.1, 2.2]))
    assert np.allclose(spin.D, np.diag([1.0, 2.0, 3.0]))
    assert np.allclose(spin.A, np.diag([10.0, 20.0, 30.0]))


def test_s_half_clears_zfs_and_i_zero_clears_hyperfine() -> None:
    spin = quiet_spin(S=0.5, I=0)
    spin.g = 2.0
    spin.A = 100.0
    spin.identity_tensor()
    assert np.allclose(spin.D, 0.0)
    assert np.allclose(spin.A, 0.0)


def test_packed_coupling_tensor_roundtrip() -> None:
    spin = quiet_spin(S=1, I=0)
    spin.T = np.zeros((3, 6))
    spin.T[0, 0] = 1.0
    spin.T[2, 5] = 2.0
    spin.identity_tensor()
    assert spin.T.shape == (3, 3, 3)
    assert spin.Tx.shape == (3, 3)
    assert spin.Tx[0, 0] == 1.0
    assert spin.Tz[0, 1] == 2.0


@pytest.mark.parametrize('bad_g', [np.ones((2, 3)), np.ones(4), 'abc'])
def test_invalid_g_shape_is_rejected(bad_g) -> None:
    spin = quiet_spin()
    spin.g = bad_g
    with pytest.raises(ValueError):
        spin.identity_tensor()


@pytest.mark.parametrize('bad_a', [np.ones((1, 3)), np.ones(4)])
def test_invalid_a_shape_is_rejected(bad_a) -> None:
    spin = quiet_spin()
    spin.g = 2.0
    spin.A = bad_a
    with pytest.raises(ValueError):
        spin.identity_tensor()


def test_invalid_packed_tensor_shape_is_rejected() -> None:
    spin = quiet_spin(S=1, I=0)
    spin.g = 2.0
    spin.T = np.ones((3, 3))
    with pytest.raises(ValueError):
        spin.identity_tensor()


def test_invalid_component_matrix_is_rejected() -> None:
    spin = quiet_spin(S=1, I=0)
    spin.g = 2.0
    spin.Tx = np.ones(3)
    with pytest.raises(ValueError):
        spin.identity_tensor()


def test_b2_length_and_d_conflict_are_rejected() -> None:
    spin = quiet_spin(S=1, I=0)
    spin.g = 2.0
    spin.B2 = np.zeros(4)
    with pytest.raises(ValueError):
        spin.identity_tensor()

    spin = quiet_spin(S=1, I=0)
    spin.g = 2.0
    spin.D = 30.0
    spin.B2[2] = 1.0
    with pytest.raises(ValueError):
        spin.identity_tensor()


def test_invalid_spin_quantum_numbers_are_rejected() -> None:
    with pytest.raises(ValueError):
        quiet_spin(S=0, I=0).identity_tensor()
    with pytest.raises(ValueError):
        quiet_spin(S=-1, I=0).identity_tensor()
    with pytest.raises(ValueError):
        quiet_spin(S=0.5, I=-1).identity_tensor()


def test_non_finite_tensor_is_rejected() -> None:
    spin = quiet_spin()
    spin.g = np.array([2.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        spin.identity_tensor()


def test_r_accepts_3x5_layout() -> None:
    spin = quiet_spin(S=2, I=0)
    spin.R = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                       [6.0, 7.0, 8.0, 9.0, 10.0],
                       [11.0, 12.0, 13.0, 14.0, 15.0]])
    spin.identity_tensor()

    D, E = 1.0, 2.0
    expected = np.array([[-D / 3 + E, 5.0, 4.0],
                         [5.0, -D / 3 - E, 3.0],
                         [4.0, 3.0, 2 * D / 3]])
    assert spin.R.shape == (3, 3, 3)
    assert np.allclose(spin.Rx, expected)
    assert np.allclose(spin.R[0], expected)


def test_r_3x5_and_3x6_give_the_same_components() -> None:
    rows = [(1.0, 2.0, 3.0, 4.0, 5.0),
            (6.0, 7.0, 8.0, 9.0, 10.0),
            (11.0, 12.0, 13.0, 14.0, 15.0)]
    packed = []
    for D, E, yz, xz, xy in rows:
        packed.append([-D / 3 + E, -D / 3 - E, 2 * D / 3, yz, xz, xy])

    short = quiet_spin(S=2, I=0)
    short.R = np.array(rows)
    short.identity_tensor()

    long = quiet_spin(S=2, I=0)
    long.R = np.array(packed)
    long.identity_tensor()

    assert np.allclose(short.Rx, long.Rx)
    assert np.allclose(short.Ry, long.Ry)
    assert np.allclose(short.Rz, long.Rz)


def test_r_with_invalid_shape_is_rejected() -> None:
    spin = quiet_spin(S=2, I=0)
    spin.R = np.ones((3, 4))
    with pytest.raises(ValueError):
        spin.identity_tensor()


def test_identity_tensor_stores_3x3x3_stacks_and_is_idempotent() -> None:
    spin = quiet_spin(S=2, I=1)
    spin.g = 2.0
    spin.D = 10.0
    spin.A = 50.0
    spin.T = np.zeros((3, 6))
    spin.T[0, 0] = 1.0
    spin.F = np.zeros((3, 6))
    spin.F[1, 1] = 2.0
    spin.identity_tensor()
    spin.identity_tensor()

    assert spin.T.shape == (3, 3, 3)
    assert spin.R.shape == (3, 3, 3)
    assert spin.F.shape == (3, 3, 3)
    assert spin.Tx[0, 0] == 1.0
    assert spin.Fy[1, 1] == 2.0


def test_clean_spin_does_not_warn(caplog) -> None:
    logging.disable(logging.NOTSET)
    spin = Spin(0.5, 0)
    spin.g = 2.0
    with caplog.at_level(logging.WARNING):
        spin.identity_tensor()
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_clearing_a_nonzero_zfs_warns(caplog) -> None:
    logging.disable(logging.NOTSET)
    spin = Spin(0.5, 0)
    spin.g = 2.0
    spin.D = 30.0
    with caplog.at_level(logging.WARNING):
        spin.identity_tensor()
    assert any('zero-field' in record.getMessage() for record in caplog.records)


def test_clearing_a_nonzero_hyperfine_warns(caplog) -> None:
    logging.disable(logging.NOTSET)
    spin = Spin(1, 0)
    spin.g = 2.0
    spin.A = 50.0
    with caplog.at_level(logging.WARNING):
        spin.identity_tensor()
    assert any('hyperfine' in record.getMessage() for record in caplog.records)
