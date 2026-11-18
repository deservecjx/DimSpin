"""Tests for tensor construction helpers."""

import numpy as np

from dimspin.Tensor import Atensor, Dtensor, Tijk2Txyz, gtensor, unpack_tensor_3x5, unpack_tensor_3x6


def test_gtensor() -> None:
    assert np.allclose(gtensor(2.0, 2.1, 1.9), np.diag([2.0, 2.1, 1.9]))


def test_dtensor_is_traceless_and_has_expected_eigenvalues() -> None:
    D, E = 9.0, 3.0
    tensor = Dtensor(D, E)
    assert np.isclose(np.trace(tensor), 0.0)
    expected = sorted([-D / 3 + E, -D / 3 - E, 2 * D / 3])
    assert np.allclose(sorted(np.linalg.eigvalsh(tensor)), expected)


def test_atensor() -> None:
    assert np.allclose(Atensor(100.0, 110.0, 120.0), np.diag([100.0, 110.0, 120.0]))


def test_tijk2txyz_mapping_and_symmetry() -> None:
    T = np.arange(1.0, 19.0).reshape(3, 6)
    Tx, Ty, Tz = Tijk2Txyz(T)

    assert Tx[0, 1] == T[0, 5]
    assert Tx[0, 2] == T[0, 4]
    assert Tx[1, 2] == T[0, 3]
    assert Ty[1, 0] == T[1, 5]
    assert Tz[2, 2] == T[2, 2]

    for matrix in (Tx, Ty, Tz):
        assert np.allclose(matrix, matrix.T)


def test_unpack_tensor_3x6_matches_component_split() -> None:
    packed = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]])
    stack = unpack_tensor_3x6(packed)
    assert stack.shape == (3, 3, 3)
    assert np.allclose(stack[0], [[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])


def test_unpack_tensor_3x5_is_traceless() -> None:
    short = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0, 9.0, 10.0],
                      [11.0, 12.0, 13.0, 14.0, 15.0]])
    stack = unpack_tensor_3x5(short)
    assert stack.shape == (3, 3, 3)
    assert np.isclose(np.trace(stack[0]), 0.0)
    assert np.allclose(stack[0], [[-1.0 / 3 + 2, 5.0, 4.0],
                                  [5.0, -1.0 / 3 - 2, 3.0],
                                  [4.0, 3.0, 2.0 / 3]])
