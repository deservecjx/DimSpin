"""Tests for spin and nuclear-spin operator matrices."""

import numpy as np

from dimspin.Spin_Operators import (
    Ix_get,
    Iy_get,
    Iz_get,
    Sm,
    Sp,
    Sx_get,
    Sy_get,
    Sz_get,
)


def test_electron_spin_commutation() -> None:
    S, I = 1.0, 0.0
    Sx = Sx_get(S, I)
    Sy = Sy_get(S, I)
    Sz = Sz_get(S, I)
    assert np.allclose(Sx @ Sy - Sy @ Sx, 1j * Sz)
    assert np.allclose(Sy @ Sz - Sz @ Sy, 1j * Sx)
    assert np.allclose(Sz @ Sx - Sx @ Sz, 1j * Sy)


def test_nuclear_spin_commutation() -> None:
    S, I = 1/2, 1
    Ix = Ix_get(S, I)
    Iy = Iy_get(S, I)
    Iz = Iz_get(S, I)
    assert np.allclose(Ix @ Iy - Iy @ Ix, 1j * Iz)
    assert np.allclose(Iy @ Iz - Iz @ Iy, 1j * Ix)
    assert np.allclose(Iz @ Ix - Ix @ Iz, 1j * Iy)


def test_sz_eigenvalues() -> None:
    S = 2.0
    diagonal = np.diag(Sz_get(S, 0.0))
    assert np.allclose(diagonal, [2.0, 1.0, 0.0, -1.0, -2.0])


def test_ladder_operators_build_sx() -> None:
    S = 1.0
    assert np.allclose(Sx_get(S, 0.0), (Sp(S) + Sm(S)) / 2)


def test_operator_dimensions() -> None:
    S, I = 1.5, 2.0
    expected = int((2 * S + 1) * (2 * I + 1))
    for operator in (Sx_get, Sy_get, Sz_get, Ix_get, Iy_get, Iz_get):
        matrix = operator(S, I)
        assert matrix.shape == (expected, expected)
        assert np.allclose(matrix, matrix.conj().T)
