"""Tests for the Hamiltonian terms and the high-order Stevens operators."""

import numpy as np
import pytest

from dimspin.ETeam_Hyperfine import H_EMod_F
from dimspin.ETeam_Zeeman import H_EMod_T
from dimspin.ETeam_ZFS import H_EMod_R
from dimspin.Rodrigues_Rotation import rotation_matrix
from dimspin.Steven_Operator import stev_op
from dimspin.Team_Hyperfine import H_Hyperfine
from dimspin.Team_Zeeman import MU_B_H, H_Zeeman
from dimspin.Team_ZFS import H_ZFS
from dimspin.Team_ZFS_HighOrder import B_arrays_to_dict, H_ZFS_HighOrder


def random_tensor(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(3, 3))


def random_symmetric_tensor(seed: int) -> np.ndarray:
    tensor = random_tensor(seed)
    return (tensor + tensor.T) / 2


def test_zeeman_free_electron_splitting() -> None:
    g = 2.0 * np.eye(3)
    field = 3300.0
    energies = np.linalg.eigvalsh(H_Zeeman(field, g, 0.5, 0))
    assert np.allclose(energies, [-MU_B_H * 2.0 * field / 2, MU_B_H * 2.0 * field / 2])


def test_zeeman_uses_only_the_z_row_of_g() -> None:
    field = 1200.0
    g1 = np.zeros((3, 3))
    g2 = np.ones((3, 3))
    g1[2] = [0.1, 0.2, 2.0]
    g2[2] = [0.1, 0.2, 2.0]
    assert np.allclose(H_Zeeman(field, g1, 1.5, 0.5), H_Zeeman(field, g2, 1.5, 0.5))


def test_zeeman_is_linear_in_field() -> None:
    g = random_tensor(1)
    assert np.allclose(H_Zeeman(200.0, g, 2, 1), 2.0 * H_Zeeman(100.0, g, 2, 1))


def test_zfs_axial_eigenvalues() -> None:
    D = 30.0
    tensor = np.diag([-D / 3, -D / 3, 2 * D / 3])
    energies = np.sort(np.linalg.eigvalsh(H_ZFS(tensor, 1, 0)))
    assert np.allclose(energies, [-2 * D / 3, D / 3, D / 3])


def test_isotropic_hyperfine_eigenvalues() -> None:
    a = 40.0
    energies = np.sort(np.linalg.eigvalsh(H_Hyperfine(a * np.eye(3), 0.5, 0.5)))
    assert np.allclose(energies, [-3 * a / 4, a / 4, a / 4, a / 4])


def test_terms_are_hermitian() -> None:
    S, I = 2, 1
    matrices = [
        H_ZFS(random_symmetric_tensor(2), S, I),
        H_Hyperfine(random_symmetric_tensor(3), S, I),
        H_EMod_T(300.0, random_symmetric_tensor(4), random_symmetric_tensor(5), random_symmetric_tensor(6), S, I),
        H_EMod_R(random_symmetric_tensor(7), random_symmetric_tensor(8), random_symmetric_tensor(9), S, I),
        H_EMod_F(random_symmetric_tensor(10), random_symmetric_tensor(11), random_symmetric_tensor(12), S, I),
    ]
    for matrix in matrices:
        assert np.allclose(matrix, matrix.conj().T)


def test_zfs_matches_rank_two_stevens_operators() -> None:
    D, E, S = 30.0, 5.0, 2
    tensor = np.diag([-D / 3 + E, -D / 3 - E, 2 * D / 3])
    expected = (D / 3) * stev_op(2, 0, S) + E * stev_op(2, 2, S)
    assert np.allclose(H_ZFS(tensor, S, 0), expected)


def test_stevens_operators_are_hermitian() -> None:
    S, I = 3, 0.5
    R = rotation_matrix([1.0, 2.0, 3.0], 0.6)
    for k in (2, 4, 6):
        for q in range(-k, k + 1):
            operator = stev_op(k, q, S, I, R)
            assert np.allclose(operator, operator.conj().T)


def test_stevens_operators_vanish_above_twice_the_spin() -> None:
    for S in (0.5, 1.5, 2.5):
        for k in (2, 4, 6):
            if k > 2 * S:
                for q in range(-k, k + 1):
                    assert np.allclose(stev_op(k, q, S), 0.0)


def test_stevens_rank_six_operator_is_nonzero_for_s_three() -> None:
    assert np.linalg.norm(stev_op(6, 6, 3)) > 1.0


def test_stevens_axial_operators_are_traceless() -> None:
    for k in (2, 4, 6):
        assert np.isclose(np.trace(stev_op(k, 0, 3)), 0.0)


def test_stevens_commutator_with_sz() -> None:
    S = 3
    sz = np.diag([S - i for i in range(2 * S + 1)]).astype(complex)
    for k in (2, 4, 6):
        for q in range(1, k + 1):
            operator = stev_op(k, q, S)
            commutator = sz @ operator - operator @ sz
            assert np.allclose(commutator, 1.0j * q * stev_op(k, -q, S))


def test_b_arrays_to_dict_maps_orders() -> None:
    B = B_arrays_to_dict(B2=[0.0, 0.0, 1.0, 2.0, 0.0], B4=np.arange(9))
    assert B[(2, 0)] == 1.0
    assert B[(2, 1)] == 2.0
    assert (2, 2) not in B
    assert (4, -4) not in B
    assert B[(4, 4)] == 8.0


def test_b_arrays_to_dict_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        B_arrays_to_dict(B4=[1.0, 2.0, 3.0])


def test_high_order_matches_d_tensor() -> None:
    D, E, S = 30.0, 5.0, 2
    B = B_arrays_to_dict(B2=[0.0, 0.0, D / 3, 0.0, E])
    tensor = np.diag([-D / 3 + E, -D / 3 - E, 2 * D / 3])
    assert np.allclose(H_ZFS_HighOrder(B, S, 0), H_ZFS(tensor, S, 0))


def test_high_order_with_identity_rotation() -> None:
    B = {(4, 0): 1.5, (6, 4): -0.3}
    assert np.allclose(H_ZFS_HighOrder(B, 2.5, 0, R=np.eye(3)), H_ZFS_HighOrder(B, 2.5, 0))


def test_high_order_axial_term_is_invariant_under_rotation_about_z() -> None:
    B = {(4, 0): 2.0, (6, 0): 0.5}
    R = rotation_matrix([0.0, 0.0, 1.0], 0.7)
    assert np.allclose(H_ZFS_HighOrder(B, 3, 0, R=R), H_ZFS_HighOrder(B, 3, 0))


def test_emod_t_matches_zeeman_with_a_combined_g_row() -> None:
    S, I = 1.5, 0.5
    Tx, Ty, Tz = random_tensor(13), random_tensor(14), random_tensor(15)
    g = np.zeros((3, 3))
    g[2] = (Tx + Ty + Tz)[:, 2] / MU_B_H
    assert np.allclose(H_EMod_T(250.0, Tx, Ty, Tz, S, I), H_Zeeman(250.0, g, S, I))
    assert np.allclose(H_EMod_T(-250.0, Tx, Ty, Tz, S, I), -H_EMod_T(250.0, Tx, Ty, Tz, S, I))


def test_emod_r_matches_zfs_with_a_summed_tensor() -> None:
    Rx, Ry, Rz = random_tensor(16), random_tensor(17), random_tensor(18)
    assert np.allclose(H_EMod_R(Rx, Ry, Rz, 2, 1), H_ZFS(Rx + Ry + Rz, 2, 1))


def test_emod_f_matches_hyperfine_with_a_summed_tensor() -> None:
    Fx, Fy, Fz = random_tensor(19), random_tensor(20), random_tensor(21)
    assert np.allclose(H_EMod_F(Fx, Fy, Fz, 1.5, 1), H_Hyperfine(Fx + Fy + Fz, 1.5, 1))
