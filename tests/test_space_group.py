"""Tests for the space-group data, lattice matrices and orientation expansion."""

import numpy as np
import pytest
from toy_spin import ToySpin

from dimspin.Space_Group import crystal_system_for, get_space_group
from dimspin.Space_Group_Expand import build_space_group_orientations
from dimspin.Space_Group_L import get_L, get_lattice_info, lattice_matrix, standard_lattice


def test_known_space_group_orders_and_symbols() -> None:
    expected = {
        1: ("P1", 1, "triclinic"),
        2: ("P-1", 2, "triclinic"),
        3: ("P2", 2, "monoclinic"),
        62: ("Pnma", 8, "orthorhombic"),
        194: ("P6_3/mmc", 24, "hexagonal"),
        195: ("P23", 12, "cubic"),
    }
    for number, (symbol, order, system) in expected.items():
        group = get_space_group(number)
        assert group.symbol == symbol
        assert group.order == order
        assert group.crystal_system == system


def test_unknown_space_group_raises() -> None:
    with pytest.raises(ValueError):
        get_space_group(0)
    with pytest.raises(ValueError):
        get_space_group(231)


def test_crystal_system_boundaries() -> None:
    assert crystal_system_for(2) == "triclinic"
    assert crystal_system_for(3) == "monoclinic"
    assert crystal_system_for(74) == "orthorhombic"
    assert crystal_system_for(75) == "tetragonal"
    assert crystal_system_for(143) == "trigonal"
    assert crystal_system_for(168) == "hexagonal"
    assert crystal_system_for(195) == "cubic"
    with pytest.raises(ValueError):
        crystal_system_for(231)


def test_hexagonal_lattice_has_120_degree_gamma() -> None:
    L = standard_lattice("hexagonal")
    metric = L.T @ L
    assert np.allclose(metric, [[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.isclose(np.linalg.det(L), np.sin(np.radians(120.0)))


def test_orthorhombic_lattice_is_orthonormal() -> None:
    assert np.allclose(standard_lattice("orthorhombic"), np.eye(3))


def test_lattice_matrix_from_cell_parameters() -> None:
    L = lattice_matrix(2.0, 3.0, 4.0, 90.0, 90.0, 90.0)
    assert np.allclose(L, np.diag([2.0, 3.0, 4.0]))


def test_get_lattice_info_and_get_L() -> None:
    info = get_lattice_info(194)
    assert info["number"] == 194
    assert info["crystal_system"] == "hexagonal"
    assert np.allclose(info["lattice"], get_L(194))


def test_general_and_special_positions() -> None:
    p2 = get_space_group(3)
    assert len(p2.equivalent_fractional_positions(np.array([0.1, 0.2, 0.3]))) == 2

    inversion = get_space_group(2)
    assert len(inversion.equivalent_fractional_positions(np.array([0.1, 0.2, 0.3]))) == 2
    assert len(inversion.equivalent_fractional_positions(np.array([0.5, 0.5, 0.5]))) == 1


def test_cartesian_rotation_for_p2_maps_axes() -> None:
    p2 = get_space_group(3)
    rotations = p2.cartesian_rotations(np.eye(3))
    assert any(np.allclose(R, np.diag([-1.0, 1.0, -1.0])) for R in rotations)


def make_ref_spin(g) -> ToySpin:
    spin = ToySpin(1/2, 0.5, np.eye(3))
    spin.g = g
    spin.A = 245.0 * np.eye(3)
    return spin


def test_p2_general_position_splits_tilted_g_into_two_sites() -> None:
    g = np.array([[2.0, 0.1, 0.0],
                  [0.1, 2.2, 0.0],
                  [0.0, 0.0, 2.4]])
    orientations, multiplicities = build_space_group_orientations(make_ref_spin(g), 3)
    assert len(orientations) == 2
    assert multiplicities == [1, 1]


def test_p2_diagonal_g_is_magnetically_equivalent() -> None:
    g = np.diag([2.0, 2.2, 2.4])
    orientations, multiplicities = build_space_group_orientations(make_ref_spin(g), 3)
    assert len(orientations) == 1
    assert multiplicities == [2]


def test_inversion_centre_special_position_gives_one_site() -> None:
    g = np.array([[2.0, 0.1, 0.2],
                  [0.1, 2.2, 0.3],
                  [0.2, 0.3, 2.4]])
    orientations, multiplicities = build_space_group_orientations(
        make_ref_spin(g), 2, frac_pos=(0.5, 0.5, 0.5))
    assert len(orientations) == 1
    assert multiplicities == [1]


def test_cubic_group_multiplicity_sum_equals_group_order() -> None:
    g = np.array([[2.0, 0.15, 0.05],
                  [0.15, 2.2, 0.25],
                  [0.05, 0.25, 2.4]])
    orientations, multiplicities = build_space_group_orientations(make_ref_spin(g), 195)
    assert sum(multiplicities) == 12
    assert len(orientations) == len(multiplicities)


def test_rhombohedral_group_uses_hexagonal_axes() -> None:
    group = get_space_group(146)
    assert group.order == 9
    L = get_L(146)
    for R in group.cartesian_rotations(L):
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(abs(np.linalg.det(R)), 1.0)


def test_rhombohedral_expansion_multiplicity_sum() -> None:
    g = np.array([[2.0, 0.15, 0.05],
                  [0.15, 2.2, 0.25],
                  [0.05, 0.25, 2.4]])
    orientations, multiplicities = build_space_group_orientations(make_ref_spin(g), 146)
    assert sum(multiplicities) == 9
    assert len(orientations) == len(multiplicities)


def test_acentric_orthorhombic_expansion_multiplicity_sum() -> None:
    g = np.array([[2.0, 0.1, 0.0],
                  [0.1, 2.2, 0.05],
                  [0.0, 0.05, 2.4]])
    orientations, multiplicities = build_space_group_orientations(make_ref_spin(g), 38)
    assert sum(multiplicities) == 8


def test_all_groups_give_orthogonal_cartesian_rotations() -> None:
    for number in range(1, 231):
        L = get_L(number)
        for R in get_space_group(number).cartesian_rotations(L):
            assert np.allclose(R @ R.T, np.eye(3)), number
            assert np.isclose(abs(np.linalg.det(R)), 1.0), number


def test_hexagonal_group_orientations_are_orthogonal() -> None:
    g = np.array([[2.0, 0.15, 0.05],
                  [0.15, 2.2, 0.25],
                  [0.05, 0.25, 2.4]])
    orientations, multiplicities = build_space_group_orientations(make_ref_spin(g), 194)
    assert sum(multiplicities) == 24
    for R in orientations:
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(abs(np.linalg.det(R)), 1.0)
