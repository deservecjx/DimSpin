"""Standard lattice matrices for the crystal systems.

Space-group operations are given as integer matrices R in fractional
coordinates, while g, D and A tensors live in Cartesian coordinates. The two
are related by

    R_cart = L @ R @ L^-1,

with L the lattice matrix whose columns are the vectors a, b, c. Absolute
lengths cancel in this expression, so standard cell parameters are enough.

Conventions: a lies along Cartesian x, b lies in the xy plane, and c follows
from the angles alpha = b.c, beta = a.c, gamma = a.b in degrees. Only the
trigonal and hexagonal systems need a non-trivial L (gamma = 120 degrees);
the other systems use an orthonormal L. Rhombohedral groups are expressed in
hexagonal axes, which is the standard setting used by Space_Group.

Public entry points: :func:`get_L` and :func:`get_lattice_info`.
"""

from __future__ import annotations

import numpy as np


def crystal_system_for(number: int) -> str:
    """
    Inputs: space group number.

    Returns: the crystal system name.
    """
    number = int(number)
    if number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "trigonal"
    if number <= 194:
        return "hexagonal"
    if number <= 230:
        return "cubic"
    raise ValueError(f"invalid space group number: {number}")


def lattice_matrix(a, b, c, alpha, beta, gamma):
    """
    Inputs: cell lengths a, b, c; cell angles alpha, beta, gamma in degrees.

    Returns: the 3x3 lattice matrix whose columns are a, b, c.

    Notes: a is along x and b lies in the xy plane.
    """
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sg = np.sin(gamma)

    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * cg, b * sg, 0.0])
    cy = (ca - cb * cg) / sg
    cz = np.sqrt(max(0.0, 1.0 - cb ** 2 - cy ** 2))
    c_vec = np.array([c * cb, c * cy, c * cz])

    return np.array([a_vec, b_vec, c_vec]).T


STANDARD_PARAMETERS = {
    "triclinic":    dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0),
    "monoclinic":   dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0),
    "orthorhombic": dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0),
    "tetragonal":   dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0),
    "trigonal":     dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=120.0),
    "hexagonal":    dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=120.0),
    "cubic":        dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0),
}


def standard_lattice(crystal_system: str) -> np.ndarray:
    """
    Inputs: crystal system name.

    Returns: the 3x3 lattice matrix in the standard setting.
    """
    key = str(crystal_system).lower()
    if key not in STANDARD_PARAMETERS:
        raise ValueError(f"unknown crystal system: {crystal_system}")
    p = STANDARD_PARAMETERS[key]
    return lattice_matrix(p["a"], p["b"], p["c"], p["alpha"], p["beta"], p["gamma"])


ALL_STANDARD_LATTICES = {
    system: standard_lattice(system) for system in STANDARD_PARAMETERS
}


def get_L(number):
    """
    Inputs: space group number.

    Returns: the standard lattice matrix L for that group.
    """
    return standard_lattice(crystal_system_for(number))


def get_lattice_info(number):
    """
    Inputs: space group number.

    Returns: a dict with the number, crystal system, lattice matrix and cell
    parameters.
    """
    number = int(number)
    system = crystal_system_for(number)
    return {
        "number": number,
        "crystal_system": system,
        "lattice": standard_lattice(system),
        "parameters": dict(STANDARD_PARAMETERS[system]),
    }


__all__ = [
    "crystal_system_for",
    "lattice_matrix",
    "standard_lattice",
    "ALL_STANDARD_LATTICES",
    "get_L",
    "get_lattice_info",
]
