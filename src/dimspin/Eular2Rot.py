"""Euler-angle and orientation helpers.

Euler angles use the zyz convention and are expressed in radians.
"""

import numpy as np


def sin(a):
    return np.sin(a)


def cos(a):
    return np.cos(a)


def eular2rotzyz(alpha, beta, gamma):
    """Return the zyz rotation matrix for angles in radians."""
    Rz_alpha = np.array([[cos(alpha), sin(alpha), 0], [-sin(alpha), cos(alpha), 0], [0, 0, 1]])
    Ry_beta = np.array([[cos(beta), 0, -sin(beta)], [0, 1, 0], [sin(beta), 0, cos(beta)]])
    Rz_gamma = np.array([[cos(gamma), sin(gamma), 0], [-sin(gamma), cos(gamma), 0], [0, 0, 1]])
    return Rz_alpha @ Ry_beta @ Rz_gamma


def resolve_orientation(ori):
    """Normalize an orientation to a 3x3 orthogonal matrix.

    Accepts a 3x3 matrix or three Euler angles in the zyz convention (radians).
    ``None`` or a zero vector gives the identity matrix.

    The determinant may be +1 for a proper rotation or -1 for an improper
    operation such as inversion or a mirror. Second-order tensors transform as
    ``R @ T @ R.T`` for either sign.

    High-order zero-field terms rotate the spin operators with ``R``
    (``S_tensor = R @ S_lab``). The implemented ranks k = 2, 4, 6 are even,
    so det(R) = -1 is acceptable.
    """
    if ori is None:
        return np.eye(3)
    ori = np.array(ori)
    if ori.shape == (3, 3):
        R = ori
    elif ori.shape == (3,) or ori.size == 3:
        R = eular2rotzyz(ori[0], ori[1], ori[2])
    else:
        raise ValueError('Orientation must be a 3x3 matrix or 3 Euler angles, got %s' % (ori.shape,))
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError('Orientation matrix is not orthogonal.')
    if not np.isclose(abs(np.linalg.det(R)), 1.0, atol=1e-6):
        raise ValueError('Orientation must be an orthogonal matrix with det=±1.')
    return R
