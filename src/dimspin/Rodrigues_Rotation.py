"""Rotation matrices built with the Rodrigues formula."""

import numpy as np


def rotation_matrix(axis, angle):
    """Return the rotation matrix around ``axis`` by ``angle`` radians."""
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
