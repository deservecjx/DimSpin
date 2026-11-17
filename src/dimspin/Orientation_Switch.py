"""Rotation that carries one orientation direction onto another."""

import math

import numpy as np

from .Rodrigues_Rotation import rotation_matrix


def orientation_switch(ori, iniori):
    """Return the rotation matrix that maps ``iniori`` onto ``ori``."""
    ori = np.asarray(ori, dtype=float)
    iniori = np.asarray(iniori, dtype=float)

    rotaxis1 = np.cross(iniori, ori)
    cos_angle = np.dot(ori, iniori) / np.sqrt(np.dot(ori, ori) * np.dot(iniori, iniori))
    rotang1 = math.acos(float(np.clip(cos_angle, -1.0, 1.0)))

    if np.linalg.norm(rotaxis1) < 1e-12:
        if cos_angle >= 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, iniori)) > 0.9 * np.linalg.norm(iniori):
            helper = np.array([0.0, 1.0, 0.0])
        return rotation_matrix(np.cross(iniori, helper), math.pi)

    return rotation_matrix(rotaxis1, rotang1)
