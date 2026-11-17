"""Normalize a three-component vector."""

import numpy as np


def vec_normalize(a):
    """Return ``a`` scaled to unit length."""
    m = np.sqrt((a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]))
    return a/m
