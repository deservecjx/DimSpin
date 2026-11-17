"""Fibonacci-sphere sampling of crystallite orientations."""

import math

import numpy as np


def fibonacci_orientation(n):
    """Return ``n`` approximately uniform unit vectors on the sphere."""
    data = np.zeros((n, 3))
    pi = math.pi
    phi = (np.sqrt(5) - 1) / 2
    for i in range(0, n):
        data[i][2] = (2 * i + 1) / n - 1
        item1 = np.sqrt(1 - data[i][2] * data[i][2])
        item2 = 2 * pi * (i + 1) * phi
        data[i][0] = item1 * math.cos(item2)
        data[i][1] = item1 * math.sin(item2)
    return data
