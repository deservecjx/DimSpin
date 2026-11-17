"""Spin and nuclear-spin operator matrices in the product basis."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def Sz_get(S, I):
    """Return the electron-spin z operator in the product basis."""
    levels = int(2 * S + 1)
    sz = np.zeros((levels, levels))
    for ii in range(levels):
        sz[ii, ii] = S - ii
    return np.kron(sz, np.identity(int(2 * I + 1)))


def Sp(S):
    """Return the electron-spin raising operator."""
    levels = int(2 * S + 1)
    sp = np.zeros((levels, levels))
    for ii in range(levels - 1):
        jj = ii + 1
        sp[ii, jj] = np.sqrt(S * (S + 1) - (S - ii) * (S - jj))
    return sp


def Sm(S):
    """Return the electron-spin lowering operator."""
    levels = int(2 * S + 1)
    sm = np.zeros((levels, levels))
    for jj in range(levels - 1):
        ii = jj + 1
        sm[ii, jj] = np.sqrt(S * (S + 1) - (S - ii) * (S - jj))
    return sm


@lru_cache(maxsize=None)
def Sx_get(S, I):
    """Return the electron-spin x operator in the product basis."""
    return np.kron((Sp(S) + Sm(S)) / 2, np.identity(int(2 * I + 1)))


@lru_cache(maxsize=None)
def Sy_get(S, I):
    """Return the electron-spin y operator in the product basis."""
    return np.kron((Sp(S) - Sm(S)) / (2j), np.identity(int(2 * I + 1)))


@lru_cache(maxsize=None)
def Iz_get(S, I):
    """Return the nuclear-spin z operator in the product basis."""
    levels = int(2 * I + 1)
    iz = np.zeros((levels, levels))
    for ii in range(levels):
        iz[ii, ii] = I - ii
    return np.kron(np.identity(int(2 * S + 1)), iz)


def Ip(I):
    """Return the nuclear-spin raising operator."""
    levels = int(2 * I + 1)
    ip = np.zeros((levels, levels))
    for ii in range(levels - 1):
        jj = ii + 1
        ip[ii, jj] = np.sqrt(I * (I + 1) - (I - ii) * (I - jj))
    return ip


def Im(I):
    """Return the nuclear-spin lowering operator."""
    levels = int(2 * I + 1)
    im = np.zeros((levels, levels))
    for jj in range(levels - 1):
        ii = jj + 1
        im[ii, jj] = np.sqrt(I * (I + 1) - (I - ii) * (I - jj))
    return im


@lru_cache(maxsize=None)
def Ix_get(S, I):
    """Return the nuclear-spin x operator in the product basis."""
    return np.kron(np.identity(int(2 * S + 1)), (Ip(I) + Im(I)) / 2)


@lru_cache(maxsize=None)
def Iy_get(S, I):
    """Return the nuclear-spin y operator in the product basis."""
    return np.kron(np.identity(int(2 * S + 1)), (Ip(I) - Im(I)) / (2j))
