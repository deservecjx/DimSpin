"""Electron-nuclear hyperfine term built from an A tensor."""

import numpy as np

from .Spin_Operators import Ix_get, Iy_get, Iz_get, Sx_get, Sy_get, Sz_get


def H_Hyperfine(A, S, I):
    """
    Inputs: 3x3 A tensor (MHz), electron spin S, nuclear spin I.

    Returns: the hyperfine Hamiltonian.
    """
    S_ops = (Sx_get(S, I), Sy_get(S, I), Sz_get(S, I))
    I_ops = (Ix_get(S, I), Iy_get(S, I), Iz_get(S, I))
    A = np.asarray(A, dtype=float)

    H = np.zeros_like(S_ops[0])
    for i in range(3):
        for j in range(3):
            if A[i, j] != 0.0:
                H = H + A[i, j] * (S_ops[i] @ I_ops[j])
    return H
