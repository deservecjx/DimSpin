"""Electric-field modulation of the zero-field splitting term through an R tensor."""

import numpy as np

from .Spin_Operators import Sx_get, Sy_get, Sz_get


def H_EMod_R(Rx, Ry, Rz, S, I):
    """
    Inputs: 3x3 tensors Rx, Ry, Rz, the derivatives of D with respect to the
    electric-field components along x, y, z; electron spin S; nuclear spin I.

    Returns: the R-tensor part of the electric-field modulated ZFS term.
    """
    S_ops = (Sx_get(S, I), Sy_get(S, I), Sz_get(S, I))
    tensors = (np.asarray(Rx, dtype=float), np.asarray(Ry, dtype=float), np.asarray(Rz, dtype=float))

    H = np.zeros_like(S_ops[0])
    for R in tensors:
        for i in range(3):
            for j in range(3):
                if R[i, j] != 0.0:
                    H = H + R[i, j] * (S_ops[i] @ S_ops[j])
    return H
