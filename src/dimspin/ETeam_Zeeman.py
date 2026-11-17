"""Electric-field modulation of the Zeeman term through a T tensor."""

import numpy as np

from .Spin_Operators import Sx_get, Sy_get, Sz_get


def H_EMod_T(Bz, Tx, Ty, Tz, S, I):
    """
    Inputs: field Bz (G); 3x3 tensors Tx, Ty, Tz, the derivatives of g with
    respect to the electric-field components along x, y, z; electron spin S;
    nuclear spin I.

    Returns: the T-tensor part of the electric-field modulated Zeeman term.

    Notes: the magnetic field is along z only, so the third column of each
    tensor enters and the other columns do not contribute.
    """
    S_ops = (Sx_get(S, I), Sy_get(S, I), Sz_get(S, I))
    tensors = (np.asarray(Tx, dtype=float), np.asarray(Ty, dtype=float), np.asarray(Tz, dtype=float))

    H = np.zeros_like(S_ops[0])
    for T in tensors:
        for i in range(3):
            if T[i, 2] != 0.0:
                H = H + T[i, 2] * S_ops[i]
    return Bz * H
