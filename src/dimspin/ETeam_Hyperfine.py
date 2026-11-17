"""Electric-field modulation of the hyperfine term through an F tensor."""

import numpy as np

from .Spin_Operators import Ix_get, Iy_get, Iz_get, Sx_get, Sy_get, Sz_get


def H_EMod_F(Fx, Fy, Fz, S, I):
    """
    Inputs: 3x3 tensors Fx, Fy, Fz, the derivatives of A with respect to the
    electric-field components along x, y, z; electron spin S; nuclear spin I.

    Returns: the F-tensor part of the electric-field modulated hyperfine term.
    """
    S_ops = (Sx_get(S, I), Sy_get(S, I), Sz_get(S, I))
    I_ops = (Ix_get(S, I), Iy_get(S, I), Iz_get(S, I))
    tensors = (np.asarray(Fx, dtype=float), np.asarray(Fy, dtype=float), np.asarray(Fz, dtype=float))

    H = np.zeros_like(S_ops[0])
    for F in tensors:
        for i in range(3):
            for j in range(3):
                if F[i, j] != 0.0:
                    H = H + F[i, j] * (S_ops[i] @ I_ops[j])
    return H
