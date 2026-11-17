"""Zero-field splitting term built from a D tensor."""

import numpy as np

from .Spin_Operators import Sx_get, Sy_get, Sz_get


def H_ZFS(D, S, I):
    """
    Inputs: 3x3 D tensor (MHz), electron spin S, nuclear spin I.

    Returns: the zero-field splitting Hamiltonian.

    Notes: the tensor is assumed symmetric; a nonsymmetric D tensor gives a non-Hermitian matrix.
    """
    S_ops = (Sx_get(S, I), Sy_get(S, I), Sz_get(S, I))
    D = np.asarray(D, dtype=float)

    H = np.zeros_like(S_ops[0])
    for i in range(3):
        for j in range(3):
            if D[i, j] != 0.0:
                H = H + D[i, j] * (S_ops[i] @ S_ops[j])
    return H
