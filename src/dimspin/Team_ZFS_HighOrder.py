"""Assembly of the high-order zero-field splitting Hamiltonian.

H = sum_{k,q} B_k^q * O_k^q(S), with the Stevens operators from
Steven_Operator.stev_op.
"""

import numpy as np

from .Steven_Operator import stev_op


def B_arrays_to_dict(B2=None, B4=None, B6=None):
    """
    Inputs: optional B2, B4, B6 arrays with 2k+1 entries each, ordered by
    q = -k ... k.

    Returns: a {(k, q): B_k^q} dict holding only the nonzero coefficients; an
    array of the wrong length raises ValueError.
    """
    B = {}
    for k, Bk in [(2, B2), (4, B4), (6, B6)]:
        if Bk is None:
            continue
        Bk = np.asarray(Bk, dtype=float).reshape(-1)
        if Bk.size != 2 * k + 1:
            raise ValueError(f"B{k} must contain exactly {2 * k + 1} elements (q = -{k} ... {k}).")
        for i, value in enumerate(Bk):
            q = i - k
            if value != 0:
                B[(k, q)] = float(value)
    return B


def H_ZFS_HighOrder(B, S, I, R=None):
    """
    Inputs: dict {(k, q): B_k^q} in MHz, electron and nuclear spins S and I,
    optional 3x3 rotation R taking laboratory operators to the tensor frame.

    Returns: the high-order zero-field splitting Hamiltonian.
    """
    nS = int(round(2 * S + 1))
    nI = int(round(2 * I + 1))
    H = np.zeros((nS * nI, nS * nI), dtype=complex)
    for (k, q), Bkq in B.items():
        if Bkq == 0:
            continue
        H = H + Bkq * stev_op(k, q, S, I, R)
    return H
