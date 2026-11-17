"""Extended Stevens operators for high-order zero-field splitting.

The Hamiltonian is written as H = sum_{k,q} B_k^q * O_k^q(S), with ranks
k = 2, 4, 6 and orders q = -k ... k. The operators are real Hermitian
matrices built from Sx, Sy, Sz, and the coefficients B_k^q carry the energy
unit (MHz here).

q > 0 gives cosine-type operators and q < 0 sine-type operators, following the
Rudowicz/EasySpin convention. Sign conventions for q < 0 differ between
sources, so coefficients taken from the literature should be checked.

Rank k = 2 is equivalent to Team_ZFS.H_ZFS with B_2^0 = D/3 and B_2^2 = E.
O_k^q vanishes identically when k > 2S.
"""

import numpy as np

from .Spin_Operators import Sx_get, Sy_get, Sz_get


def _sym(A, B):
    """
    Inputs: two square matrices A and B.

    Returns: the anticommutator A @ B + B @ A.
    """
    return A @ B + B @ A


def stev_op(k, q, S, I=0, R=None):
    """
    Inputs: rank k in {2, 4, 6}; order q with -k <= q <= k; electron and
    nuclear spins S and I; optional 3x3 rotation R taking laboratory operators
    to the tensor frame (S_tensor = R @ S_lab).

    Returns: the extended Stevens operator O_k^q as a (2S+1)(2I+1) Hermitian
    matrix, identity on the nuclear space.

    Notes: q > 0 gives cosine-type and q < 0 sine-type operators, and the sign
    convention for q < 0 differs between sources. O_k^q is zero for k > 2S.
    """
    n = int(round((2 * S + 1) * (2 * I + 1)))
    if k > 2 * S:
        return np.zeros((n, n))

    Sx = Sx_get(S, I)
    Sy = Sy_get(S, I)
    Sz = Sz_get(S, I)

    if R is not None:
        R = np.asarray(R, dtype=float)
        Sx, Sy, Sz = (
            R[0, 0] * Sx + R[0, 1] * Sy + R[0, 2] * Sz,
            R[1, 0] * Sx + R[1, 1] * Sy + R[1, 2] * Sz,
            R[2, 0] * Sx + R[2, 1] * Sy + R[2, 2] * Sz,
        )

    Sp_mat = Sx + 1.0j * Sy
    Sm_mat = Sx - 1.0j * Sy
    Id = np.eye(n)
    s = S

    if k == 2:
        if q == 0:
            return 3.0 * (Sz @ Sz) - s * (s + 1.0) * Id
        if q == 1:
            return 0.5 * _sym(Sz, Sx)
        if q == -1:
            return 0.5 * _sym(Sz, Sy)
        if q == 2:
            return Sx @ Sx - Sy @ Sy
        if q == -2:
            return _sym(Sx, Sy)

    if k == 4:
        Sz2 = Sz @ Sz
        Sz3 = Sz2 @ Sz
        Sz4 = Sz2 @ Sz2
        if q == 0:
            return (35.0 * Sz4
                    - (30.0 * s * (s + 1.0) - 25.0) * Sz2
                    + (3.0 * s ** 2 * (s + 1.0) ** 2 - 6.0 * s * (s + 1.0)) * Id)
        A1 = 7.0 * Sz3 - 3.0 * s * (s + 1.0) * Sz - Sz
        if q == 1:
            return 0.5 * _sym(A1, Sx)
        if q == -1:
            return 0.5 * _sym(A1, Sy)
        B2 = 7.0 * Sz2 - s * (s + 1.0) * Id - 5.0 * Id
        if q == 2:
            return 0.5 * _sym(B2, Sx @ Sx - Sy @ Sy)
        if q == -2:
            return 0.5 * _sym(B2, _sym(Sx, Sy))
        P3 = Sp_mat @ Sp_mat @ Sp_mat
        M3 = Sm_mat @ Sm_mat @ Sm_mat
        if q == 3:
            return 0.25 * _sym(Sz, P3 + M3)
        if q == -3:
            return 0.25 * _sym(Sz, (P3 - M3) / 1.0j)
        P4 = Sp_mat @ P3
        M4 = Sm_mat @ M3
        if q == 4:
            return 0.5 * (P4 + M4)
        if q == -4:
            return 0.5 * (P4 - M4) / 1.0j

    if k == 6:
        Sz2 = Sz @ Sz
        Sz3 = Sz2 @ Sz
        Sz4 = Sz2 @ Sz2
        Sz5 = Sz4 @ Sz
        Sz6 = Sz3 @ Sz3
        if q == 0:
            return (231.0 * Sz6
                    - (315.0 * s * (s + 1.0) - 735.0) * Sz4
                    + (105.0 * s ** 2 * (s + 1.0) ** 2
                       - 525.0 * s * (s + 1.0) + 294.0) * Sz2
                    - (5.0 * s ** 3 * (s + 1.0) ** 3
                       - 40.0 * s ** 2 * (s + 1.0) ** 2
                       + 60.0 * s * (s + 1.0)) * Id)
        A1 = (33.0 * Sz5
              - (30.0 * s * (s + 1.0) - 15.0) * Sz3
              - (5.0 * s * (s + 1.0) - 10.0) * Sz)
        if q == 1:
            return 0.5 * _sym(A1, Sx)
        if q == -1:
            return 0.5 * _sym(A1, Sy)
        B2 = (33.0 * Sz4
              - (18.0 * s * (s + 1.0) - 123.0) * Sz2
              + (s ** 2 * (s + 1.0) ** 2 - 10.0 * s * (s + 1.0) + 102.0) * Id)
        if q == 2:
            return 0.5 * _sym(B2, Sx @ Sx - Sy @ Sy)
        if q == -2:
            return 0.5 * _sym(B2, _sym(Sx, Sy))
        C3 = 11.0 * Sz3 - (3.0 * s * (s + 1.0) + 59.0) * Sz
        P3 = Sp_mat @ Sp_mat @ Sp_mat
        M3 = Sm_mat @ Sm_mat @ Sm_mat
        if q == 3:
            return 0.25 * _sym(C3, P3 + M3)
        if q == -3:
            return 0.25 * _sym(C3, (P3 - M3) / 1.0j)
        D4 = 11.0 * Sz2 - s * (s + 1.0) * Id - 38.0 * Id
        P4 = Sp_mat @ P3
        M4 = Sm_mat @ M3
        if q == 4:
            return 0.25 * _sym(D4, P4 + M4)
        if q == -4:
            return 0.25 * _sym(D4, (P4 - M4) / 1.0j)
        P5 = Sp_mat @ P4
        M5 = Sm_mat @ M4
        if q == 5:
            return 0.25 * _sym(Sz, P5 + M5)
        if q == -5:
            return 0.25 * _sym(Sz, (P5 - M5) / 1.0j)
        P6 = Sp_mat @ P5
        M6 = Sm_mat @ M5
        if q == 6:
            return 0.5 * (P6 + M6)
        if q == -6:
            return 0.5 * (P6 - M6) / 1.0j

    raise ValueError(f"unsupported (k, q) = ({k}, {q}); only k in {{2, 4, 6}} is available")
