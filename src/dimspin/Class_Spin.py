"""Spin centre: tensors, Hamiltonian assembly and modulation.

The Spin object stores the g, D and A tensors, the spin-electric coupling
tensors T, R and F, the high-order zero-field coefficients B2/B4/B6, and the
parameter-modulation amplitudes dg/dD/dA. :meth:`Spin.identity_tensor`
validates and normalizes them; :meth:`Spin.update_Btensor` and
:meth:`Spin.update_H_initial` prepare the crystal-frame tensors and the
field-independent Hamiltonian; the ``modulate_*`` methods set ``H_total`` for
a given field and modulation sign.
"""

import logging

import numpy as np


def _array_any(value):
    """
    Inputs: tensor-like value.

    Returns: True when the value contains a nonzero entry.

    Notes: unreadable values count as nonzero so that the caller still warns.
    """
    try:
        return bool(np.asarray(value).any())
    except (TypeError, ValueError):
        return True


def _as_matrix(value, name, allow_scalar=False, scalar_factory=None,
               vector_factory=None, base_factory=None, two_factory=None):
    """
    Inputs: tensor-like value; tensor name; conversion callables for a scalar,
    a length-2 pair and a 3-vector.

    Returns: a 3x3 float array.

    Notes: None or an empty value gives zeros. An unrecognized shape raises
    ValueError.
    """
    if value is None:
        return np.zeros((3, 3))
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.zeros((3, 3))
    if arr.shape == () and allow_scalar and scalar_factory is not None:
        result = scalar_factory(arr.item())
    elif arr.shape == (2,) and two_factory is not None:
        result = two_factory(arr)
    elif arr.shape == (3,) and vector_factory is not None:
        result = vector_factory(arr)
    elif arr.shape == (3, 3) and base_factory is not None:
        result = base_factory(arr)
    else:
        raise ValueError(f'The {name} tensor has an unrecognized shape {arr.shape}.')
    result = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f'The {name} tensor contains non-finite values.')
    return result


def _as_sec_tensor(value, name, allow_short=False):
    """
    Inputs: packed tensor-like value; tensor name; whether the 3x5 layout is
    accepted.

    Returns: a (3, 3, 3) float array.

    Notes: None or an empty value gives zeros. A 3x6 input uses the columns
    (xx, yy, zz, yz, xz, xy); a 3x5 input uses (D, E, yz, xz, xy) with the
    traceless diagonal block; an already unpacked (3, 3, 3) stack is accepted
    unchanged, which keeps the method idempotent. Any other shape or
    non-finite values raise ValueError.
    """
    from .Tensor import unpack_tensor_3x5, unpack_tensor_3x6
    if value is None:
        return np.zeros((3, 3, 3))
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.zeros((3, 3, 3))
    if arr.shape == (3, 3, 3):
        stack = arr
    elif arr.shape == (3, 6):
        stack = unpack_tensor_3x6(arr)
    elif allow_short and arr.shape == (3, 5):
        stack = unpack_tensor_3x5(arr)
    else:
        expected = '(3, 6) or (3, 5)' if allow_short else '(3, 6)'
        raise ValueError(f'{name} must have shape {expected}.')
    if not np.all(np.isfinite(stack)):
        raise ValueError(f'{name} contains non-finite values.')
    return stack


class Spin:
    """A spin centre with its tensors and Hamiltonian state."""

    def __init__(self, S, I):
        self.name = 'Spin'
        self.S = S
        self.I = I
        self.g = np.zeros(3)
        self.D = np.zeros((3, 3))
        self.A = np.zeros(3)
        self.dg = np.zeros(3)
        self.dD = np.zeros((3, 3))
        self.dA = np.zeros(3)
        self.B2 = np.zeros(5)
        self.B4 = np.zeros(9)
        self.B6 = np.zeros(13)
        self.g1 = np.zeros((3, 3))
        self.D1 = np.zeros((3, 3))
        self.A1 = np.zeros((3, 3))
        self.dg1 = np.zeros((3, 3))
        self.dD1 = np.zeros((3, 3))
        self.dA1 = np.zeros((3, 3))
        self.T = np.zeros((3, 6))
        self.R = np.zeros((3, 6))
        self.F = np.zeros((3, 6))
        self.Tx = np.zeros((3, 3))
        self.Ty = np.zeros((3, 3))
        self.Tz = np.zeros((3, 3))
        self.Rx = np.zeros((3, 3))
        self.Ry = np.zeros((3, 3))
        self.Rz = np.zeros((3, 3))
        self.Fx = np.zeros((3, 3))
        self.Fy = np.zeros((3, 3))
        self.Fz = np.zeros((3, 3))
        self.Tx1 = np.zeros((3, 3))
        self.Ty1 = np.zeros((3, 3))
        self.Tz1 = np.zeros((3, 3))
        self.Rx1 = np.zeros((3, 3))
        self.Ry1 = np.zeros((3, 3))
        self.Rz1 = np.zeros((3, 3))
        self.Fx1 = np.zeros((3, 3))
        self.Fy1 = np.zeros((3, 3))
        self.Fz1 = np.zeros((3, 3))
        self.total_Level = int((2 * self.S + 1) * (2 * self.I + 1))
        self.H_initial = np.zeros((self.total_Level, self.total_Level))
        self.H_Einitial = np.zeros((self.total_Level, self.total_Level))
        self.H_total = np.zeros((self.total_Level, self.total_Level))
        self.H_g_grad = None
        self.H_T_grad = None
        self.H_dg_grad = None
        self.H_delta_const = None

        self.tensor_orientation = None
        self.Etensor_orientation = None
        self.delta_orientation = None
        self.R_tf2lf = np.eye(3)
        self.R_tf2lf_rotated = np.eye(3)
        self.R_etf2lf = np.eye(3)
        self.R_etf2lf_rotated = np.eye(3)
        self.R_dtf2lf = np.eye(3)
        self.R_dtf2lf_rotated = np.eye(3)

    def clone(self):
        """
        Inputs: none.

        Returns: a deep copy, used by parallel tasks to isolate mutable state.
        """
        from copy import deepcopy
        return deepcopy(self)

    def identity_tensor(self, crystal_orientation=np.eye(3), molecular_orientation=np.eye(3)):
        """
        Inputs: crystal orientation and molecular orientation.

        Returns: nothing; g, D, A, the coupling tensors and their orientations
        are validated and normalized in place.

        Notes: a scalar, 3-vector or 3x3 tensor is accepted for g, A and the
        modulation amplitudes. A scalar D is treated as the axial case
        (D, E = 0); a (D, E) pair, a 3-vector or a 3x3 tensor is also accepted.
        D is cleared for S = 1/2 and A for I = 0. T and F accept the 3x6
        packed layout (or are empty). R accepts the 3x6 layout or a 3x5 layout
        whose columns are (D, E, yz, xz, xy) for the three electric-field
        directions, with the traceless diagonal block
        (-D/3 + E, -D/3 - E, 2D/3). T, R and F are stored as (3, 3, 3) stacks
        and the component matrices Tx..Fz must be 3x3. The coupling tensors use
        ``Etensor_orientation`` and the modulation amplitudes use
        ``delta_orientation``, which defaults to ``tensor_orientation``.
        B2^0/B2^2 duplicate the D tensor and cannot be given together with a
        nonzero D. Unrecognized shapes raise ValueError instead of being
        silently dropped.
        """
        from .Tensor import Dtensor
        if not np.isscalar(self.S) or self.S <= 0:
            raise ValueError("S must be a positive spin quantum number.")
        if not np.isscalar(self.I) or self.I < 0:
            raise ValueError("I must be a non-negative spin quantum number.")

        self.B2 = np.asarray(self.B2, dtype=float)
        self.B4 = np.asarray(self.B4, dtype=float)
        self.B6 = np.asarray(self.B6, dtype=float)
        if self.B2.size != 5:
            raise ValueError("B2 must contain exactly 5 elements (q = -2 ... 2).")
        if self.B4.size != 9:
            raise ValueError("B4 must contain exactly 9 elements (q = -4 ... 4).")
        if self.B6.size != 13:
            raise ValueError("B6 must contain exactly 13 elements (q = -6 ... 6).")

        self.g = _as_matrix(self.g, 'g', allow_scalar=True,
                            scalar_factory=lambda x: x * np.eye(3),
                            vector_factory=np.diag,
                            base_factory=lambda x: x)

        self.T = _as_sec_tensor(self.T, 'T')
        self.R = _as_sec_tensor(self.R, 'R', allow_short=True)
        self.F = _as_sec_tensor(self.F, 'F')
        if self.T.any():
            self.Tx, self.Ty, self.Tz = self.T[0], self.T[1], self.T[2]
        if self.R.any():
            self.Rx, self.Ry, self.Rz = self.R[0], self.R[1], self.R[2]
        if self.F.any():
            self.Fx, self.Fy, self.Fz = self.F[0], self.F[1], self.F[2]

        for name in ('Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Fx', 'Fy', 'Fz'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3, 3):
                raise ValueError(f'{name} must be a 3x3 matrix.')
            setattr(self, name, value)

        if self.S == 1 / 2:
            had_zfs = any(_array_any(value) for value in
                          (self.D, self.Rx, self.Ry, self.Rz,
                           self.B2, self.B4, self.B6))
            self.D = np.zeros((3, 3))
            self.Rx = np.zeros((3, 3))
            self.Ry = np.zeros((3, 3))
            self.Rz = np.zeros((3, 3))
            self.B2 = np.zeros(5)
            self.B4 = np.zeros(9)
            self.B6 = np.zeros(13)
            if had_zfs:
                logging.warning('S = 1/2: zero-field splitting is not applicable and has been cleared.')
        else:
            self.D = _as_matrix(self.D, 'D', allow_scalar=True,
                                scalar_factory=lambda x: Dtensor(x, 0.0),
                                vector_factory=np.diag,
                                base_factory=lambda x: x,
                                two_factory=lambda x: Dtensor(x[0], x[1]))

        if not (self.Tx.any() or self.Ty.any() or self.Tz.any()
                or self.Rx.any() or self.Ry.any() or self.Rz.any()
                or self.Fx.any() or self.Fy.any() or self.Fz.any()):
            logging.debug('No electric-field coupling tensor is set.')

        if self.D.any():
            conflicts = []
            if self.B2[2] != 0:
                conflicts.append('B2[2] (B2^0)')
            if self.B2[4] != 0:
                conflicts.append('B2[4] (B2^2)')
            if conflicts:
                raise ValueError('B2^0/B2^2 are equivalent to the D tensor, and cannot be '
                                 'defined together with a non-zero D: ' + ', '.join(conflicts) + '.')

        if self.I == 0:
            had_hyperfine = any(_array_any(value) for value in
                                (self.A, self.Fx, self.Fy, self.Fz))
            self.A = np.zeros((3, 3))
            self.Fx = np.zeros((3, 3))
            self.Fy = np.zeros((3, 3))
            self.Fz = np.zeros((3, 3))
            if had_hyperfine:
                logging.warning('I = 0: hyperfine coupling is not applicable and has been cleared.')
        else:
            self.A = _as_matrix(self.A, 'A', allow_scalar=True,
                                scalar_factory=lambda x: x * np.eye(3),
                                vector_factory=np.diag,
                                base_factory=lambda x: x)

        self.dg = _as_matrix(self.dg, 'Delta g', allow_scalar=True,
                             scalar_factory=lambda x: x * np.eye(3),
                             vector_factory=np.diag,
                             base_factory=lambda x: x)
        if self.S == 1 / 2:
            self.dD = np.zeros((3, 3))
        else:
            self.dD = _as_matrix(self.dD, 'Delta D', allow_scalar=True,
                                 scalar_factory=lambda x: Dtensor(x, 0.0),
                                 vector_factory=np.diag,
                                 base_factory=lambda x: x,
                                 two_factory=lambda x: Dtensor(x[0], x[1]))
        if self.I == 0:
            self.dA = np.zeros((3, 3))
        else:
            self.dA = _as_matrix(self.dA, 'Delta A', allow_scalar=True,
                                 scalar_factory=lambda x: x * np.eye(3),
                                 vector_factory=np.diag,
                                 base_factory=lambda x: x)

        from .Eular2Rot import resolve_orientation
        R_sf2lf = resolve_orientation(crystal_orientation)
        R_mf2sf = resolve_orientation(molecular_orientation)
        R_tf2mf = resolve_orientation(self.tensor_orientation)
        if self.Etensor_orientation is None:
            R_etf2mf = R_tf2mf
        else:
            R_etf2mf = resolve_orientation(self.Etensor_orientation)
        if self.delta_orientation is None:
            R_dtf2mf = R_tf2mf
        else:
            R_dtf2mf = resolve_orientation(self.delta_orientation)

        self.R_tf2lf = R_sf2lf @ (R_mf2sf @ R_tf2mf)
        self.R_etf2lf = R_sf2lf @ (R_mf2sf @ R_etf2mf)
        self.R_dtf2lf = R_sf2lf @ (R_mf2sf @ R_dtf2mf)
        self.R_tf2lf_rotated = self.R_tf2lf
        self.R_etf2lf_rotated = self.R_etf2lf
        self.R_dtf2lf_rotated = self.R_dtf2lf

    def update_Btensor(self):
        """
        Inputs: none.

        Returns: nothing; the rotated tensors and field gradients are updated.

        Notes: call this before assembling a Hamiltonian. For the first angle
        or a single-crystal one-side simulation the rotation is the identity.
        The modulation amplitudes use their own rotation ``R_dtf2lf_rotated``.
        """
        R = self.R_tf2lf_rotated
        R_delta = self.R_dtf2lf_rotated
        self.g1 = R @ self.g @ R.T
        self.D1 = R @ self.D @ R.T
        self.A1 = R @ self.A @ R.T
        self.dg1 = R_delta @ self.dg @ R_delta.T
        self.dD1 = R_delta @ self.dD @ R_delta.T
        self.dA1 = R_delta @ self.dA @ R_delta.T

        from .Team_Zeeman import H_Zeeman
        self.H_g_grad = H_Zeeman(1.0, self.g1, self.S, self.I)
        self.H_dg_grad = H_Zeeman(1.0, self.dg1, self.S, self.I)
        self.H_delta_const = np.zeros((self.total_Level, self.total_Level))
        if self.dD1.any():
            from .Team_ZFS import H_ZFS
            self.H_delta_const = self.H_delta_const + H_ZFS(self.dD1, self.S, self.I)
        if self.dA1.any():
            from .Team_Hyperfine import H_Hyperfine
            self.H_delta_const = self.H_delta_const + H_Hyperfine(self.dA1, self.S, self.I)

    def update_H_initial(self):
        """
        Inputs: none.

        Returns: nothing; the field-independent Hamiltonian is stored in
        ``H_initial``.
        """
        self.H_initial = np.zeros((self.total_Level, self.total_Level))
        if self.A1.any():
            from .Team_Hyperfine import H_Hyperfine
            self.H_initial = self.H_initial + H_Hyperfine(self.A1, self.S, self.I)
        if self.D1.any():
            from .Team_ZFS import H_ZFS
            self.H_initial = self.H_initial + H_ZFS(self.D1, self.S, self.I)
        if self.B2.any() or self.B4.any() or self.B6.any():
            from .Team_ZFS_HighOrder import B_arrays_to_dict, H_ZFS_HighOrder
            B = B_arrays_to_dict(self.B2, self.B4, self.B6)
            self.H_initial = self.H_initial + H_ZFS_HighOrder(B, self.S, self.I, R=self.R_tf2lf_rotated.T)

    def modulate_B(self, Bz, Bmod, modulation_symbol='positive'):
        """
        Inputs: swept field Bz (G); modulation amplitude Bmod (G);
        modulation_symbol ("positive" or "negative").

        Returns: nothing; ``H_total`` is set.

        Notes: the Hamiltonian is linear in the field, so the Zeeman term is
        built from the cached gradient ``H_g_grad``.
        """
        if Bmod == 0:
            raise ValueError("The modulated magnetic field is zero, the procedure will be closed!")
        if modulation_symbol == 'negative':
            self.H_total = self.H_initial + (Bz - Bmod) * self.H_g_grad
        elif modulation_symbol == 'positive':
            self.H_total = self.H_initial + (Bz + Bmod) * self.H_g_grad
        else:
            raise ValueError('An unknown error has occurred. The modulated magnetic field '
                             'is neither positive nor negative.')

    def modulate_param(self, Bz, modulation_symbol='positive'):
        """
        Inputs: swept field Bz (G); modulation_symbol ("positive" or
        "negative").

        Returns: nothing; ``H_total`` is set.

        Notes: dg/dD/dA are added as the modulation term; only nonzero parts
        contribute.
        """
        if modulation_symbol == 'positive':
            sign = 1
        elif modulation_symbol == 'negative':
            sign = -1
        else:
            raise ValueError('An unknown error has occurred. The modulated parameter is neither positive nor negative.')
        self.H_total = (self.H_initial + sign * self.H_delta_const
                        + Bz * self.H_g_grad + sign * Bz * self.H_dg_grad)

    def update_Etensor(self, E):
        """
        Inputs: electric-field vector E.

        Returns: nothing; the frame-rotated coupling tensors and the T-term
        field gradient are updated.
        """
        Rrot_etf2lf = self.R_etf2lf_rotated

        RRR = np.einsum('ip,jq,kr,pqr->ijk', Rrot_etf2lf, Rrot_etf2lf, Rrot_etf2lf,
                        [self.Rx, self.Ry, self.Rz])
        self.Rx1 = E[0] * RRR[0][:][:]
        self.Ry1 = E[1] * RRR[1][:][:]
        self.Rz1 = E[2] * RRR[2][:][:]

        FFF = np.einsum('ip,jq,kr,pqr->ijk', Rrot_etf2lf, Rrot_etf2lf, Rrot_etf2lf,
                        [self.Fx, self.Fy, self.Fz])
        self.Fx1 = E[0] * FFF[0][:][:]
        self.Fy1 = E[1] * FFF[1][:][:]
        self.Fz1 = E[2] * FFF[2][:][:]

        TTT = np.einsum('ip,jq,kr,pqr->ijk', Rrot_etf2lf, Rrot_etf2lf, Rrot_etf2lf,
                        [self.Tx, self.Ty, self.Tz])
        self.Tx1 = E[0] * TTT[0][:][:]
        self.Ty1 = E[1] * TTT[1][:][:]
        self.Tz1 = E[2] * TTT[2][:][:]

        from .ETeam_Zeeman import H_EMod_T
        self.H_T_grad = H_EMod_T(1.0, self.Tx1, self.Ty1, self.Tz1, self.S, self.I)

    def modulate_E_initial(self, E1):
        """
        Inputs: electric-field vector E1.

        Returns: nothing; the field-independent electric modulation term is
        stored in ``H_Einitial``.
        """
        from .ETeam_Hyperfine import H_EMod_F
        from .ETeam_ZFS import H_EMod_R
        self.H_Einitial = np.zeros((self.total_Level, self.total_Level))

        found = False
        if self.Rx1.any() or self.Ry1.any() or self.Rz1.any():
            self.H_Einitial = self.H_Einitial + H_EMod_R(self.Rx1, self.Ry1, self.Rz1, self.S, self.I)
            found = True
        if self.Fx1.any() or self.Fy1.any() or self.Fz1.any():
            self.H_Einitial = self.H_Einitial + H_EMod_F(self.Fx1, self.Fy1, self.Fz1, self.S, self.I)
            found = True
        if self.Tx1.any() or self.Ty1.any() or self.Tz1.any():
            found = True
        if not found:
            logging.error("Electric field parameter(T,R,F) not found, please reconfirm entry!")

    def modulate_E_T(self, Bz, E, modulation_symbol='positive'):
        """
        Inputs: swept field Bz (G); electric-field vector E;
        modulation_symbol ("positive" or "negative").

        Returns: nothing; ``H_total`` is set.
        """
        if E[0] == 0 and E[1] == 0 and E[2] == 0:
            raise ValueError("The modulated electric field is zero, the procedure will be closed!")

        if modulation_symbol == 'negative':
            self.H_total = (self.H_initial - self.H_Einitial
                            + Bz * self.H_g_grad - Bz * self.H_T_grad)
        elif modulation_symbol == 'positive':
            self.H_total = (self.H_initial + self.H_Einitial
                            + Bz * self.H_g_grad + Bz * self.H_T_grad)
        else:
            raise ValueError('An unknown error has occurred. The modulated electric field '
                             'is neither positive nor negative.')
