"""Minimal Spin/Spec stand-ins shared by the method tests."""

from types import SimpleNamespace

import numpy as np

from dimspin.ETeam_Zeeman import H_EMod_T
from dimspin.Team_Hyperfine import H_Hyperfine
from dimspin.Team_Zeeman import H_Zeeman
from dimspin.Team_ZFS import H_ZFS

G_ISO = 2.0023


class ToySpin:
    """Minimal stand-in for Class_Spin exposing what the methods read."""

    def __init__(self, S, I, g, D=None, A=None, dg=None, T=None):
        self.S, self.I = S, I
        self.total_Level = int(round((2 * S + 1) * (2 * I + 1)))
        self.g1 = np.asarray(g, dtype=float)
        self.g = g
        self.D = np.zeros((3, 3)) if D is None else D
        self.A = np.zeros((3, 3)) if A is None else A
        self.dg1 = np.zeros((3, 3)) if dg is None else np.asarray(dg, dtype=float)
        self.dD1 = np.zeros((3, 3))
        self.dA1 = np.zeros((3, 3))
        self.H_initial = np.zeros((self.total_Level, self.total_Level))
        if D is not None:
            self.H_initial = self.H_initial + H_ZFS(D, S, I)
        if A is not None:
            self.H_initial = self.H_initial + H_Hyperfine(A, S, I)
        self.H_g_grad = H_Zeeman(1.0, self.g1, S, I)
        self.H_dg_grad = H_Zeeman(1.0, self.dg1, S, I)
        self.H_delta_const = np.zeros((self.total_Level, self.total_Level))
        self._E_tensors = T
        self.H_total = self.H_initial

    def modulate_B(self, Bz, Bmod, symbol='positive'):
        sign = -1.0 if symbol == 'negative' else 1.0
        self.H_total = self.H_initial + (Bz + sign * Bmod) * self.H_g_grad

    def modulate_param(self, Bz, symbol='positive'):
        sign = -1.0 if symbol == 'negative' else 1.0
        self.H_total = (self.H_initial + sign * self.H_delta_const
                        + Bz * self.H_g_grad + sign * Bz * self.H_dg_grad)

    def update_Etensor(self, E):
        Tx, Ty, Tz = self._E_tensors
        self.H_T_grad = H_EMod_T(1.0, E[0] * Tx, E[1] * Ty, E[2] * Tz, self.S, self.I)

    def modulate_E_initial(self, E1):
        self.H_Einitial = np.zeros((self.total_Level, self.total_Level))

    def modulate_E_T(self, Bz, E, symbol='positive'):
        sign = -1.0 if symbol == 'negative' else 1.0
        self.H_total = (self.H_initial + sign * self.H_Einitial
                        + Bz * self.H_g_grad + sign * Bz * self.H_T_grad)

    def identity_tensor(self):
        """Normalize g, D and A to 3x3 as the real Spin does before expansion."""
        g = np.asarray(self.g, dtype=float)
        if g.ndim == 0:
            self.g = g * np.eye(3)
        elif g.shape == (3,):
            self.g = np.diag(g)
        elif g.shape != (3, 3):
            raise ValueError("g tensor must be a scalar, a 3-vector or a 3x3 matrix")

        D = np.asarray(self.D, dtype=float)
        if self.S == 1 / 2:
            self.D = np.zeros((3, 3))
        elif D.shape == (2,):
            self.D = np.diag([-D[0] / 3 + D[1], -D[0] / 3 - D[1], 2 * D[0] / 3])
        elif D.shape == (3,):
            self.D = np.diag(D)
        elif D.shape != (3, 3):
            self.D = np.zeros((3, 3))

        A = np.asarray(self.A, dtype=float)
        if self.I == 0:
            self.A = np.zeros((3, 3))
        elif A.shape == (3,):
            self.A = np.diag(A)
        elif A.shape != (3, 3):
            self.A = np.zeros((3, 3))


def make_spec(modtype, **overrides):
    params = dict(mwfreq=9800.0, fieldrange=np.array([3000.0, 4000.0]),
                  modtype=modtype, Bmod=5.0, E1=None, Temperature=300.0)
    params.update(overrides)
    return SimpleNamespace(**params)
