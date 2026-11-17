"""Construct g, D, A and spin-electric coupling tensors."""

import numpy as np


def gtensor(gx, gy, gz):
    """Return the diagonal g tensor."""
    return np.array([[gx, 0, 0],
                     [0, gy, 0],
                     [0, 0, gz]])


def Dtensor(D, E):
    """Return the traceless zero-field-splitting tensor from D and E."""
    return np.array([[-D/3 + E, 0, 0],
                     [0, -D/3 - E, 0],
                     [0, 0, 2*D/3]])


def Atensor(Ax, Ay, Az):
    """Return the diagonal hyperfine tensor."""
    return np.array([[Ax, 0, 0],
                     [0, Ay, 0],
                     [0, 0, Az]])


def Tijk2Txyz(T):
    """Split a 3x6 SEC tensor into its Tx, Ty and Tz components.

    Used for every spin-electric coupling tensor (T, R and F), not only T.
    Setting Tx, Ty and Tz directly is recommended when the component
    matrices are already available.
    """
    Tx = np.zeros((3, 3))
    Ty = np.zeros((3, 3))
    Tz = np.zeros((3, 3))
    Tx[0][0] = T[0][0]
    Tx[0][1] = T[0][5]
    Tx[0][2] = T[0][4]
    Tx[1][0] = T[0][5]
    Tx[1][1] = T[0][1]
    Tx[1][2] = T[0][3]
    Tx[2][0] = T[0][4]
    Tx[2][1] = T[0][3]
    Tx[2][2] = T[0][2]
    Ty[0][0] = T[1][0]
    Ty[0][1] = T[1][5]
    Ty[0][2] = T[1][4]
    Ty[1][0] = T[1][5]
    Ty[1][1] = T[1][1]
    Ty[1][2] = T[1][3]
    Ty[2][0] = T[1][4]
    Ty[2][1] = T[1][3]
    Ty[2][2] = T[1][2]
    Tz[0][0] = T[2][0]
    Tz[0][1] = T[2][5]
    Tz[0][2] = T[2][4]
    Tz[1][0] = T[2][5]
    Tz[1][1] = T[2][1]
    Tz[1][2] = T[2][3]
    Tz[2][0] = T[2][4]
    Tz[2][1] = T[2][3]
    Tz[2][2] = T[2][2]
    return Tx, Ty, Tz
