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


def unpack_tensor_3x6(T):
    """Return the (3, 3, 3) stack of blocks from a 3x6 packed tensor.

    Each row holds one electric-field direction with the columns
    (xx, yy, zz, yz, xz, xy) of a symmetric 3x3 block.
    """
    arr = np.asarray(T, dtype=float)
    if arr.shape != (3, 6):
        raise ValueError('packed tensor must have shape (3, 6)')
    stack = np.zeros((3, 3, 3))
    for a in range(3):
        xx, yy, zz, yz, xz, xy = arr[a]
        stack[a] = ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))
    return stack


def unpack_tensor_3x5(R):
    """Return the (3, 3, 3) stack of traceless blocks from a 3x5 tensor.

    Each row holds one electric-field direction with the columns
    (D, E, yz, xz, xy), giving xx = -D/3 + E, yy = -D/3 - E, zz = 2D/3. The
    trace part is not representable here; it shifts all levels by a constant
    and does not affect the spectrum.
    """
    arr = np.asarray(R, dtype=float)
    if arr.shape != (3, 5):
        raise ValueError('short tensor must have shape (3, 5)')
    stack = np.zeros((3, 3, 3))
    for a in range(3):
        D, E, yz, xz, xy = arr[a]
        stack[a] = ((-D / 3 + E, xy, xz),
                    (xy, -D / 3 - E, yz),
                    (xz, yz, 2 * D / 3))
    return stack


def Tijk2Txyz(T):
    """Split a 3x6 SEC tensor into its Tx, Ty and Tz components.

    Used for every spin-electric coupling tensor (T, R and F), not only T.
    Setting Tx, Ty and Tz directly is recommended when the component
    matrices are already available.
    """
    stack = unpack_tensor_3x6(T)
    return stack[0], stack[1], stack[2]
