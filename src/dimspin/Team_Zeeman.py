"""Electron Zeeman term for a static magnetic field along the laboratory z axis."""

from .Spin_Operators import Sx_get, Sy_get, Sz_get

# mu_B / h in MHz/G, converts g * B (G) to frequency (MHz).
MU_B_H = 1.399624493607271


def H_Zeeman(Bz, g, S, I):
    """
    Inputs: field Bz (G), 3x3 g tensor, electron spin S, nuclear spin I.

    Returns: the electron Zeeman Hamiltonian for a field along z.

    Notes: only the z row of ``g`` enters, because the field is along the
    laboratory z axis.
    """
    Sx = Sx_get(S, I)
    Sy = Sy_get(S, I)
    Sz = Sz_get(S, I)

    return MU_B_H * Bz * (g[2][0] * Sx + g[2][1] * Sy + g[2][2] * Sz)
