"""Powder EFM spectrum of the Ce(III) complex [NEt4][Ce{Pt(SAc)4}2].

The electric field couples through the T tensor. Inversion-related sites are
not included here; even so, each orientation is evaluated together with its
oppositely oriented partner, and the first-order EFM signal of the powder
cancels. A nonzero powder EFM signal therefore points to an artifact such as
a leaked magnetic field. Parameters follow Song et al., Angew. Chem. Int.
Ed. 2025, 64, e202513081 (doi:10.1002/anie.202513081).
"""

import numpy as np

from dimspin import Spec, Spin, autosim


def main():
    spin = Spin(1/2, 0)
    spin.g = [2.485, 2.485, 0.695]
    spin.Tx = [[0.0, 0.0, 0.0], [0.0, 0.0, 2e-10], [0.0, 2e-10, 0.0]]
    spin.Ty = [[0.0, 0.0, -2e-10], [0.0, 0.0, 0.0], [-2e-10, 0.0, 0.0]]

    spec = Spec()
    spec.method = "High_Field_Method"
    spec.sample = "Powder"
    spec.modtype = "Emod"
    spec.Emod = 1.5e6
    spec.Edirection = [0.0, 1.0, 0.0]
    spec.mwfreq = 9560.0
    spec.fieldrange = [1000.0, 11000.0]
    spec.specpoints = 1000
    spec.lw_lorentzian = 50.0
    spec.lw_gaussian = 80.0
    spec.npowder = 10000
    spec.Temperature = 5.0

    spectrum = autosim(spin, spec, show=True)
    print("max |spectrum|:", np.max(np.abs(spectrum)))


if __name__ == "__main__":
    main()
