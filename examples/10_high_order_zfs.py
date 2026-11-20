"""High-order zero-field splitting of a S = 5/2 centre with B4 terms."""

import numpy as np

from dimspin import Spec, Spin, autosim


def main():
    spin = Spin(5/2, 0)
    spin.g = [2.0, 2.0, 2.0]
    spin.B4 = np.zeros(9)
    spin.B4[4] = 12.0
    spin.B4[8] = 3.0

    spec = Spec()
    spec.method = "Adaptive_Grid_Method"
    spec.sample = "Crystal_Oneside"
    spec.modtype = "Bmod"
    spec.Bmod = 1.0
    spec.mwfreq = 9500.0
    spec.fieldrange = [0.0, 8000.0]
    spec.specpoints = 2000
    spec.lw_lorentzian = 5.0
    spec.lw_gaussian = 2.0
    spec.Temperature = 298.0

    spectrum = autosim(spin, spec, show=True)
    print("spectrum shape:", spectrum.shape)


if __name__ == "__main__":
    main()
