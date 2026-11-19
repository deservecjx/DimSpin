"""Electric-field modulated spectrum through a spin-electric coupling T tensor."""

import numpy as np

from dimspin import Spec, Spin, autosim


def main():
    spin = Spin(1/2, 0)
    spin.g = 2.0023
    spin.T = np.zeros((3, 6))
    spin.T[2, 2] = 0.2

    spec = Spec()
    spec.method = "Adaptive_Grid_Method"
    spec.sample = "Crystal_Oneside"
    spec.mwfreq = 9800.0
    spec.fieldrange = [3000.0, 4000.0]
    spec.specpoints = 800
    spec.lw_lorentzian = 5.0
    spec.Emod = 1.0
    spec.Edirection = [0.0, 0.0, 1.0]
    spec.Temperature = 298.0

    spectrum = autosim(spin, spec, show=True)
    print("spectrum shape:", spectrum.shape)


if __name__ == "__main__":
    main()
