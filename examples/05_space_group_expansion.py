"""Space-group expansion: sum the magnetically inequivalent orientations of P2."""

import numpy as np

from dimspin import Spec, Spin, autosim


def main():
    spin = Spin(1/2, 0.5)
    spin.g = [[2.0, 0.1, 0.0],
              [0.1, 2.2, 0.0],
              [0.0, 0.0, 2.4]]
    spin.A = 100.0 * np.eye(3)

    spec = Spec()
    spec.method = "High_Field_Method"
    spec.sample = "Crystal_Oneside"
    spec.modtype = "Bmod"
    spec.Bmod = 1.0
    spec.mwfreq = 9800.0
    spec.fieldrange = [2500.0, 3500.0]
    spec.specpoints = 800
    spec.lw_lorentzian = 1.0
    spec.group_number = 3
    spec.Temperature = 298.0

    spectrum = autosim(spin, spec, show=True)
    print("spectrum shape:", spectrum.shape)


if __name__ == "__main__":
    main()
