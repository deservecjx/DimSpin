"""Crystal rotation from a non-default initial crystal orientation."""

import numpy as np

from dimspin import Spec, Spin, autosim


def main():
    spin = Spin(1/2, 0)
    spin.g = [2.0, 2.1, 1.9]

    spec = Spec()
    spec.method = "Adaptive_Grid_Method"
    spec.sample = "Crystal_Rotation"
    spec.modtype = "Bmod"
    spec.Bmod = 1.0
    spec.mwfreq = 9560.0
    spec.fieldrange = [3200.0, 3800.0]
    spec.specpoints = 600
    spec.lw_lorentzian = 0.5
    spec.lw_gaussian = 0.2
    spec.anglerange = [0, 180, 5]
    spec.crystal_orientation = [0.0, np.pi / 3, 0.0]
    spec.Temperature = 298.0

    spectra = autosim(spin, spec, show=True)
    print("spectrum map shape:", spectra.shape)


if __name__ == "__main__":
    main()
