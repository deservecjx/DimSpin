"""Energy-level diagram of an S = 1 centre, with the resonances of the spectrum."""

import matplotlib.pyplot as plt

from dimspin import Spec, Spin, plot_energy_levels


def main():
    spin = Spin(1, 0)
    spin.g = [2.002, 2.002, 2.002]
    spin.D = 300.0
    spin.E = 30.0

    spec = Spec()
    spec.method = "Adaptive_Grid_Method"
    spec.sample = "Crystal_Oneside"
    spec.modtype = "Bmod"
    spec.Bmod = 1.0
    spec.mwfreq = 9500.0
    spec.fieldrange = [0.0, 6000.0]
    spec.specpoints = 2000
    spec.lw_lorentzian = 3.0
    spec.lw_gaussian = 1.0
    spec.Temperature = 298.0

    plot_energy_levels(spin, spec)
    plt.show()


if __name__ == "__main__":
    main()
