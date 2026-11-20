"""Parallel powder simulation of a high-spin centre with dense sampling.

The orientational average needs one resonance search per crystallite
orientation, so ``Spec.npowder`` sets the cost of a powder simulation. With
``parallel=True`` (the default) the orientations are distributed over a
process pool, and ``max_workers`` sets the number of worker processes; when
it is omitted the built-in default is used, and ``Spec.max_workers`` carries
the same setting for the other sample modes.
"""

from dimspin import Spec, Spin, autosim


def main():
    spin = Spin(5/2, 0)
    spin.g = [2.001, 2.001, 2.002]
    spin.D = 300.0
    spin.E = 30.0

    spec = Spec()
    spec.method = "Adaptive_Grid_Method"
    spec.sample = "Powder"
    spec.modtype = "Bmod"
    spec.Bmod = 1.0
    spec.mwfreq = 9500.0
    spec.fieldrange = [0.0, 7000.0]
    spec.specpoints = 2000
    spec.lw_lorentzian = 3.0
    spec.lw_gaussian = 1.0
    spec.npowder = 5000
    spec.Temperature = 298.0

    spectrum = autosim(spin, spec, show=True, parallel=True, max_workers=4)
    print("spectrum shape:", spectrum.shape)


if __name__ == "__main__":
    main()
