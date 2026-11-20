"""Hyperfine splitting of a S = 1/2 centre coupled to a I = 1 nucleus."""

from dimspin import Spec, Spin, autosim


def main():
    spin = Spin(1/2, 1)
    spin.g = [2.009, 2.006, 2.002]
    spin.A = [20.0, 20.0, 55.0]

    spec = Spec()
    spec.method = "High_Field_Method"
    spec.sample = "Crystal_Oneside"
    spec.modtype = "Bmod"
    spec.Bmod = 1.0
    spec.mwfreq = 9560.0
    spec.fieldrange = [3300.0, 3600.0]
    spec.specpoints = 800
    spec.lw_lorentzian = 0.3
    spec.lw_gaussian = 0.1
    spec.Temperature = 298.0

    spectrum = autosim(spin, spec, show=True)
    print("spectrum shape:", spectrum.shape)


if __name__ == "__main__":
    main()
