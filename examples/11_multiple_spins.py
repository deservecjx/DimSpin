"""Several spin centres in one simulation; their spectra are summed."""

from dimspin import Spec, Spin, autosim


def main():
    radical = Spin(1/2, 0)
    radical.g = [2.009, 2.006, 2.002]

    metal = Spin(1/2, 0)
    metal.g = [2.05, 2.05, 2.25]

    spec = Spec()
    spec.method = "High_Field_Method"
    spec.sample = "Crystal_Oneside"
    spec.modtype = "Bmod"
    spec.Bmod = 1.0
    spec.mwfreq = 9560.0
    spec.fieldrange = [2900.0, 3600.0]
    spec.specpoints = 1200
    spec.lw_lorentzian = 0.5
    spec.lw_gaussian = 0.2
    spec.Temperature = 298.0

    spectrum = autosim(radical, metal, spec, show=True)
    print("spectrum shape:", spectrum.shape)


if __name__ == "__main__":
    main()
