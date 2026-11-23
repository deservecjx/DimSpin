"""Powder MFM spectrum of the Ce(III) complex [NEt4][Ce{Pt(SAc)4}2].

The strongly anisotropic g tensor (2.485, 2.485, 0.695) spreads the powder
pattern over a wide field range at 5 K. Parameters follow Song et al.,
Angew. Chem. Int. Ed. 2025, 64, e202513081 (doi:10.1002/anie.202513081).
"""

from dimspin import Spec, Spin, autosim


def main():
    spin = Spin(1/2, 0)
    spin.g = [2.485, 2.485, 0.695]

    spec = Spec()
    spec.method = "High_Field_Method"
    spec.sample = "Powder"
    spec.modtype = "Bmod"
    spec.Bmod = 1.0
    spec.mwfreq = 9560.0
    spec.fieldrange = [1000.0, 11000.0]
    spec.specpoints = 1000
    spec.lw_lorentzian = 50.0
    spec.lw_gaussian = 80.0
    spec.npowder = 10000
    spec.Temperature = 5.0

    spectrum = autosim(spin, spec, show=True)
    print("spectrum shape:", spectrum.shape)


if __name__ == "__main__":
    main()
