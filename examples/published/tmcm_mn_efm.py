"""Angle-dependent EFM spectra of the TMCM-Mn complex.

Turning the single crystal about the laboratory x axis, with the electric
field fixed in the crystal frame, modulates the ZFS tensor through dD, whose
principal axes are tilted relative to D. Parameters from Liu et al.,
J. Am. Chem. Soc. 2024, 146, 19397 (doi:10.1021/jacs.4c05655).
"""

import numpy as np

from dimspin import Spec, Spin, autosim


def main():
    spin1 = Spin(5/2, 5/2)
    spin1.g = 2.0045
    spin1.A = 246.4
    spin1.D = (116.0, 28.5)
    spin1.tensor_orientation = np.deg2rad([30.0, -95.0, 100.0])
    spin1.dD = [-0.36, -0.99, 1.35]
    spin1.delta_orientation = np.deg2rad([176.0, 53.0, 58.0])

    spin2 = Spin(5/2, 5/2)
    spin2.g = 2.0045
    spin2.A = 246.4
    spin2.D = (116.0, 28.5)
    spin2.tensor_orientation = np.deg2rad([150.0, -85.0, 100.0])
    spin2.dD = [0.33, 0.55, -0.88]
    spin2.delta_orientation = np.deg2rad([4.0, 127.0, 58.0])

    spec = Spec()
    spec.method = "High_Field_Method"
    spec.sample = "Crystal_Rotation"
    spec.modtype = "param"
    spec.mwfreq = 9560.0
    spec.fieldrange = [2800.0, 3900.0]
    spec.specpoints = 800
    spec.lw_lorentzian = 3.0
    spec.lw_gaussian = 3.0
    spec.crystal_orientation = [0.0, np.pi / 2, 0.0]
    spec.anglerange = [0, 180, 1]
    spec.Temperature = 298.0

    spectra = autosim(spin1, spin2, spec, show=True)
    print("spectrum map shape:", spectra.shape)


if __name__ == "__main__":
    main()
