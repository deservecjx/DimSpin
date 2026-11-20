"""Experimental parameters for a spectrum simulation.

Spec stores the method, modulation type, sample mode, field grid, linewidths
and the settings for the crystal-rotation and powder modes.
:meth:`Spec.check_simpara` validates the settings and expands the field range
and the electric-field vector into their usable forms.
"""

import logging

import numpy as np


class Spec:
    """Experimental parameters for one simulation."""

    def __init__(self):
        self.name = 'Spec'
        self.method = 'High_Field_Method'
        self.modtype = 'Bmod'
        self.sample = 'Crystal_Oneside'

        self.mwfreq = 9560.000

        self.lw = np.zeros(2)
        self.lw_lorentzian = 1
        self.lw_gaussian = 0
        self.lw_type = ''

        self.fieldrange = np.zeros(2)
        self.Bmod = 0
        self.Emod = 0
        self.Edirection = np.zeros(3)
        self.E1 = np.zeros(3)
        self.Temperature = 298
        self.specpoints = 0

        self.anglerange = [0, 180, 1]
        self.rotaxis = np.array([0, 0, 0])
        self.crystal_orientation = None
        self.group_number = None
        self.frac_pos = None

        self.npowder = 2000
        self.max_workers = None
        self.linecolor = 'red'

    def clone(self):
        """
        Inputs: none.

        Returns: a deep copy, used by parallel tasks to isolate mutable state.
        """
        from copy import deepcopy
        return deepcopy(self)

    def check_simpara(self):
        """
        Inputs: none; the parameters are read from the instance.

        Returns: nothing; ``modtype``, ``E1``, ``lw_type``, ``specpoints`` and
        the expanded ``fieldrange`` are set in place.

        Notes: the modulation type is detected from Bmod/Emod unless
        ``modtype`` is explicitly 'param'; giving both fields at once is
        rejected. The method must be 'High_Field_Method' or
        'Adaptive_Grid_Method'. A field range given as two values is expanded
        to ``specpoints`` points; an already expanded axis must be uniformly
        spaced, and it sets ``specpoints`` when that is zero. The method is
        idempotent. ``max_workers``, when given, must be a positive integer.
        ``rotaxis`` is deprecated and unused: the rotation mode turns the
        crystal about the laboratory x axis, and the initial orientation comes
        from ``crystal_orientation``.
        """
        self.mwfreq = _positive_number(self.mwfreq, 'mwfreq')
        self.Temperature = _positive_number(self.Temperature, 'Temperature')

        if self.group_number is not None:
            try:
                number = int(self.group_number)
            except (TypeError, ValueError):
                raise ValueError("'group_number' must be an integer between 1 and 230.")
            if number != self.group_number or not 1 <= number <= 230:
                raise ValueError("'group_number' must be an integer between 1 and 230.")
            self.group_number = number
        if self.frac_pos is not None:
            self.frac_pos = np.asarray(self.frac_pos, dtype=float)
            if self.frac_pos.shape != (3,):
                raise ValueError("'frac_pos' must be a three-dimensional fractional coordinate.")
            if not np.all(np.isfinite(self.frac_pos)):
                raise ValueError("'frac_pos' must contain finite values.")

        self.Bmod = _number(self.Bmod, 'Bmod')
        self.Emod = _number(self.Emod, 'Emod')
        self.Edirection = np.asarray(self.Edirection, dtype=float)
        if self.Edirection.shape != (3,):
            raise ValueError("'Edirection' must be a three-dimensional vector.")
        if not np.all(np.isfinite(self.Edirection)):
            raise ValueError("'Edirection' must contain finite values.")

        if self.modtype != 'param':
            if self.Bmod != 0 and self.Emod != 0:
                raise ValueError('Both Bmod and Emod are nonzero; please give only one modulation field.')
            if self.Bmod != 0:
                self.modtype = 'Bmod'
                logging.warning('Magnetic field modulation is recognized.')
            if self.Emod != 0:
                self.modtype = 'Emod'
                logging.warning('Electric field modulation is recognized.')

        if self.modtype == 'Bmod' and self.Bmod == 0:
            raise ValueError('Magnetic field modulation is selected, but modulating magnetic field is not given')
        elif self.modtype == 'Emod' and (self.Emod == 0 or np.all(self.Edirection == 0)):
            raise ValueError('Electric field modulation is selected, but modulating electric field is not given')
        elif self.modtype == 'param':
            logging.warning('Parameter modulation (Delta g/Delta D/Delta A) is recognized.')
        if self.modtype not in ('Bmod', 'Emod', 'param'):
            raise ValueError("Can't recognize modtype, please check it again."
                             "(Only 'Bmod', 'Emod' or 'param' is allowed)")
        self.E1 = self.Emod * self.Edirection

        self.lw_lorentzian = _number(self.lw_lorentzian, 'lw_lorentzian')
        self.lw_gaussian = _number(self.lw_gaussian, 'lw_gaussian')
        self.lw = np.asarray(self.lw, dtype=float)
        if self.lw.size not in (0, 2):
            raise ValueError("'lw' must contain two elements (Lorentzian, Gaussian).")
        if self.lw.any():
            if self.lw_lorentzian == 0 and self.lw_gaussian == 0:
                self.lw_lorentzian = float(self.lw[0])
                self.lw_gaussian = float(self.lw[1])
                self.lw = np.zeros(2)
            else:
                raise ValueError('The linewidth has been entered twice, please double check and delete one of them.')
        elif self.lw_lorentzian == 0 and self.lw_gaussian == 0:
            raise ValueError('Lorentzian and Gaussian linewidths cannot be zero at the same time.')
        self.lw_lorentzian = _nonnegative_number(self.lw_lorentzian, 'lw_lorentzian')
        self.lw_gaussian = _nonnegative_number(self.lw_gaussian, 'lw_gaussian')
        if self.lw_lorentzian == 0 and self.lw_gaussian != 0:
            logging.warning('The spectrogram will be broadened using Gaussian linewidth.')
            self.lw_type = 'Gaussian'
        if self.lw_lorentzian != 0 and self.lw_gaussian == 0:
            logging.warning('The spectrogram will be broadened using lorentzian linewidth.')
            self.lw_type = 'Lorentzian'
        if self.lw_lorentzian != 0 and self.lw_gaussian != 0:
            logging.warning('The spectrogram will be broadened using both Lorentzian and Gaussian linewidths.')
            self.lw_type = 'Voigt'

        if self.sample not in ('Crystal_Oneside', 'Crystal_Rotation', 'Powder'):
            raise ValueError("Only 'Crystal_Oneside', 'Crystal_Rotation' or 'Powder' is allowed for 'sample'.")

        if self.sample == 'Crystal_Rotation':
            self.rotaxis = np.asarray(self.rotaxis, dtype=float)
            if self.rotaxis.shape != (3,):
                raise ValueError("'rotaxis' must be a three-dimensional vector")
            if not np.all(np.isfinite(self.rotaxis)):
                raise ValueError("'rotaxis' must contain finite values")
            if self.rotaxis.any():
                from .Vec_normalize import vec_normalize
                self.rotaxis = vec_normalize(self.rotaxis)
            self.anglerange = np.asarray(self.anglerange, dtype=float)
            if self.anglerange.shape == (2,):
                self.anglerange = np.array([self.anglerange[0], self.anglerange[1], 1.0])
            elif self.anglerange.shape != (3,):
                raise ValueError("Incorrect initial and ending angles for the given rotation!")
            if not np.all(np.isfinite(self.anglerange)):
                raise ValueError("'anglerange' must contain finite values")
            if self.anglerange[2] <= 0:
                raise ValueError("The angle step must be positive.")
            if self.anglerange[1] < self.anglerange[0]:
                raise ValueError("The ending angle must not be smaller than the initial angle.")

        if self.sample == 'Powder':
            try:
                count = int(self.npowder)
            except (TypeError, ValueError):
                raise ValueError("'npowder' must be a positive integer.")
            if count != self.npowder or count <= 0:
                raise ValueError("'npowder' must be a positive integer.")
            self.npowder = count

        if self.max_workers is not None:
            self.max_workers = _positive_int(self.max_workers, 'max_workers')

        if self.method not in ('High_Field_Method', 'Adaptive_Grid_Method'):
            raise ValueError("Only 'High_Field_Method' or 'Adaptive_Grid_Method' is allowed.")

        self.fieldrange = np.asarray(self.fieldrange, dtype=float)
        if self.fieldrange.ndim != 1 or self.fieldrange.size < 2:
            raise ValueError("Please enter an interval for legal magnetic field range in 'fieldrange'")
        if not np.all(np.isfinite(self.fieldrange)):
            raise ValueError("'fieldrange' must contain finite values")
        if self.fieldrange.size == 2:
            if self.fieldrange[0] >= self.fieldrange[1]:
                raise ValueError("Please enter an interval for legal magnetic field range in 'fieldrange'")
        else:
            steps = np.diff(self.fieldrange)
            if not np.all(steps > 0):
                raise ValueError("Please enter an interval for legal magnetic field range in 'fieldrange'")
            if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
                raise ValueError("An expanded 'fieldrange' must be uniformly spaced.")

        if self.fieldrange.size > 2:
            if self.specpoints == 0:
                self.specpoints = int(self.fieldrange.size)
            else:
                self.specpoints = _positive_int(self.specpoints, 'specpoints')
                if self.specpoints != self.fieldrange.size:
                    raise ValueError("The length of 'fieldrange' does not match 'specpoints'")
        elif self.specpoints == 0:
            from .Pointsnum_Optimise import get_points_spectrum
            self.specpoints = get_points_spectrum(self.fieldrange, self.lw_lorentzian, self.lw_gaussian)
        else:
            self.specpoints = _positive_int(self.specpoints, 'specpoints')

        if self.fieldrange.size == 2:
            self.fieldrange = np.linspace(self.fieldrange[0], self.fieldrange[1], self.specpoints)


def _number(value, name):
    """
    Inputs: value; parameter name.

    Returns: the value as a finite float.

    Notes: a non-numeric or non-finite value raises ValueError.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number.")
    if not np.isfinite(result):
        raise ValueError(f"'{name}' must be finite.")
    return result


def _positive_number(value, name):
    """
    Inputs: value; parameter name.

    Returns: the value as a positive finite float.
    """
    result = _number(value, name)
    if result <= 0:
        raise ValueError(f"'{name}' must be positive.")
    return result


def _nonnegative_number(value, name):
    """
    Inputs: value; parameter name.

    Returns: the value as a non-negative finite float.
    """
    result = _number(value, name)
    if result < 0:
        raise ValueError(f"'{name}' cannot be negative.")
    return result


def _positive_int(value, name):
    """
    Inputs: value; parameter name.

    Returns: the value as a positive int.

    Notes: a non-integer or non-positive value raises ValueError.
    """
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a positive integer.")
    if float(value) != result or result <= 0:
        raise ValueError(f"'{name}' must be a positive integer.")
    return result
