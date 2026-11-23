"""Powder spectra: average over crystallite orientations.

Orientations are sampled on a Fibonacci sphere. Each orientation is computed
without line broadening, and all transitions are broadened once at the end,
which is equivalent to broadening every orientation separately because the
convolution is linear.

For an electric-field modulated spectrum each sampled orientation is
evaluated together with its oppositely oriented partner, obtained by a half
turn about an axis perpendicular to the electric field. When the electric
field is zero there is no partner and only the sampled orientations are
used.
"""

import math

import numpy as np

from .Fibonacci_Orientation import fibonacci_orientation
from .Orientation_Switch import orientation_switch
from .Parallel import parallel_map
from .Rodrigues_Rotation import rotation_matrix
from .Spec_Oneside import broaden_transitions, spec_transitions

_worker_spins = None
_worker_spec = None
_worker_Rrot_average = None


def _init_worker(spins, spec, Rrot_average):
    global _worker_spins, _worker_spec, _worker_Rrot_average
    _worker_spins = spins
    _worker_spec = spec
    _worker_Rrot_average = Rrot_average


def _orientation_task(spin_Rrot):
    spin_idx, Rrot = spin_Rrot
    return _one_orientation(_worker_spins[spin_idx], _worker_spec, Rrot, _worker_Rrot_average)


def _one_orientation(spin, spec, Rrot, Rrot_average):
    """
    Inputs: Spin and Spec objects; orientation rotation; optional partner
    rotation (None when the electric field is zero).

    Returns: (positions, intensities) accumulated over the sampled orientation
    and, when present, its partner.

    Notes: spin is copied so that parallel workers never share mutable state.
    """
    spin = spin.clone()
    spin.R_tf2lf_rotated = Rrot @ spin.R_tf2lf
    spin.R_etf2lf_rotated = Rrot @ spin.R_etf2lf
    spin.R_dtf2lf_rotated = Rrot @ spin.R_dtf2lf
    pos1, int1 = spec_transitions(spin, spec)
    if Rrot_average is None:
        return pos1, int1

    spin.R_tf2lf_rotated = Rrot_average @ (Rrot @ spin.R_tf2lf)
    spin.R_etf2lf_rotated = Rrot_average @ (Rrot @ spin.R_etf2lf)
    spin.R_dtf2lf_rotated = Rrot_average @ (Rrot @ spin.R_dtf2lf)
    pos2, int2 = spec_transitions(spin, spec)

    return np.concatenate([pos1, pos2]), np.concatenate([int1, int2])


def powder(spins, spec, parallel=True):
    """
    Inputs: list of Spin objects; Spec object; whether to use a process pool.

    Returns: an array of shape (len(spins), specpoints).

    Notes: tasks are (spin index, orientation) pairs. Each orientation only
    produces transitions; the main process accumulates them per spin and
    broadens once.
    """
    n_spins = len(spins)
    oreientations = fibonacci_orientation(spec.npowder)

    if np.all(spec.Edirection == 0):
        Rrot_average = None
    else:
        averageaxis = np.array([-spec.Edirection[1], spec.Edirection[0], 0])
        if averageaxis[0] == 0 and averageaxis[1] == 0:
            averageaxis = np.array([0, -spec.Edirection[2], spec.Edirection[1]])
        Rrot_average = rotation_matrix(averageaxis, math.pi)
        for Ri in range(0, 3):
            for Rj in range(0, 3):
                if abs(Rrot_average[Ri][Rj]) < 1e-8:
                    Rrot_average[Ri][Rj] = 0

    Rrot_list = []
    for i in range(spec.npowder):
        if i == 0:
            Rrot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            Rrot = orientation_switch(np.array(oreientations[i]), np.array(oreientations[0]))
        Rrot_list.append(Rrot)

    tasks = [(si, Rrot) for si in range(n_spins) for Rrot in Rrot_list]
    if parallel:
        partials = parallel_map(_orientation_task, tasks, initializer=_init_worker,
                                initargs=(spins, spec, Rrot_average), max_workers=spec.max_workers)
    else:
        partials = (_one_orientation(spins[si], spec, Rrot, Rrot_average) for si, Rrot in tasks)

    pos_buckets = [[] for _ in range(n_spins)]
    int_buckets = [[] for _ in range(n_spins)]
    for (si, _), (pos, inten) in zip(tasks, partials):
        pos_buckets[si].append(pos)
        int_buckets[si].append(inten)

    spectra = np.zeros((n_spins, spec.specpoints))
    for si in range(n_spins):
        if pos_buckets[si]:
            positions = np.concatenate(pos_buckets[si])
            intensities = np.concatenate(int_buckets[si])
            spectra[si] = broaden_transitions(spec, positions, intensities)
    return spectra
