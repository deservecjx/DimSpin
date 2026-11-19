"""DimSpin: numerical EPR/ESR spectrum simulator.

The public API is:

- :class:`Spin`: spin centre, tensors and Hamiltonian state.
- :class:`Spec`: experimental parameters.
- :func:`autosim`: run a simulation and return the spectrum.
- :func:`plot_energy_levels`: draw an energy-level diagram for one
  crystal orientation.

Two resonance-search methods are available: ``High_Field_Method`` and
``Adaptive_Grid_Method``.
"""

from .AutoSim import autosim
from .Class_Spec import Spec
from .Class_Spin import Spin
from .Levels_Plot import plot_energy_levels

__version__ = "0.1.0"

__all__ = [
    "Spin",
    "Spec",
    "autosim",
    "plot_energy_levels",
    "__version__",
]
