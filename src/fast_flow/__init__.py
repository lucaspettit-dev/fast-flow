"""fast-flow: a fast 2D incompressible flow solver for rapid design iteration."""

__version__ = "0.2.0"

from .solver import FlowSolver, Direction, Boundry, ForceHandler
from .solver import ElectrostaticForceHandler
from .solver import geometric_widths

__all__ = ["FlowSolver",
           "Direction",
           "Boundry",
           "ForceHandler",
           "ElectrostaticForceHandler",
           "geometric_widths",
           "__version__"]
