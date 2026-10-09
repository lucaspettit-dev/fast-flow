"""fast-flow: a fast 2D incompressible flow solver for rapid design iteration."""

__version__ = "0.2.0"

from .solver import FlowSolver, Direction, ForceHandler, ConstantVelocityForceHandler
from .solver import ElectrostaticForceHandler

__all__ = ["FlowSolver",
           "Direction",
           "ForceHandler",
           "ConstantVelocityForceHandler",
           "ElectrostaticForceHandler",
           "__version__"]
