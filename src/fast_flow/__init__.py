"""fast-flow: a fast 2D incompressible flow solver for rapid design iteration."""

__version__ = "0.2.0"

from .solver import FlowSolver, Direction, ForceHandler, ConstantVelocityForceHandler

__all__ = ["FlowSolver",
           "Direction",
           "ForceHandler",
           "ConstantVelocityForceHandler",
           "__version__"]
