"""fast-flow: a fast 2D incompressible flow solver for rapid design iteration."""

__version__ = "0.2.0"

from .solver import FlowSolver, Direction, ForceHandler, ConstantVelocityForceHandler
from .solver import ElectrostaticForceHandler
from .polygons import PolygonExtractor, extract_layered_polygons

__all__ = ["FlowSolver",
           "Direction",
           "ForceHandler",
           "ConstantVelocityForceHandler",
           "ElectrostaticForceHandler",
           "PolygonExtractor",
           "extract_layered_polygons",
           "__version__"]
