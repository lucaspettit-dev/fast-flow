"""FlowSolver: public Python API over the compiled Cython core."""

from __future__ import annotations

import numpy as np

from ._core import SolverCore


class FlowSolver:
    """2D incompressible flow past polygonal obstacles.

    Numerics (Jos Stam, "Stable Fluids", 1999): per step, implicit
    velocity diffusion, pressure projection, semi-Lagrangian
    self-advection, projection again -- all in compiled Cython.
    Boundary conditions: uniform inflow on the left edge,
    zero-gradient outflow on the right, no-slip walls top/bottom,
    no-slip on every obstacle polygon.
    """

    def __init__(self,
                 nx: int = 256,
                 ny: int = 128,
                 nit: int = 50,
                 rho: float = 1.0,
                 nu: float = 0.1,
                 dt: float = 0.02):
        self._cy = SolverCore(nx=nx, ny=ny, dt=dt)
        self.obstacles: list = []

    @property
    def nx(self): return self._cy.nx

    @property
    def ny(self): return self._cy.ny

    @property
    def dt(self): return self._cy.dt

    @property
    def horizontal_velocity(self) -> np.ndarray:
        return self._cy.horizontal_velocity

    @property
    def vertical_velocity(self) -> np.ndarray:
        return self._cy.vertical_velocity

    @property
    def pressure(self) -> np.ndarray:
        return self._cy.pressure

    def add_velocity(self, value):
        self._cy.add_velocity(value)

    def add_obstacle(self, polygon) -> None:
        """Add a solid obstacle as an iterable of (x, y) vertices (y up)."""
        xs = np.array([p[0] for p in polygon], dtype=np.float64)
        ys = np.array([p[1] for p in polygon], dtype=np.float64)
        self._cy.add_obstacle(xs, ys)
        self.obstacles.append([(float(x), float(y)) for x, y in polygon])

    def step(self) -> None:
        """Advance the flow by exactly one timestep (runs natively)."""
        self._cy.step()

