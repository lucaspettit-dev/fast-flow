"""FlowSolver: public Python API over the compiled Cython core."""

from __future__ import annotations

import numpy as np

from ._core import SolverCore


class FlowSolver:
    """2D incompressible flow solver.

    Numerics: explicit advection/diffusion with a Jacobi pressure
    Poisson solve each step (Barba-style collocated formulation),
    all in compiled Cython.

    Boundary modes:
      "cavity" (default) -- closed box; drive it with add_velocity()
        (moving lid on the top edge), no-slip walls elsewhere.
      "throughflow" -- the ehd-flow "infinite flow" setup: uniform
        inflow of inflow_velocity on the left edge, zero-gradient
        outflow on the right, no-slip top/bottom walls.
    """

    def __init__(self,
                 nx: int = 256,
                 ny: int = 128,
                 nit: int = 50,
                 rho: float = 1.0,
                 nu: float = 0.1,
                 dt: float = 0.02,
                 boundary: str = "cavity",
                 inflow_velocity: float = 1.0,
                 lx: float = 2.0,
                 ly: float = 2.0):
        if boundary not in ("cavity", "throughflow"):
            raise ValueError(
                'boundary must be "cavity" or "throughflow"')
        self._cy = SolverCore(
            nx=nx, ny=ny, nit=nit, rho=rho, nu=nu, dt=dt,
            flow_mode=1 if boundary == "throughflow" else 0,
            inflow_u=inflow_velocity, lx=lx, ly=ly)
        self.boundary = boundary
        self.inflow_velocity = inflow_velocity
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

    @property
    def solid(self) -> np.ndarray:
        """Read-only (ny, nx) mask; 1 where a cell is solid obstacle."""
        return self._cy.solid_mask

    def add_velocity(self, value):
        """Cavity mode: set the moving-lid speed on the top edge."""
        self._cy.add_velocity(value)

    def set_inflow_velocity(self, value):
        """Throughflow mode: set the uniform inflow speed."""
        self.inflow_velocity = value
        self._cy.set_inflow(value)

    def add_obstacle(self, polygon) -> None:
        """Add a solid obstacle as an iterable of (x, y) vertices (y up)."""
        xs = np.array([p[0] for p in polygon], dtype=np.float64)
        ys = np.array([p[1] for p in polygon], dtype=np.float64)
        self._cy.add_obstacle(xs, ys)
        self.obstacles.append([(float(x), float(y)) for x, y in polygon])

    def step(self) -> None:
        """Advance the flow by exactly one timestep (runs natively)."""
        self._cy.step()

