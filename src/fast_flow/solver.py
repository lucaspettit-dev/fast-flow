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

    def __init__(self, nx: int = 256, ny: int = 128,
                 lx: float = 2.0, ly: float = 1.0,
                 viscosity: float = 1.5e-5, vx0: float = 1.0,
                 dt: float = 0.02):
        self._cy = SolverCore(nx=nx, ny=ny, lx=lx, ly=ly,
                                viscosity=viscosity, vx0=vx0, dt=dt)
        self.obstacles: list = []

    @property
    def nx(self): return self._cy.nx

    @property
    def ny(self): return self._cy.ny

    @property
    def lx(self): return self._cy.lx

    @property
    def ly(self): return self._cy.ly

    @property
    def dt(self): return self._cy.dt

    @property
    def t(self): return self._cy.t

    @property
    def step_count(self): return self._cy.step_count

    def add_obstacle(self, polygon) -> None:
        """Add a solid obstacle as an iterable of (x, y) vertices (y up)."""
        xs = np.array([p[0] for p in polygon], dtype=np.float64)
        ys = np.array([p[1] for p in polygon], dtype=np.float64)
        self._cy.add_obstacle(xs, ys)
        self.obstacles.append([(float(x), float(y)) for x, y in polygon])

    def add_density_source(self, x: float, y: float, radius: float = 0.05,
                           color=(1.0, 0.0, 0.0), rate: float = 10.0) -> int:
        """Add a dye source; returns its id. Color is an (r, g, b) tuple."""
        return self._cy.add_density_source(x, y, radius, tuple(color), rate)

    def step(self) -> None:
        """Advance the flow by exactly one timestep (runs natively)."""
        self._cy.step()

    def run(self, steps: int, callback=None) -> None:
        """Advance ``steps`` timesteps.

        ``callback``, if given, is called as ``callback(step, self)``
        after each step -- rendering stays lazy unless the callback
        asks for it.
        """
        for step in range(1, steps + 1):
            self.step()
            if callback is not None:
                callback(step, self)

    @property
    def speed(self):
        """Speed magnitude as a read-only (ny, nx) float64 array."""
        return self._cy.render_speed()

    def render_density(self):
        """Composite dye field as a read-only (ny, nx, 3) uint8 RGB image."""
        return self._cy.render_density()

    def render_direction(self):
        """Flow-direction color wheel as a read-only (ny, nx, 3) uint8 RGB.

        Red = up, cyan = down; brightness scales with speed.
        """
        return self._cy.render_direction()
