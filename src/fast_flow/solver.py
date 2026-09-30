"""FlowSolver: fast 2D incompressible Navier-Stokes solver.

Status: skeleton. The public API below is fixed; the numerics are not
implemented yet. Planned algorithm: explicit operator splitting on a
staggered (MAC) grid -- semi-Lagrangian advection, explicit viscous
diffusion, pressure projection via an iterative Poisson solve --
with the hot loops compiled (Numba) so a design-iteration run finishes
in seconds, not minutes.
"""

from __future__ import annotations


class FlowSolver:
    """2D incompressible flow past polygonal obstacles.

    Boundary conditions (planned): uniform inflow on the left edge,
    zero-gradient outflow on the right edge, no-slip walls top/bottom,
    no-slip on every obstacle polygon.
    """

    def __init__(self, nx: int = 256, ny: int = 128,
                 lx: float = 2.0, ly: float = 1.0,
                 viscosity: float = 1.5e-5, vx0: float = 1.0):
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx, ly
        self.viscosity = viscosity
        self.vx0 = vx0
        self.obstacles: list = []

    def add_obstacle(self, polygon) -> None:
        """Add a solid obstacle as an iterable of (x, y) vertices."""
        self.obstacles.append([tuple(p) for p in polygon])

    def step(self) -> None:
        """Advance the flow by exactly one timestep."""
        raise NotImplementedError("solver numerics are not implemented yet")

    def run(self, steps: int, callback=None) -> None:
        """Advance ``steps`` timesteps, calling ``callback(step, self)`` after each."""
        raise NotImplementedError("solver numerics are not implemented yet")

    @property
    def speed(self):
        """Speed magnitude at cell centers as a ``(ny, nx)`` array."""
        raise NotImplementedError("solver numerics are not implemented yet")
