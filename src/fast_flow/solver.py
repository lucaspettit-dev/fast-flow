"""FlowSolver: public Python API over the compiled Cython core."""

from __future__ import annotations

import numpy as np
from enum import Enum
from .core.flow import FlowSolverCore
from .core.handlers.forcehandler import ConstantVelocityForceHandlerCore
from .core.electrostaticforcehandler import ElectrostaticForceHandler as ElectrostaticForceHandlerCore


class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class ForceHandler:
    pass


class ConstantVelocityForceHandler(ForceHandler):
    def __init__(self,
                 direction: Direction,
                 velocity: float):
        self._direction = direction
        self._velocity = velocity

    def _set_solver(self, solver: FlowSolverCore):
        self._handler = ConstantVelocityForceHandlerCore(
            solver, self._direction.value, self._velocity)


class ElectrostaticForceHandler(ForceHandler):
    """Electrostatic forcing from the charged regions of an image.

    Give the handler an image -- required, a numpy array of
    float64 (DTYPE_f) with shape (ny, nx, k) and both spatial
    dimensions at least 10, e.g. values 0.0/1.0 per layer --
    plus `layermap`, a dict mapping layer index -> charge (e.g. {0: 20000, 2: -20000}: red shapes positive,
    blue negative, every other layer neutral).  When the
    handler binds to a solver, the compiled core flattens the
    image and, per sign group, saves the non-solid cells that
    border a solid horizontally or vertically as that group's
    `edges` in a Shape struct (positive group first), with the
    group's charge in the struct's extradata.
    """

    def __init__(self, image, layermap: dict = None):
        self._image = image
        self._layermap = layermap

    def _set_solver(self, solver: FlowSolverCore):
        # The core constructor flattens the image, builds the
        # edge structs, and takes in the solver's solids bitmask
        # itself (a DTYPE_f copy made inside the core).
        self._handler = ElectrostaticForceHandlerCore(
            solver, self._image, self._layermap)

    @property
    def shape_charges(self) -> list:
        """The structs' charges (voltages): positive group first,
        then negative.  Before binding, derived from layermap."""
        if getattr(self, "_handler", None) is None:
            if not self._layermap:
                return []
            charges = []
            pos = [float(v) for v in self._layermap.values() if v > 0]
            neg = [float(v) for v in self._layermap.values() if v < 0]
            if pos:
                charges.append(pos[0])
            if neg:
                charges.append(neg[0])
            return charges
        return [self._handler.shape_charge(i)
                for i in range(self._handler.num_shapes())]

    @property
    def num_shapes(self) -> int:
        """Number of sign-group structs (0 before binding)."""
        if getattr(self, "_handler", None) is None:
            return 0
        return self._handler.num_shapes()

    def shape_edges(self, i: int) -> list:
        """Struct i's edge cells as (x, y) physical coordinates."""
        if getattr(self, "_handler", None) is None:
            raise RuntimeError("handler is not bound to a solver yet")
        return self._handler.shape_edges(i)

    def coulomb_field(self):
        """(u, v) Coulomb force components at the solver grid's
        cell centres, as read-only (ny, nx) arrays.  Computed
        once when the handler binds (static field, no ion
        movement)."""
        if getattr(self, "_handler", None) is None:
            raise RuntimeError("handler is not bound to a solver yet")
        return self._handler.coulomb_field()


class FlowSolver:
    """2D incompressible flow solver.

    Numerics: explicit advection/diffusion with a Jacobi pressure
    Poisson solve each step (Barba-style collocated formulation),
    all in compiled Cython.

    Timesteps are dynamic: the nominal ``dt`` is only a base
    value.  Every step takes the largest power-of-two scaling of
    ``dt`` (``dt`` * 2**k for integer k, up or down) that respects
    the explicit advection-diffusion stability limit for the
    current grid and maximum speed.  Prefer ``run(total_time)``,
    which advances a set amount of *simulated seconds*, over
    counting fixed steps.

    Boundary modes:
      "cavity" (default) -- closed box; drive it with a force
        handler (see add_force_handler()), no-slip walls elsewhere.
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
                 ly: float = 2.0,
                 force_handlers: [ForceHandler] = None):
        if boundary not in ("cavity", "throughflow"):
            raise ValueError(
                'boundary must be "cavity" or "throughflow"')
        if dt <= 0:
            raise ValueError("dt must be > 0")
        self._cy = FlowSolverCore(
            nx=nx, ny=ny, nit=nit, rho=rho, nu=nu, dt=dt,
            flow_mode=1 if boundary == "throughflow" else 0,
            inflow_u=inflow_velocity, lx=lx, ly=ly)
        self.boundary = boundary
        self.inflow_velocity = inflow_velocity
        self.lx = float(lx)
        self.ly = float(ly)
        self.obstacles: list = []
        if force_handlers is not None:
            for handler in force_handlers:
                handler._set_solver(self._cy)
                self._cy.add_force_handler(handler._handler)

    @property
    def nx(self): return self._cy.nx

    @property
    def ny(self): return self._cy.ny

    @property
    def dt(self):
        """Nominal (base) dt.  Actual steps dynamically scale this
        by powers of two; see ``last_dt`` for the most recent one."""
        return self._cy.dt_base

    @property
    def last_dt(self):
        """Dt actually used by the most recent step (0 before any)."""
        return self._cy.last_dt

    @property
    def time(self):
        """Total simulated seconds advanced so far."""
        return self._cy.sum_dt

    @property
    def dx(self): return self.lx / (self.nx - 1)

    @property
    def dy(self): return self.ly / (self.ny - 1)

    def stable_dt(self):
        """Current stability limit on dt for this grid/flow."""
        return self._cy.stable_dt()

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

    def add_force_handler(self, handler: ForceHandler):
        if getattr(handler, "_handler", None) is None:
            handler._set_solver(self._cy)
        self._cy.add_force_handler(handler._handler)

    def set_inflow_velocity(self, value):
        """Throughflow mode: set the uniform inflow speed."""
        self.inflow_velocity = value
        self._cy.set_inflow(value)

    def add_obstacle(self, polygon) -> None:
        """Add a solid obstacle as an iterable of (x, y) vertices (y up).

        Coordinates are physical, on [0, lx] x [0, ly].
        """
        xs = np.array([p[0] for p in polygon], dtype=np.float64)
        ys = np.array([p[1] for p in polygon], dtype=np.float64)
        self._cy.add_obstacle(xs, ys)
        self.obstacles.append([(float(x), float(y)) for x, y in polygon])

    def step(self) -> float:
        """Advance by one dynamically-chosen timestep.

        Returns the dt actually used (a power-of-two scaling of
        the nominal dt, the largest one currently stable).
        """
        self._cy.step()
        return self._cy.last_dt

    def run(self, total_time: float) -> int:
        """Advance ``total_time`` seconds of simulated time.

        Each step dynamically takes the largest stable dt (a
        power-of-two scaling of the nominal dt); the final step
        is trimmed to land exactly on ``total_time``.  Returns
        the number of steps taken.
        """
        if total_time <= 0:
            raise ValueError("total_time must be > 0")
        return self._cy.run(float(total_time))
