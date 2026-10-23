"""FlowSolver: public Python API over the compiled Cython core."""

from __future__ import annotations

import numpy as np
from enum import Enum
from .core.flow import FlowSolverCore
from .core.handlers.electrostaticforcehandler import ElectrostaticForceHandler as ElectrostaticForceHandlerCore
from .core.handlers.forcehandler import ForceHandlerCore
from .core.renderer import Renderer
from typing import Callable


class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Boundry(Enum):
    CAVITY = 0
    THROUGHFLOW = 1
    NONE = 2


def geometric_widths(n: int, h_min: float, growth: float,
                     h_max: float, core_fraction: float = 0.25):
    """Cell widths for a symmetric geometrically-stretched axis.

    The central ``core_fraction`` of the cells sit at
    ``h_min``; outside the core each cell grows by at most
    ``growth`` over its neighbour toward the core (the
    per-cell ratio never exceeds ``growth``, the usual CFD
    grading limit), capped at ``h_max``.  Returns
    ``(widths, span)`` where span is the centre-to-centre
    extent of the axis -- the solver's lx convention, ready
    to pass back as ``lx``/``ly`` alongside the widths.
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    if h_min <= 0 or growth < 1.0 or h_max < h_min:
        raise ValueError("need h_min > 0, growth >= 1, h_max >= h_min")
    if not 0.0 < core_fraction <= 1.0:
        raise ValueError("core_fraction must be in (0, 1]")
    core = max(1, int(round(n * core_fraction)))
    center = 0.5 * (n - 1)
    half_core = 0.5 * core
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = abs(i - center) - half_core + 1.0
        if d <= 0.0:
            w[i] = h_min
        else:
            w[i] = min(h_max, h_min * growth ** d)
    span = float(w.sum() - 0.5 * (w[0] + w[-1]))
    return w, span


class ForceHandler:

    def __init__(self, constructor: Callable[..., ForceHandler]):
        self._f = constructor

    def _build(self, solver: FlowSolverCore) -> ForceHandlerCore:
        return self._f(solver)


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
        super().__init__(lambda s: ElectrostaticForceHandlerCore(s, self._image, self._layermap))

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

    def ion_density(self):
        """Current +ion density at the solver grid's cells,
        as a read-only (ny, nx) array.  Zeros until the
        solver's first step advances the ions."""
        if getattr(self, "_handler", None) is None:
            raise RuntimeError("handler is not bound to a solver yet")
        return self._handler.ion_density()

    def ion_velocity(self):
        """Current ion velocity (field drift + air velocity)
        at the solver grid's cells, as (u, v) read-only
        (ny, nx) arrays."""
        if getattr(self, "_handler", None) is None:
            raise RuntimeError("handler is not bound to a solver yet")
        return self._handler.ion_velocity()

    def o2_density(self):
        """Current O2- (negative corona) density at the
        solver grid's cells, as a read-only (ny, nx)
        array.  Zeros until the solver's first step
        advances the ions, and zeros if no negative
        electrode is in corona (see neg_corona_active)."""
        if getattr(self, "_handler", None) is None:
            raise RuntimeError("handler is not bound to a solver yet")
        return self._handler.o2_density()

    def o2_velocity(self):
        """Current O2- velocity (field drift opposite to
        the +ions, + air velocity) at the solver grid's
        cells, as (u, v) read-only (ny, nx) arrays."""
        if getattr(self, "_handler", None) is None:
            raise RuntimeError("handler is not bound to a solver yet")
        return self._handler.o2_velocity()

    @property
    def pixel_size(self):
        """Real-world size of one grid cell in metres
        (metres per pixel), propagated from the solver
        this handler is bound to.  The handler's field
        solve, drift, transport and diffusion work on
        the solver's dx/dy, which this sizing defines
        when the solver is built from a pixel size."""
        if getattr(self, "_handler", None) is None:
            raise RuntimeError("handler is not bound to a solver yet")
        return self._handler.pixel_size

    @property
    def neg_corona_active(self):
        """True if any negative electrode is in corona
        (its surface field exceeds breakdown), i.e. O2-
        is being generated."""
        if getattr(self, "_handler", None) is None:
            return False
        return self._handler.neg_corona_active


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
                 img: np.ndarray,
                 nit: int = 50,
                 rho: float = 1.0,
                 nu: float = 0.1,
                 dt: float = 0.02,
                 boundary: Boundry = Boundry.CAVITY,
                 inflow_velocity: float = 1.0,
                 lx: float = None,
                 ly: float = None,
                 pixel_size: float = None,
                 x_widths: np.ndarray = None,
                 y_widths: np.ndarray = None,
                 force_handlers: [ForceHandler] = None,
                 renderer: Renderer = None):
        if dt <= 0:
            raise ValueError("dt must be > 0")
        # Physical scale: either `pixel_size` (the real-world
        # size of one grid cell, in metres per pixel, the
        # same in x and y) or the domain size `lx`/`ly`.
        # pixel_size defines lx/ly from the grid shape
        # (lx = pixel_size * (nx - 1), so the core's
        # dx = lx / (nx - 1) is exactly pixel_size); the
        # sizing is stored on the core, where bound force
        # handlers read it for their equations.  With
        # neither, the historical 2.0 x 2.0 domain is used.
        ny, nx = img.shape[0], img.shape[1]
        # Grid stretching (feature/grid-stretching): per-axis
        # cell widths (see geometric_widths).  Mutually
        # exclusive with pixel_size; lx / ly are derived from
        # the widths (the centre-to-centre span) unless given,
        # in which case they must agree with the span.
        wxa = None
        wya = None
        if x_widths is not None or y_widths is not None:
            if pixel_size is not None:
                raise ValueError(
                    "pass either pixel_size or widths, not both")
            if x_widths is not None:
                wxa = np.ascontiguousarray(x_widths, dtype=np.float64)
                if wxa.ndim != 1 or wxa.shape[0] != nx:
                    raise ValueError("x_widths must have length nx")
                span = float(wxa.sum() - 0.5 * (wxa[0] + wxa[-1]))
                if lx is not None and \
                        abs(lx - span) > 1e-9 * max(1.0, span):
                    raise ValueError("lx disagrees with the x_widths span")
                lx = span
            if y_widths is not None:
                wya = np.ascontiguousarray(y_widths, dtype=np.float64)
                if wya.ndim != 1 or wya.shape[0] != ny:
                    raise ValueError("y_widths must have length ny")
                span = float(wya.sum() - 0.5 * (wya[0] + wya[-1]))
                if ly is not None and \
                        abs(ly - span) > 1e-9 * max(1.0, span):
                    raise ValueError("ly disagrees with the y_widths span")
                ly = span
            if lx is None:
                lx = 2.0
            if ly is None:
                ly = 2.0
        elif pixel_size is not None:
            if lx is not None or ly is not None:
                raise ValueError(
                    "pass either pixel_size or lx/ly, not both")
            if pixel_size <= 0:
                raise ValueError("pixel_size must be > 0")
            lx = float(pixel_size) * (nx - 1)
            ly = float(pixel_size) * (ny - 1)
        elif lx is None and ly is None:
            lx = 2.0
            ly = 2.0
        elif lx is None or ly is None:
            raise ValueError("lx and ly must be given together")
        self.boundary = boundary
        self.inflow_velocity = inflow_velocity
        self.lx = float(lx)
        self.ly = float(ly)
        self.pixel_size = self.lx / (nx - 1) if nx > 1 else 0.0
        self.obstacles: list = []
        if force_handlers is None:
            force_handlers = []

        self.x_widths = wxa
        self.y_widths = wya
        self._cy = FlowSolverCore(
            img=img, nit=nit, rho=rho, nu=nu, dt=dt,
            flow_mode=boundary.value,
            inflow_u=inflow_velocity, lx=lx, ly=ly,
            x_widths=wxa, y_widths=wya,
            n_force_handlers=len(force_handlers))

        for handler in force_handlers:
            handler._set_solver(self._cy)
            self._cy.add_force_handler(handler._handler)

        self._renderer = renderer
        if self._renderer is not None:
            self._cy.set_renderer(self._renderer)

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

    def set_renderer(self, renderer: Renderer):
        """Attach a Renderer (a fast_flow.core.renderor.Renderer,
        or None to detach).  After every step the core calls
        the renderer's render_frame(dt) with that step's dt;
        the renderer tracks the total time and draws a frame
        only when one is due at its fps."""
        self._renderer = renderer
        self._cy.set_renderer(renderer)

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
