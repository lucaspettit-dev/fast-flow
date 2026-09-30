# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Compiled core of fast-flow: MAC-grid (staggered) stable-fluids solver.

Grid layout (j = 0 is the physical bottom, y up):
  - u : (ny, nx+1)  x-velocity on vertical faces;   u[j,i] at (i*hx, (j+0.5)*hy)
  - v : (ny+1, nx)  y-velocity on horizontal faces; v[j,i] at ((i+0.5)*hx, j*hy)
  - p, div, solid, dye : (ny, nx) cell centers;      c[j,i] at ((i+0.5)*hx, (j+0.5)*hy)

On a staggered grid div(grad(p)) is exactly the compact 5-point Laplacian,
so the pressure projection is a true (energy-decreasing) projection --
unlike the collocated layout, which is unstable at higher resolutions.

All fields live in C-typed NumPy arrays; every timestep runs as compiled C
loops. Rendering is lazy: ``render_*`` computes on demand (cached until the
next step) and returns read-only NumPy arrays with image row 0 at the top.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, atan2, fabs, floor

cnp.import_array()


cdef inline double _bilin(double[:, :] f, double x, double y,
                          int nx, int ny) noexcept nogil:
    """Bilinear sample of f at its own grid coordinates (x, y), clamped."""
    cdef int i0, j0, i1, j1
    cdef double fx, fy, a, b
    if x < 0.0:
        x = 0.0
    elif x > nx - 1.0:
        x = nx - 1.0
    if y < 0.0:
        y = 0.0
    elif y > ny - 1.0:
        y = ny - 1.0
    i0 = <int>x
    j0 = <int>y
    i1 = i0 + 1
    if i1 >= nx:
        i1 = nx - 1
    j1 = j0 + 1
    if j1 >= ny:
        j1 = ny - 1
    fx = x - i0
    fy = y - j0
    a = f[j0, i0] * (1.0 - fx) + f[j0, i1] * fx
    b = f[j1, i0] * (1.0 - fx) + f[j1, i1] * fx
    return a * (1.0 - fy) + b * fy


cdef inline void _hsv_to_rgb(double h, double s, double v,
                             double* r, double* g, double* b) noexcept nogil:
    """h in [0, 360), s/v in [0, 1] -> rgb in [0, 1]."""
    cdef double c = v * s
    cdef double hh = h / 60.0
    cdef double x = c * (1.0 - fabs(hh - 2.0 * floor(hh / 2.0) - 1.0))
    cdef double m = v - c
    cdef double rr, gg, bb
    cdef int sector = <int>hh
    if sector <= 0:
        rr = c; gg = x; bb = 0.0
    elif sector == 1:
        rr = x; gg = c; bb = 0.0
    elif sector == 2:
        rr = 0.0; gg = c; bb = x
    elif sector == 3:
        rr = 0.0; gg = x; bb = c
    elif sector == 4:
        rr = x; gg = 0.0; bb = c
    else:
        rr = c; gg = 0.0; bb = x
    r[0] = rr + m
    g[0] = gg + m
    b[0] = bb + m


cdef class CyFlowSolver:
    """Fast 2D incompressible Navier-Stokes solver (Stam 1999, MAC grid).

    Per step: implicit diffusion of each velocity component, pressure
    projection to a divergence-free field, semi-Lagrangian advection of
    velocity, projection again, then per dye source inject/diffuse/advect.

    Boundary conditions: uniform inflow (u=vx0) on the left, zero-gradient
    outflow on the right, no-slip walls top/bottom, no-penetration on
    obstacle polygons.
    """

    cdef readonly int nx, ny
    cdef readonly double lx, ly, hx, hy, dt, viscosity, vx0, dens_diff
    cdef readonly double t
    cdef readonly long step_count
    cdef int diff_iters, proj_iters

    cdef object _u, _v, _u0, _v0, _p, _div, _solid
    cdef double[:, :] u, v, u0, v0, p, div
    cdef unsigned char[:, :] solid

    cdef list _dens       # current density field per source (np arrays)
    cdef list _dens0      # scratch density field per source
    cdef list _src        # (cx, cy, radius, r, g, b, rate) per source
    cdef dict _cache      # lazy render cache, cleared on every step

    def __init__(self, int nx=256, int ny=128,
                 double lx=2.0, double ly=1.0,
                 double viscosity=1.5e-5, double vx0=1.0, double dt=0.02,
                 int diff_iters=8, int proj_iters=30,
                 double dens_diff=0.0):
        if nx < 4 or ny < 4:
            raise ValueError("nx and ny must be >= 4")
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx, ly
        self.hx = lx / nx
        self.hy = ly / ny
        self.dt = dt
        self.viscosity = viscosity
        self.vx0 = vx0
        self.dens_diff = dens_diff
        self.diff_iters = diff_iters
        self.proj_iters = proj_iters
        self.t = 0.0
        self.step_count = 0

        self._u = np.full((ny, nx + 1), vx0, dtype=np.float64)
        self._v = np.zeros((ny + 1, nx), dtype=np.float64)
        self._u0 = np.empty((ny, nx + 1), dtype=np.float64)
        self._v0 = np.empty((ny + 1, nx), dtype=np.float64)
        self._p = np.zeros((ny, nx), dtype=np.float64)
        self._div = np.empty((ny, nx), dtype=np.float64)
        self._solid = np.zeros((ny, nx), dtype=np.uint8)
        self.u = self._u
        self.v = self._v
        self.u0 = self._u0
        self.v0 = self._v0
        self.p = self._p
        self.div = self._div
        self.solid = self._solid

        self._dens = []
        self._dens0 = []
        self._src = []
        self._cache = {}
        self._enforce_bcs()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def add_obstacle(self, cnp.ndarray[double, ndim=1] xs,
                     cnp.ndarray[double, ndim=1] ys):
        """Rasterize a polygon (physical coords, y up) into the solid mask."""
        cdef int n = xs.shape[0]
        if n != ys.shape[0] or n < 3:
            raise ValueError("need >= 3 vertices")
        cdef int i, j, k, crossings
        cdef double px, py, x1, y1, x2, y2, xinters
        for j in range(self.ny):
            py = (j + 0.5) * self.hy
            for i in range(self.nx):
                px = (i + 0.5) * self.hx
                crossings = 0
                for k in range(n):
                    x1 = xs[k]; y1 = ys[k]
                    x2 = xs[(k + 1) % n]; y2 = ys[(k + 1) % n]
                    if (y1 > py) != (y2 > py):
                        xinters = (x2 - x1) * (py - y1) / (y2 - y1) + x1
                        if px < xinters:
                            crossings += 1
                if crossings & 1:
                    self.solid[j, i] = 1
        self._enforce_bcs()
        self._cache.clear()

    def add_density_source(self, double cx, double cy, double radius,
                           tuple color=(1.0, 0.0, 0.0), double rate=10.0):
        """Add a dye source; returns its id. Color is an (r, g, b) tuple."""
        self._dens.append(np.zeros((self.ny, self.nx), dtype=np.float64))
        self._dens0.append(np.zeros((self.ny, self.nx), dtype=np.float64))
        self._src.append((cx, cy, radius,
                          float(color[0]), float(color[1]), float(color[2]),
                          rate))
        return len(self._src) - 1

    # ------------------------------------------------------------------
    # timestep
    # ------------------------------------------------------------------
    def step(self):
        """Advance the flow by exactly one timestep (all in C)."""
        cdef double ax = self.dt * self.viscosity / (self.hx * self.hx)
        cdef double ay = self.dt * self.viscosity / (self.hy * self.hy)
        cdef double dax = self.dt * self.dens_diff / (self.hx * self.hx)
        cdef double day = self.dt * self.dens_diff / (self.hy * self.hy)
        cdef int s, nsrc = len(self._src)
        cdef double[:, :] d, d0
        cdef double cx, cy, radius, rate

        # velocity: diffuse, project, advect, project (Stam 1999)
        self._copy(self.u0, self.u)
        self._copy(self.v0, self.v)
        self._diffuse_u(ax, ay)
        self._diffuse_v(ax, ay)
        self._enforce_bcs()
        self._project()
        self._copy(self.u0, self.u)
        self._copy(self.v0, self.v)
        self._advect_u()
        self._advect_v()
        self._enforce_bcs()
        self._project()

        # dye sources: inject, diffuse, advect
        for s in range(nsrc):
            cx, cy, radius = self._src[s][0], self._src[s][1], self._src[s][2]
            rate = self._src[s][6]
            d = self._dens[s]
            d0 = self._dens0[s]
            self._inject_one(cx, cy, radius, rate, d)
            self._copy(d0, d)
            if dax > 0.0 or day > 0.0:
                self._diffuse_c(d, d0, dax, day)
                self._neumann(d)
            self._copy(d0, d)
            self._advect_c(d, d0)
            self._neumann(d)
            self._zero_solid(d)

        self.step_count += 1
        self.t += self.dt
        self._cache.clear()

    cdef void _copy(self, double[:, :] dst, double[:, :] src) noexcept nogil:
        cdef int i, j, ni = dst.shape[1], nj = dst.shape[0]
        for j in range(nj):
            for i in range(ni):
                dst[j, i] = src[j, i]

    cdef void _enforce_bcs(self) noexcept nogil:
        # Dirichlet faces only: inlet, walls, solids. Everything else is a
        # free face owned by the pressure correction -- overwriting a
        # corrected free face would destroy the divergence-free field and
        # inject energy, so tangential mirrors / outlet copies of v are
        # deliberately NOT applied here.
        cdef int i, j
        cdef int nx = self.nx, ny = self.ny
        # solid cells: no penetration on all four faces
        for j in range(ny):
            for i in range(nx):
                if self.solid[j, i]:
                    self.u[j, i] = 0.0
                    self.u[j, i + 1] = 0.0
                    self.v[j, i] = 0.0
                    self.v[j + 1, i] = 0.0
        # inlet: prescribed inflow (face on the boundary, never corrected)
        for j in range(ny):
            self.u[j, 0] = self.vx0
        # outlet: zero gradient, slaved to the corrected interior face so
        # div stays zero in the outlet column
        for j in range(ny):
            self.u[j, nx] = self.u[j, nx - 1]
        # walls: no penetration (faces on the walls, never corrected);
        # tangential velocity is free-slip
        for i in range(nx):
            self.v[0, i] = 0.0
            self.v[ny, i] = 0.0

    cdef void _neumann(self, double[:, :] f) noexcept nogil:
        """Zero-gradient boundary for cell-centered scalars."""
        cdef int i, j
        cdef int nx = self.nx, ny = self.ny
        for i in range(nx):
            f[0, i] = f[1, i]
            f[ny - 1, i] = f[ny - 2, i]
        for j in range(ny):
            f[j, 0] = f[j, 1]
            f[j, nx - 1] = f[j, nx - 2]

    cdef void _zero_solid(self, double[:, :] f) noexcept nogil:
        cdef int i, j
        for j in range(self.ny):
            for i in range(self.nx):
                if self.solid[j, i]:
                    f[j, i] = 0.0

    cdef void _diffuse_u(self, double ax, double ay) noexcept nogil:
        """Implicit diffusion of u on its (ny, nx+1) face grid (Jacobi)."""
        cdef int i, j, k
        cdef double denom = 1.0 + 2.0 * ax + 2.0 * ay
        for k in range(self.diff_iters):
            for j in range(1, self.ny - 1):
                for i in range(1, self.nx):
                    self.u[j, i] = (self.u0[j, i]
                                    + ax * (self.u[j, i - 1] + self.u[j, i + 1])
                                    + ay * (self.u[j - 1, i] + self.u[j + 1, i])) / denom

    cdef void _diffuse_v(self, double ax, double ay) noexcept nogil:
        """Implicit diffusion of v on its (ny+1, nx) face grid (Jacobi)."""
        cdef int i, j, k
        cdef double denom = 1.0 + 2.0 * ax + 2.0 * ay
        for k in range(self.diff_iters):
            for j in range(1, self.ny):
                for i in range(1, self.nx - 1):
                    self.v[j, i] = (self.v0[j, i]
                                    + ax * (self.v[j, i - 1] + self.v[j, i + 1])
                                    + ay * (self.v[j - 1, i] + self.v[j + 1, i])) / denom

    cdef void _diffuse_c(self, double[:, :] x, double[:, :] x0,
                         double ax, double ay) noexcept nogil:
        """Implicit diffusion of a cell-centered scalar (Jacobi)."""
        cdef int i, j, k
        cdef double denom = 1.0 + 2.0 * ax + 2.0 * ay
        for k in range(self.diff_iters):
            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    x[j, i] = (x0[j, i]
                               + ax * (x[j, i - 1] + x[j, i + 1])
                               + ay * (x[j - 1, i] + x[j + 1, i])) / denom

    cdef void _project(self) noexcept nogil:
        """Pressure projection: solve lap(p) = div(u,v), subtract grad(p).

        On the MAC grid div(grad(p)) is exactly the compact 5-point
        Laplacian, so this is a true projection. The Laplacian uses the
        proper symmetric homogeneous-Neumann discretization (solid and
        domain-boundary neighbors drop out of both the diagonal and the
        off-diagonals). div is zero-meaned over fluid cells so the
        singular Neumann system is consistent, and p is zero-meaned to
        remove the constant null space.
        """
        cdef int i, j, k, nfluid
        cdef int nx = self.nx, ny = self.ny
        cdef double hx = self.hx, hy = self.hy
        cdef double px = 1.0 / (hx * hx), py = 1.0 / (hy * hy)
        cdef double diag, rhs, pmean, dmean
        for j in range(ny):
            for i in range(nx):
                if self.solid[j, i]:
                    self.div[j, i] = 0.0
                else:
                    self.div[j, i] = ((self.u[j, i + 1] - self.u[j, i]) / hx
                                      + (self.v[j + 1, i] - self.v[j, i]) / hy)
        # zero-mean div over fluid cells: consistency for the singular system
        dmean = 0.0
        nfluid = 0
        for j in range(ny):
            for i in range(nx):
                if not self.solid[j, i]:
                    dmean += self.div[j, i]
                    nfluid += 1
        if nfluid > 0:
            dmean /= nfluid
            for j in range(ny):
                for i in range(nx):
                    self.div[j, i] -= dmean
        # Gauss-Seidel on the symmetric Neumann Laplacian (p persists as
        # the initial guess across steps)
        for k in range(self.proj_iters):
            for j in range(ny):
                for i in range(nx):
                    if self.solid[j, i]:
                        continue
                    diag = 0.0
                    rhs = self.div[j, i]
                    if i > 0 and not self.solid[j, i - 1]:
                        rhs += px * self.p[j, i - 1]
                        diag += px
                    if i < nx - 1 and not self.solid[j, i + 1]:
                        rhs += px * self.p[j, i + 1]
                        diag += px
                    if j > 0 and not self.solid[j - 1, i]:
                        rhs += py * self.p[j - 1, i]
                        diag += py
                    if j < ny - 1 and not self.solid[j + 1, i]:
                        rhs += py * self.p[j + 1, i]
                        diag += py
                    if diag > 0.0:
                        self.p[j, i] = rhs / diag
        # zero-mean p: removes the constant null space
        pmean = 0.0
        nfluid = 0
        for j in range(ny):
            for i in range(nx):
                if not self.solid[j, i]:
                    pmean += self.p[j, i]
                    nfluid += 1
        if nfluid > 0:
            pmean /= nfluid
            for j in range(ny):
                for i in range(nx):
                    self.p[j, i] -= pmean
        # subtract the pressure gradient on interior faces
        for j in range(ny):
            for i in range(1, nx):
                self.u[j, i] -= (self.p[j, i] - self.p[j, i - 1]) / hx
        for j in range(1, ny):
            for i in range(nx):
                self.v[j, i] -= (self.p[j, i] - self.p[j - 1, i]) / hy
        self._enforce_bcs()

    cdef void _advect_u(self) noexcept nogil:
        """Semi-Lagrangian advection of u on its face grid."""
        cdef int i, j, im, ip
        cdef double uu, vv, xd, yd
        cdef double dtx = self.dt / self.hx, dty = self.dt / self.hy
        cdef int nx = self.nx, ny = self.ny
        for j in range(ny):
            for i in range(nx + 1):
                im = i - 1 if i > 0 else 0
                ip = i if i < nx else nx - 1
                uu = self.u0[j, i]
                vv = 0.25 * (self.v0[j, im] + self.v0[j, ip]
                             + self.v0[j + 1, im] + self.v0[j + 1, ip])
                xd = i - dtx * uu
                yd = (j + 0.5) - dty * vv
                self.u[j, i] = _bilin(self.u0, xd, yd, nx + 1, ny)

    cdef void _advect_v(self) noexcept nogil:
        """Semi-Lagrangian advection of v on its face grid."""
        cdef int i, j, jm, jp
        cdef double uu, vv, xd, yd
        cdef double dtx = self.dt / self.hx, dty = self.dt / self.hy
        cdef int nx = self.nx, ny = self.ny
        for j in range(ny + 1):
            for i in range(nx):
                jm = j - 1 if j > 0 else 0
                jp = j if j < ny else ny - 1
                uu = 0.25 * (self.u0[jm, i] + self.u0[jm, i + 1]
                             + self.u0[jp, i] + self.u0[jp, i + 1])
                vv = self.v0[j, i]
                xd = (i + 0.5) - dtx * uu
                yd = j - dty * vv
                self.v[j, i] = _bilin(self.v0, xd, yd, nx, ny + 1)

    cdef void _advect_c(self, double[:, :] d, double[:, :] d0) noexcept nogil:
        """Semi-Lagrangian advection of a cell-centered scalar."""
        cdef int i, j
        cdef double uu, vv, xd, yd
        cdef double dtx = self.dt / self.hx, dty = self.dt / self.hy
        for j in range(self.ny):
            for i in range(self.nx):
                uu = 0.5 * (self.u[j, i] + self.u[j, i + 1])
                vv = 0.5 * (self.v[j, i] + self.v[j + 1, i])
                xd = (i + 0.5) - dtx * uu
                yd = (j + 0.5) - dty * vv
                d[j, i] = _bilin(d0, xd, yd, self.nx, self.ny)

    cdef void _inject_one(self, double cx, double cy, double radius,
                          double rate, double[:, :] d) noexcept nogil:
        cdef int i, j, i0, i1, j0, j1
        cdef double px, py, dist, fall, add
        i0 = <int>((cx - radius) / self.hx - 0.5)
        if i0 < 0:
            i0 = 0
        i1 = <int>((cx + radius) / self.hx - 0.5) + 1
        if i1 >= self.nx:
            i1 = self.nx - 1
        j0 = <int>((cy - radius) / self.hy - 0.5)
        if j0 < 0:
            j0 = 0
        j1 = <int>((cy + radius) / self.hy - 0.5) + 1
        if j1 >= self.ny:
            j1 = self.ny - 1
        add = rate * self.dt
        for j in range(j0, j1 + 1):
            py = (j + 0.5) * self.hy
            for i in range(i0, i1 + 1):
                px = (i + 0.5) * self.hx
                dist = sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))
                if dist < radius:
                    fall = 1.0 - (dist / radius) * (dist / radius)
                    d[j, i] += add * fall

    # ------------------------------------------------------------------
    # lazy renders (computed on demand, cached until the next step)
    # ------------------------------------------------------------------
    cdef inline void _cell_vel(self, int j, int i,
                               double* uc, double* vc) noexcept nogil:
        uc[0] = 0.5 * (self.u[j, i] + self.u[j, i + 1])
        vc[0] = 0.5 * (self.v[j, i] + self.v[j + 1, i])

    def render_speed(self):
        """Speed magnitude as a read-only (ny, nx) float64 array."""
        cached = self._cache.get("speed")
        if cached is not None:
            return cached
        cdef cnp.ndarray out = np.empty((self.ny, self.nx), dtype=np.float64)
        cdef double[:, :] o = out
        cdef int i, j, r
        cdef double uc, vc
        for r in range(self.ny):
            j = self.ny - 1 - r  # image row 0 = physical top
            for i in range(self.nx):
                self._cell_vel(j, i, &uc, &vc)
                o[r, i] = sqrt(uc * uc + vc * vc)
        out.flags.writeable = False
        self._cache["speed"] = out
        return out

    def render_density(self):
        """Composite dye field as a read-only (ny, nx, 3) uint8 RGB image.

        Each source is normalized by its own current max and tinted with
        its color; sources blend additively.
        """
        cached = self._cache.get("density")
        if cached is not None:
            return cached
        cdef cnp.ndarray out = np.zeros((self.ny, self.nx, 3), dtype=np.uint8)
        cdef unsigned char[:, :, :] o = out
        cdef int s, i, j, r, nsrc = len(self._src)
        cdef double[:, :] d
        cdef double m, t, rr, gg, bb, nv
        for s in range(nsrc):
            d = self._dens[s]
            m = 0.0
            for j in range(self.ny):
                for i in range(self.nx):
                    if d[j, i] > m:
                        m = d[j, i]
            if m <= 1e-12:
                continue
            rr = self._src[s][3]; gg = self._src[s][4]; bb = self._src[s][5]
            for r in range(self.ny):
                j = self.ny - 1 - r
                for i in range(self.nx):
                    t = d[j, i] / m
                    if t > 1.0:
                        t = 1.0
                    nv = o[r, i, 0] + rr * 255.0 * t
                    o[r, i, 0] = 255 if nv > 255.0 else <unsigned char>nv
                    nv = o[r, i, 1] + gg * 255.0 * t
                    o[r, i, 1] = 255 if nv > 255.0 else <unsigned char>nv
                    nv = o[r, i, 2] + bb * 255.0 * t
                    o[r, i, 2] = 255 if nv > 255.0 else <unsigned char>nv
        out.flags.writeable = False
        self._cache["density"] = out
        return out

    def render_direction(self):
        """Flow-direction color wheel as a read-only (ny, nx, 3) uint8 RGB.

        Red = up, cyan = down; brightness scales with speed (still
        regions are black).
        """
        cached = self._cache.get("direction")
        if cached is not None:
            return cached
        cdef cnp.ndarray out = np.zeros((self.ny, self.nx, 3), dtype=np.uint8)
        cdef unsigned char[:, :, :] o = out
        cdef int i, j, r
        cdef double sp, vmax = 1e-12, t, theta, hue, uc, vc, rr, gg, bb
        for j in range(self.ny):
            for i in range(self.nx):
                self._cell_vel(j, i, &uc, &vc)
                sp = sqrt(uc * uc + vc * vc)
                if sp > vmax:
                    vmax = sp
        for r in range(self.ny):
            j = self.ny - 1 - r
            for i in range(self.nx):
                self._cell_vel(j, i, &uc, &vc)
                sp = sqrt(uc * uc + vc * vc)
                if sp <= 1e-12:
                    continue
                t = sp / vmax
                if t > 1.0:
                    t = 1.0
                # v > 0 is physically up; red at up, cyan at down
                theta = atan2(vc, uc) * 57.29577951308232
                hue = 90.0 - theta
                if hue < 0.0:
                    hue += 360.0
                elif hue >= 360.0:
                    hue -= 360.0
                _hsv_to_rgb(hue, 1.0, t, &rr, &gg, &bb)
                o[r, i, 0] = <unsigned char>(rr * 255.0)
                o[r, i, 1] = <unsigned char>(gg * 255.0)
                o[r, i, 2] = <unsigned char>(bb * 255.0)
        out.flags.writeable = False
        self._cache["direction"] = out
        return out
