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
cimport cython
from cython cimport view
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, atan2, fabs, floor

ctypedef cnp.float64_t DTYPE_f


cdef class SolverCore:
    """Fast 2D incompressible Navier-Stokes solver (Stam 1999, MAC grid).

    Per step: implicit diffusion of each velocity component, pressure
    projection to a divergence-free field, semi-Lagrangian advection of
    velocity, projection again, then per dye source inject/diffuse/advect.

    Boundary conditions depend on flow_mode:
      0 (cavity)     -- moving lid on top (see add_velocity), no-slip
                        walls elsewhere.
      1 (throughflow)-- uniform inflow on the left, zero-gradient
                        outflow on the right, no-slip top/bottom
                        (the ehd-flow "infinite flow" setup).

    Solid obstacles (either mode): add_obstacle() rasterizes
    polygons into a cell mask.  Solid cells carry exactly zero
    velocity and zero pressure, and fluid stencils ignore solid
    neighbours (they substitute the fluid cell's own pressure,
    i.e. zero normal gradient at the wall).
    """

    cdef readonly int nx
    cdef readonly int ny
    cdef readonly DTYPE_f rho
    cdef readonly DTYPE_f nu
    cdef readonly DTYPE_f dt
    cdef readonly DTYPE_f dt_base
    cdef readonly DTYPE_f last_dt
    cdef readonly DTYPE_f sum_dt
    cdef readonly DTYPE_f lx
    cdef readonly DTYPE_f ly
    cdef DTYPE_f dx
    cdef DTYPE_f dy
    cdef int nit
    cdef int N
    cdef int flow_mode
    cdef DTYPE_f inflow_u

    cdef object _solid
    cdef DTYPE_f* b
    cdef DTYPE_f** u
    cdef DTYPE_f** v
    cdef DTYPE_f** p
    cdef Py_ssize_t pk
    cdef Py_ssize_t uvk
    cdef unsigned char[:, :] solid

    def __init__(
            self,
            int nx=256,
            int ny=128,
            int nit = 50,
            double rho = 1.0,
            double nu = 0.1,
            double dt=0.001,
            int flow_mode=0,
            double inflow_u=1.0,
            double lx=2.0,
            double ly=2.0):

        if nx < 4 or ny < 4:
            raise ValueError("nx and ny must be >= 4")

        self.nx = nx
        self.ny = ny
        self.N = nx * ny
        self.rho = rho
        self.nu = nu
        self.dt = dt
        self.dt_base = dt
        self.last_dt = 0.0
        self.nit = nit
        self.flow_mode = flow_mode
        self.inflow_u = inflow_u
        self.lx = lx
        self.ly = ly

        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)

        # init data structures
        self.u = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        self.u[0] = <DTYPE_f*> malloc(self.N * sizeof(DTYPE_f))
        self.u[1] = <DTYPE_f*> malloc(self.N * sizeof(DTYPE_f))

        self.v = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        self.v[0] = <DTYPE_f*> malloc(self.N * sizeof(DTYPE_f))
        self.v[1] = <DTYPE_f*> malloc(self.N * sizeof(DTYPE_f))

        self.p = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        self.p[0] = <DTYPE_f*> malloc(self.N * sizeof(DTYPE_f))
        self.p[1] = <DTYPE_f*> malloc(self.N * sizeof(DTYPE_f))

        self.b = <DTYPE_f*> malloc(self.N * sizeof(DTYPE_f))
        self.pk = 0
        self.uvk = 0
        self.sum_dt = 0.0

        self._solid = np.zeros((ny, nx), dtype=np.uint8)
        self.solid = self._solid

        self.init_arrays()

    def __dealloc__(self):
        if self.b != NULL:
            free(self.b)

        if self.u != NULL:
            if self.u[0] != NULL:
                free(self.u[0])
            if self.u[1] != NULL:
                free(self.u[1])
            free(self.u)
            self.u = NULL

        if self.v != NULL:
            if self.v[0] != NULL:
                free(self.v[0])
            if self.v[1] != NULL:
                free(self.v[1])
            free(self.v)
            self.v = NULL

        if self.p != NULL:
            if self.p[0] != NULL:
                free(self.p[0])
            if self.p[1] != NULL:
                free(self.p[1])
            free(self.p)
            self.p = NULL

    cdef init_arrays(self):
        # Populate your data here
        cdef Py_ssize_t i, j
        for i in range(self.N):
            self.b[i] = 0.0
            for j in range(2):
                self.u[j][i] = 0.0
                self.v[j][i] = 0.0
                self.p[j][i] = 0.0

    @property
    def pressure(self) -> np.ndarray:
        return self.to_numpy(self.p[self.pk])

    @property
    def horizontal_velocity(self) -> np.ndarray:
        return self.to_numpy(self.u[self.uvk])

    @property
    def vertical_velocity(self) -> np.ndarray:
        return self.to_numpy(self.v[self.uvk])

    @property
    def solid_mask(self) -> np.ndarray:
        """Read-only (ny, nx) uint8 mask; 1 = solid obstacle cell."""
        arr = np.array(self._solid, copy=True)
        arr.flags.writeable = False
        return arr

    cdef inline cnp.ndarray to_numpy(self, DTYPE_f* a):
        cdef DTYPE_f[:] view = <DTYPE_f[:self.N]> a
        cdef cnp.ndarray arr = np.asarray(view)
        arr = arr.reshape((self.ny, self.nx))
        arr.flags.writeable = False
        return arr

    cdef inline Py_ssize_t idx(
            self,
            Py_ssize_t y,
            Py_ssize_t x
    ) noexcept nogil:
        return y * self.nx + x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void build_up_pressure_step(
            self,
            Py_ssize_t y,
            Py_ssize_t x,
            DTYPE_f dy2,
            DTYPE_f dx2,
            DTYPE_f inv_dt) noexcept nogil:
            DTYPE_f inv_dt):
        if self.solid[y + 1, x + 1]:
            return

        cdef DTYPE_f* u = self.u[self.uvk]
        cdef DTYPE_f* v = self.v[self.uvk]

        cdef DTYPE_f hor = (u[self.idx(y+1, x+2)] - u[self.idx(y+1, x)]) / dx2
        cdef DTYPE_f vert = (v[self.idx(y+2, x+1)] - v[self.idx(y, x+1)]) / dy2

        cdef DTYPE_f c = 2 * (
            (u[self.idx(y+2, x+1)] - u[self.idx(y, x+1)]) / dy2
            * (v[self.idx(y+1, x+2)] - v[self.idx(y+1, x)]) / dx2
        )
        cdef DTYPE_f a = hor + vert
        cdef DTYPE_f b = hor * hor
        cdef DTYPE_f d = vert * vert

        self.b[self.idx(y+1, x+1)] = self.rho * (inv_dt * a - b - c - d)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void pressure_poisson_step(
            self,
            Py_ssize_t y,
            Py_ssize_t x,
            DTYPE_f* p,
            DTYPE_f* p_,
            DTYPE_f dy_squared,
            DTYPE_f dx_squared) noexcept nogil:
        cdef Py_ssize_t idx = self.idx(y+1, x+1)
        cdef DTYPE_f pc = p_[idx]
        cdef DTYPE_f pe = pc if self.solid[y + 1, x + 2] \
            else p_[self.idx(y+1, x+2)]
        cdef DTYPE_f pw = pc if self.solid[y + 1, x] \
            else p_[self.idx(y+1, x)]
        cdef DTYPE_f pn = pc if self.solid[y + 2, x + 1] \
            else p_[self.idx(y+2, x+1)]
        cdef DTYPE_f ps = pc if self.solid[y, x + 1] \
            else p_[self.idx(y, x+1)]
        cdef DTYPE_f hor = (pe + pw) * dy_squared
        cdef DTYPE_f vert = (pn + ps) * dx_squared

        cdef DTYPE_f d = 2 * (dx_squared + dy_squared)

        if not self.solid[y + 1, x + 1]:
            p[idx] = (hor + vert) / d - (dx_squared * dy_squared / d) * self.b[idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef pressure_set_boundry_conditions(self):
        cdef DTYPE_f* p = self.p[self.pk ^ 1]
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        cdef Py_ssize_t n = self.nx
        if self.ny > self.nx:
            n = self.ny

        # solid cells: pressure is clamped to zero, nothing diffuses in
        for j in prange(self.ny, nogil=True):
            for i in range(self.nx):
                if self.solid[j, i]:
                    p[self.idx(j, i)] = 0.0

        if self.flow_mode == 1:
            # throughflow: zero-gradient pressure on every side
            for i in range(1, self.nx - 1):
                p[self.idx(0, i)] = p[self.idx(1, i)]
                p[self.idx(self.ny-1, i)] = p[self.idx(self.ny-2, i)]
            for i in range(1, self.ny - 1):
                p[self.idx(i, 0)] = p[self.idx(i, 1)]
                p[self.idx(i, self.nx-1)] = p[self.idx(i, self.nx-2)]
            p[0] = p[self.idx(1, 1)]
            p[self.idx(0, self.nx-1)] = p[self.idx(1, self.nx-2)]
            p[self.idx(self.ny-1, 0)] = p[self.idx(self.ny-2, 1)]
            p[self.idx(self.ny-1, self.nx-1)] = \
                p[self.idx(self.ny-2, self.nx-2)]
            return

        for i in range(n):
            # left and right boundaries, excluding corners
            if 0 < i and i < self.ny - 1:
                p[self.idx(i, 0)] = p[self.idx(i, 1)]
                p[self.idx(i, self.nx-1)] = p[self.idx(i, self.nx-2)]

            # bottom boundary, excluding corners
            if 0 < i and i < self.nx - 1:
                p[self.idx(0, i)] = p[self.idx(1, i)]

            # top boundary (dirichlet condition)
            if i < self.nx:
                p[self.idx(self.ny-1, i)] = 0

        # bottom-left corner
        p[0] = p[self.idx(1, 1)]

        # bottom-right corner
        p[self.idx(0, self.nx-1)] = p[self.idx(1, self.nx-2)]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef pressure_poisson(self):
        cdef DTYPE_f dy_squared = self.dy * self.dy
        cdef DTYPE_f dx_squared = self.dx * self.dx
        cdef DTYPE_f dy2 = 2 * self.dy
        cdef DTYPE_f dx2 = 2 * self.dx
        cdef DTYPE_f inv_dt = 1 / self.dt
        cdef Py_ssize_t y
        cdef Py_ssize_t x
        cdef Py_ssize_t q
        cdef DTYPE_f* pp
        cdef DTYPE_f* p_src
        cdef DTYPE_f* p_dest
        cdef DTYPE_f pmean
        cdef Py_ssize_t jj, ii, base
        cdef int nfluid

        # determine which p is written to and which is read from.
        cdef Py_ssize_t src = self.pk
        cdef Py_ssize_t dest = src ^ 1

        # do the first loop w/build-pressure step (rows are
        # independent: each writes only its own p/b cells and reads
        # the source pressure buffer, so this runs GIL-free in
        # parallel via OpenMP)
        p_dest = self.p[dest]
        p_src = self.p[src]
        for y in prange(self.ny - 2, nogil=True):
            for x in range(self.nx - 2):
                self.build_up_pressure_step(y, x, dy2, dx2, inv_dt)
                self.pressure_poisson_step(y, x, p_dest, p_src, dy_squared, dx_squared)
        self.pressure_set_boundry_conditions()

        self.pk ^= 1
        src = self.pk
        dest = src ^ 1

        for q in range(self.nit - 1):
            p_dest = self.p[dest]
            p_src = self.p[src]
            for y in prange(self.ny - 2, nogil=True):
                for x in range(self.nx - 2):
                    self.pressure_poisson_step(y, x, p_dest, p_src, dy_squared, dx_squared)
            self.pressure_set_boundry_conditions()

            # rotate src/dest
            self.pk ^= 1
            src = self.pk
            dest = src ^ 1

        if self.flow_mode == 1:
            # all-Neumann pressure is defined only up to a constant;
            # pin the fluid mean to zero so it cannot drift between
            # steps (solids are exactly zero and stay out of this)
            pp = self.p[self.pk]
            pmean = 0.0
            nfluid = 0
            for jj in range(self.ny):
                base = jj * self.nx
                for ii in range(self.nx):
                    if not self.solid[jj, ii]:
                        pmean += pp[base + ii]
                        nfluid += 1
            if nfluid > 0:
                pmean /= nfluid
            for jj in prange(self.ny, nogil=True):
                base = jj * self.nx
                for ii in range(self.nx):
                    if not self.solid[jj, ii]:
                        pp[base + ii] -= pmean

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef update_momentum(self):
        cdef Py_ssize_t src = self.uvk
        cdef Py_ssize_t dest = src ^ 1

        cdef DTYPE_f* u_src = self.u[src]
        cdef DTYPE_f* v_src = self.v[src]
        cdef DTYPE_f* u_dest = self.u[dest]
        cdef DTYPE_f* v_dest = self.v[dest]
        cdef DTYPE_f* p = self.p[self.pk]

        cdef Py_ssize_t nx = self.nx
        cdef Py_ssize_t ny = self.ny
        cdef Py_ssize_t x, y, idx

        cdef DTYPE_f dt = self.dt
        cdef DTYPE_f dx = self.dx
        cdef DTYPE_f dy = self.dy
        cdef DTYPE_f rho = self.rho
        cdef DTYPE_f nu = self.nu
        cdef DTYPE_f p_c, p_e, p_w, p_n, p_s
        cdef DTYPE_f uc, vc, dudx, dudy, dvdx, dvdy

        # rows are independent (read src buffers, write own dest
        # row), so this runs GIL-free in parallel via OpenMP.
        # idx is recomputed per cell rather than carried with += :
        # a carried counter looks like a reduction variable to
        # prange and is rejected.
        for y in prange(1, ny - 1, nogil=True):
            for x in range(1, nx - 1):
                idx = y * nx + x

                if self.solid[y, x]:
                    u_dest[idx] = 0.0
                    v_dest[idx] = 0.0
                    continue

                # pressure at the centre and each neighbour; a
                # neighbour inside a solid is ignored (own value
                # used, i.e. zero normal gradient at the wall)
                p_c = p[idx]
                p_e = p_c if self.solid[y, x + 1] else p[idx + 1]
                p_w = p_c if self.solid[y, x - 1] else p[idx - 1]
                p_n = p_c if self.solid[y + 1, x] else p[idx + nx]
                p_s = p_c if self.solid[y - 1, x] else p[idx - nx]

                # Upwind convection: each derivative is differenced
                # toward the side the flow comes from, chosen by the
                # sign of the advecting velocity at this cell.  (The
                # old code always used the backward difference, which
                # is upwind only for positive velocities and downwind
                # -- anti-dissipative -- wherever the flow reverses.)
                # Neighbour values inside solids are the enforced 0,
                # as before.  Diffusion and pressure stay centred.
                uc = u_src[idx]
                vc = v_src[idx]
                if uc >= 0.0:
                    dudx = (u_src[idx] - u_src[idx - 1]) / dx
                    dvdx = (v_src[idx] - v_src[idx - 1]) / dx
                else:
                    dudx = (u_src[idx + 1] - u_src[idx]) / dx
                    dvdx = (v_src[idx + 1] - v_src[idx]) / dx
                if vc >= 0.0:
                    dudy = (u_src[idx] - u_src[idx - nx]) / dy
                    dvdy = (v_src[idx] - v_src[idx - nx]) / dy
                else:
                    dudy = (u_src[idx + nx] - u_src[idx]) / dy
                    dvdy = (v_src[idx + nx] - v_src[idx]) / dy

                # Horizontal velocity u
                u_dest[idx] = (
                    u_src[idx]

                    # convection (upwind)
                    - dt * (uc * dudx + vc * dudy)

                    # x pressure gradient
                    - dt / (2.0 * rho * dx)
                    * (p_e - p_w)

                    # viscosity
                    + nu * (
                        dt / (dx * dx)
                        * (
                            u_src[idx + 1]
                            - 2.0 * u_src[idx]
                            + u_src[idx - 1]
                        )
                        +
                        dt / (dy * dy)
                        * (
                            u_src[idx + nx]
                            - 2.0 * u_src[idx]
                            + u_src[idx - nx]
                        )
                    )
                )

                # Vertical velocity v
                v_dest[idx] = (
                    v_src[idx]

                    # convection (upwind)
                    - dt * (uc * dvdx + vc * dvdy)

                    # y pressure gradient
                    - dt / (2.0 * rho * dy)
                    * (p_n - p_s)

                    # viscosity
                    + nu * (
                        dt / (dx * dx)
                        * (
                            v_src[idx + 1]
                            - 2.0 * v_src[idx]
                            + v_src[idx - 1]
                        )
                        +
                        dt / (dy * dy)
                        * (
                            v_src[idx + nx]
                            - 2.0 * v_src[idx]
                            + v_src[idx - nx]
                        )
                    )
                )

        # dest now contains the newest velocity field.
        self.uvk = dest

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef clamp_momentum_boundary(self):
        cdef Py_ssize_t nx = self.nx
        cdef Py_ssize_t ny = self.ny
        cdef Py_ssize_t last_row = (ny - 1) * nx
        cdef Py_ssize_t n = nx if nx > ny else ny

        cdef Py_ssize_t i
        cdef Py_ssize_t left
        cdef Py_ssize_t right

        cdef DTYPE_f* u = self.u[self.uvk]
        cdef DTYPE_f* v = self.v[self.uvk]

        if self.flow_mode == 1:
            # throughflow: prescribed uniform inflow on the left,
            # zero-gradient outflow on the right, no-slip top/bottom
            for i in range(ny):
                left = i * nx
                right = left + nx - 1
                u[left] = self.inflow_u
                v[left] = 0.0
                u[right] = u[right - 1]
                v[right] = v[right - 1]
            for i in range(nx):
                u[i] = 0.0
                v[i] = 0.0
                u[last_row + i] = 0.0
                v[last_row + i] = 0.0
            return

        for i in range(n):

            # Bottom boundary.
            if i < nx:
                u[i] = 0.0
                v[i] = 0.0

                # Top boundary:
                # Don't modify u here because that's the moving lid.
                v[last_row + i] = 0.0

            # Left/right boundaries, excluding lid corners.
            if i < ny - 1:
                left = i * nx
                right = left + nx - 1

                u[left] = 0.0
                u[right] = 0.0

                v[left] = 0.0
                v[right] = 0.0


    cdef enforce_solids(self):
        """No-slip: velocity is exactly zero inside solid cells."""
        cdef Py_ssize_t i, j, idx
        cdef DTYPE_f* u = self.u[self.uvk]
        cdef DTYPE_f* v = self.v[self.uvk]
        for j in prange(self.ny, nogil=True):
            idx = j * self.nx
            for i in range(self.nx):
                if self.solid[j, i]:
                    u[idx + i] = 0.0
                    v[idx + i] = 0.0

    cdef DTYPE_f _max_speed(self):
        cdef DTYPE_f* u = self.u[self.uvk]
        cdef DTYPE_f* v = self.v[self.uvk]
        cdef DTYPE_f m = 0.0
        cdef DTYPE_f a
        cdef Py_ssize_t k
        for k in range(self.N):
            a = fabs(u[k])
            if a > m:
                m = a
            a = fabs(v[k])
            if a > m:
                m = a
        return m

    cdef DTYPE_f _stable_dt_limit(self):
        """Largest explicit step allowed by the combined
        advection-diffusion stability limit for this grid and the
        current maximum speed (the two limits combine -- their
        reciprocals add), with a 0.9 safety factor."""
        cdef DTYPE_f denom = 0.0
        cdef DTYPE_f umax
        if self.nu > 0.0:
            denom += 2.0 * self.nu * (1.0 / (self.dx * self.dx)
                                      + 1.0 / (self.dy * self.dy))
        umax = self._max_speed()
        if self.flow_mode == 1 and self.inflow_u > umax:
            umax = self.inflow_u
        if umax > 0.0:
            denom += umax * (1.0 / self.dx + 1.0 / self.dy)
            if self.nu > 0.0:
                # FTCS centred-advection condition C^2 <= 2r, i.e.
                # dt <= 2*nu/umax^2 -- independent of grid spacing;
                # without it, fast jets run away no matter how far
                # dt shrinks in response.
                denom += (umax * umax) / (2.0 * self.nu)
        if denom <= 0.0:
            return self.dt_base
        return 0.9 / denom

    cpdef DTYPE_f stable_dt(self):
        """Public read of the current stability limit on dt."""
        return self._stable_dt_limit()

    cdef DTYPE_f _choose_dt(self, DTYPE_f limit):
        """Largest power-of-two scaling of the nominal dt that is
        still <= the stability limit: halve while too big, double
        while the doubled value would still be stable."""
        cdef DTYPE_f chosen = self.dt_base
        if chosen <= 0.0:
            return limit
        while chosen > limit:
            chosen *= 0.5
        while chosen * 2.0 <= limit:
            chosen *= 2.0
        return chosen

    cdef _step_once(self):
        self.pressure_poisson()
        self.update_momentum()

        # update_momentum() has already changed self.uvk to the
        # newly calculated destination buffer.
        self.clamp_momentum_boundary()
        self.enforce_solids()

    cpdef step(self):
        """Advance by one dynamically-chosen timestep.

        dt is not fixed: each step takes the largest power-of-two
        scaling of the nominal dt that respects the explicit
        advection-diffusion stability limit for the current flow.
        Use run(total_time) to advance a set amount of simulated
        time instead of a set number of steps.
        """
        cdef DTYPE_f chosen = self._choose_dt(self._stable_dt_limit())
        self.dt = chosen
        self._step_once()
        self.last_dt = chosen
        self.sum_dt += chosen
        self.dt = self.dt_base

    cpdef int run(self, DTYPE_f total_time):
        """Advance total_time seconds of simulated time.

        Every step dynamically takes the largest stable dt (a
        power-of-two scaling of the nominal dt, recomputed from
        the current maximum speed); the final step is trimmed so
        the accumulated time lands exactly on total_time.
        Returns the number of steps taken.
        """
        cdef DTYPE_f elapsed = 0.0
        cdef DTYPE_f remaining
        cdef DTYPE_f chosen
        cdef int steps = 0
        if total_time <= 0.0:
            raise ValueError("total_time must be > 0")
        while elapsed < total_time - 1e-15:
            remaining = total_time - elapsed
            chosen = self._choose_dt(self._stable_dt_limit())
            if chosen > remaining:
                chosen = remaining
            self.dt = chosen
            self._step_once()
            self.last_dt = chosen
            self.sum_dt += chosen
            elapsed += chosen
            steps += 1
        self.dt = self.dt_base
        return steps

    cpdef set_inflow(self, DTYPE_f velocity):
        """Set the uniform inflow speed (throughflow mode only)."""
        self.inflow_u = velocity

    def add_obstacle(self, double[::1] xs, double[::1] ys):
        """Rasterize a polygon into the solid mask (ehd-flow style).

        Vertices are physical coordinates (y up) on the [0, lx] x
        [0, ly] domain; a grid node is solid when it lies inside the
        polygon (ray-casting test).  Calls accumulate.  Velocities
        inside newly solid cells are zeroed immediately.
        """
        cdef Py_ssize_t n = xs.shape[0]
        cdef Py_ssize_t i, j, k, idx
        cdef int crossings
        cdef double px, py, x1, y1, x2, y2, xinters

        if n != ys.shape[0] or n < 3:
            raise ValueError("need >= 3 vertices")

        for j in range(self.ny):
            py = j * self.dy
            for i in range(self.nx):
                if self.solid[j, i]:
                    continue
                px = i * self.dx
                crossings = 0
                for k in range(n):
                    x1 = xs[k]
                    y1 = ys[k]
                    x2 = xs[(k + 1) % n]
                    y2 = ys[(k + 1) % n]
                    if (y1 > py) != (y2 > py):
                        xinters = (x2 - x1) * (py - y1) / (y2 - y1) + x1
                        if px < xinters:
                            crossings += 1
                if crossings & 1:
                    self.solid[j, i] = 1

        # kill any pre-existing flow inside the new solid
        for j in range(self.ny):
            idx = j * self.nx
            for i in range(self.nx):
                if self.solid[j, i]:
                    self.u[0][idx + i] = 0.0
                    self.u[1][idx + i] = 0.0
                    self.v[0][idx + i] = 0.0
                    self.v[1][idx + i] = 0.0
                    self.p[0][idx + i] = 0.0
                    self.p[1][idx + i] = 0.0


cdef class ForceHandlerCore:
    cdef readonly SolverCore solver
    def __init__(self, SolverCore solver):
        self.solver = solver

    cdef DTYPE_f get_u(
        self,
        Py_ssize_t y,
        Py_ssize_t x
    ) noexcept:
        return 0.0

    cdef DTYPE_f get_v(
        self,
        Py_ssize_t y,
        Py_ssize_t x
    ) noexcept:
        return 0.0

    cdef DTYPE_f get_p(
        self,
        Py_ssize_t y,
        Py_ssize_t x
    ) noexcept:
        return 0.0


cdef class ConstantVelocityForceHandlerCore(ForceHandlerCore):
    cdef Direction direction
    cdef DTYPE_f velocity

    def __init__(self, SolverCore solver, Direction direction, DTYPE_f velocity):
        ForceHandlerCore.__init__(self, solver)
        self.velocity = velocity
        self.direction = direction

    cdef DTYPE_f get_u(self, Py_ssize_t y, Py_ssize_t x) noexcept:
        if self.direction == Direction.RIGHT:
            if x == 0:
                return self.velocity
        elif self.direction == Direction.LEFT:
            if x == self.solver.nx - 1:
                return self.velocity
        return 0.0

    cdef DTYPE_f get_v(self, Py_ssize_t y, Py_ssize_t x) noexcept:
        if self.direction == Direction.DOWN:
            if y == 0:
                return self.velocity
        elif self.direction == Direction.UP:
            if y == self.solver.ny - 1:
                return self.velocity
        return 0.0


cdef HydroelectricForceHandlerCore(ForceHandlerCore):

    # Coulomb field values
    cdef DTYPE_f** cu
    cdef DTYPE_f** cv
    cdef Py_ssize_t ck

    # Density of ion particles
    cdef DTYPE_f** d
    cdef Py_ssize_t dk

    def __init__(self, SolverCore solver):
        self.super.__init__(solver)

    # TODO: look through solver's objects and identify positive/negative
    # then populate Coulumbs field force vector
    def init_field():
        pass
