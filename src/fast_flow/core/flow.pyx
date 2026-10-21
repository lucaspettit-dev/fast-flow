# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""FlowSolverCore: the compiled 2D Navier-Stokes solver core.

Split out of the old single-file _core.pyx: this module holds
only the solver; force handlers live in core/handlers/.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, atan2, fabs, floor

from fast_flow.core.common cimport DTYPE_f, build_solids_mask, to_numpy
from fast_flow.core.handlers.forcehandler cimport ForceHandlerCore

# Native-C solver (c/solver.c). The C struct is opaque here
# (void* c_solver on the class); these are its constructors /
# accessors. The C code references the Cython-owned buffers
# (same memory the Python accessors view) and owns the
# thread pool.
cdef extern from "c/solver.h":
    ctypedef void solver_c_t
    solver_c_t* solver_c_create(long nx, long ny,
                                double dx, double dy,
                                double lx, double ly,
                                double rho, double nu,
                                double dt_base, long nit, long flow_mode,
                                double inflow_u,
                                double* u0, double* u1,
                                double* v0, double* v1,
                                double* p0, double* p1,
                                double* b, unsigned char* solid) noexcept nogil
    void solver_c_destroy(solver_c_t* s) noexcept nogil
    void solver_c_set_inflow(solver_c_t* s, double inflow_u) noexcept nogil
    void solver_c_set_dt(solver_c_t* s, double dt) noexcept nogil
    void solver_c_set_uvk(solver_c_t* s, long uvk) noexcept nogil
    void solver_c_set_pk(solver_c_t* s, long pk) noexcept nogil
    # void solver_c_set_no_boundaries(solver_c_t* s, int no_boundaries) noexcept nogil
    long solver_c_pk(const solver_c_t* s) noexcept nogil
    long solver_c_uvk(const solver_c_t* s) noexcept nogil
    # Sub-steps (the wrapper orchestrates them with the
    # handler passes interleaved, all nogil).
    void solver_c_poisson(solver_c_t* s) noexcept nogil
    void solver_c_momentum(solver_c_t* s) noexcept nogil
    void solver_c_clamp(solver_c_t* s) noexcept nogil
    void solver_c_enforce(solver_c_t* s) noexcept nogil


cdef class FlowSolverCore:
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


    def __init__(
            self,
            object img,
            int nit = 50,
            DTYPE_f rho = 1.0,
            DTYPE_f nu = 0.1,
            DTYPE_f dt=0.001,
            int flow_mode=0,
            DTYPE_f inflow_u=1.0,
            DTYPE_f lx=2.0,
            DTYPE_f ly=2.0,
            Py_ssize_t n_force_handlers):

        self.nx = img.shape[1]
        self.ny = img.shape[0]
        self.N = self.nx * self.ny
        self.solid = <unsigned char*> malloc(self.N * sizeof(unsigned char*))
        build_solids_mask(img, self.solid)
        self._solid = to_numpy(self.solid, self.ny, self.nx, False)

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

        self.dx = lx / (self.nx - 1)
        self.dy = ly / (self.ny - 1)

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
        self.init_arrays()

        self.nforces = n_force_handlers
        self.forces = <ForceHandlerCore*> malloc(self.nforces * sizeof(ForceHandlerCore*))

        # Native-C solver: references the buffers above (same
        # memory the Python accessors view); owns the thread
        # pool. Created after the buffers (it stores their
        # addresses, it does not copy them).
        self.c_solver = <void*>solver_c_create(
            self.nx, self.ny, self.dx, self.dy,
            self.lx, self.ly, self.rho, self.nu,
            self.dt_base, self.nit, self.flow_mode,
            self.inflow_u,
            self.u[0], self.u[1],
            self.v[0], self.v[1],
            self.p[0], self.p[1],
            self.b, self.solid)
        if self.c_solver == NULL:
            raise MemoryError("could not allocate native-C solver")

    def __dealloc__(self):
        if self.c_solver != NULL:
            solver_c_destroy(<solver_c_t*>self.c_solver)
            self.c_solver = NULL

        if self.solid != NULL:
            free(self.solid)

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

    cpdef void set_renderer(self, Renderer renderer):
        """Attach a Renderer (or None to detach).  After
        every step, _step_once calls the renderer's
        render_frame(self.dt); the renderer itself decides
        whether a frame is due at its fps."""
        self.renderer = renderer

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
        return to_numpy(self.p[self.pk], self.ny, self.nx, True)

    @property
    def horizontal_velocity(self) -> np.ndarray:
        return to_numpy(self.u[self.uvk], self.ny, self.nx, True)

    @property
    def vertical_velocity(self) -> np.ndarray:
        return to_numpy(self.v[self.uvk], self.ny, self.nx, True)

    @property
    def solid_mask(self) -> np.ndarray:
        """Read-only (ny, nx) uint8 mask; 1 = solid obstacle cell."""
        arr = np.array(self._solid, copy=True)
        arr.flags.writeable = False
        return arr

    cdef inline Py_ssize_t idx(
            self,
            Py_ssize_t y,
            Py_ssize_t x
    ) noexcept nogil:
        return y * self.nx + x

    cdef DTYPE_f _max_speed(self) noexcept nogil:
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

    cdef DTYPE_f _stable_dt_limit(self) noexcept nogil:
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

    cpdef DTYPE_f stable_dt(self) noexcept:
        """Public read of the current stability limit on dt."""
        return self._stable_dt_limit()

    cdef DTYPE_f _choose_dt(self, DTYPE_f limit) noexcept nogil:
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

    cdef void _full_step(self) noexcept nogil:
        """
        One full step at self.dt
        (already set on the C solver by the caller):

        C sub-steps + the handler passes, all nogil.
        Order mirrors the old _step_once:
            1. Poisson
            2. momentum
            3. handler interior
            4. flip uvk
            5. clamp
            6. handler boundary
            7. handler pressure (serial, exactly-once)
            8. enforce.

        The handler passes live here in Cython (they call
        typed handler methods); the C code is pure-fluid.
        The momentum does not flip uvk -- this function
        flips it after the handler interior pass, so the
        handler still reads the source buffer.
        """

        cdef solver_c_t* cs = <solver_c_t*>self.c_solver
        cdef Py_ssize_t x, y, i, idx, dest
        cdef DTYPE_f* u_dest
        cdef DTYPE_f* v_dest
        cdef DTYPE_f* u
        cdef DTYPE_f* v
        cdef DTYPE_f* p

        solver_c_poisson(cs)
        self.pk = solver_c_pk(cs)
        solver_c_momentum(cs)

        # Handler interior pass (what update_momentum used to
        # call per cell): non-solid interior cells only.
        if self.nforces > 0:
            dest = self.uvk ^ 1
            u_dest = self.u[dest]
            v_dest = self.v[dest]
            for y in range(1, self.ny - 1):
                for x in range(1, self.nx - 1):
                    idx = y * self.nx + x
                    if not self.solid[idx]:
                        for i in range(self.nforces):
                           self.u_dest += self.forces[i].get_u(y, x, idx)
                           self.v_dest += self.forces[i].get_v(y, x, idx)

        # Flip uvk (both sides: Cython + C struct).
        self.uvk ^= 1
        solver_c_set_uvk(cs, self.uvk)

        solver_c_clamp(cs)

        # Handler boundary pass (what apply_boundary_forces
        # did): every boundary cell exactly once. The top
        # (lid) row's u is zeroed first so handlers set it
        # fresh (see the old .pyx comment).
        if self.nforces > 0:
            u = self.u[self.uvk]
            v = self.v[self.uvk]
            for x in range(self.nx):
                u[(self.ny - 1) * self.nx + x] = 0.0
            for x in range(self.nx):
                for i in range(self.nforces):
                    u[x] += self.forces[i].get_u(0, x, x)
                    u[(self.ny - 1) * self.nx + x] += self.forces[i].get_u(self.ny - 1, x, (self.ny - 1) * self.nx + x)
            for y in range(1, self.ny - 1):
                for i in range(self.nforces):
                    v[y * self.nx] += self.forces[i].get_v(y, 0, y * self.nx)
                    v[y, self.nx - 1] = self.forces[i].get_v(y, self.nx-1, y * self.nx + self.nx - 1)

        # Handler pressure pass (serial, exactly-once: a
        # handler's get_p may advance per-cell state).
        if self.nforces > 0:
            for i in range(self.nforces):
                self.forces[i].prepare_pressure()

            p = self.p[self.pk]
            for y in range(self.ny):
                for x in range(self.nx):
                    for i in range(self.nforces):
                        p[y * self.nx + x] += self.forces[i].get_p(y, x, y * self.nx + x)

            for i in range(self.nforces):
                self.forces[i].finish_pressure()

        solver_c_enforce(cs)

    cdef void _step_once(self) noexcept:
        """
            One full step at self.dt (already chosen by the
            caller). The computation runs nogil (C sub-steps +
            handler passes on the thread pool); the GIL is
            reacquired only afterwards for the renderer hook
            (which needs Python) and the read-buffer sync.
        """

        cdef solver_c_t* cs = <solver_c_t*>self.c_solver
        solver_c_set_dt(cs, self.dt)
        with nogil:
            self._full_step()

        # Read-buffer sync: the Cython-side indices (which
        # the accessors and handlers read) follow the C
        # struct's (uvk was flipped in _full_step; pk by
        # the C Poisson).
        self.pk = solver_c_pk(cs)
        self.uvk = solver_c_uvk(cs)

        # Render hook: the renderer accumulates this step's
        # dt and draws a frame only when one is due at its
        # fps, so this costs a comparison on most steps.
        if self.renderer is not None:
            self.renderer.render_frame(self.dt, self.N, self.u[self.uvk], self.v[self.uvk])

    cdef void _step_nogil(self) noexcept nogil:
        # Nogil core of step(): dt selection + the full step
        # + read-buffer sync. No renderer hook here (it needs
        # the GIL).
        cdef DTYPE_f chosen = self._choose_dt(self._stable_dt_limit())
        self.dt = chosen
        solver_c_set_dt(<solver_c_t*>self.c_solver, chosen)
        self._full_step()
        self.last_dt = chosen
        self.sum_dt += chosen
        self.dt = self.dt_base

    cpdef void step(self) noexcept:
        """Advance by one dynamically-chosen timestep.

        The computation runs entirely nogil (dt selection +
        the native-C step on the thread pool); the GIL is
        held only to enter/exit. No renderer hook here (it
        needs the GIL) -- use run(total_time) for the
        rendered path; run() calls _step_once, which does
        the hook after reacquiring the GIL. The read buffer
        (the numpy views over the C buffers) is live: the
        C step writes the buffers directly, and the GIL
        reacquire in the caller is the synchronization point.

        dt is not fixed: each step takes the largest power-of-two
        scaling of the nominal dt that respects the explicit
        advection-diffusion stability limit for the current flow.
        Use run(total_time) to advance a set amount of simulated
        time instead of a set number of steps.
        """
        with nogil:
            self._step_nogil()

    cpdef int run(self, DTYPE_f total_time) noexcept:
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
        if self.c_solver != NULL:
            solver_c_set_inflow(<solver_c_t*>self.c_solver, velocity)

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
                if self.solid[self.idx(j, i)]:
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
                    self.solid[self.idx(j, i)] = 1

        # kill any pre-existing flow inside the new solid
        for i in range(self.N):
            if self.solid[i]:
                self.u[0][i] = 0.0
                self.u[1][i] = 0.0
                self.v[0][i] = 0.0
                self.v[1][i] = 0.0
                self.p[0][i] = 0.0
                self.p[1][i] = 0.0

    cdef VectorBinding velocity_binding(self) noexcept:
        cdef VectorBinding field = VectorBinding()
        field.ny = self.ny
        field.nx = self.nx
        field.v_buffers = self.v
        field.u_buffers = self.u
        field.active = &self.uvk

    cdef ScalarBinding density_binding(self) noexcept:
        cdef ScalarBinding field = ScalarBinding()
        field.ny = self.ny
        field.nx = self.nx
        field.buffers = self.p
        field.active = &self.pk
