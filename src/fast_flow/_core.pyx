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
from libc.math cimport sqrt, atan2, fabs, floor

ctypedef cnp.float64_t DTYPE_f


cdef class SolverCore:
    """Fast 2D incompressible Navier-Stokes solver (Stam 1999, MAC grid).

    Per step: implicit diffusion of each velocity component, pressure
    projection to a divergence-free field, semi-Lagrangian advection of
    velocity, projection again, then per dye source inject/diffuse/advect.

    Boundary conditions: uniform inflow (u=vx0) on the left, zero-gradient
    outflow on the right, no-slip walls top/bottom, no-penetration on
    obstacle polygons.
    """

    cdef readonly int nx, ny
    cdef readonly double rho, nu, dt, sum_dt
    cdef int nit, c

    cdef object _solid
    cdef DTYPE_f[:, ::1] u0, v0, u1, v1, p0, p1, b
    cdef DTYPE_f* u, u_, v, v_, p, p_
    cdef unsigned char[:, :] solid

    def __init__(
            self,
            int nx=256,
            int ny=128,
            int c = 1,
            double rho = 1.0,
            double nu = 0.1,
            double dt=0.02):

        if nx < 4 or ny < 4:
            raise ValueError("nx and ny must be >= 4")

        self.nx = nx
        self.ny = ny
        self.c = c
        self.rho = rho
        self.nu = nu
        self.dt = dt

        self.dx = 2.0 / (nx - 1)
        self.dy = 2.0 / (ny - 1)
        self.u0 = view.array(
            shape=(ny, nx),
            itemsize=sizeof(DTYPE_f),
            format="d",
            mode="c",
            allocate_buffer=True
        )
        self.u1 = view.array(
            shape=(ny, nx),
            itemsize=sizeof(DTYPE_f),
            format="d",
            mode="c",
            allocate_buffer=True
        )
        self.v0 = view.array(
            shape=(ny, nx),
            itemsize=sizeof(DTYPE_f),
            format="d",
            mode="c",
            allocate_buffer=True
        )
        self.v1 = view.array(
            shape=(ny, nx),
            itemsize=sizeof(DTYPE_f),
            format="d",
            mode="c",
            allocate_buffer=True
        )
        self.p0 = view.array(
            shape=(ny, nx),
            itemsize=sizeof(DTYPE_f),
            format="d",
            mode="c",
            allocate_buffer=True
        )
        self.p1 = view.array(
            shape=(ny, nx),
            itemsize=sizeof(DTYPE_f),
            format="d",
            mode="c",
            allocate_buffer=True
        )
        self.d = view.array(
            shape=(ny, nx),
            itemsize=sizeof(DTYPE_f),
            format="d",
            mode="c",
            allocate_buffer=True
        )

        self.init_arrays()

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    cdef init_arrays(self):
        cdef Py_ssize_t ny = self.ny
        cdef Py_ssize_t nx = self.nx

        # Populate your data here
        cdef Py_ssize_t x, y
        for y in range(ny):
            for x in range(nx):
                self.u0[y, x] = 0.0
                self.u1[y, x] = 0.0
                self.v0[y, x] = 0.0
                self.v1[y, x] = 0.0
                self.p0[y, x] = 0.0
                self.p1[y, x] = 0.0
                self.d[y, x] = 0.0

        # 3. Convert to NumPy when returning (NumPy will now own the C pointer)
        # This prevents memory leaks so you don't manually have to call free()
        # return np.asarray(arr_view)

    cdef idx(self, y, x):
        return y * self.nx + x


#    def add_obstacle(self, cnp.ndarray[double, ndim=1] xs,
#                     cnp.ndarray[double, ndim=1] ys):
#        """Rasterize a polygon (physical coords, y up) into the solid mask."""
#        cdef int n = xs.shape[0]
#        if n != ys.shape[0] or n < 3:
#            raise ValueError("need >= 3 vertices")
#        cdef int i, j, k, crossings
#        cdef double px, py, x1, y1, x2, y2, xinters
#        for j in range(self.ny):
#            py = (j + 0.5) * self.hy
#            for i in range(self.nx):
#                px = (i + 0.5) * self.hx
#                crossings = 0
#                for k in range(n):
#                    x1 = xs[k]; y1 = ys[k]
#                    x2 = xs[(k + 1) % n]; y2 = ys[(k + 1) % n]
#                    if (y1 > py) != (y2 > py):
#                        xinters = (x2 - x1) * (py - y1) / (y2 - y1) + x1
#                        if px < xinters:
#                            crossings += 1
#                if crossings & 1:
#                    self.solid[j, i] = 1
#        self._enforce_bcs()
#        self._cache.clear()
