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
from libc.stdlib cimport malloc, free
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
            int c = 1,
            double rho = 1.0,
            double nu = 0.1,
            double dt=0.02):

        if nx < 4 or ny < 4:
            raise ValueError("nx and ny must be >= 4")

        self.nx = nx
        self.ny = ny
        self.N = nx * ny
        self.c = c
        self.rho = rho
        self.nu = nu
        self.dt = dt

        self.dx = 2.0 / (nx - 1)
        self.dy = 2.0 / (ny - 1)

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

    cdef to_numpy(self, DTYPE_f* a):
        cdef DTYPE_f[:] view = <DTYPE_f[:self.N]> a
        cdef cnp.ndarray arr = np.asarray(view)
        arr.flags.writeable = False
        return arr

    cdef idx(self, y, x):
        return y * self.nx + x

    cdef build_up_pressure_step(self, y, x, dy2, dx2, inv_dt):
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
