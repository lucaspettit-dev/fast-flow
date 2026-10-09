# Declarations for FlowSolverCore (implementation in flow.pyx).
# The attribute layout lives here so the handler modules can
# cimport the class; note the deliberate circularity with
# handlers/forcehandler.pxd (FlowSolverCore holds a
# ForceHandlerCore, handlers hold a FlowSolverCore).

cimport numpy as cnp

from fast_flow.core.common cimport DTYPE_f
from fast_flow.core.handlers.forcehandler cimport ForceHandlerCore


cdef class FlowSolverCore:
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
    cdef unsigned char* solid

    cdef ForceHandlerCore force

    cpdef void add_force_handler(self, ForceHandlerCore next)
    cdef init_arrays(self)
    cdef inline cnp.ndarray to_numpy(self, DTYPE_f* a)
    cdef inline Py_ssize_t idx(self, Py_ssize_t y,
                               Py_ssize_t x) noexcept nogil
    cdef void build_up_pressure_step(self, Py_ssize_t y,
                                     Py_ssize_t x, DTYPE_f dy2,
                                     DTYPE_f dx2,
                                     DTYPE_f inv_dt) noexcept nogil
    cdef void pressure_poisson_step(self, Py_ssize_t y,
                                    Py_ssize_t x, DTYPE_f* p,
                                    DTYPE_f* p_, DTYPE_f dy_squared,
                                    DTYPE_f dx_squared) noexcept nogil
    cdef void pressure_set_boundry_conditions(self) noexcept nogil
    cdef void pressure_poisson(self) noexcept
    cdef void update_momentum(self) noexcept
    cdef void apply_uv_force(self, Py_ssize_t y, Py_ssize_t x,
                             Py_ssize_t idx, DTYPE_f* u_dest,
                             DTYPE_f* v_dest) noexcept nogil
    cdef void apply_boundary_forces(self) noexcept
    cdef void clamp_momentum_boundary(self) noexcept
    cdef void enforce_solids(self) noexcept
    cdef DTYPE_f _max_speed(self) noexcept
    cdef DTYPE_f _stable_dt_limit(self) noexcept
    cpdef DTYPE_f stable_dt(self) noexcept
    cdef DTYPE_f _choose_dt(self, DTYPE_f limit) noexcept
    cdef void _step_once(self) noexcept
    cpdef void step(self) noexcept
    cpdef int run(self, DTYPE_f total_time) noexcept
    cpdef set_inflow(self, DTYPE_f velocity)
