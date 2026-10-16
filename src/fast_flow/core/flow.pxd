# Declarations for FlowSolverCore (implementation in flow.pyx).
# The attribute layout lives here so the handler modules can
# cimport the class; note the deliberate circularity with
# handlers/forcehandler.pxd (FlowSolverCore holds a
# ForceHandlerCore, handlers hold a FlowSolverCore).
#
# perf/native-c-threadpool: the solver's computation
# (Poisson, momentum, BCs, enforce) lives in native C
# (solver_c.c) behind the opaque c_solver pointer; the
# Cython side owns the buffers (same memory the Python
# accessors view) and the dt selection. The old Cython
# compute methods are gone (replaced by the C step).

cimport numpy as cnp

from fast_flow.core.common cimport DTYPE_f
from fast_flow.core.handlers.forcehandler cimport ForceHandlerCore
from fast_flow.core.renderer cimport Renderer


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
    # no_boundaries: when True, the solver does not
    # enforce domain boundaries (velocity clamp + pressure
    # BCs are skipped). Solids are still enforced.
    cdef bint no_boundaries

    cdef object _solid
    cdef DTYPE_f* b
    cdef DTYPE_f** u
    cdef DTYPE_f** v
    cdef DTYPE_f** p
    cdef Py_ssize_t pk
    cdef Py_ssize_t uvk
    cdef unsigned char* solid

    # Opaque native-C solver state (solver_c_t*). The C struct
    # references the buffers above (it does not own them);
    # it owns the thread pool.
    cdef void* c_solver

    cdef ForceHandlerCore force
    cdef Renderer renderer

    cpdef void add_force_handler(self, ForceHandlerCore next)
    cpdef void set_renderer(self, Renderer renderer)
    cdef init_arrays(self)
    cdef inline Py_ssize_t idx(self, Py_ssize_t y,
                               Py_ssize_t x) noexcept nogil
    cdef DTYPE_f _max_speed(self) noexcept nogil
    cdef DTYPE_f _stable_dt_limit(self) noexcept nogil
    cpdef DTYPE_f stable_dt(self) noexcept
    cdef DTYPE_f _choose_dt(self, DTYPE_f limit) noexcept nogil
    cdef void _step_once(self) noexcept
    cdef void _full_step(self) noexcept nogil
    cdef void _step_nogil(self) noexcept nogil
    cpdef void step(self) noexcept
    cpdef int run(self, DTYPE_f total_time) noexcept
    cpdef set_inflow(self, DTYPE_f velocity)
