# Declarations for the force-handler classes (implementation
# in forcehandler.pyx).  Declared here so FlowSolverCore
# (core/flow.pxd) and the other handler modules can cimport
# them; see flow.pxd for the note on circularity.
#
# perf/native-c-threadpool: ff_handler_accumulate /
# ff_handler_accumulate_p are `cdef api` functions (C API
# header forcehandler_api.h) so the native-C solver can
# call the handler chain per cell, nogil, without a
# void* -> Python-object cast (which would need the GIL
# for the refcount). The handler pointer is borrowed; the
# solver holds the owning reference.

from fast_flow.core.common cimport DTYPE_f
from fast_flow.core.flow cimport FlowSolverCore


cdef class ForceHandlerCore:
    cdef FlowSolverCore solver
    cdef ForceHandlerCore next_handler

    cdef add_next(self, ForceHandlerCore handler)
    cdef ForceHandlerCore next(self)
    cdef DTYPE_f get_u(self, Py_ssize_t y, Py_ssize_t x,
                       Py_ssize_t idx) noexcept nogil
    cdef DTYPE_f get_v(self, Py_ssize_t y, Py_ssize_t x,
                       Py_ssize_t idx) noexcept nogil
    cdef DTYPE_f get_p(self, Py_ssize_t y, Py_ssize_t x,
                       Py_ssize_t idx) noexcept nogil
    cdef void accumulate(self, Py_ssize_t y, Py_ssize_t x,
                         Py_ssize_t idx, DTYPE_f* u_dest,
                         DTYPE_f* v_dest) noexcept nogil
    cdef void prepare_pressure(self) noexcept nogil
    cdef void finish_pressure(self) noexcept nogil
    cdef void accumulate_p(self, Py_ssize_t y, Py_ssize_t x,
                           Py_ssize_t idx,
                           DTYPE_f* p_dest) noexcept nogil


cdef class ConstantVelocityForceHandlerCore(ForceHandlerCore):
    cdef Py_ssize_t direction
    cdef DTYPE_f velocity

    cdef DTYPE_f get_u(self, Py_ssize_t y, Py_ssize_t x,
                       Py_ssize_t idx) noexcept nogil
    cdef DTYPE_f get_v(self, Py_ssize_t y, Py_ssize_t x,
                       Py_ssize_t idx) noexcept nogil
