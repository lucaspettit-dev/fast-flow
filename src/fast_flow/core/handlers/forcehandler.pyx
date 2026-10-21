# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""ForceHandlerCore: abstract parent of the force handlers,
plus the general-purpose handlers (constant velocity).
"""

from fast_flow.core.common cimport DTYPE_f
from fast_flow.core.flow cimport FlowSolverCore


cdef class ForceHandlerCore:

    def __init__(self, FlowSolverCore solver):
        self.solver = solver

    cdef DTYPE_f get_u(
        self,
        Py_ssize_t y,
        Py_ssize_t x,
        Py_ssize_t idx
    ) noexcept nogil:
        return 0.0

    cdef DTYPE_f get_v(
        self,
        Py_ssize_t y,
        Py_ssize_t x,
        Py_ssize_t idx
    ) noexcept nogil:
        return 0.0

    cdef DTYPE_f get_p(
        self,
        Py_ssize_t y,
        Py_ssize_t x,
        Py_ssize_t idx
    ) noexcept nogil:
        return 0.0

