# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""ForceHandlerCore: abstract parent of the force handlers,
plus the general-purpose handlers (constant velocity).

Handlers are chained (each holds the next) and contribute
per-cell u/v increments through get_u/get_v; the chain is
walked nogil via accumulate().  Implementations of concrete
handlers live in their own modules (see
core/electrostaticforcehandler.pyx).
"""

from fast_flow.core.common cimport DTYPE_f
from fast_flow.core.flow cimport FlowSolverCore


cdef class ForceHandlerCore:

    def __init__(self, FlowSolverCore solver):
        self.solver = solver
        self.next_handler = None

    cdef add_next(self, ForceHandlerCore handler):
        if self.next_handler is not None:
            raise ValueError("Next handler already assigned")
        self.next_handler = handler

    cdef ForceHandlerCore next(self):
        return self.next_handler

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
    ) noexcept:
        return 0.0

    cdef void accumulate(
        self,
        Py_ssize_t y,
        Py_ssize_t x,
        Py_ssize_t idx,
        DTYPE_f* u_dest,
        DTYPE_f* v_dest
    ) noexcept nogil:
        """Add this handler's contribution at one cell, then the
        next handler's.  The chain is walked by recursion through
        the stored attribute rather than a local variable:
        assigning a handler (a Python object) to a local would
        need the GIL, and apply_uv_force runs nogil inside
        update_momentum's parallel loop."""
        u_dest[idx] += self.get_u(y, x, idx)
        v_dest[idx] += self.get_v(y, x, idx)
        if self.next_handler is not None:
            self.next_handler.accumulate(y, x, idx, u_dest, v_dest)


cdef class ConstantVelocityForceHandlerCore(ForceHandlerCore):

    def __init__(self, FlowSolverCore solver, Py_ssize_t direction, DTYPE_f velocity):
        ForceHandlerCore.__init__(self, solver)
        self.velocity = velocity
        self.direction = direction

    cdef DTYPE_f get_u(self, Py_ssize_t y, Py_ssize_t x, Py_ssize_t idx) noexcept nogil:
        if self.direction == 0:
            # LEFT: flow moves in -x, entering at the right edge
            if x == self.solver.nx - 1:
                return -self.velocity
        elif self.direction == 1:
            # RIGHT: flow moves in +x, entering at the left edge
            if x == 0:
                return self.velocity
        return 0.0

    cdef DTYPE_f get_v(self, Py_ssize_t y, Py_ssize_t x, Py_ssize_t idx) noexcept nogil:
        if self.direction == 2:
            # UP: flow moves in +y, entering at the bottom edge
            if y == 0:
                return self.velocity
        elif self.direction == 3:
            # DOWN: flow moves in -y, entering at the top edge
            if y == self.solver.ny - 1:
                return -self.velocity
        return 0.0


