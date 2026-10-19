# Declarations for the renderer classes (implementation
# in renderer.pyx).  Declared here so subclasses in other
# modules can cimport Renderer and override draw_frame;
# see flow.pxd for the same pattern.

cimport numpy as cnp
from fast_flow.core.common cimport DTYPE_f


cdef class Renderer:
    cdef str winname
    cdef Py_ssize_t fps
    cdef object window
    # The last frame rendered: a cnp.ndarray[unsigned char,
    # ndim=3] (buffer types cannot be declared on class
    # attributes, so the dtype/shape are by contract).
    cdef cnp.ndarray img
    cdef unsigned char* buffer
    cdef Py_ssize_t ny
    cdef Py_ssize_t nx
    cdef Py_ssize_t N
    cdef Py_ssize_t current_frame_id
    cdef Py_ssize_t shown_frame_id
    cdef DTYPE_f total_time
    cdef DTYPE_f next_frame_time
    cdef DTYPE_f frame_interval
    cdef DTYPE_f max_magnitude
    # Fixed brightness scale: if > 0, draw_frame scales by
    # this (so runs at different speeds are directly
    # comparable); if 0, each frame scales by its own max
    # (the old behaviour, which makes a 2x-faster flow
    # render pixel-identical).
    cdef DTYPE_f scale_max
    cdef bint _running
    # Record flag: when True, render_frame() saves a copy of
    # every drawn frame plus the scale it was drawn with,
    # for post-processing by save_video(). Off by default
    # (recording costs one ny*nx*3 copy per frame).
    cdef bint record
    # Recorded frames (only when record=True): a list of
    # (frame ndarray (ny, nx, 3) uint8 copy, scale) tuples,
    # where scale is the max_magnitude the frame was drawn
    # with (scale_max if > 0, else that frame's own max).
    cdef object _recorded

    cdef void render_frame(self, DTYPE_f dt, Py_ssize_t n, DTYPE_f* u, DTYPE_f* v) noexcept
    cdef void draw_frame(self, Py_ssize_t n, DTYPE_f* u, DTYPE_f* v) noexcept
    cdef void _publish(self)
    cdef Py_ssize_t idx(self, Py_ssize_t x, Py_ssize_t y, Py_ssize_t z) noexcept nogil
    cpdef void refresh(self, const DTYPE_f[:, :] u, const DTYPE_f[:, :] v)
    cpdef void run(self)
    cpdef void save(self, str filepath)
    cpdef void save_video(self, str filepath, Py_ssize_t fps)
    cpdef void close(self)
    cpdef cnp.ndarray colorwheel(self)
