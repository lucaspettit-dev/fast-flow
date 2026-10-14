# Declarations for the renderer classes (implementation
# in renderor.pyx).  Declared here so subclasses in other
# modules can cimport Renderer and override render_frame;
# see flow.pxd for the same pattern.

cimport numpy as cnp


cdef class Renderer:
    cdef str winname
    cdef Py_ssize_t fps
    cdef object window
    # The last frame rendered: a cnp.ndarray[unsigned char,
    # ndim=3] (buffer types cannot be declared on class
    # attributes, so the dtype/shape are by contract).
    cdef cnp.ndarray buffer
    cdef Py_ssize_t current_guid
    cdef Py_ssize_t shown_guid
    cdef double total_time
    cdef double next_frame_time
    cdef double frame_interval
    cdef bint _running
    cdef object _thread

    cdef void render_frame(self, double dt) noexcept
    cdef void draw_frame(self) noexcept
    cdef void _publish(self)
    cdef void _publish(self)
    cdef void _loop(self) noexcept nogil
    cpdef void refresh(self)
    cpdef void run(self)
    cpdef void save(self, str filepath)
    cpdef void close(self)
