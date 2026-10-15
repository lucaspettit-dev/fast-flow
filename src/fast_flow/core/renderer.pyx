# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Renderer: abstract base for live cv2-window renderers.

A Renderer owns a cv2 window and displays the frames a
subclass draws.  The solver calls render_frame(dt) after
each step; render_frame accumulates the run's total_time
and, only when that time crosses the next frame boundary
(spaced 1/fps apart), calls the subclass's draw_frame()
hook to draw the new frame into self.buffer (a uint8
ndarray of shape (ny, nx, 3)) and publishes it with a
fresh guid -- calls between boundaries are no-ops.
run() opens the window and starts a background display
loop: the loop is nogil and only takes the GIL to hand cv2
a frame whose guid differs from the one last shown, so
producing frames never waits on the display.  save()
writes the last rendered frame to a file; close() stops
the loop and destroys the window.

The Renderer class tracks the max magnitude per frame using
the `fast_hypot` approximation. Each frame needs to be
scaled between 0-255. For the live run, each frame's max
magnitude scale will be unique. But in later iterations, we'll
keep a buffer of all rendered frames, then re-calculate the
scale according to the overall max magnitude of the run, then
save the video. This'll let the user see the properly scaled
render after the run.
"""

import threading

import numpy as np
import cv2
from time import sleep
cimport numpy as cnp
from libc.math cimport floor
from libc.stdlib cimport malloc, free
from fast_flow.core.common cimport vector_to_rgb, to_numpy_3d, fast_hypot

"""
Since I cannot do enums, this is the render-style mapping:

MAGNITUDE = 0 // works for p,b, and (u,v)
RAINBOW = 1 // directional color wheel
HEATMAP = 2 // cool blue to hot red for higher values
"""

cdef class Renderer:
    # The window is identified by its name in cv2; `window`
    # holds that name once run() has created the window,
    # and is None before and after close().

    def __init__(self, str winname, Py_ssize_t fps, Py_ssize_t ny, Py_ssize_t nx):
        self.winname = winname
        self.fps = fps if fps > 0 else 30
        self.ny = ny
        self.nx = nx
        self.N = ny * nx * 3
        self.window = None
        self.current_frame_id = 0
        self.shown_frame_id = -1
        self.max_magnitude = 0
        self.total_time = 0.0
        self.next_frame_time = 0.0
        self.frame_interval = 1.0 / self.fps
        self._running = 0
        self._thread = None
        self.img = None
        # make buffer
        self.buffer = <unsigned char*>malloc(self.N * sizeof(unsigned char*))
        if self.buffer is NULL:
            raise MemoryError("could not allocate ion state")
        cdef Py_ssize_t i = 0
        for i in range(self.N):
            self.buffer[i] = 0

    def __dealloc__(self):
       if self.buffer is not NULL:
           free(self.buffer)

    cdef inline Py_ssize_t idx(self, Py_ssize_t x, Py_ssize_t y, Py_ssize_t z):
        return (y * nx + x) * 3 + z

    cdef void render_frame(self, DTYPE_f dt, Py_ssize_t n, DTYPE_f* u, DTYPE_f* v) noexcept:
        """Advance the run clock by dt and render a frame
        if one is due.

        Called after each solver step with that step's dt.
        total_time accumulates every dt; when total_time
        reaches the next frame boundary (boundaries are
        frame_interval = 1/fps apart, the first at t=0) the
        frame is drawn (draw_frame()) and published with a
        fresh guid, and the boundary advances past
        total_time.  Otherwise the call is a no-op, so the
        solver can call it every step regardless of how
        small dt is compared to the frame interval."""

        self.total_time += dt
        if self.total_time >= self.next_frame_time:
            self.next_frame_time += (
                floor((self.total_time - self.next_frame_time)
                      / self.frame_interval) + 1.0) \
                * self.frame_interval
            self.draw_frame(n, u, v)
            self._publish()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void draw_frame(self, Py_ssize_t n, DTYPE_f* u, DTYPE_f* v) noexcept:
        """Draw the current frame into self.buffer.

        Abstract: the base class draws nothing.  Subclasses
        override this to render their state; render_frame()
        calls it (and publishes the frame) when a frame is
        due."""
        cdef DTYPE_f frame_max_mag = 0
        cdef DTYPE_f m = 0
        cdef Py_ssize_t i = 0
        for i in range(n):
            m = fast_hypot(u[i], v[i])
            if m > frame_max_mag:
                frame_max_mag = m

        if frame_max_mag > self.max_magnitude:
            self.max_magnitude = frame_max_mag

        for i in range(n):
            vector_to_rgb(u[i], v[i], frame_max_mag, self.buffer[i * 3])

    cdef void _publish(self):
        """Stamp the frame just drawn into self.buffer as
        the current frame: it gets a new guid, so the
        display loop knows to show it."""
        self.current_frame_id += 1

    cpdef void refresh(self):
        """Render a frame if one is due at the current
        total_time, without advancing the clock (calls
        render_frame(0)).  Right after construction a frame
        is always due (the first boundary is t=0), so this
        draws the initial state."""
        self.render_frame(0.0)

    cpdef void run(self):
        if self._running == 1:
            return
        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
        self.window = self.winname
        self._running = 0
        self._thread = threading.Thread(
            target=self._thread_main, daemon=True)
        self._thread.start()

    def _thread_main(self):
        self._loop()

    cdef void _loop(self) noexcept nogil:
        """Display loop: whenever the current frame's guid
        differs from the last shown frame's, update the
        window image with self.buffer and mark the new
        guid as shown.  Runs nogil; the GIL is taken only
        for the cv2 calls themselves.  cv2.waitKey paces
        the loop to fps and pumps the window's events;
        the loop ends when close() clears _running, when
        ESC is pressed, or when the window is closed."""
        cdef Py_ssize_t delay_ms = 1000 // self.fps
        if delay_ms < 1:
            delay_ms = 1

        while self._running == 1:
            # todo: requesting GIL each loop is expensive.
            # investigate if we can have the data-processing no-gil and have update_frame request GIL
            # after rendering a new frame so it can update the img
            with gil:
                try:
                    if cv2.getWindowProperty(
                            self.winname,
                            cv2.WND_PROP_VISIBLE) < 1:
                        self._running = 0
                        break
                    if cv2.waitKey(delay_ms) == 27:
                        self._running = 0
                    if self.current_frame_id != self.shown_frame_id:
                        self.img = to_numpy_3d(self.buffer, self.nx, self.ny, 3, True)
                        self.shown_frame_id = self.current_frame_id
                        cv2.imshow(self.winname, self.img)
                except cv2.error:
                    self._running = False


    cpdef void save(self, str filepath):
        if self.img is None:
            raise RuntimeError(
                "no frame has been rendered yet (buffer is None)")
        if not cv2.imwrite(filepath, self.img):
            raise IOError(f"cv2.imwrite failed for {filepath!r}")

    cpdef void close(self):
        """Stop the display loop and destroy the window.
        Safe to call when not running."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.window is not None:
            try:
                self._cv2.destroyWindow(self.winname)
            except Exception:
                pass
            self.window = None
