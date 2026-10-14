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

cv2 is imported lazily, only when run() or save() is
called -- never at module import.  The class declaration
lives in renderor.pxd (no cv2 there either), and the
solver cimports only that declaration, so importing the
solver does not require cv2; cv2 is needed only when a
window is actually opened or a frame saved.
"""

import threading

import numpy as np
cimport numpy as cnp
from libc.math cimport floor


cdef class Renderer:
    # The window is identified by its name in cv2; `window`
    # holds that name once run() has created the window,
    # and is None before and after close().

    def __init__(self, str winname, Py_ssize_t fps):
        self.winname = winname
        self.fps = fps if fps > 0 else 30
        self.window = None
        self.current_guid = 0
        self.shown_guid = 0
        self.total_time = 0.0
        self.next_frame_time = 0.0
        self.frame_interval = 1.0 / self.fps
        self._cv2 = None
        self._running = False
        self._thread = None

    cdef void render_frame(self, double dt) noexcept:
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
            self.draw_frame()
            self._publish()

    cdef void draw_frame(self) noexcept:
        """Draw the current frame into self.buffer.

        Abstract: the base class draws nothing.  Subclasses
        override this to render their state; render_frame()
        calls it (and publishes the frame) when a frame is
        due."""
        pass

    cdef void _publish(self):
        """Stamp the frame just drawn into self.buffer as
        the current frame: it gets a new guid, so the
        display loop knows to show it."""
        self.current_guid += 1

    cpdef void refresh(self):
        """Render a frame if one is due at the current
        total_time, without advancing the clock (calls
        render_frame(0)).  Right after construction a frame
        is always due (the first boundary is t=0), so this
        draws the initial state."""
        self.render_frame(0.0)

    cpdef void run(self):
        """Open the cv2 window named by the constructor's
        winname and start the asynchronous display loop on
        a background thread.  No-op if already running.
        Imports cv2 here (lazily), on first use."""
        if self._running:
            return
        import cv2
        self._cv2 = cv2
        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
        self.window = self.winname
        self._running = True
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
        while self._running:
            with gil:
                # cv2 was imported by run(); bind it locally
                # for the calls and the error handler below.
                cv2 = self._cv2
                try:
                    if cv2.getWindowProperty(
                            self.winname,
                            cv2.WND_PROP_VISIBLE) < 1:
                        self._running = False
                        break
                    if self.buffer is not None and \
                            self.current_guid != self.shown_guid:
                        cv2.imshow(self.winname, self.buffer)
                        self.shown_guid = self.current_guid
                    if cv2.waitKey(delay_ms) == 27:
                        self._running = False
                except cv2.error:
                    self._running = False

    cpdef void save(self, str filepath):
        """Write the last rendered frame (self.buffer) to
        filepath with cv2.imwrite.  Imports cv2 here
        (lazily), on first use."""
        if self.buffer is None:
            raise RuntimeError(
                "no frame has been rendered yet (buffer is None)")
        import cv2
        if not cv2.imwrite(filepath, self.buffer):
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
