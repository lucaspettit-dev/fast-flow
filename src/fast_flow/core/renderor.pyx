# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Renderer: abstract base for live cv2-window renderers.

A Renderer owns a cv2 window and displays the frames a
subclass draws.  The subclass implements render_frame():
draw the new frame into self.buffer (a uint8 ndarray of
shape (ny, nx, 3)) and call self._publish(), which stamps
the frame with a fresh guid.  run() opens the window and
starts a background display loop: the loop is nogil and
only takes the GIL to hand cv2 a frame whose guid differs
from the one last shown, so producing frames never waits
on the display.  save() writes the last rendered frame to
a file; close() stops the loop and destroys the window.
"""

import threading

import cv2
import numpy as np
cimport numpy as cnp


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
        self._running = False
        self._thread = None

    cdef void render_frame(self):
        """Draw a new frame into self.buffer and publish
        it (call self._publish() last, which gives the
        frame a fresh guid).

        Abstract: the base class draws nothing.  Subclasses
        override this to render their state; the display
        loop in run() picks the frame up by its guid."""
        pass

    cdef void _publish(self):
        """Stamp the frame just drawn into self.buffer as
        the current frame: it gets a new guid, so the
        display loop knows to show it."""
        self.current_guid += 1

    cpdef void refresh(self):
        """Produce a new frame now (calls render_frame()).
        The display loop shows it on its next pass."""
        self.render_frame()

    cpdef void run(self):
        """Open the cv2 window named by the constructor's
        winname and start the asynchronous display loop on
        a background thread.  No-op if already running."""
        if self._running:
            return
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
        filepath with cv2.imwrite."""
        if self.buffer is None:
            raise RuntimeError(
                "no frame has been rendered yet (buffer is None)")
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
                cv2.destroyWindow(self.winname)
            except cv2.error:
                pass
            self.window = None
