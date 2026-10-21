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
run() opens the window; render_frame() shows each due
frame inline in the solver's thread -- no background
thread, no GIL handoff per frame.  save() writes the
last rendered frame to a file; close() destroys the
window.

The Renderer class tracks the max magnitude per frame using
the `fast_hypot` approximation. Each frame needs to be
scaled between 0-255. For the live run, each frame's max
magnitude scale will be unique. When the renderer is
constructed with record=True, every drawn frame is saved
(a copy of the buffer plus the scale it was drawn with);
save_video() then unwinds each frame's scale and re-scales
all frames to the run's global max, so the video's
brightness is consistent across frames (a dim frame stays
dim instead of being normalized up to full brightness).
save_video()'s fps must be <= the renderer's fps (slower
or equal -- slow-mo); faster is rejected.
"""

import numpy as np
import cv2
cimport numpy as cnp
from libc.math cimport floor, sqrt
from libc.stdlib cimport malloc, free
from fast_flow.core.common cimport DTYPE_f, vector_to_rgb, to_numpy_3d, fast_hypot


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

    def __init__(self, str winname, Py_ssize_t fps, Py_ssize_t ny, Py_ssize_t nx,
                 DTYPE_f scale_max=0.0, bint record=False, bint legend=False):
        self.winname = winname
        self.fps = fps if fps > 0 else 30
        self.ny = ny
        self.nx = nx
        self.N = ny * nx * 3
        self.window = None
        self.current_frame_id = 0
        self.shown_frame_id = -1
        self.max_magnitude = 0
        self.scale_max = scale_max
        self.record = record
        self._recorded = [] if record else None
        self.total_time = 0.0
        self.next_frame_time = 0.0
        self.frame_interval = 1.0 / self.fps
        self._running = 0
        self.img = None
        # make buffer
        self.buffer = <unsigned char*>malloc(self.N * sizeof(unsigned char))
        if self.buffer is NULL:
            raise MemoryError("could not allocate ion state")
        cdef Py_ssize_t i = 0
        for i in range(self.N):
            self.buffer[i] = 0
        # legend wheel (precomputed once, overlaid per frame)
        self.legend = False
        self.legend_buf = NULL
        self.legend_mask = NULL
        self.legend_size = 0
        if legend:
            self._build_legend()

    def __dealloc__(self):
       if self.buffer is not NULL:
           free(self.buffer)
       if self.legend_buf is not NULL:
           free(self.legend_buf)
       if self.legend_mask is not NULL:
           free(self.legend_mask)

    cdef void _build_legend(self) except *:
        """Precompute the legend color wheel (once, at init).

        The wheel is a disc of diameter D = ny // 4 (no
        more than 1/4 of the image height), clipped so
        the footprint (D + 7) still fits with 5px padding
        on all sides; if it cannot fit, the legend stays
        off.  Each wheel pixel's color comes from the
        same vector_to_rgb() as the flow pixels: the
        vector from the wheel center, normalized so the
        edge has magnitude 1.0 (angle = direction, radius
        = relative magnitude, center = black), with dy
        negated to match draw_frame's -v.  A 3px
        white ring borders the wheel.  Outside the ring
        the mask is 0 (leave the flow pixel alone), so
        the overlay writes only the wheel + border."""
        cdef Py_ssize_t D, S
        cdef double R, c
        cdef Py_ssize_t lx, ly, k
        cdef double dx, dy, dist
        cdef unsigned char* buf
        cdef unsigned char* mask
        D = self.ny // 4
        if D > self.ny - 17:
            D = self.ny - 17
        if D > self.nx - 17:
            D = self.nx - 17
        if D < 1:
            return
        S = D + 7
        buf = <unsigned char*>malloc(S * S * 3 * sizeof(unsigned char))
        mask = <unsigned char*>malloc(S * S * sizeof(unsigned char))
        if buf is NULL or mask is NULL:
            if buf is not NULL:
                free(buf)
            if mask is not NULL:
                free(mask)
            raise MemoryError("could not allocate legend wheel")
        R = D / 2.0
        c = (S - 1) / 2.0
        for ly in range(S):
            for lx in range(S):
                dx = lx - c
                dy = ly - c
                dist = sqrt(dx * dx + dy * dy)
                k = ly * S + lx
                if dist <= R:
                    # Negate dy to match draw_frame's -v (and
                    # colorwheel's -y_c): the flow's "up" must
                    # read the same hue as the wheel's "up".
                    vector_to_rgb(dx / R, -dy / R, 1.0, buf + k * 3)
                    mask[k] = 1
                elif dist <= R + 3.0:
                    buf[k * 3] = 255
                    buf[k * 3 + 1] = 255
                    buf[k * 3 + 2] = 255
                    mask[k] = 1
                else:
                    mask[k] = 0
        self.legend = True
        self.legend_buf = buf
        self.legend_mask = mask
        self.legend_size = S

    cdef void _overlay_legend(self) noexcept:
        """Copy the precomputed legend wheel into the
        buffer's bottom-left corner (5px padding from the
        left and bottom edges).  Only wheel + border
        pixels are written; the rest keep the flow."""
        cdef Py_ssize_t S
        cdef Py_ssize_t lx, ly, src, dst
        if not self.legend:
            return
        S = self.legend_size
        for ly in range(S):
            for lx in range(S):
                if self.legend_mask[ly * S + lx]:
                    src = (ly * S + lx) * 3
                    dst = ((5 + ly) * self.nx + (5 + lx)) * 3
                    self.buffer[dst] = self.legend_buf[src]
                    self.buffer[dst + 1] = self.legend_buf[src + 1]
                    self.buffer[dst + 2] = self.legend_buf[src + 2]

    cdef inline Py_ssize_t idx(self, Py_ssize_t x, Py_ssize_t y, Py_ssize_t z) noexcept nogil:
        return (y * self.nx + x) * 3 + z

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
            if self._running == 1:
                # No display thread: show the frame inline in
                # the solver's thread -- no GIL juggling, no
                # second thread competing with the solver.
                # Display is best-effort: if it fails, close
                # the window and keep solving.
                try:
                    if cv2.getWindowProperty(
                            self.winname,
                            cv2.WND_PROP_VISIBLE) < 1:
                        self.close()
                    else:
                        self.img = to_numpy_3d(
                            self.buffer, self.ny, self.nx, 3, True)
                        self.shown_frame_id = self.current_frame_id
                        cv2.imshow(self.winname, cv2.cvtColor(cv2.flip(self.img, 0),cv2.COLOR_BGR2RGB))
                        if cv2.waitKey(1) == 27:
                            self.close()
                except Exception:
                    self.close()

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

        # Fixed scale (if set) so a faster run renders
        # brighter instead of pixel-identical; otherwise
        # the frame's own max (old behaviour).
        cdef DTYPE_f scale = self.scale_max if self.scale_max > 0 \
            else frame_max_mag
        for i in range(n):
            vector_to_rgb(u[i], -v[i], scale, self.buffer + i * 3)
        # Legend overlay (no-op when legend is off): runs
        # before the record copy, so recorded frames and
        # save_video() include the wheel.
        self._overlay_legend()
        if self.record:
            # Save a copy of this frame plus the scale it
            # was drawn with, for save_video() to unwind and
            # re-scale. to_numpy_3d copies, so the recorded
            # frame is safe from the next draw overwriting
            # the buffer. Recording must never break the
            # render, so a failure here is swallowed.
            try:
                self._recorded.append(
                    (to_numpy_3d(self.buffer, self.ny, self.nx, 3, True),
                     scale))
            except Exception:
                pass

    cdef void _publish(self):
        """Stamp the frame just drawn into self.buffer as
        the current frame: it gets a new guid, so the
        display knows to show it."""
        self.current_frame_id += 1

    cpdef void refresh(self, const DTYPE_f[:, :] u, const DTYPE_f[:, :] v):
        """Render a frame if one is due at the current
        total_time, without advancing the clock (calls
        render_frame(0, ...)).  Right after construction a
        frame is always due (the first boundary is t=0),
        so this draws the initial state.  Takes the (u, v)
        fields as (ny, nx) arrays, since render_frame now
        needs the field data to draw."""
        self.render_frame(0.0, u.shape[0] * u.shape[1], &u[0, 0], &v[0, 0])

    cpdef void run(self):
        if self._running == 1:
            return
        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
        self.window = self.winname
        self._running = 1

    cpdef void save(self, str filepath):
        """Write the last rendered frame to filepath with
        cv2.imwrite.  Converts the raw buffer to the image
        here if the display has not already done so, so
        save() also works without a running window
        (headless runs, demos)."""
        if self.current_frame_id == 0:
            raise RuntimeError(
                "no frame has been rendered yet (buffer is empty)")
        if self.img is None or \
                self.current_frame_id != self.shown_frame_id:
            self.img = to_numpy_3d(self.buffer, self.ny, self.nx, 3, True)
        if not cv2.imwrite(filepath, cv2.cvtColor(cv2.flip(self.img, 0),cv2.COLOR_BGR2RGB)):
            raise IOError(f"cv2.imwrite failed for {filepath!r}")

    cpdef void save_video(self, str filepath, Py_ssize_t fps):
        """Write the recorded frames to a video at `fps`.

        Only available when the renderer was constructed with
        record=True. `fps` must be <= the renderer's fps
        (slower or equal -- slow-mo); faster is rejected
        because there are no frames to fill the gaps.

        Every recorded frame was drawn scaled by its own
        scale (scale_max if set, else that frame's own max).
        Here each frame is unwound and re-scaled to the
        run's global max, so brightness is consistent across
        frames: a frame whose own max was small is dimmed
        back down instead of staying normalized to full
        brightness. The unwind is exact for the renderer's
        fourth-root gamma: brightness = (mag/scale)^0.25,
        so re-scaling to the global max multiplies the RGB
        channels by (scale/global_max)^0.25 (RGB scales
        linearly with brightness at fixed hue/saturation,
        and the factor is <= 1, so nothing clips). The
        frames get the same flip + BGR->RGB transform as
        save(), so the video matches the saved PNGs."""
        if not self.record:
            raise RuntimeError(
                "renderer was not constructed with record=True; "
                "no frames were saved for post-processing")
        if not self._recorded:
            raise RuntimeError("no frames recorded yet")
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if fps > self.fps:
            raise ValueError(
                f"save fps {fps} exceeds rendered fps {self.fps}; "
                "save fps must be <= rendered fps (slower or equal)")
        cdef DTYPE_f global_max = 0.0
        for _, scale in self._recorded:
            if scale > global_max:
                global_max = scale
        writer = cv2.VideoWriter(
            filepath, cv2.VideoWriter_fourcc(*'mp4v'),
            float(fps), (self.nx, self.ny))
        if not writer.isOpened():
            raise IOError(
                f"could not open video writer for {filepath!r} "
                "(check the extension/codec)")
        try:
            for frame, scale in self._recorded:
                if global_max > 0.0:
                    factor = (scale / global_max) ** 0.25
                else:
                    factor = 1.0
                out = np.clip(
                    frame.astype(np.float32) * factor,
                    0.0, 255.0).astype(np.uint8)
                out = cv2.cvtColor(cv2.flip(out, 0), cv2.COLOR_BGR2RGB)
                writer.write(out)
        finally:
            writer.release()

    cpdef void close(self):
        """Destroy the window. Safe to call when not running."""
        self._running = False
        if self.window is not None:
            try:
                cv2.destroyWindow(self.winname)
            except cv2.error:
                pass
            self.window = None

    cpdef cnp.ndarray colorwheel(self):
        cdef Py_ssize_t ny = 56
        cdef Py_ssize_t nx = 56
        cdef Py_ssize_t nz = 4
        cdef Py_ssize_t c = 28
        cdef DTYPE_f max_mag = 25.0
        cdef DTYPE_f mag = 0.0
        cdef Py_ssize_t n = ny * nx * nz
        cdef unsigned char* a = <unsigned char*>malloc(n * sizeof(unsigned char))
        if a is NULL:
            raise MemoryError("could not allocate ion state")

        cdef Py_ssize_t x, y, i
        cdef DTYPE_f x_c, y_c
        cdef cnp.ndarray img
        try:
            for y in range(ny):
                y_c = <DTYPE_f>(y - c)
                for x in range(nx):
                    i = (y * nx + x) * nz
                    x_c = <DTYPE_f>(x - c)
                    mag = sqrt(x_c * x_c + y_c * y_c)
                    if mag > 28:
                        a[i] = 0
                        a[i+1] = 0
                        a[i+2] = 0
                        a[i+3] = 0
                    if mag > max_mag:
                        a[i] = 255
                        a[i+1] = 255
                        a[i+2] = 255
                        a[i+3] = 255
                    else:
                        vector_to_rgb(x_c, -y_c, max_mag, a + (i * sizeof(unsigned char)))
                        a[i+3] = 255
            img = to_numpy_3d(a, ny, nx, nz, True)
        finally:
            free(a)
        return img
