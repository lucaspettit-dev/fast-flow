# Shared C types and helpers for the fast_flow.core modules.

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_f

import numpy as np
cimport numpy as cnp
from libc.math cimport fmod, floor, sqrt
cimport cython

# NOTE: do not declare module-level `cdef` variables here
# (e.g. `cdef double PI = ...`).  Cython then generates a
# runtime `import fast_flow.core.common` in every module
# that cimports from this pxd to fetch them -- but common
# is pxd-only (no compiled module), so that import fails.
# Constants belong as locals inside the functions that use
# them (see TWO_PI in vector_to_rgb).

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void build_solids_mask(
        object src,
        unsigned char* dest) except *:
    cdef Py_ssize_t ny, nx, nz
    cdef Py_ssize_t y, x, z, idx
    cdef unsigned char[:, :] src2
    cdef unsigned char[:, :, :] src3
    cdef unsigned char solid
    cdef unsigned char thresh = 127

    if dest == NULL:
        raise ValueError("build_solids_mask: dest is NULL")
    if not isinstance(src, np.ndarray):
        raise TypeError("build_solids_mask: src must be a NumPy array")
    if src.dtype != np.uint8:
        raise TypeError("build_solids_mask: src must have dtype uint8")
    if src.ndim != 2 and src.ndim != 3:
        raise ValueError("build_solids_mask: src must be 2D or 3D")

    ny = src.shape[0]
    nx = src.shape[1]
    if ny < 10 or nx < 10:
        raise ValueError("build_solids_mask: src must be at least 10x10")

    if src.ndim == 2:
        src2 = src
        for y in range(ny):
            for x in range(nx):
                idx = y * nx + x
                dest[idx] = 1 if src2[y, x] > thresh else 0

    else:
        src3 = src
        nz = src.shape[2]
        for y in range(ny):
            for x in range(nx):
                idx = y * nx + x
                solid = 0
                for z in range(nz):
                    if src3[y, x, z] > thresh:
                        solid = 1
                        break
                dest[idx] = solid

ctypedef fused ArrayType:
    unsigned char
    DTYPE_f

cdef inline cnp.ndarray to_numpy(
    ArrayType* a,
    Py_ssize_t ny,
    Py_ssize_t nx,
    bint readonly
):
    cdef ArrayType[:] view = <ArrayType[:nx * ny]> a

    cdef cnp.ndarray arr = np.array(view, copy=True)
    arr = arr.reshape((ny, nx))
    arr.flags.writeable = not readonly

    return arr

cdef inline cnp.ndarray to_numpy_3d(
    ArrayType* a,
    Py_ssize_t ny,
    Py_ssize_t nx,
    Py_ssize_t nz,
    bint readonly
):
    cdef ArrayType[:] view = <ArrayType[:nx * ny * nz]> a

    cdef cnp.ndarray arr = np.array(view, copy=True)
    arr = arr.reshape((ny, nx, nz))
    arr.flags.writeable = not readonly

    return arr

cdef inline double fast_hypot(
    DTYPE_f x,
    DTYPE_f y
) noexcept nogil:
    cdef DTYPE_f ax = -x if x < 0 else x
    cdef DTYPE_f ay = -y if y < 0 else y

    if ax > ay:
        return 0.9604 * ax + 0.3978 * ay
    else:
        return 0.9604 * ay + 0.3978 * ax

from libc.math cimport fabs

cdef inline DTYPE_f fast_atan2(
    DTYPE_f y,
    DTYPE_f x
) noexcept nogil:
    cdef DTYPE_f ax = -x if x < 0 else x
    cdef DTYPE_f ay = -y if y < 0 else y
    cdef DTYPE_f z
    cdef DTYPE_f angle

    if ax == 0.0 and ay == 0.0:
        return 0.0

    if ax >= ay:
        z = ay / ax
        angle = z * (0.7853981633974483 + 0.273 * (1.0 - z))
    else:
        z = ax / ay
        angle = 1.5707963267948966 - \
                z * (0.7853981633974483 + 0.273 * (1.0 - z))

    if x < 0.0:
        angle = 3.141592653589793 - angle

    if y < 0.0:
        angle = -angle

    return angle

cdef inline double clamp01(double x) noexcept nogil:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

cdef inline void vector_to_rgb(
    DTYPE_f x,
    DTYPE_f y,
    DTYPE_f max_magnitude,
    unsigned char* o,
) noexcept nogil:
    cdef DTYPE_f hue
    cdef DTYPE_f magnitude
    cdef DTYPE_f value
    cdef DTYPE_f TWO_PI = 6.28318530717958647692

    # Direction determines hue [0, 1)
    hue = fast_atan2(y, x) / TWO_PI
    if hue < 0.0:
        hue += 1.0

    # Magnitude determines brightness [0, 1].  The
    # brightness is gamma-compressed (fourth root --
    # gamma 0.25): EHD fields span orders of magnitude
    # (median speed ~0.07% of max), and with a linear
    # scale every cell below 1/255 of max truncates to
    # pure black -- at t=0.05 only 65 of 150,000 cells
    # survived.  A square root (gamma 0.5, the old demo's
    # PowerNorm) still left 99% of the visible pixels at
    # <= 2/255, i.e. the frame still read as black; the
    # fourth root lifts the bulk recirculation into view
    # while the jet stays brightest.
    magnitude = fast_hypot(x, y)

    if max_magnitude > 0.0:
        value = clamp01(sqrt(sqrt(magnitude / max_magnitude)))
    else:
        value = 0.0

    hsv_to_rgb(
        hue,
        1.0,  # Full saturation
        value,
        o
    )

cdef inline void hsv_to_rgb(
    DTYPE_f h,
    DTYPE_f s,
    DTYPE_f v,
    unsigned char* o
) noexcept nogil:
    cdef DTYPE_f r
    cdef DTYPE_f g
    cdef DTYPE_f b
    cdef DTYPE_f i
    cdef DTYPE_f f
    cdef DTYPE_f p
    cdef DTYPE_f q
    cdef DTYPE_f t
    cdef int sector

    # h is normalized to [0, 1)
    i = floor(h * 6.0)
    sector = <int>i
    f = h * 6.0 - i

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    if sector == 0:
        r = v
        g = t
        b = p
    elif sector == 1:
        r = q
        g = v
        b = p
    elif sector == 2:
        r = p
        g = v
        b = t
    elif sector == 3:
        r = p
        g = q
        b = v
    elif sector == 4:
        r = t
        g = p
        b = v
    else:
        r = v
        g = p
        b = q

    # Matches int(c * 255) from the original Python code.
    # o is a cv2 BGR triplet (the buffer feeds
    # cv2.imshow/imwrite, which expect BGR), so the blue
    # channel goes first.
    o[0] = <unsigned char>(b * 255.0)
    o[1] = <unsigned char>(g * 255.0)
    o[2] = <unsigned char>(r * 255.0)

