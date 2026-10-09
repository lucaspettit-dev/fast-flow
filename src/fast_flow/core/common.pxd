# Shared C types and helpers for the fast_flow.core modules.

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_f


cdef inline void build_solids_mask(object src,
                                   unsigned char* dest) except *:
    """Build a 2D solids bitmask from a 2D or 3D image array.

    `src` is a numpy array of unsigned char, either 2D (ny, nx)
    or 3D (ny, nx, k); `dest` is a caller-owned unsigned char*
    with room for ny*nx values, filled row-major with 1 where
    a cell is solid and 0 elsewhere.  A cell is solid when its
    value -- for 3D input, the maximum across layers -- is
    greater than the threshold 127.

    Raises ValueError if `dest` is NULL, if src is not 2D/3D,
    or if either spatial dimension is below 10.  (src is an
    object parameter rather than a fixed-ndim buffer because
    Cython buffer types pin the dimension count; the typed
    views below enforce unsigned char for both arities.)
    """
    cdef Py_ssize_t ny, nx, k, y, x, c
    cdef unsigned char m
    cdef cnp.ndarray[unsigned char, ndim=2] src2
    cdef cnp.ndarray[unsigned char, ndim=3] src3

    if dest == NULL:
        raise ValueError("build_solids_mask: dest is a NULL pointer")
    if src.ndim == 2:
        src2 = src
        ny = src2.shape[0]
        nx = src2.shape[1]
        if ny < 10 or nx < 10:
            raise ValueError(
                f"build_solids_mask: src must be at least 10x10 "
                f"(got ny={ny}, nx={nx})")
        for y in range(ny):
            for x in range(nx):
                dest[y * nx + x] = 1 if src2[y, x] > 127 else 0
    elif src.ndim == 3:
        src3 = src
        ny = src3.shape[0]
        nx = src3.shape[1]
        k = src3.shape[2]
        if ny < 10 or nx < 10:
            raise ValueError(
                f"build_solids_mask: src must be at least 10x10 "
                f"(got ny={ny}, nx={nx})")
        for y in range(ny):
            for x in range(nx):
                m = 0
                for c in range(k):
                    if src3[y, x, c] > m:
                        m = src3[y, x, c]
                dest[y * nx + x] = 1 if m > 127 else 0
    else:
        raise ValueError(
            f"build_solids_mask: src must be 2D or 3D "
            f"(got {src.ndim}D)")
