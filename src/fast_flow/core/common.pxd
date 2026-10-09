# Shared C types and helpers for the fast_flow.core modules.

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_f

import numpy as np
cimport numpy as cnp
cimport cython

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
