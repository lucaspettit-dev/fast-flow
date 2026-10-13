# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, atan2, fabs, floor


cdef class Renderer:
    cdef unsigned char** buffer
    cdef Py_ssize_t ny
    cdef Py_ssize_t nx
    cdef readonly Py_ssize_t nz = 3

    def __init__(self):
        pass


