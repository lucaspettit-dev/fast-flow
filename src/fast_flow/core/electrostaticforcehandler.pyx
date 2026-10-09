# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""ElectrostaticForceHandler: charged-image forcing.

Owns the Shape structs (edge cells of the positive/negative
electrode groups in an image, plus each group's charge in
extradata) and will populate the Coulomb field from them.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, realloc

from fast_flow.core.common cimport DTYPE_f, build_solids_mask
from fast_flow.core.flow cimport FlowSolverCore
from fast_flow.core.handlers.forcehandler cimport ForceHandlerCore


cdef struct ShapeExtraData:
    # Metadata carried alongside a shape's geometry.  For now the
    # only entry is `charge`: the electric potential (voltage) on
    # the shape.
    DTYPE_f charge


cdef struct Shape:
    # The edge cells of one sign group's electrodes, owned by
    # ElectrostaticForceHandler.  `edges` is a flat buffer of
    # interleaved physical coordinates (x0, y0, x1, y1, ...,
    # y up, on the solver's [0, lx] x [0, ly] domain) of the
    # non-solid cells that border a solid cell horizontally or
    # vertically (diagonals do not count); `n_edges` is the
    # point count (buffer length 2*n).  This is separate from
    # FlowSolverCore's solid bitmask: obstacles still rasterize
    # into the mask, which is what voids pressure/velocity
    # inside solids.
    DTYPE_f* edges
    Py_ssize_t n_edges
    ShapeExtraData extradata




cdef class ElectrostaticForceHandler(ForceHandlerCore):

    # Coulomb field values
    cdef DTYPE_f** cu
    cdef DTYPE_f** cv
    cdef Py_ssize_t ck

    # Density of ion particles
    cdef DTYPE_f** d
    cdef Py_ssize_t dk

    # One Shape struct per sign group present in the image: the
    # positive group's struct first, then the negative group's.
    # Each holds the edge cells of that group's solids.
    cdef Shape* shapes
    cdef Py_ssize_t n_shapes
    cdef Py_ssize_t shapes_cap

    # Solids bitmask on the image grid, as a flat unsigned
    # char* (idx(y, x) layout, 1 = solid, 0 = not) -- the same
    # element type and layout as FlowSolverCore's mask.  It is
    # generated from the image at construction by
    # build_solids_mask (max across layers > 127) and freed in
    # __dealloc__; solids_nx/solids_ny are its dimensions.
    cdef unsigned char* solids
    cdef Py_ssize_t solids_ny
    cdef Py_ssize_t solids_nx

    def __init__(self, FlowSolverCore solver,
                 cnp.ndarray[DTYPE_f, ndim=3] image,
                 dict layermap=None):
        """Bind to `solver` and build edge structs from an image.

        `image` is required: a numpy array of DTYPE_f (float64)
        with shape (ny, nx, k), whose y (ny) and x (nx)
        dimensions must both be at least 10.  It is flattened into an unsigned char
        buffer (layer-major planes, one byte per pixel marking
        solid or not), and each layer is processed in turn: a
        pixel above half the array's maximum value is a solid
        in that layer (so both 0.0/1.0 and 0.0/255.0 data
        binarize the same way).  `layermap` maps layer index ->
        charge, e.g. {0: 20000, 2: -20000} for an RGB image whose
        red shapes are positive and blue shapes negative;
        layers missing from the map are neutral and ignored.

        Per sign group (all layers whose charge is positive,
        and all whose charge is negative), the solids are
        unioned and every cell that is NOT solid but has a
        solid horizontal or vertical neighbour -- diagonals do
        not count -- is saved as an edge.  The result is at
        most two Shape structs, positive first, each holding
        its edge cells' physical coordinates (x = col mapped
        onto [0, lx], y = row flipped onto [0, ly], y up) in
        `edges` and the group's charge in
        `extradata.charge` (if a group spans layers with
        different charges, the lowest layer's charge is kept).

        The handler also keeps a solids bitmask of its own --
        an unsigned char* on the image grid, the same element
        type as FlowSolverCore's mask -- generated from the
        image by common.build_solids_mask: a cell is solid
        when the max across layers exceeds 127 (on the byte
        scale the image is converted to internally).
        """
        ForceHandlerCore.__init__(self, solver)
        self.solids = NULL
        self.solids_ny = 0
        self.solids_nx = 0
        self.shapes = NULL
        self.n_shapes = 0
        self.shapes_cap = 0
        self._build_edges(image, layermap)

    cdef Py_ssize_t _new_shape_slot(self, Py_ssize_t n,
                                    DTYPE_f charge):
        """Grow the shape array if full and allocate the next
        slot's edges buffer for n points; returns the new
        shape's index.  The caller fills the buffer."""
        cdef Shape* grown
        cdef Py_ssize_t new_cap
        cdef Py_ssize_t slot
        if self.n_shapes == self.shapes_cap:
            new_cap = 8 if self.shapes_cap == 0 else 2 * self.shapes_cap
            grown = <Shape*> realloc(self.shapes,
                                     new_cap * sizeof(Shape))
            if grown == NULL:
                raise MemoryError("could not grow shape array")
            self.shapes = grown
            self.shapes_cap = new_cap
        slot = self.n_shapes
        self.shapes[slot].edges = \
            <DTYPE_f*> malloc(2 * n * sizeof(DTYPE_f))
        if self.shapes[slot].edges == NULL:
            raise MemoryError("could not allocate shape edges")
        self.shapes[slot].n_edges = n
        self.shapes[slot].extradata.charge = charge
        self.n_shapes += 1
        return slot

    cdef _build_edges(self, cnp.ndarray[DTYPE_f, ndim=3] image,
                      object layermap):
        """Flatten the image and extract per-sign-group edges
        (see __init__ for the semantics)."""
        cdef Py_ssize_t ny, nx, k, c, y, x, i
        cdef unsigned char* flat
        cdef unsigned char* group
        cdef DTYPE_f amax, thresh
        cdef DTYPE_f charge, group_charge
        cdef Py_ssize_t count, slot, e
        cdef list layers
        cdef DTYPE_f sx, sy

        ny = image.shape[0]
        nx = image.shape[1]
        k = image.shape[2]

        if ny < 10 or nx < 10:
            raise ValueError(
                f"image y and x dimensions must both be >= 10 "
                f"(got ny={ny}, nx={nx})")

        amax = np.max(image) if image.size else 0.0

        # the handler's solids bitmask, generated from the
        # image on its byte scale (max across layers > 127)
        if amax > 0.0:
            img_u8 = np.rint(255.0 * image / amax).astype(np.uint8)
        else:
            img_u8 = np.zeros((ny, nx, k), dtype=np.uint8)
        self.solids = <unsigned char*> malloc(
            ny * nx * sizeof(unsigned char))
        if self.solids == NULL:
            raise MemoryError("could not allocate solids bitmask")
        self.solids_ny = ny
        self.solids_nx = nx
        build_solids_mask(img_u8, self.solids)

        if amax <= 0.0:
            return  # no positive pixels anywhere -> no edges
        thresh = 0.5 * amax

        # flatten into an unsigned char buffer, layer-major,
        # binarizing as we go:
        # flat[(c * ny + y) * nx + x] = 1 if image[y, x, c] is solid
        flat = <unsigned char*> malloc(ny * nx * k * sizeof(unsigned char))
        if flat == NULL:
            raise MemoryError("could not allocate flattened image")
        for c in range(k):
            for y in range(ny):
                for x in range(nx):
                    flat[(c * ny + y) * nx + x] = \
                        1 if image[y, x, c] > thresh else 0

        group = <unsigned char*> malloc(ny * nx * sizeof(unsigned char))
        if group == NULL:
            free(flat)
            raise MemoryError("could not allocate group mask")

        sx = self.solver.lx / (nx - 1) if nx > 1 else 0.0
        sy = self.solver.ly / (ny - 1) if ny > 1 else 0.0

        try:
            if layermap is None:
                return
            for sign in (1, -1):
                layers = []
                group_charge = 0.0
                for layer in sorted(layermap.keys()):
                    charge = float(layermap[layer])
                    if (sign > 0 and charge > 0.0) or \
                            (sign < 0 and charge < 0.0):
                        if not layers:
                            group_charge = charge
                        layers.append(int(layer))
                if not layers:
                    continue

                # union this group's solids
                for i in range(ny * nx):
                    group[i] = 0
                for c in layers:
                    if c < 0 or c >= k:
                        raise ValueError(
                            f"layermap layer {c} out of range for "
                            f"image with {k} layer(s)")
                    for y in range(ny):
                        for x in range(nx):
                            if flat[(c * ny + y) * nx + x]:
                                group[y * nx + x] = 1

                # count, then fill: non-solid cells with a solid
                # horizontal or vertical neighbour
                count = 0
                for y in range(ny):
                    for x in range(nx):
                        if group[y * nx + x]:
                            continue
                        if (x > 0 and group[y * nx + x - 1]) or \
                                (x + 1 < nx and group[y * nx + x + 1]) or \
                                (y > 0 and group[(y - 1) * nx + x]) or \
                                (y + 1 < ny and group[(y + 1) * nx + x]):
                            count += 1
                if count == 0:
                    continue
                slot = self._new_shape_slot(count, group_charge)
                e = 0
                for y in range(ny):
                    for x in range(nx):
                        if group[y * nx + x]:
                            continue
                        if (x > 0 and group[y * nx + x - 1]) or \
                                (x + 1 < nx and group[y * nx + x + 1]) or \
                                (y > 0 and group[(y - 1) * nx + x]) or \
                                (y + 1 < ny and group[(y + 1) * nx + x]):
                            self.shapes[slot].edges[2 * e] = x * sx
                            self.shapes[slot].edges[2 * e + 1] = \
                                self.solver.ly - y * sy
                            e += 1
        finally:
            free(flat)
            free(group)

    def __dealloc__(self):
        if self.solids != NULL:
            free(self.solids)
            self.solids = NULL
        if self.shapes != NULL:
            for i in range(self.n_shapes):
                if self.shapes[i].edges != NULL:
                    free(self.shapes[i].edges)
            free(self.shapes)
            self.shapes = NULL

    @property
    def solids_mask(self) -> np.ndarray:
        """Read-only (ny, nx) uint8 copy of the solids bitmask
        generated from the image at construction; 1 where a
        cell is solid."""
        if self.solids == NULL:
            arr = np.zeros((self.solids_ny, self.solids_nx),
                           dtype=np.uint8)
            arr.flags.writeable = False
            return arr
        return self.to_mask_numpy(self.solids)

    cdef cnp.ndarray to_mask_numpy(self, unsigned char* a):
        cdef unsigned char[:] view = \
            <unsigned char[:self.solids_ny * self.solids_nx]> a
        cdef cnp.ndarray arr = np.array(view, copy=True)
        arr = arr.reshape((self.solids_ny, self.solids_nx))
        arr.flags.writeable = False
        return arr

    cpdef Py_ssize_t num_shapes(self):
        """Number of sign-group structs (0, 1, or 2)."""
        return self.n_shapes

    cpdef DTYPE_f shape_charge(self, Py_ssize_t i):
        """The `charge` (voltage) stored in struct i's extradata."""
        if i < 0 or i >= self.n_shapes:
            raise IndexError("shape index out of range")
        return self.shapes[i].extradata.charge

    cpdef list shape_edges(self, Py_ssize_t i):
        """Struct i's edge cells as a list of (x, y) physical
        coordinates, read back from the struct's interleaved
        DTYPE_f* buffer."""
        cdef Py_ssize_t k
        if i < 0 or i >= self.n_shapes:
            raise IndexError("shape index out of range")
        return [(self.shapes[i].edges[2 * k],
                 self.shapes[i].edges[2 * k + 1])
                for k in range(self.shapes[i].n_edges)]

    # TODO: populate the Coulomb field from the positive and
    # negative edge structs
    cdef void init_field(self):
        pass
