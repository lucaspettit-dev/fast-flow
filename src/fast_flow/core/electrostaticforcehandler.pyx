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

cdef const DTYPE_f ION_MASS_GMOL = 28.97
cdef const DTYPE_f AVOGADRO = 6.02214076e23
cdef const DTYPE_f ELEMENTARY_CHARGE = 1.602176634e-19
cdef const DTYPE_f ION_MASS_KG = 4.81058168e-26
cdef const DTYPE_f ION_Q_OVER_M_SI = 3330525.78789183

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
                 cnp.ndarray[unsigned char, ndim=3] image,
                 dict layermap=None):
        """Bind to `solver` and build edge structs from an image.

        `image` is required: a numpy array of unsigned char (uint8)
        with shape (ny, nx, nz), whose y (ny) and x (nx)
        dimensions must both be at least 10.  It is flattened into an unsigned char
        buffer (channel-major planes, one byte per pixel marking
        solid or not), and each channel is processed in turn using
        build_solids_mask which assumes 0-255 value range.
        `layermap` maps layer index -> charge, e.g. {0: 20000, 2: -20000}
        for an RGB image whose red shapes are positive and blue shapes negative;
        layers missing from the map are neutral and ignored.

        Each channel is assumed to be an independent power source, so a charge in one
        channel is distributed evenly across the surface of all solids in that channel.
        Multiple positive/negative power sources can be added in separate channels.
        Every cell in a channel, that IS solid but has a NON solid horizontal or
        vertical neighbour is saved as an edge. The result is at most zn Shape structs,
        each holding its edge cells' physical coordinates (x = col mapped
        onto [0, lx], y = row flipped onto [0, ly], y up) in
        `edges` and the group's charge in `extradata.charge` (if a group spans layers with
        different charges, the lowest layer's charge is kept).

        The handler also keeps a solids bitmask of its own --
        an unsigned char* on the image grid, the same element
        type as FlowSolverCore's mask -- generated from the
        image by common.build_solids_mask: a cell is solid
        when the max across layers exceeds 127 (on the byte
        scale the image is converted to internally).
        """
        ForceHandlerCore.__init__(self, solver)

        # build solid mask
        self.solids = <unsigned char*> malloc(image.shape[0] * image.shape[1] * sizeof(unsigned char))
        if self.solids == NULL:
            raise MemoryError("could not allocate solids bitmask")
        self.solids_ny = image.shape[0]
        self.solids_nx = image.shape[1]
        build_solids_mask(image, self.solids)

        self.shapes = NULL
        self.n_shapes = 0
        self.shapes_cap = 0
        self._build_edges(image, layermap)

    cdef Py_ssize_t _new_shape_slot(self, Py_ssize_t n, DTYPE_f charge):
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

    cdef _build_edges(self, cnp.ndarray[unsigned char, ndim=3] image, object layermap):
        """Flatten the image and extract per-sign-group edges
        (see __init__ for the semantics)."""
        cdef Py_ssize_t nx = image.shape[1]
        cdef Py_ssize_t ny = image.shape[0]
        cdef Py_ssize_t nz = image.shape[2]
        cdef Py_ssize_t x
        cdef Py_ssize_t y
        cdef Py_ssize_t z
        cdef Py_ssize_t i
        cdef unsigned char** channels
        cdef unsigned char* channel
        cdef DTYPE_f charge
        cdef Py_ssize_t count
        cdef Py_ssize_t slot
        cdef Py_ssize_t e
        cdef list layers
        cdef DTYPE_f sx = self.solver.lx / (nx - 1) if nx > 1 else 0.0
        cdef DTYPE_f sy = self.solver.ly / (ny - 1) if ny > 1 else 0.0

        # get each channels bitmask of solids
        channels = <unsigned char**> malloc(nz * sizeof(unsigned char*))
        if channels == NULL:
            raise MemoryError("could not allocate flattened image")
        for z in range(nz):
            channels[z] = <unsigned char*> malloc(ny * nx * sizeof(unsigned char))
            if channels[z] == NULL:
                for i in range(z):
                    free(channels[i])
                free(channels)
                raise MemoryError("could not allocate channel mask image")
            build_solids_mask(image[:, :, z], channels[z])

        try:
            if layermap is None:
                return

            # iterate over each layer, if there's a non-zero charge
            # then add the electrod edges
            for layer in sorted(layermap.keys()):
                if layer >= image.ndim:
                    raise ValueError(
                        f"layermap keys must reference a real channel in "
                        f"the source image (image.ndim={image.ndim} but got key={layer})")
                charge = float(layermap[layer])
                if charge == 0:
                    continue

                # count, then fill: non-solid cells with a solid
                # horizontal or vertical neighbour
                channel = channels[layer]
                count = 0
                for y in range(ny):
                    i = y * nx
                    for x in range(nx):
                        # we want to take the solid layer that's touching air
                        # we skip if channel[i] == 0 because that's air
                        if channel[i] == 0:
                            i += 1
                            continue
                        if (x > 0 and channel[i-1] == 0) or \
                                (x + 1 < nx and channel[i+1] == 0) or \
                                (y > 0 and channel[i-nx] == 0) or \
                                (y + 1 < ny and channel[i+nx] == 0):
                            count += 1
                        i += 1

                if count == 0:
                    continue

                slot = self._new_shape_slot(count, charge)
                e = 0
                for y in range(ny):
                    i = y * nx
                    for x in range(nx):
                        # we want to take the solid layer that's touching air
                        # we skip if channel[i] == 0 because that's air
                        if channel[i] == 0:
                            i += 1
                            continue
                        if (x > 0 and channel[i-1] == 0) or \
                                (x + 1 < nx and channel[i+1] == 0) or \
                                (y > 0 and channel[i-nx] == 0) or \
                                (y + 1 < ny and channel[i+nx] == 0):
                            self.shapes[slot].edges[2 * e] = x * sx
                            self.shapes[slot].edges[2 * e + 1] = \
                                self.solver.ly - y * sy
                            e += 1
                        i += 1
        finally:
            for z in range(nz):
                free(channels[z])
            free(channels)

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

cdef class CoulombField:
    cdef DTYPE_f cell_size_cm
    cdef Py_ssize_t ny
    cdef Py_ssize_t nx

    # represented as array [y0, x0, y1, x1, ..., yn, xn]
    cdef Py_ssize_t* positive_electrodes
    cdef Py_ssize_t* negative_electrodes
    cdef DTYPE_f positive_charge_per_electrode
    cdef DTYPE_f negative_charge_per_electrode

    def __init__(
            self,
            Py_ssize_t ny,
            Py_ssize_t nx,
            unsigned char[:, :] positive,
            unsigned char[:, :] negative,
            # cell_size_cm is the width/height distance each pixel/grid cell
            # represents in the real-world. Measured in centimeters (cm)
            DTYPE_f cell_size_cm):
        self.ny = ny
        self.nx = nx
        self.cell_size_cm = cell_size_cm

    def set_electrodes(self, unsigned char[:, :] mask):
        cdef Py_ssize_t y = 0
        cdef Py_ssize_t x = 0
        cdef Py_ssize_t idx = 0
        coordinates = []
        for y in range(self.ny):
            idx = y * self.ny
            for x in range(self.nx):
                if mask[y, x] != 0:
                    if x != 0 and mask[y, x-1] == 0:
                        coordinates.append(y)
                        coordinates.append(x)
                # todo finish
