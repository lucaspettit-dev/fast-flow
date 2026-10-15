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
from libc.math cimport sqrt, fabs

from fast_flow.core.common cimport DTYPE_f, build_solids_mask, to_numpy
from fast_flow.core.flow cimport FlowSolverCore
from fast_flow.core.handlers.forcehandler cimport ForceHandlerCore

# Native-C handler (native/handler_c.c) + the solver's pool
# (native/solver_c.c). The C advance references the
# Cython-owned buffers (same memory the accessors view)
# and re-uses the solver's thread pool.
cdef extern from "solver_c.h":
    ctypedef void solver_c_t
    ctypedef void ff_pool_t
    ff_pool_t* solver_c_pool(solver_c_t* s) noexcept nogil
cdef extern from "native/handler_c.h":
    ctypedef void handler_c_t
    handler_c_t* handler_c_create(long nx, long ny,
                                  double dx, double dy,
                                  double* d0, double* d1,
                                  double* p0, double* p1,
                                  double* u0, double* u1,
                                  double* v0, double* v1,
                                  double* cu0, double* cu1,
                                  double* cv0, double* cv1,
                                  unsigned char* ion_zone,
                                  unsigned char* solids,
                                  long solids_nx, long solids_ny,
                                  ff_pool_t* pool) noexcept nogil
    void handler_c_destroy(handler_c_t* h) noexcept nogil
    long handler_c_dk(const handler_c_t* h) noexcept nogil
    void handler_c_set_pk(handler_c_t* h, long pk) noexcept nogil
    void handler_c_advance(handler_c_t* h,
                           const double* au, const double* av,
                           const unsigned char* solid,
                           double dt, double dt_base) noexcept nogil

cdef const DTYPE_f ION_MASS_GMOL = 28.97
cdef const DTYPE_f AVOGADRO = 6.02214076e23
cdef const DTYPE_f ELEMENTARY_CHARGE = 1.602176634e-19
cdef const DTYPE_f ION_MASS_KG = 4.81058168e-26
cdef const DTYPE_f ION_Q_OVER_M_SI = 3330525.78789183

# Ion-step calibration, in the solver's folded units (the
# Coulomb field folds the Coulomb constant and the ion
# mobility to 1, so these constants map the folded
# electrostatics into momentum/pressure units).
cdef const DTYPE_f ION_ZONE_TARGET = 1.0
cdef const DTYPE_f ION_PRODUCTION_TAU = 0.01
cdef const DTYPE_f ION_DIFFUSION = 0.01
cdef const DTYPE_f ION_AIR_COUPLING = 0.001
cdef const DTYPE_f ION_PRESSURE_COUPLING = 1.0e-5
cdef const DTYPE_f ION_PRESSURE_MAX = 100.0
# Space-charge shielding density: as the local ion
# density approaches this value, the cloud's own charge
# cancels the applied field, so the drift (not the air
# advection) fades to zero.  Without it the static field
# drives ions into concave sinks (slots, notches) without
# limit -- real space charge is self-limiting.
cdef const DTYPE_f ION_SHIELD_DENSITY = 3.0
cdef const DTYPE_f ION_AMBIENT_PRESSURE = 1.0
cdef const DTYPE_f ION_RESISTANCE_MAX = 250.0
cdef const DTYPE_f ION_SPEED_MAX = 1000.0

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
    # tracks the force at the center of each grid-cell
    cdef DTYPE_f** cu
    cdef DTYPE_f** cv
    cdef Py_ssize_t ck

    # Ion density (p) tracks the +ion density at each grid-cell
    cdef DTYPE_f** d
    cdef Py_ssize_t dk

    # Ion's electrostatic pressure at each grid-cell
    # I'm not sure if we'll need it, might be too small to really make a difference.
    # idea here is that the +ions will repel each other, so if there's a lot in a neighbor cell
    # then they should diffuse similar to regular air pressure, but under different laws
    cdef DTYPE_f** p
    cdef Py_ssize_t pk

    # Ion velocity at each grid-cell
    # I'm not sure how to handle if there's no Ions in a cell. Maybe np.nan or 0?
    cdef DTYPE_f** u
    cdef DTYPE_f** v
    cdef Py_ssize_t uvk

    # Air cells next to a positive electrode: the
    # ionization zone, where new +ions are generated.
    cdef unsigned char* ion_zone

    # get_p calls seen so far in the solver's current
    # pressure pass; the ion state advances on the first
    # call of each pass.
    cdef Py_ssize_t p_calls

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

    # Opaque native-C handler state (handler_c_t*). The C
    # struct references the buffers above (it does not own
    # them) and re-uses the solver's thread pool.
    cdef void* c_handler

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

        # Coulomb field buffers (double-buffered like the
        # solver's fields), then the static field itself
        self.cu = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        self.cv = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        if self.cu == NULL or self.cv == NULL:
            raise MemoryError("could not allocate Coulomb field")

        cdef Py_ssize_t i, j
        for i in range(2):
            self.cu[i] = <DTYPE_f*> malloc(
                self.solver.N * sizeof(DTYPE_f))
            self.cv[i] = <DTYPE_f*> malloc(
                self.solver.N * sizeof(DTYPE_f))
            if self.cu[i] == NULL or self.cv[i] == NULL:
                raise MemoryError("could not allocate Coulomb field")
            for j in range(self.solver.N):
                self.cu[i][j] = 0
                self.cv[i][j] = 0
        self.ck = 0

        # Ion state buffers, double-buffered like the
        # solver's fields and zeroed: d = +ion density,
        # p = the ion cloud's electrostatic pressure,
        # u/v = ion velocity (drift + air, filled by the
        # first advance).  The velocity is defined in
        # every air cell -- it is the velocity an ion
        # there would have; the density weights its
        # effect, so no NaN/empty-cell case arises.
        self.d = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        self.p = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        self.u = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        self.v = <DTYPE_f**> malloc(2 * sizeof(DTYPE_f*))
        if self.d == NULL or self.p == NULL or \
                self.u == NULL or self.v == NULL:
            raise MemoryError("could not allocate ion state")
        for i in range(2):
            self.d[i] = <DTYPE_f*> malloc(
                self.solver.N * sizeof(DTYPE_f))
            self.p[i] = <DTYPE_f*> malloc(
                self.solver.N * sizeof(DTYPE_f))
            self.u[i] = <DTYPE_f*> malloc(
                self.solver.N * sizeof(DTYPE_f))
            self.v[i] = <DTYPE_f*> malloc(
                self.solver.N * sizeof(DTYPE_f))
            if self.d[i] == NULL or self.p[i] == NULL or \
                    self.u[i] == NULL or self.v[i] == NULL:
                raise MemoryError("could not allocate ion state")
            for j in range(self.solver.N):
                self.d[i][j] = 0
                self.p[i][j] = 0
                self.u[i][j] = 0
                self.v[i][j] = 0
        self.dk = 0
        self.pk = 0
        self.uvk = 0
        self.p_calls = 0
        self._build_ion_zone(image, layermap)

        self.init_field()

        # Native-C handler: references the buffers above
        # (same memory the accessors view); re-uses the
        # solver's thread pool within the solver step.
        cdef ff_pool_t* pool = NULL
        if self.solver.c_solver != NULL:
            pool = solver_c_pool(<solver_c_t*>self.solver.c_solver)
        self.c_handler = <void*>handler_c_create(
            self.solver.nx, self.solver.ny,
            self.solver.dx, self.solver.dy,
            self.d[0], self.d[1],
            self.p[0], self.p[1],
            self.u[0], self.u[1],
            self.v[0], self.v[1],
            self.cu[0], self.cu[1],
            self.cv[0], self.cv[1],
            self.ion_zone, self.solids,
            self.solids_nx, self.solids_ny,
            pool)
        if self.c_handler == NULL:
            raise MemoryError("could not allocate native-C handler")

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
                            self.shapes[slot].edges[2 * e + 1] = y * sy
                            e += 1
                        i += 1
        finally:
            for z in range(nz):
                free(channels[z])
            free(channels)

    cdef void _build_ion_zone(self,
            cnp.ndarray[unsigned char, ndim=3] image,
            object layermap):
        """Mark the ionization zone: air cells (in the
        handler's own solids mask) that border a solid of
        any positively-charged layer.  Corona discharge at
        a positive electrode ionizes the air right at its
        surface, so new +ions appear in exactly these
        cells."""
        cdef Py_ssize_t nx = image.shape[1]
        cdef Py_ssize_t ny = image.shape[0]
        cdef Py_ssize_t x, y, i
        cdef unsigned char* channel

        self.ion_zone = <unsigned char*> malloc(
            ny * nx * sizeof(unsigned char))
        if self.ion_zone == NULL:
            raise MemoryError("could not allocate ion zone")
        for i in range(ny * nx):
            self.ion_zone[i] = 0
        if layermap is None:
            return

        channel = <unsigned char*> malloc(
            ny * nx * sizeof(unsigned char))
        if channel == NULL:
            raise MemoryError("could not allocate channel mask")
        try:
            for layer in sorted(layermap.keys()):
                if layer < 0 or layer >= image.shape[2]:
                    raise ValueError(
                        "layermap keys must reference a real "
                        "channel in the source image")
                if float(layermap[layer]) <= 0.0:
                    continue
                build_solids_mask(image[:, :, layer], channel)
                for y in range(ny):
                    i = y * nx
                    for x in range(nx):
                        if not self.solids[i]:
                            if (x > 0 and channel[i - 1]) or \
                                    (x + 1 < nx and channel[i + 1]) or \
                                    (y > 0 and channel[i - nx]) or \
                                    (y + 1 < ny and channel[i + nx]):
                                self.ion_zone[i] = 1
                        i += 1
        finally:
            free(channel)

    def __dealloc__(self):
        if self.c_handler != NULL:
            handler_c_destroy(<handler_c_t*>self.c_handler)
            self.c_handler = NULL
        if self.d != NULL:
            for i in range(2):
                if self.d[i] != NULL:
                    free(self.d[i])
            free(self.d)
            self.d = NULL
        if self.p != NULL:
            for i in range(2):
                if self.p[i] != NULL:
                    free(self.p[i])
            free(self.p)
            self.p = NULL
        if self.u != NULL:
            for i in range(2):
                if self.u[i] != NULL:
                    free(self.u[i])
            free(self.u)
            self.u = NULL
        if self.v != NULL:
            for i in range(2):
                if self.v[i] != NULL:
                    free(self.v[i])
            free(self.v)
            self.v = NULL
        if self.ion_zone != NULL:
            free(self.ion_zone)
            self.ion_zone = NULL
        if self.cu != NULL:
            for i in range(2):
                if self.cu[i] != NULL:
                    free(self.cu[i])
            free(self.cu)
            self.cu = NULL
        if self.cv != NULL:
            for i in range(2):
                if self.cv[i] != NULL:
                    free(self.cv[i])
            free(self.cv)
            self.cv = NULL
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
        return to_numpy(self.solids, self.solids_ny, self.solids_nx, True)

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

    cdef void init_field(self):
        """Compute the Coulomb force at the centre of every
        solver grid cell and store the components in cu/cv.

        Each struct's charge is distributed evenly over its
        edge points; an edge at e contributes to the force at
        cell centre c:

            F(c) += Q_edge * (c - e) / |c - e|^3

        (Coulomb's law with the constant folded to 1, the same
        convention as ehd-flow's coulomb.py: this is the force
        per unit charge on a test ion sitting at the centre.)
        Cell centres are at ((x + 0.5) * dx, (y + 0.5) * dy) in
        the coordinate frame the edge coordinates are stored
        in.  Static batch: no ions move and nothing is
        time-stepped; contributions from an edge that coincides
        with a centre (zero distance) are skipped.
        """
        cdef Py_ssize_t x, y, s, e, idx
        cdef DTYPE_f xc, yc, rx, ry, r2, inv_r3, q
        cdef DTYPE_f* cu = self.cu[self.ck]
        cdef DTYPE_f* cv = self.cv[self.ck]
        cdef Py_ssize_t nx = self.solver.nx
        cdef Py_ssize_t ny = self.solver.ny

        for idx in range(self.solver.N):
            cu[idx] = 0.0
            cv[idx] = 0.0

        for s in range(self.n_shapes):
            if self.shapes[s].n_edges == 0:
                continue
            q = self.shapes[s].extradata.charge / self.shapes[s].n_edges
            for y in range(ny):
                yc = (y + 0.5) * self.solver.dy
                for x in range(nx):
                    xc = (x + 0.5) * self.solver.dx
                    idx = y * nx + x
                    for e in range(self.shapes[s].n_edges):
                        rx = xc - self.shapes[s].edges[2 * e]
                        ry = yc - self.shapes[s].edges[2 * e + 1]
                        r2 = rx * rx + ry * ry
                        if r2 == 0.0:
                            continue
                        inv_r3 = 1.0 / (r2 * sqrt(r2))
                        cu[idx] += q * rx * inv_r3
                        cv[idx] += q * ry * inv_r3

    cpdef tuple coulomb_field(self):
        """The Coulomb force components at the solver grid's
        cell centres, as (u, v) read-only (ny, nx) arrays, as
        computed by init_field() at construction."""
        return (to_numpy(self.cu[self.ck], self.solver.ny, self.solver.nx, True),
                to_numpy(self.cv[self.ck], self.solver.ny, self.solver.nx, True))

    cpdef cnp.ndarray ion_density(self):
        """The current +ion density at the solver grid's
        cells, as a read-only (ny, nx) array.  All zeros
        until the solver's first step advances the ions."""
        return to_numpy(self.d[self.dk], self.solver.ny,
                        self.solver.nx, True)

    cpdef tuple ion_velocity(self):
        """The current ion velocity (field drift + air
        velocity) at the solver grid's cells, as (u, v)
        read-only (ny, nx) arrays."""
        return (to_numpy(self.u[self.uvk], self.solver.ny, self.solver.nx, True),
                to_numpy(self.v[self.uvk], self.solver.ny, self.solver.nx, True))

    cdef bint _solid_at(self, Py_ssize_t idx) noexcept nogil:
        """Solid for the ion physics: a solver solid, or --
        when the image and solver grids match -- one of
        the handler's own electrode cells."""
        if self.solver.solid[idx]:
            return True
        if self.solids_nx == self.solver.nx and \
                self.solids_ny == self.solver.ny:
            return self.solids[idx] != 0
        return False

    cdef DTYPE_f _drag(self, Py_ssize_t idx, DTYPE_f ion_v,
                       DTYPE_f air_v) noexcept nogil:
        """The change in air velocity the ions in this cell
        cause (the get_u/get_v contract): the ions drag the
        air toward the ion velocity.  The drag rate grows
        with the ion density and with a resistance read
        from the cell's solver pressure; the increment is
        the relaxation fraction frac * (ion_v - air_v) with
        frac capped at 1, so a step never overshoots the
        ion velocity."""
        cdef DTYPE_f density = self.d[self.dk][idx]
        cdef DTYPE_f resistance, frac, dt
        if density <= 0.0:
            return 0.0
        resistance = (ION_AMBIENT_PRESSURE +
                      self.solver.p[self.solver.pk][idx]) / \
            ION_AMBIENT_PRESSURE
        if resistance < 0.1:
            resistance = 0.1
        if resistance > ION_RESISTANCE_MAX:
            resistance = ION_RESISTANCE_MAX
        dt = self.solver.dt
        if dt <= 0.0:
            dt = self.solver.dt_base
        frac = dt * density * ION_AIR_COUPLING * resistance / \
            self.solver.rho
        if frac > 1.0:
            frac = 1.0
        return frac * (ion_v - air_v)

    cdef DTYPE_f get_u(self, Py_ssize_t y, Py_ssize_t x,
                       Py_ssize_t idx) noexcept nogil:
        if self._solid_at(idx):
            return 0.0
        return self._drag(idx, self.u[self.uvk][idx],
                          self.solver.u[self.solver.uvk][idx])

    cdef DTYPE_f get_v(self, Py_ssize_t y, Py_ssize_t x,
                       Py_ssize_t idx) noexcept nogil:
        if self._solid_at(idx):
            return 0.0
        return self._drag(idx, self.v[self.uvk][idx],
                          self.solver.v[self.solver.uvk][idx])

    cdef DTYPE_f get_p(self, Py_ssize_t y, Py_ssize_t x,
                       Py_ssize_t idx) noexcept nogil:
        """The change in the ion cloud's electrostatic
        pressure since last step, which the solver adds
        into its pressure -- so the ion contribution held
        in the solver's pressure telescopes to the current
        ion pressure field (bounded by ION_PRESSURE_MAX)
        instead of accumulating step over step.  The new
        field is computed into the p buffer's back slot
        during the advance; when the pass wraps, pk flips
        to it.  The solver's pressure pass calls get_p
        exactly once per cell per step; the first call of
        a pass advances the whole ion state (generation,
        transport, diffusion, pressure) for the step."""
        cdef DTYPE_f val
        if self.p_calls == 0:
            # Native-C advance on the solver's thread pool
            # (re-used within the solver step).
            handler_c_advance(<handler_c_t*>self.c_handler,
                              self.solver.u[self.solver.uvk],
                              self.solver.v[self.solver.uvk],
                              self.solver.solid,
                              self.solver.dt, self.solver.dt_base)
            self.dk = handler_c_dk(<handler_c_t*>self.c_handler)
        self.p_calls += 1
        val = self.p[1 - self.pk][idx] - self.p[self.pk][idx]
        if self.p_calls >= self.solver.N:
            self.p_calls = 0
            self.pk = 1 - self.pk
            handler_c_set_pk(<handler_c_t*>self.c_handler, self.pk)
        if self._solid_at(idx):
            return 0.0
        return val
