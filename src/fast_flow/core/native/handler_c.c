/* handler_c.c — native-C electrostatic handler (ion advance).
 *
 * Faithful port of ElectrostaticForceHandler._advance_ion_state
 * (electrostaticforcehandler.pyx): same formulas, same loop
 * order, same buffer flips, so results match the Cython
 * handler to floating-point tolerance. The advance runs on
 * the solver's thread pool (borrowed), re-used within the
 * solver step.
 *
 * Parallelization (all row/flat-index independent, Jacobi):
 *  - velocity: parallel-for + a max reduction (max is exact
 *    in FP, so the parallel reduction is bitwise-exact);
 *  - transport: each subcycle is a parallel-for (the
 *    subcycles stay sequential -- each depends on the last);
 *  - generation / diffusion / pressure: parallel-for.
 * The pinning-style serial passes are none here; every
 * pass is independent per cell.
 *
 * Floating point: no -ffast-math; expressions are written
 * in the same form as the .pyx source.
 */

#include "handler_c.h"

#include <stdlib.h>
#include <math.h>

/* Constants (must match electrostaticforcehandler.pyx). */
#define ION_ZONE_TARGET 1.0
#define ION_PRODUCTION_TAU 0.01
#define ION_DIFFUSION 0.01
#define ION_PRESSURE_COUPLING 1.0e-5
#define ION_PRESSURE_MAX 100.0
#define ION_SHIELD_DENSITY 3.0
#define ION_SPEED_MAX 1000.0

struct handler_c {
    long nx, ny, N;
    double dx, dy;
    double *d[2], *p[2];
    double *u[2], *v[2];
    double *cu[2], *cv[2];
    unsigned char *ion_zone;
    unsigned char *solids;
    long solids_nx, solids_ny;
    long dk, pk, uvk, ck;
    ff_pool_t *pool;
    int owns_pool;
};

static inline int h_solid_at(const handler_c_t *h,
                             const unsigned char *solid, long idx) {
    if (solid[idx])
        return 1;
    if (h->solids && h->solids_nx == h->nx && h->solids_ny == h->ny)
        return h->solids[idx] != 0;
    return 0;
}

handler_c_t *handler_c_create(long nx, long ny,
                              double dx, double dy,
                              double *d0, double *d1,
                              double *p0, double *p1,
                              double *u0, double *u1,
                              double *v0, double *v1,
                              double *cu0, double *cu1,
                              double *cv0, double *cv1,
                              unsigned char *ion_zone,
                              unsigned char *solids,
                              long solids_nx, long solids_ny,
                              ff_pool_t *pool) {
    handler_c_t *h = (handler_c_t *)calloc(1, sizeof(handler_c_t));
    if (!h)
        return NULL;
    h->nx = nx;
    h->ny = ny;
    h->N = nx * ny;
    h->dx = dx;
    h->dy = dy;
    h->d[0] = d0;
    h->d[1] = d1;
    h->p[0] = p0;
    h->p[1] = p1;
    h->u[0] = u0;
    h->u[1] = u1;
    h->v[0] = v0;
    h->v[1] = v1;
    h->cu[0] = cu0;
    h->cu[1] = cu1;
    h->cv[0] = cv0;
    h->cv[1] = cv1;
    h->ion_zone = ion_zone;
    h->solids = solids;
    h->solids_nx = solids_nx;
    h->solids_ny = solids_ny;
    h->dk = 0;
    h->pk = 0;
    h->uvk = 0;
    h->ck = 0;
    if (pool) {
        h->pool = pool;
        h->owns_pool = 0;
    } else {
        h->pool = ff_pool_create(0);
        h->owns_pool = 1;
        if (!h->pool) {
            free(h);
            return NULL;
        }
    }
    return h;
}

void handler_c_destroy(handler_c_t *h) {
    if (!h)
        return;
    if (h->owns_pool)
        ff_pool_destroy(h->pool);
    free(h);
}

long handler_c_dk(const handler_c_t *h) {
    return h ? h->dk : 0;
}

void handler_c_set_pk(handler_c_t *h, long pk) {
    if (h)
        h->pk = pk;
}

/* ------------------------------------------------------------------ */
/* Advance passes                                                      */
/* ------------------------------------------------------------------ */

/* Velocity pass ctx (partial max per participant). */
typedef struct {
    handler_c_t *h;
    const double *au, *av;
    const unsigned char *solid;
    double *partial_max;       /* [nthreads] */
} velocity_ctx_t;

static void velocity_rows(void *vctx, long y0, long y1, int id) {
    velocity_ctx_t *ctx = (velocity_ctx_t *)vctx;
    handler_c_t *h = ctx->h;
    const double *au = ctx->au, *av = ctx->av;
    const unsigned char *solid = ctx->solid;
    double *iu = h->u[h->uvk];
    double *iv = h->v[h->uvk];
    double *cu = h->cu[h->ck];
    double *cv = h->cv[h->ck];
    double *dd = h->d[h->dk];
    long nx = h->nx;
    double dx = h->dx, dy = h->dy;
    double local_rate = 0.0;
    long y, x, idx;
    double vx, vy, sp2, scale, shield, cell_rate;

    for (y = y0; y < y1; y++) {
        for (x = 0; x < nx; x++) {
            idx = y * nx + x;
            if (h_solid_at(h, solid, idx)) {
                iu[idx] = 0.0;
                iv[idx] = 0.0;
                continue;
            }
            vx = au[idx];
            vy = av[idx];
            shield = 1.0 - dd[idx] / ION_SHIELD_DENSITY;
            if (shield > 0.0) {
                vx += cu[idx] * shield;
                vy += cv[idx] * shield;
            }
            sp2 = vx * vx + vy * vy;
            if (sp2 > ION_SPEED_MAX * ION_SPEED_MAX) {
                scale = ION_SPEED_MAX / sqrt(sp2);
                vx *= scale;
                vy *= scale;
            }
            iu[idx] = vx;
            iv[idx] = vy;
            cell_rate = fabs(vx) / dx + fabs(vy) / dy;
            if (cell_rate > local_rate)
                local_rate = cell_rate;
        }
    }
    ctx->partial_max[id] = local_rate;
}

/* Transport subcycle ctx. */
typedef struct {
    handler_c_t *h;
    const unsigned char *solid;
    double *sbuf, *dbuf;
    double tau;
} transport_ctx_t;

static void transport_rows(void *vctx, long y0, long y1, int id) {
    transport_ctx_t *ctx = (transport_ctx_t *)vctx;
    handler_c_t *h = ctx->h;
    const unsigned char *solid = ctx->solid;
    double *sbuf = ctx->sbuf, *dbuf = ctx->dbuf;
    double tau = ctx->tau;
    double *iu = h->u[h->uvk];
    double *iv = h->v[h->uvk];
    long nx = h->nx, ny = h->ny;
    double dx = h->dx, dy = h->dy;
    long y, x, idx;
    double ue, uw, vn, vs, fx_e, fx_w, fy_n, fy_s;
    double src_e, src_w, src_n, src_s, val, shield;
    (void)id;

    for (y = y0; y < y1; y++) {
        for (x = 0; x < nx; x++) {
            idx = y * nx + x;
            if (h_solid_at(h, solid, idx)) {
                dbuf[idx] = 0.0;
                continue;
            }
            /* east face */
            if (x + 1 < nx && !h_solid_at(h, solid, idx + 1)) {
                ue = 0.5 * (iu[idx] + iu[idx + 1]);
                src_e = sbuf[idx + 1];
            } else {
                ue = iu[idx];
                src_e = 0.0;
            }
            fx_e = ue * (ue > 0.0 ? sbuf[idx] : src_e);
            /* west face */
            if (x > 0 && !h_solid_at(h, solid, idx - 1)) {
                uw = 0.5 * (iu[idx] + iu[idx - 1]);
                src_w = sbuf[idx - 1];
            } else {
                uw = iu[idx];
                src_w = 0.0;
            }
            fx_w = uw * (uw > 0.0 ? src_w : sbuf[idx]);
            /* north face */
            if (y + 1 < ny && !h_solid_at(h, solid, idx + nx)) {
                vn = 0.5 * (iv[idx] + iv[idx + nx]);
                src_n = sbuf[idx + nx];
            } else {
                vn = iv[idx];
                src_n = 0.0;
            }
            fy_n = vn * (vn > 0.0 ? sbuf[idx] : src_n);
            /* south face */
            if (y > 0 && !h_solid_at(h, solid, idx - nx)) {
                vs = 0.5 * (iv[idx] + iv[idx - nx]);
                src_s = sbuf[idx - nx];
            } else {
                vs = iv[idx];
                src_s = 0.0;
            }
            fy_s = vs * (vs > 0.0 ? src_s : sbuf[idx]);
            /* Space-charge choking (see the .pyx comment). */
            if (ue > 0.0) {
                if (x + 1 < nx && !h_solid_at(h, solid, idx + 1)) {
                    shield = 1.0 - sbuf[idx + 1] / ION_SHIELD_DENSITY;
                    fx_e *= shield > 0.0 ? shield : 0.0;
                }
            } else {
                shield = 1.0 - sbuf[idx] / ION_SHIELD_DENSITY;
                fx_e *= shield > 0.0 ? shield : 0.0;
            }
            if (uw > 0.0) {
                shield = 1.0 - sbuf[idx] / ION_SHIELD_DENSITY;
                fx_w *= shield > 0.0 ? shield : 0.0;
            } else if (x > 0 && !h_solid_at(h, solid, idx - 1)) {
                shield = 1.0 - sbuf[idx - 1] / ION_SHIELD_DENSITY;
                fx_w *= shield > 0.0 ? shield : 0.0;
            }
            if (vn > 0.0) {
                if (y + 1 < ny && !h_solid_at(h, solid, idx + nx)) {
                    shield = 1.0 - sbuf[idx + nx] / ION_SHIELD_DENSITY;
                    fy_n *= shield > 0.0 ? shield : 0.0;
                }
            } else {
                shield = 1.0 - sbuf[idx] / ION_SHIELD_DENSITY;
                fy_n *= shield > 0.0 ? shield : 0.0;
            }
            if (vs > 0.0) {
                shield = 1.0 - sbuf[idx] / ION_SHIELD_DENSITY;
                fy_s *= shield > 0.0 ? shield : 0.0;
            } else if (y > 0 && !h_solid_at(h, solid, idx - nx)) {
                shield = 1.0 - sbuf[idx - nx] / ION_SHIELD_DENSITY;
                fy_s *= shield > 0.0 ? shield : 0.0;
            }
            val = sbuf[idx] - tau * (
                (fx_e - fx_w) / dx + (fy_n - fy_s) / dy);
            dbuf[idx] = val > 0.0 ? val : 0.0;
        }
    }
}

/* Generation ctx (flat index range). */
typedef struct {
    handler_c_t *h;
    double *sbuf;
    double gen;
} generation_ctx_t;

static void generation_range(void *vctx, long i0, long i1, int id) {
    generation_ctx_t *ctx = (generation_ctx_t *)vctx;
    handler_c_t *h = ctx->h;
    double *sbuf = ctx->sbuf;
    double gen = ctx->gen;
    long idx;
    (void)id;
    for (idx = i0; idx < i1; idx++) {
        if (h->ion_zone[idx] && sbuf[idx] < ION_ZONE_TARGET) {
            sbuf[idx] += (ION_ZONE_TARGET - sbuf[idx]) * gen;
        }
    }
}

/* Diffusion ctx. */
typedef struct {
    handler_c_t *h;
    const unsigned char *solid;
    double *sbuf, *dbuf;
} diffusion_ctx_t;

static void diffusion_rows(void *vctx, long y0, long y1, int id) {
    diffusion_ctx_t *ctx = (diffusion_ctx_t *)vctx;
    handler_c_t *h = ctx->h;
    const unsigned char *solid = ctx->solid;
    double *sbuf = ctx->sbuf, *dbuf = ctx->dbuf;
    long nx = h->nx, ny = h->ny;
    long y, x, idx;
    double val, ce, cw, cn, cs;
    (void)id;

    for (y = y0; y < y1; y++) {
        for (x = 0; x < nx; x++) {
            idx = y * nx + x;
            if (h_solid_at(h, solid, idx)) {
                dbuf[idx] = 0.0;
                continue;
            }
            val = sbuf[idx];
            ce = (x + 1 >= nx || h_solid_at(h, solid, idx + 1))
                ? val : sbuf[idx + 1];
            cw = (x == 0 || h_solid_at(h, solid, idx - 1))
                ? val : sbuf[idx - 1];
            cn = (y + 1 >= ny || h_solid_at(h, solid, idx + nx))
                ? val : sbuf[idx + nx];
            cs = (y == 0 || h_solid_at(h, solid, idx - nx))
                ? val : sbuf[idx - nx];
            dbuf[idx] = val + ION_DIFFUSION * (
                ce + cw + cn + cs - 4.0 * val);
        }
    }
}

/* Pressure ctx (flat index range). */
typedef struct {
    handler_c_t *h;
    const unsigned char *solid;
    double *sbuf, *pp;
} pressure_ctx_t;

static void pressure_range(void *vctx, long i0, long i1, int id) {
    pressure_ctx_t *ctx = (pressure_ctx_t *)vctx;
    handler_c_t *h = ctx->h;
    const unsigned char *solid = ctx->solid;
    double *sbuf = ctx->sbuf, *pp = ctx->pp;
    double *cu = h->cu[h->ck];
    double *cv = h->cv[h->ck];
    long idx;
    double e2, val;
    (void)id;
    for (idx = i0; idx < i1; idx++) {
        if (h_solid_at(h, solid, idx)) {
            pp[idx] = 0.0;
            continue;
        }
        e2 = cu[idx] * cu[idx] + cv[idx] * cv[idx];
        if (e2 > ION_SPEED_MAX * ION_SPEED_MAX)
            e2 = ION_SPEED_MAX * ION_SPEED_MAX;
        val = ION_PRESSURE_COUPLING * sbuf[idx] * e2;
        pp[idx] = val < ION_PRESSURE_MAX ? val : ION_PRESSURE_MAX;
    }
}

/* ------------------------------------------------------------------ */
/* Advance                                                             */
/* ------------------------------------------------------------------ */

void handler_c_advance(handler_c_t *h,
                       const double *au, const double *av,
                       const unsigned char *solid,
                       double dt, double dt_base) {
    long ny = h->ny;
    double rate = 0.0;
    double tau, gen;
    long nsub, k, cur, i;
    double *sbuf, *dbuf, *tbuf, *sbuf_final;
    velocity_ctx_t vctx;
    transport_ctx_t tctx;
    generation_ctx_t gctx;
    diffusion_ctx_t dctx;
    pressure_ctx_t pctx;
    double *partial_max;
    int nthreads;

    if (dt <= 0.0)
        dt = dt_base;

    /* 1. ion velocity + transport rate (parallel max) */
    nthreads = ff_pool_nthreads(h->pool);
    partial_max = (double *)calloc((size_t)nthreads, sizeof(double));
    if (!partial_max)
        return;
    vctx.h = h;
    vctx.au = au;
    vctx.av = av;
    vctx.solid = solid;
    vctx.partial_max = partial_max;
    ff_pool_for(h->pool, ny, velocity_rows, &vctx);
    for (i = 0; i < nthreads; i++) {
        if (partial_max[i] > rate)
            rate = partial_max[i];
    }
    free(partial_max);

    /* 2. conservative flux transport, subcycled */
    nsub = 1;
    if (rate > 0.0) {
        nsub = (long)(dt * rate / 0.8) + 1;
        if (nsub > 8192)
            nsub = 8192;
    }
    tau = dt / nsub;
    cur = h->dk;
    sbuf = h->d[cur];
    dbuf = h->d[1 - cur];
    tctx.h = h;
    tctx.solid = solid;
    tctx.tau = tau;
    for (k = 0; k < nsub; k++) {
        tctx.sbuf = sbuf;
        tctx.dbuf = dbuf;
        ff_pool_for(h->pool, ny, transport_rows, &tctx);
        tbuf = sbuf;
        sbuf = dbuf;
        dbuf = tbuf;
        cur = 1 - cur;
    }

    /* 3. ionization-zone generation */
    gen = dt / ION_PRODUCTION_TAU;
    if (gen > 1.0)
        gen = 1.0;
    gctx.h = h;
    gctx.sbuf = sbuf;
    gctx.gen = gen;
    ff_pool_for(h->pool, h->N, generation_range, &gctx);

    /* 4. diffusion of the density into the other buffer */
    dctx.h = h;
    dctx.solid = solid;
    dctx.sbuf = sbuf;
    dctx.dbuf = dbuf;
    ff_pool_for(h->pool, ny, diffusion_rows, &dctx);
    h->dk = 1 - cur;
    sbuf_final = h->d[h->dk];

    /* 5. electrostatic pressure of the ion cloud.
     * pp = p[1 - pk]; pk is passed by the caller via the
     * Cython side (get_p) -- here we use h->pk, which the
     * Cython side keeps in sync (it only flips in get_p,
     * never in the advance). */
    pctx.h = h;
    pctx.solid = solid;
    pctx.sbuf = sbuf_final;
    pctx.pp = h->p[1 - h->pk];
    ff_pool_for(h->pool, h->N, pressure_range, &pctx);
}
