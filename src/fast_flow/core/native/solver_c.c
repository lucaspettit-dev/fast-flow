/* solver_c.c — native-C solver core + thread pool.
 *
 * See ../solver_c.h for the design. The solver functions are
 * a faithful port of FlowSolverCore (flow.pyx): same formulas,
 * same loop order, same buffer flips, so results match the
 * Cython solver to floating-point tolerance. Two deliberate
 * deviations, both order-preserving: the pinning sum stays
 * serial (row-major, bitwise-exact); the pressure-force pass
 * stays serial (the handler's exactly-once get_p contract).
 * Everything else that is row-independent runs on the pool.
 *
 * Floating point: no -ffast-math anywhere, so the compiler
 * must preserve the operation order as written (IEEE double,
 * round-to-nearest). Expressions are written in the same
 * form as the .pyx source, so the FP results match.
 */

#include "solver_c.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

/* ================================================================== */
/* Thread pool                                                         */
/* ================================================================== */

struct ff_pool {
    int nthreads;              /* participants (main + workers) */
    int nworkers;              /* nthreads - 1 */
    pthread_t *threads;        /* worker threads */
    pthread_mutex_t mutex;
    pthread_cond_t go;         /* task ready */
    pthread_cond_t done;       /* all workers finished */
    unsigned long generation;  /* task epoch */
    unsigned long *last_seen;  /* per-worker last processed epoch */
    int finished;              /* workers done with current task */
    int shutdown;
    ff_task_fn func;
    void *ctx;
    long n;                    /* range [0, n) */
};

int ff_pool_max_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1)
        n = 1;
    return (int)n;
}

/* Chunk for participant `id` (0 = main, 1..nworkers = workers):
 * static split, ceil(n / nthreads) rows per chunk. */
static void pool_chunk(const ff_pool_t *pool, long n, int id,
                       long *start, long *end) {
    long chunk = (n + pool->nthreads - 1) / pool->nthreads;
    *start = (long)id * chunk;
    *end = *start + chunk;
    if (*end > n)
        *end = n;
    if (*start > n)
        *start = n;
}

typedef struct {
    ff_pool_t *pool;
    int index;                 /* chunk id for this worker */
} worker_arg_t;

static void *pool_worker_indexed(void *arg) {
    worker_arg_t *warg = (worker_arg_t *)arg;
    ff_pool_t *pool = warg->pool;
    int id = warg->index;
    unsigned long gen;
    ff_task_fn func;
    void *ctx;
    long n, start, end;
    free(warg);

    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->generation == pool->last_seen[id] && !pool->shutdown)
            pthread_cond_wait(&pool->go, &pool->mutex);
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        gen = pool->generation;
        func = pool->func;
        ctx = pool->ctx;
        n = pool->n;
        pool->last_seen[id] = gen;
        pthread_mutex_unlock(&pool->mutex);

        pool_chunk(pool, n, id, &start, &end);
        if (func && end > start)
            func(ctx, start, end, id);

        pthread_mutex_lock(&pool->mutex);
        if (gen == pool->generation) {
            pool->finished++;
            if (pool->finished >= pool->nworkers)
                pthread_cond_signal(&pool->done);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
}

ff_pool_t *ff_pool_create(int nthreads) {
    ff_pool_t *pool;
    int i;

    if (nthreads <= 0)
        nthreads = ff_pool_max_threads();
    if (nthreads < 1)
        nthreads = 1;

    pool = (ff_pool_t *)calloc(1, sizeof(ff_pool_t));
    if (!pool)
        return NULL;
    pool->nthreads = nthreads;
    pool->nworkers = nthreads - 1;
    pool->generation = 0;
    pool->shutdown = 0;
    pool->finished = 0;

    if (pthread_mutex_init(&pool->mutex, NULL) != 0 ||
        pthread_cond_init(&pool->go, NULL) != 0 ||
        pthread_cond_init(&pool->done, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        pthread_cond_destroy(&pool->go);
        pthread_cond_destroy(&pool->done);
        free(pool);
        return NULL;
    }

    if (pool->nworkers > 0) {
        pool->threads =
            (pthread_t *)malloc(sizeof(pthread_t) * (size_t)pool->nworkers);
        pool->last_seen =
            (unsigned long *)calloc((size_t)(pool->nworkers + 1),
                                    sizeof(unsigned long));
        if (!pool->threads || !pool->last_seen) {
            free(pool->threads);
            free(pool->last_seen);
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->go);
            pthread_cond_destroy(&pool->done);
            free(pool);
            return NULL;
        }
        for (i = 0; i < pool->nworkers; i++) {
            worker_arg_t *warg =
                (worker_arg_t *)malloc(sizeof(worker_arg_t));
            if (!warg) {
                pthread_mutex_lock(&pool->mutex);
                pool->shutdown = 1;
                pthread_cond_broadcast(&pool->go);
                pthread_mutex_unlock(&pool->mutex);
                for (int j = 0; j < i; j++)
                    pthread_join(pool->threads[j], NULL);
                free(pool->threads);
                free(pool->last_seen);
                pthread_mutex_destroy(&pool->mutex);
                pthread_cond_destroy(&pool->go);
                pthread_cond_destroy(&pool->done);
                free(pool);
                return NULL;
            }
            warg->pool = pool;
            warg->index = i + 1;   /* chunk id: workers are 1..nworkers */
            if (pthread_create(&pool->threads[i], NULL,
                               pool_worker_indexed, warg) != 0) {
                free(warg);
                pthread_mutex_lock(&pool->mutex);
                pool->shutdown = 1;
                pthread_cond_broadcast(&pool->go);
                pthread_mutex_unlock(&pool->mutex);
                for (int j = 0; j < i; j++)
                    pthread_join(pool->threads[j], NULL);
                free(pool->threads);
                free(pool->last_seen);
                pthread_mutex_destroy(&pool->mutex);
                pthread_cond_destroy(&pool->go);
                pthread_cond_destroy(&pool->done);
                free(pool);
                return NULL;
            }
        }
    }
    return pool;
}

int ff_pool_nthreads(const ff_pool_t *pool) {
    return pool ? pool->nthreads : 1;
}

void ff_pool_destroy(ff_pool_t *pool) {    int i;
    if (!pool)
        return;
    if (pool->nworkers > 0) {
        pthread_mutex_lock(&pool->mutex);
        pool->shutdown = 1;
        pthread_cond_broadcast(&pool->go);
        pthread_mutex_unlock(&pool->mutex);
        for (i = 0; i < pool->nworkers; i++)
            pthread_join(pool->threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->go);
    pthread_cond_destroy(&pool->done);
    free(pool->threads);
    free(pool->last_seen);
    free(pool);
}

void ff_pool_for(ff_pool_t *pool, long n, ff_task_fn func, void *ctx) {
    long start, end;
    if (!pool || !func || n <= 0)
        return;
    if (pool->nworkers == 0) {
        /* single-threaded: run the whole range inline */
        func(ctx, 0, n, 0);
        return;
    }
    pthread_mutex_lock(&pool->mutex);
    pool->func = func;
    pool->ctx = ctx;
    pool->n = n;
    pool->finished = 0;
    pool->generation++;
    pthread_cond_broadcast(&pool->go);
    pthread_mutex_unlock(&pool->mutex);

    /* main thread runs chunk 0 */
    pool_chunk(pool, n, 0, &start, &end);
    if (end > start)
        func(ctx, start, end, 0);

    /* barrier: wait for all workers */
    pthread_mutex_lock(&pool->mutex);
    while (pool->finished < pool->nworkers)
        pthread_cond_wait(&pool->done, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

/* ================================================================== */
/* Solver                                                              */
/* ================================================================== */

struct solver_c {
    long nx, ny, N;
    double dx, dy, lx, ly;
    double rho, nu, dt, dt_base, inflow_u;
    long nit, flow_mode;
    double *u[2], *v[2], *p[2];
    double *b;
    unsigned char *solid;
    long pk, uvk;
    double sum_dt, last_dt;
    ff_pool_t *pool;
};

static inline long c_idx(const solver_c_t *s, long y, long x) {
    return y * s->nx + x;
}

solver_c_t *solver_c_create(long nx, long ny,
                            double dx, double dy,
                            double lx, double ly,
                            double rho, double nu,
                            double dt_base, long nit, long flow_mode,
                            double inflow_u,
                            double *u0, double *u1,
                            double *v0, double *v1,
                            double *p0, double *p1,
                            double *b, unsigned char *solid) {
    solver_c_t *s = (solver_c_t *)calloc(1, sizeof(solver_c_t));
    if (!s)
        return NULL;
    s->nx = nx;
    s->ny = ny;
    s->N = nx * ny;
    s->dx = dx;
    s->dy = dy;
    s->lx = lx;
    s->ly = ly;
    s->rho = rho;
    s->nu = nu;
    s->dt = dt_base;
    s->dt_base = dt_base;
    s->inflow_u = inflow_u;
    s->nit = nit;
    s->flow_mode = flow_mode;
    s->u[0] = u0;
    s->u[1] = u1;
    s->v[0] = v0;
    s->v[1] = v1;
    s->p[0] = p0;
    s->p[1] = p1;
    s->b = b;
    s->solid = solid;
    s->pk = 0;
    s->uvk = 0;
    s->sum_dt = 0.0;
    s->last_dt = 0.0;
    s->pool = ff_pool_create(0);   /* max available threads */
    if (!s->pool) {
        free(s);
        return NULL;
    }
    return s;
}

void solver_c_destroy(solver_c_t *s) {
    if (!s)
        return;
    ff_pool_destroy(s->pool);
    free(s);
}

void solver_c_set_inflow(solver_c_t *s, double inflow_u) {
    if (s)
        s->inflow_u = inflow_u;
}

void solver_c_set_dt(solver_c_t *s, double dt) {
    if (s)
        s->dt = dt;
}

void solver_c_set_uvk(solver_c_t *s, long uvk) {
    if (s)
        s->uvk = uvk;
}

void solver_c_set_pk(solver_c_t *s, long pk) {
    if (s)
        s->pk = pk;
}

long solver_c_pk(const solver_c_t *s) {
    return s ? s->pk : 0;
}

long solver_c_uvk(const solver_c_t *s) {
    return s ? s->uvk : 0;
}

ff_pool_t *solver_c_pool(solver_c_t *s) {
    return s ? s->pool : NULL;
}

/* ------------------------------------------------------------------ */
/* Per-cell kernels (faithful port of the .pyx methods)               */
/* ------------------------------------------------------------------ */

static inline void c_build_up(solver_c_t *s, long y, long x,
                              double dy2, double dx2, double inv_dt) {
    long idx = c_idx(s, y + 1, x + 1);
    double *u, *v, hor, vert, c, a, b, d;
    if (s->solid[idx])
        return;
    u = s->u[s->uvk];
    v = s->v[s->uvk];
    hor = (u[c_idx(s, y + 1, x + 2)] - u[c_idx(s, y + 1, x)]) / dx2;
    vert = (v[c_idx(s, y + 2, x + 1)] - v[c_idx(s, y, x + 1)]) / dy2;
    c = 2 * ((u[c_idx(s, y + 2, x + 1)] - u[c_idx(s, y, x + 1)]) / dy2 *
             (v[c_idx(s, y + 1, x + 2)] - v[c_idx(s, y + 1, x)]) / dx2);
    a = hor + vert;
    b = hor * hor;
    d = vert * vert;
    s->b[idx] = s->rho * (inv_dt * a - b - c - d);
}

static inline void c_poisson_step(solver_c_t *s, long y, long x,
                                  double *p_dest, double *p_src,
                                  double dy_squared, double dx_squared) {
    long idx = c_idx(s, y + 1, x + 1);
    double pc, pe, pw, pn, ps, hor, vert, d;
    if (s->solid[idx])
        return;
    pc = p_src[idx];
    pe = s->solid[c_idx(s, y + 1, x + 2)] ? pc : p_src[c_idx(s, y + 1, x + 2)];
    pw = s->solid[c_idx(s, y + 1, x)] ? pc : p_src[c_idx(s, y + 1, x)];
    pn = s->solid[c_idx(s, y + 2, x + 1)] ? pc : p_src[c_idx(s, y + 2, x + 1)];
    ps = s->solid[idx - s->nx] ? pc : p_src[c_idx(s, y, x + 1)];
    hor = (pe + pw) * dy_squared;
    vert = (pn + ps) * dx_squared;
    d = 2 * (dx_squared + dy_squared);
    p_dest[idx] = (hor + vert) / d - (dx_squared * dy_squared / d) * s->b[idx];
}

/* Forward: pinning subtract pass (defined below c_pressure_poisson). */
static void c_pin_subtract(solver_c_t *s, double *pp, double pmean);

/* Pressure boundary conditions on the dest buffer (serial:
 * perimeter-only, plus a full-domain solid-zeroing that runs
 * on the pool). Mirrors pressure_set_boundry_conditions. */
typedef struct {
    solver_c_t *s;
    double *p_dest;
} bc_ctx_t;

static void bc_zero_solids(void *vctx, long y0, long y1, int id) {
    bc_ctx_t *ctx = (bc_ctx_t *)vctx;
    (void)id;
    solver_c_t *s = ctx->s;
    double *p = ctx->p_dest;
    long y, x, idx;
    for (y = y0; y < y1; y++) {
        for (x = 0; x < s->nx; x++) {
            idx = c_idx(s, y, x);
            if (s->solid[idx])
                p[idx] = 0.0;
        }
    }
}

static void c_pressure_bcs(solver_c_t *s, double *p_dest) {
    long i, n = s->nx;
    bc_ctx_t ctx;
    if (s->ny > s->nx)
        n = s->ny;

    ctx.s = s;
    ctx.p_dest = p_dest;
    ff_pool_for(s->pool, s->ny, bc_zero_solids, &ctx);

    if (s->flow_mode == 1) {
        /* throughflow: zero-gradient pressure on every side */
        for (i = 1; i < s->nx - 1; i++) {
            p_dest[c_idx(s, 0, i)] = p_dest[c_idx(s, 1, i)];
            p_dest[c_idx(s, s->ny - 1, i)] = p_dest[c_idx(s, s->ny - 2, i)];
        }
        for (i = 1; i < s->ny - 1; i++) {
            p_dest[c_idx(s, i, 0)] = p_dest[c_idx(s, i, 1)];
            p_dest[c_idx(s, i, s->nx - 1)] = p_dest[c_idx(s, i, s->nx - 2)];
        }
        p_dest[0] = p_dest[c_idx(s, 1, 1)];
        p_dest[c_idx(s, 0, s->nx - 1)] = p_dest[c_idx(s, 1, s->nx - 2)];
        p_dest[c_idx(s, s->ny - 1, 0)] = p_dest[c_idx(s, s->ny - 2, 1)];
        p_dest[c_idx(s, s->ny - 1, s->nx - 1)] =
            p_dest[c_idx(s, s->ny - 2, s->nx - 2)];
        return;
    }

    for (i = 0; i < n; i++) {
        if (0 < i && i < s->ny - 1) {
            p_dest[c_idx(s, i, 0)] = p_dest[c_idx(s, i, 1)];
            p_dest[c_idx(s, i, s->nx - 1)] = p_dest[c_idx(s, i, s->nx - 2)];
        }
        if (0 < i && i < s->nx - 1) {
            p_dest[c_idx(s, 0, i)] = p_dest[c_idx(s, 1, i)];
        }
        if (i < s->nx) {
            p_dest[c_idx(s, s->ny - 1, i)] = 0;
        }
    }
    p_dest[0] = p_dest[c_idx(s, 1, 1)];
    p_dest[c_idx(s, 0, s->nx - 1)] = p_dest[c_idx(s, 1, s->nx - 2)];
}

/* Poisson sweep task contexts (parallel-for over row range). */
typedef struct {
    solver_c_t *s;
    double *p_dest;
    double *p_src;
    double dy_squared, dx_squared;
    double dy2, dx2, inv_dt;
    int with_build_up;
} poisson_ctx_t;

static void poisson_rows(void *vctx, long y0, long y1, int id) {
    poisson_ctx_t *ctx = (poisson_ctx_t *)vctx;
    (void)id;
    solver_c_t *s = ctx->s;
    long y, x;
    /* y here is the sweep row (0..ny-3); the kernels add +1. */
    for (y = y0; y < y1; y++) {
        for (x = 0; x < s->nx - 2; x++) {
            if (ctx->with_build_up)
                c_build_up(s, y, x, ctx->dy2, ctx->dx2, ctx->inv_dt);
            c_poisson_step(s, y, x, ctx->p_dest, ctx->p_src,
                           ctx->dy_squared, ctx->dx_squared);
        }
    }
}

static void c_pressure_poisson(solver_c_t *s) {
    double dy_squared = s->dy * s->dy;
    double dx_squared = s->dx * s->dx;
    double dy2 = 2 * s->dy;
    double dx2 = 2 * s->dx;
    double inv_dt = 1 / s->dt;
    long src = s->pk;
    long dest = src ^ 1;
    double *p_dest, *p_src;
    poisson_ctx_t ctx;
    long q, jj, ii, base;
    double pmean;
    long nfluid;
    double *pp;

    /* first sweep with build-up */
    p_dest = s->p[dest];
    p_src = s->p[src];
    ctx.s = s;
    ctx.p_dest = p_dest;
    ctx.p_src = p_src;
    ctx.dy_squared = dy_squared;
    ctx.dx_squared = dx_squared;
    ctx.dy2 = dy2;
    ctx.dx2 = dx2;
    ctx.inv_dt = inv_dt;
    ctx.with_build_up = 1;
    ff_pool_for(s->pool, s->ny - 2, poisson_rows, &ctx);
    if (!s->flow_mode == 2)
        c_pressure_bcs(s, p_dest);

    s->pk ^= 1;
    src = s->pk;
    dest = src ^ 1;

    for (q = 0; q < s->nit - 1; q++) {
        p_dest = s->p[dest];
        p_src = s->p[src];
        ctx.p_dest = p_dest;
        ctx.p_src = p_src;
        ctx.with_build_up = 0;
        ff_pool_for(s->pool, s->ny - 2, poisson_rows, &ctx);
        if (!s->flow_mode == 2)
            c_pressure_bcs(s, p_dest);
        s->pk ^= 1;
        src = s->pk;
        dest = src ^ 1;
    }

    if (s->flow_mode == 1) {
        /* Pin the fluid mean to zero. The sum is SERIAL and
         * row-major (bitwise-exact); the subtract is parallel. */
        pp = s->p[s->pk];
        pmean = 0.0;
        nfluid = 0;
        for (jj = 0; jj < s->ny; jj++) {
            base = jj * s->nx;
            for (ii = 0; ii < s->nx; ii++) {
                if (!s->solid[c_idx(s, jj, ii)]) {
                    pmean += pp[base + ii];
                    nfluid += 1;
                }
            }
        }
        if (nfluid > 0)
            pmean /= nfluid;
        c_pin_subtract(s, pp, pmean);
    }
}

/* Pinning subtract pass (file-scope task ctx). */
typedef struct {
    solver_c_t *s;
    double *pp;
    double pmean;
} pin_sub_ctx_t;

static void pin_sub_rows(void *vctx, long y0, long y1, int id) {
    pin_sub_ctx_t *ctx = (pin_sub_ctx_t *)vctx;
    (void)id;
    solver_c_t *s = ctx->s;
    long y, x, idx;
    for (y = y0; y < y1; y++) {
        for (x = 0; x < s->nx; x++) {
            idx = c_idx(s, y, x);
            if (!s->solid[idx])
                ctx->pp[idx] -= ctx->pmean;
        }
    }
}

static void c_pin_subtract(solver_c_t *s, double *pp, double pmean) {
    pin_sub_ctx_t ctx;
    ctx.s = s;
    ctx.pp = pp;
    ctx.pmean = pmean;
    ff_pool_for(s->pool, s->ny, pin_sub_rows, &ctx);
}

/* ------------------------------------------------------------------ */
/* Momentum                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    solver_c_t *s;
    double *u_src, *v_src, *u_dest, *v_dest, *p;
    double dt, dx, dy, rho, nu;
    long y_off;                /* row offset: task range maps to y+off */
} momentum_ctx_t;

static void momentum_rows(void *vctx, long y0, long y1, int id) {
    momentum_ctx_t *ctx = (momentum_ctx_t *)vctx;
    (void)id;
    solver_c_t *s = ctx->s;
    double *u_src = ctx->u_src, *v_src = ctx->v_src;
    double *u_dest = ctx->u_dest, *v_dest = ctx->v_dest;
    double *p = ctx->p;
    double dt = ctx->dt, dx = ctx->dx, dy = ctx->dy;
    double rho = ctx->rho, nu = ctx->nu;
    long nx = s->nx;
    long y, x, idx;
    double p_c, p_e, p_w, p_n, p_s;
    double uc, vc, dudx, dudy, dvdx, dvdy;

    for (y = y0 + ctx->y_off; y < y1 + ctx->y_off; y++) {
        for (x = 1; x < nx - 1; x++) {
            idx = c_idx(s, y, x);
            if (s->solid[idx]) {
                u_dest[idx] = 0.0;
                v_dest[idx] = 0.0;
                continue;
            }
            p_c = p[idx];
            p_e = s->solid[idx + 1] ? p_c : p[idx + 1];
            p_w = s->solid[idx - 1] ? p_c : p[idx - 1];
            p_n = s->solid[idx + nx] ? p_c : p[idx + nx];
            p_s = s->solid[idx - nx] ? p_c : p[idx - nx];

            uc = u_src[idx];
            vc = v_src[idx];
            if (uc >= 0.0) {
                dudx = (u_src[idx] - u_src[idx - 1]) / dx;
                dvdx = (v_src[idx] - v_src[idx - 1]) / dx;
            } else {
                dudx = (u_src[idx + 1] - u_src[idx]) / dx;
                dvdx = (v_src[idx + 1] - v_src[idx]) / dx;
            }
            if (vc >= 0.0) {
                dudy = (u_src[idx] - u_src[idx - nx]) / dy;
                dvdy = (v_src[idx] - v_src[idx - nx]) / dy;
            } else {
                dudy = (u_src[idx + nx] - u_src[idx]) / dy;
                dvdy = (v_src[idx + nx] - v_src[idx]) / dy;
            }

            u_dest[idx] = (
                u_src[idx]
                - dt * (uc * dudx + vc * dudy)
                - dt / (2.0 * rho * dx) * (p_e - p_w)
                + nu * (
                    dt / (dx * dx) *
                    (u_src[idx + 1] - 2.0 * u_src[idx] + u_src[idx - 1])
                    +
                    dt / (dy * dy) *
                    (u_src[idx + nx] - 2.0 * u_src[idx] + u_src[idx - nx])
                )
            );
            v_dest[idx] = (
                v_src[idx]
                - dt * (uc * dvdx + vc * dvdy)
                - dt / (2.0 * rho * dy) * (p_n - p_s)
                + nu * (
                    dt / (dx * dx) *
                    (v_src[idx + 1] - 2.0 * v_src[idx] + v_src[idx - 1])
                    +
                    dt / (dy * dy) *
                    (v_src[idx + nx] - 2.0 * v_src[idx] + v_src[idx - nx])
                )
            );

            /* (handler interior pass is in Cython, after this
             * loop -- the C code is pure-fluid) */
        }
    }
}

static void c_update_momentum(solver_c_t *s) {
    long src = s->uvk;
    long dest = src ^ 1;
    momentum_ctx_t ctx;
    ctx.s = s;
    ctx.u_src = s->u[src];
    ctx.v_src = s->v[src];
    ctx.u_dest = s->u[dest];
    ctx.v_dest = s->v[dest];
    ctx.p = s->p[s->pk];
    ctx.dt = s->dt;
    ctx.dx = s->dx;
    ctx.dy = s->dy;
    ctx.rho = s->rho;
    ctx.nu = s->nu;
    ctx.y_off = 1;             /* rows 1..ny-2 */
    ff_pool_for(s->pool, s->ny - 2, momentum_rows, &ctx);
    /* no uvk flip here: the Cython wrapper flips it after
     * the handler interior pass (so the handler still reads
     * the source buffer) */
}

/* ------------------------------------------------------------------ */
/* Boundary / enforce / step_once                                      */
/* ------------------------------------------------------------------ */

static void c_clamp_momentum_boundary(solver_c_t *s) {
    long nx = s->nx, ny = s->ny;
    long last_row = (ny - 1) * nx;
    long n = nx > ny ? nx : ny;
    long i, left, right;
    double *u = s->u[s->uvk];
    double *v = s->v[s->uvk];

    /* flow_mode == 2: the domain edges are not forced to
     * any condition (they evolve via the Poisson stencil
     * and the handler forces). Solids are still enforced
     * (in c_enforce_solids, separately). */
    if (s->flow_mode == 2)
        return;

    if (s->flow_mode == 1) {
        for (i = 0; i < ny; i++) {
            left = i * nx;
            right = left + nx - 1;
            u[left] = s->inflow_u;
            v[left] = 0.0;
            u[right] = u[right - 1];
            v[right] = v[right - 1];
        }
        for (i = 0; i < nx; i++) {
            u[i] = 0.0;
            v[i] = 0.0;
            u[last_row + i] = 0.0;
            v[last_row + i] = 0.0;
        }
        return;
    }

    for (i = 0; i < n; i++) {
        if (i < nx) {
            u[i] = 0.0;
            v[i] = 0.0;
            v[last_row + i] = 0.0;
        }
        if (i < ny - 1) {
            left = i * nx;
            right = left + nx - 1;
            u[left] = 0.0;
            u[right] = 0.0;
            v[left] = 0.0;
            v[right] = 0.0;
        }
    }
}

/* Boundary forces and pressure forces are Cython (they call
 * typed handler methods); the C code is pure-fluid. */

typedef struct {
    solver_c_t *s;
    double *u, *v;
} enforce_ctx_t;

static void enforce_rows(void *vctx, long y0, long y1, int id) {
    enforce_ctx_t *ctx = (enforce_ctx_t *)vctx;
    (void)id;
    solver_c_t *s = ctx->s;
    long y, x, idx;
    for (y = y0; y < y1; y++) {
        for (x = 0; x < s->nx; x++) {
            idx = c_idx(s, y, x);
            if (s->solid[idx]) {
                ctx->u[idx] = 0.0;
                ctx->v[idx] = 0.0;
            }
        }
    }
}

static void c_enforce_solids(solver_c_t *s) {
    enforce_ctx_t ctx;
    ctx.s = s;
    ctx.u = s->u[s->uvk];
    ctx.v = s->v[s->uvk];
    ff_pool_for(s->pool, s->ny, enforce_rows, &ctx);
}

/* Exposed sub-steps (the Cython wrapper orchestrates them
 * with the handler passes interleaved). */
void solver_c_poisson(solver_c_t *s) {
    c_pressure_poisson(s);
}

void solver_c_momentum(solver_c_t *s) {
    c_update_momentum(s);
}

void solver_c_clamp(solver_c_t *s) {
    c_clamp_momentum_boundary(s);
}

void solver_c_enforce(solver_c_t *s) {
    c_enforce_solids(s);
}
