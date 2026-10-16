/* solver_c.h — native-C solver core + thread pool.
 *
 * perf/native-c-threadpool: the FlowSolverCore computation
 * (pressure Poisson, momentum, boundary conditions, enforce)
 * lives here in plain C, driven by a persistent pthread pool
 * sized to the machine's online CPUs. The Cython side owns
 * the buffers (same memory the Python accessors view),
 * orchestrates the step (C sub-steps + the handler passes,
 * all nogil), and reacquires the GIL only afterwards
 * (renderer hook + read-buffer sync).
 *
 * No Python, no GIL in this file. The C code is pure-fluid:
 * the handler passes live in Cython (they call typed
 * handler methods, which the C code cannot do without a
 * void* -> Python-object cast that needs the GIL). The
 * renderer hook stays in Cython (it needs the GIL) and is
 * called by the wrapper after the C steps.
 */

#ifndef FAST_FLOW_SOLVER_C_H
#define FAST_FLOW_SOLVER_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A parallel-for task: func(ctx, start, end, id) over a row
 * range. id is the participant's chunk id (0 = the calling
 * thread, 1..nworkers = workers); tasks that need a
 * per-participant reduction (e.g. a max) use it to index a
 * per-participant slot. */
typedef void (*ff_task_fn)(void *ctx, long start, long end, int id);

/* ------------------------------------------------------------------ */
/* Thread pool                                                         */
/* ------------------------------------------------------------------ */

typedef struct ff_pool ff_pool_t;

/* Online CPUs (>= 1). Used for the pool size: the pool fetches
 * the max available threads on the system. */
int ff_pool_max_threads(void);

/* Create a pool with nthreads participants (the calling thread
 * is one of them). nthreads <= 0 means ff_pool_max_threads(). */
ff_pool_t *ff_pool_create(int nthreads);
void ff_pool_destroy(ff_pool_t *pool);
/* Number of participants in the pool (for per-participant
 * reduction slots). */
int ff_pool_nthreads(const ff_pool_t *pool);

/* Run func(ctx, start, end) over [0, n), split across the pool.
 * The calling thread participates. Blocks until all chunks are
 * done (barrier). n <= 0 runs nothing. */
void ff_pool_for(ff_pool_t *pool, long n, ff_task_fn func, void *ctx);

/* ------------------------------------------------------------------ */
/* Solver                                                              */
/* ------------------------------------------------------------------ */

typedef struct solver_c solver_c_t;

/* Create the C solver state. Buffers are owned by the Cython
 * side (same memory the Python accessors view); this struct
 * references them, it does not free them. The pool is owned
 * by this struct (created here, destroyed in solver_c_destroy).
 * u0/u1, v0/v1, p0/p1 are the double-buffered fields; b is
 * the Poisson RHS; solid is the uint8 mask. */
solver_c_t *solver_c_create(long nx, long ny,
                            double dx, double dy,
                            double lx, double ly,
                            double rho, double nu,
                            double dt_base, long nit, long flow_mode,
                            double inflow_u,
                            double *u0, double *u1,
                            double *v0, double *v1,
                            double *p0, double *p1,
                            double *b, unsigned char *solid);
void solver_c_destroy(solver_c_t *s);

void solver_c_set_inflow(solver_c_t *s, double inflow_u);
void solver_c_set_dt(solver_c_t *s, double dt);
void solver_c_set_uvk(solver_c_t *s, long uvk);
void solver_c_set_pk(solver_c_t *s, long pk);
/* When nonzero, the solver does not enforce domain
 * boundaries: the velocity clamp and the pressure BCs
 * are skipped (the edges evolve only via the Poisson
 * stencil and the handler forces). Solids are still
 * enforced (they are obstacles, not boundaries). */
void solver_c_set_no_boundaries(solver_c_t *s, int no_boundaries);
/* Buffer indices (the Cython side syncs its pk/uvk from these). */
long solver_c_pk(const solver_c_t *s);
long solver_c_uvk(const solver_c_t *s);
/* The thread pool (borrowed; the solver owns it). The
 * handler's C advance re-uses this pool within the solver
 * step. */
ff_pool_t *solver_c_pool(solver_c_t *s);

/* Sub-steps (dt must be set via solver_c_set_dt before the
 * Poisson). The Cython wrapper calls them in _step_once
 * order, with the handler passes interleaved, all nogil:
 *   poisson -> momentum -> [handler interior] -> flip uvk
 *     -> clamp -> [handler boundary] -> [handler pressure]
 *     -> enforce
 * The momentum does NOT flip uvk (the wrapper flips it
 * after the handler interior pass, so the handler still
 * reads the source buffer). The Poisson flips pk internally
 * (the wrapper syncs its pk from solver_c_pk afterwards). */
void solver_c_poisson(solver_c_t *s);
void solver_c_momentum(solver_c_t *s);
void solver_c_clamp(solver_c_t *s);
void solver_c_enforce(solver_c_t *s);

#ifdef __cplusplus
}
#endif

#endif /* FAST_FLOW_SOLVER_C_H */
