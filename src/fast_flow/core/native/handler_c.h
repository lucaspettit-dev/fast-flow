/* handler_c.h — native-C electrostatic handler (ion advance).
 *
 * perf/native-c-handler: the ElectrostaticForceHandler's
 * _advance_ion_state in plain C, on the solver's thread
 * pool (re-used within the solver step -- the pool is
 * borrowed from the solver, it is not owned here).
 *
 * The C struct references the handler's buffers (same
 * memory the Cython accessors view) and the solver's
 * per-step inputs (passed to the advance each step, since
 * the solver's current buffers/dt change per step). The
 * per-cell accessors (get_u/get_v/get_p, the drag) stay in
 * Cython; only the advance (the expensive part) is here.
 *
 * No Python, no GIL in this file.
 */

#ifndef FAST_FLOW_HANDLER_C_H
#define FAST_FLOW_HANDLER_C_H

#include "solver_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct handler_c handler_c_t;

/* Create the C handler state. All buffers are owned by the
 * Cython side (same memory the Cython accessors view);
 * this struct references them. pool is borrowed (the
 * solver's); if NULL, a private pool is created and owned
 * here (standalone use). d0/d1, p0/p1 are the
 * double-buffered density/pressure; u0/u1, v0/v1 the
 * double-buffered ion velocity (uvk selects the current);
 * cu0/cu1, cv0/cv1 the Coulomb field (ck selects the
 * current); ion_zone is the generation mask; solids is
 * the handler's own electrode mask (may be NULL).
 * dn0/dn1 are the double-buffered O2- (negative-corona)
 * density, un0/un1, vn0/vn1 its velocity (same uvk),
 * ion_zone_neg its generation mask (air cells bordering
 * a negative electrode that is in corona). */
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
                              double *dn0, double *dn1,
                              double *un0, double *un1,
                              double *vn0, double *vn1,
                              unsigned char *ion_zone_neg,
                              ff_pool_t *pool);
void handler_c_destroy(handler_c_t *h);

/* The density buffer index after the last advance (the
 * Cython side syncs its dk from this). */
long handler_c_dk(const handler_c_t *h);
/* The O2- density buffer index after the last advance. */
long handler_c_dnk(const handler_c_t *h);
/* Keep the C struct's pk in sync with the Cython side's
 * (get_p flips pk when the pass wraps; the advance reads
 * it to find the pressure back slot). */
void handler_c_set_pk(handler_c_t *h, long pk);

/* Advance the ion state by dt (the solver's current dt;
 * dt <= 0 falls back to dt_base, as in the .pyx).
 * au/av are the solver's current air velocity buffers;
 * solid is the solver's solid mask. Called once per
 * solver step (from get_p's first call), nogil, on the
 * solver's pool. */
void handler_c_advance(handler_c_t *h,
                       const double *au, const double *av,
                       const unsigned char *solid,
                       double dt, double dt_base);

#ifdef __cplusplus
}
#endif

#endif /* FAST_FLOW_HANDLER_C_H */
