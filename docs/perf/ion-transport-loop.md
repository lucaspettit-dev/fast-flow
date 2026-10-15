# Idea 8a: Ion transport loop — hoist divisions, cache face-solid flags, inline solid check

Branch: `perf/ion-transport-loop` (off main @ 826e795).
File changed: `src/fast_flow/core/electrostaticforcehandler.pyx`
(`_advance_ion_state`, pass 2 — the subcycled flux transport).

## Why this loop

Phase-split profiling of the 0.05 s benchmark: the ion advance is
86.1% of step time, and inside it the transport loop runs
`nsub` (~80) full-domain sweeps per step, each doing ~8 solid
checks + 2 divisions per cell. It is the hottest loop in the
program by an order of magnitude.

## Changes (all in the transport loop; the math is unchanged)

1. **Hoist the divisions.** `val = sbuf - tau*((fx_e-fx_w)/dx +
   (fy_n-fy_s)/dy)` divided twice per cell per subcycle.
   Now `tau_dx = tau/dx`, `tau_dy = tau/dy` are computed once
   per advance and the update is two multiplies. Rounding
   differs in the last bit (see parity below); the math is
   identical.
2. **Cache the face-solid flags.** The old code checked each
   neighbour's solidity once for the face flux and again in the
   choking block — 8 checks per cell. Now `se/sw/sn/ss` are
   computed once and reused. Exactly equivalent (same values).
3. **`_solid_at` is `cdef inline`.** It was a real function call
   per check (~100M calls/step in this loop). Inlining removes
   the call overhead. Body unchanged.

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 27.71 | **5.77** |

**2.14x speedup.**

## Parity (vs `ff_ehd_test/perf_ref/`)

Worst relative diff 2.8e-14 (u), from the division-hoisting
reordering — far below the 1e-12 tolerance. PASS.

## Verdict

Keep. The biggest single win available at the Cython level.
The remaining ion cost is the subcycled sweeps themselves —
see ideas 8b (exact empty-cell early-out) and 13 (sparse
transport) for the next steps on this loop.
