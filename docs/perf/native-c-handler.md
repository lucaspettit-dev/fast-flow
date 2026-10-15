# Native-C handler (ion advance) — 2.70 -> ~6.5 it/s, bitwise

Branch: `perf/native-c-handler` (off
`perf/native-c-threadpool` @ b73062c, which is off
main @ 826e795).

## What was built

- `src/fast_flow/core/native/handler_c.c` (+
  `handler_c.h`): the ElectrostaticForceHandler's
  `_advance_ion_state` in plain C — ion velocity +
  transport rate, the subcycled donor-cell transport,
  generation, diffusion, ion pressure. The Cython
  advance is deleted (single source of truth).
- The handler's advance **re-uses the solver's thread
  pool** within the solver step: the pool object is
  created by the solver's C code and passed to the
  handler (via `solver_c_pool`); the handler stores
  only a borrowed pointer (it owns a private pool
  only in standalone use). The solver's C sub-steps
  and the handler's C advance share the one pool —
  sequential calls, barrier between each.
- Parallelization (all Jacobi, per-cell independent):
  velocity (parallel-for + a max reduction — max is
  exact in FP, so the reduction is bitwise-exact),
  each transport subcycle (parallel-for; subcycles
  stay sequential), generation / diffusion / pressure
  (parallel-for). The pool task signature gained a
  chunk id (`func(ctx, start, end, id)`) for the
  reduction slots.
- The per-cell accessors (get_u/get_v/get_p, the drag)
  stay in Cython; get_p calls the C advance on its
  first call of a pass, then telescopes as before.
  dk is synced from the C struct after the advance;
  pk is synced to the C struct when get_p flips it.

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| C solver only (prev branch) | 55.14 | 2.9 |
| + C handler, parity run 1 | 24.52 | 6.53 |
| + C handler, parity run 2 | 24.49 | 6.53 |
| + C handler, demo | — | 6.6 |

**~6.5 it/s (2.4x over main).** The handler was 86%
of step time; putting it on the 2-thread pool is most
of this gain.

## Parity (vs `ff_ehd_test/perf_ref/`)

**PASS, bitwise identical (0.0 diff)** — the C
handler port is exact. No `-ffast-math`;
expressions match the .pyx source.

## Verdict

Keep. The C solver + C handler + one shared pool is
the architecture. The remaining constant-factor wins
(Idea 8a/8b transport loop, 8b early-out) are not
yet applied to the C handler — a follow-up would
port them into handler_c.c.
