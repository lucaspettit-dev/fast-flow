# Idea 2: Fuse update_momentum into the final pressure loop — FAILED

Branch: `perf/fuse-momentum-pressure` (off main @ 826e795).
**Do not merge.** The idea fails parity and shows no speedup.
This file documents why, so the experiment is not repeated.

## Change (as implemented)

- `update_momentum()` split into per-cell `momentum_step()`
  (pressure gradient + upwind convection + viscosity, no drag)
  plus a separate `apply_drag_forces()` pass.
- The last Jacobi sweep runs `pressure_poisson_fused_last()`:
  after the sweep finishes row y+1, momentum for row y runs
  (lag one row + one column, so every pressure it reads is
  this sweep's final value). Per-row BC copies run just
  before each momentum row; the full BC pass + a drain for
  the last interior row finish the job. Then pk and uvk flip.
- Drag stays a separate pass after pinning (it reads solver
  pressure for resistance, which must be post-pinning).

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 59.17 | 2.70 |

**No speedup.** The momentum pass is ~1% of runtime; the
new drag pass + per-row BCs + call overhead eat the saving.

## Parity (vs `ff_ehd_test/perf_ref/`)

**FAIL** — worst rel diff 2.4e-05 (u/v), 3.7e-06 (p),
tolerance 1e-12. Step 1 is bitwise identical; step 2
differs at ~1e-12; the difference amplifies ~10x every
~10 steps.

## Root cause: pinning breaks the fusion's exactness

The original momentum reads pressure AFTER mean-pinning;
the fused momentum reads it BEFORE pinning (pinning needs
the sweep's output, so it must come after). In exact math
the pressure gradient is pinning-invariant (a constant
shift cancels in `p_e - p_w`), but in floating point
`(p_e - mean) - (p_w - mean) != p_e - p_w` — a ~1 ulp
difference per gradient. There is no exact fix: making
momentum read post-pinning p requires running it after
pinning, which un-fuses it; splitting the gradient term
into a post-pinning pass re-associates the sum and is
also not bitwise. The fusion is mathematically equivalent
but not bitwise, and the rounding difference amplifies.

## Verdict

Abandon. No speedup, and the parity failure is structural
(pre- vs post-pinning gradients), not a bug that can be
fixed while keeping the fusion. The per-cell formula was
verified verbatim-identical to the original; the 1-step,
2-step, and no-handler inflow=5 bisection runs isolate the
cause to the pinning rounding (details in the experiment
log).
