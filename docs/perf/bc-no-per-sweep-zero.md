# Idea 7: BCs — drop per-sweep full-domain solid-zeroing

Branch: `perf/bc-no-per-sweep-zero` (off main @ 826e795).
File changed: `src/fast_flow/core/flow.pyx`
(`pressure_set_boundry_conditions`, `enforce_solids`).

## Change

`pressure_set_boundry_conditions` used to zero pressure at all
solid cells on every one of the 50 Jacobi sweeps per step — a
full-domain pass per sweep. Removed: the Poisson sweep never
writes solid centers (early return), the stencil substitutes
the fluid cell's own pressure for solid neighbours, mean-pinning
skips solids, `get_p` returns 0 for solids, and no reader ever
uses pressure at a solid cell mid-step. `enforce_solids` now
also zeroes solid pressure once per step, preserving the
final-field invariant (solids exactly 0 pressure).

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After, run 1 | 63.26 | 2.53 |
| After, run 2 | 58.25 | 2.75 |

**Neutral** — the two runs bracket the baseline; average is
within run noise.

## Why it didn't help (the miscalculation)

The reasoning was pass-counting: 50 BC passes vs 50 Poisson
sweeps looks like doubling the pressure cost. But a BC pass
is ~1 load + 1 branch per cell, while a Poisson sweep is a
5-point stencil with ~30 flops + 6 index multiplies + branches
per cell. The BC pass is ~10x cheaper per cell, so removing
it saves well under 1% overall — invisible next to run noise.

## Parity (vs `ff_ehd_test/perf_ref/`)

Bitwise identical (worst rel diff 0.0) on both runs.

## Verdict

Neutral — not a speedup. Kept on this branch as a cleanup
(the redundant pass is gone, the invariant is preserved and
proven by bitwise parity), but not recommended on performance
grounds. The lesson: count per-cell cost, not passes.
