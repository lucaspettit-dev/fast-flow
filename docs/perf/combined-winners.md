# Combined winners: 8a + 8b + 4 + 3 — 2.70 -> ~6.8 it/s

Branch: `perf/combined-winners` (off main @ 826e795).
Cherry-picks the four keepers, resolving two conflicts:
- 8a vs 8b: early-out first (empty cells skip the flag
  computation too), then the cached face-solid flags.
- 8a vs 3: 8a's cached flags + hoisted tau_dx/tau_dy with
  3's `fx = fx * (...)` form (not `*=` — Cython infers
  `*=` as a reduction).

## Benchmark (demo config, 0.05 s / 160 steps,
## OMP_NUM_THREADS=2)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After, run 1 | 23.90 | 6.70 |
| After, run 2 | 23.25 | 6.88 |

**~6.8 it/s, 2.5x.** Below the naive product
(2.70 x 2.14 x 1.11 x 1.03 x 1.13 = 7.46) because 8b's
early-out overlaps 8a (once 8a lowers the per-cell
cost, the work 8b skips is worth less) and 4's +2.6%
was inside the noise band.

## Parity (vs `ff_ehd_test/perf_ref/`)

PASS (worst rel diff 2.8e-14, tol 1e-12). The tiny
diff is 8a's last-bit reordering from the hoisted
divisions (tau/dx, tau/dy); the other three are
bitwise-exact.

## Verdict

Keep as the combined result. Still short of 60 it/s:
the remaining cost is the 82-subcycle ion transport,
which needs a different algorithm (fewer subcycles,
threshold sparsity, or a coarser ion grid), not more
constant-factor tuning.
