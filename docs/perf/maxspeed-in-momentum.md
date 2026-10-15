# Idea 10: Fuse _max_speed into update_momentum

Branch: `perf/maxspeed-in-momentum` (off main @ 826e795).
File changed: `src/fast_flow/core/flow.pyx`,
`src/fast_flow/core/flow.pxd`.

## Change

`_max_speed()` was a full-domain pass per step (called via
`_stable_dt_limit` -> `_choose_dt` to pick dt). Now:
- `update_momentum()` tracks the component-wise max
  `max(|u|, |v|)` (the same quantity `_max_speed` computed)
  into `cached_max_speed`, AFTER the drag (the drag can
  accelerate a cell past its momentum value).
- `_stable_dt_limit()` takes `max(cached interior max,
  perimeter scan)`. Clamp, boundary forces, and
  enforce_solids only touch boundary/solid cells after
  momentum, so the interior cache stays exact and only the
  boundary needs re-scanning — O(perimeter) instead of
  O(N). The existing `inflow_u` floor is kept.
- `_max_speed()` deleted (subsumed).

Exactness note: the first step's cache is 0 (zero-init),
which matches `_max_speed` on the all-zero initial field.

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 60.12 | 2.66 |

**Neutral** (within run noise).

## Why ~0%: one pass out of ~86

The removed pass is 1 of ~86 domain passes per step —
~1%, below the run-noise floor.

## Parity (vs `ff_ehd_test/perf_ref/`)

Bitwise identical (worst rel diff 0.0) — the cached +
perimeter max equals the old full-scan max exactly, so
every dt choice is identical.

## Verdict

Neutral — exact, but no measurable speedup. Keep as a
cleanup (one less full-domain pass per step), not as a
speedup.
