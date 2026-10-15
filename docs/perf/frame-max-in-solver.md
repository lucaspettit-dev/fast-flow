# Idea 1: frame_max_mag into the solver velocity loop

Branch: `perf/frame-max-in-solver` (off main @ 826e795).
Files changed: `src/fast_flow/core/flow.pxd`,
`src/fast_flow/core/flow.pyx`,
`src/fast_flow/core/renderer.pxd`,
`src/fast_flow/core/renderer.pyx`.

## Change

`draw_frame()` used to re-scan the whole field to find the
frame's max velocity magnitude before the color pass.
Now the solver tracks it while the velocities are in hand:
- `FlowSolverCore.max_velocity_mag`, set in
  `update_momentum()` (interior cells, `fast_hypot` per cell)
  and folded with the perimeter in `_track_boundary_max()`
  (called from `clamp_momentum_boundary()` — covers inflow,
  copies, lid).
- The `_step_once` hook passes it to
  `render_frame(dt, n, u, v, max_mag)`; `draw_frame()` uses
  it when > 0 and falls back to scanning when 0 (the
  `refresh()` path, which has no solver step).
- Known gap (documented in code): handler-driven boundary
  velocities from `apply_boundary_forces` (runs after clamp)
  are not tracked. In the EHD demo the boundary drag is ~0,
  so the tracked max is exact here; a handler that drove a
  boundary cell past the interior max would clip the
  renderer's scale there.

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 58.27 | 2.75 |

**Neutral** — within run noise.

## Why ~0%: the max pass only runs on due frames

The fps gate means `draw_frame()` runs ~2x per 0.05 s run
(~31x per 1.0 s run), not every step. The eliminated pass
is ~2 x 150k hypot computations against ~160 steps x ~8M
cell-updates — ~0.02% of the run. The idea is
architecturally right (don't recompute what's known) but
performance-irrelevant for this benchmark.

## Parity

- u/v/p vs `ff_ehd_test/perf_ref/`: bitwise identical.
- Rendered PNG vs main's: pixel-identical (same colors),
  proving the tracked max equals the scanned max on this
  benchmark.

## Verdict

Neutral on speed; correct. Keep as a cleanup (one less
redundant scan, renderer API now carries the scale
explicitly), not as a speedup.
