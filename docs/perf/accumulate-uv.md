# Idea 11: Single-call accumulate_uv

Branch: `perf/accumulate-uv` (off main @ 826e795).
Files changed: `src/fast_flow/core/handlers/forcehandler.pxd`,
`src/fast_flow/core/handlers/forcehandler.pyx`,
`src/fast_flow/core/electrostaticforcehandler.pyx`,
`src/fast_flow/core/flow.pyx`.

## Change

`apply_uv_force` called `accumulate()`, which called
`get_u` and `get_v` separately — and the electrostatic
handler's `get_u`/`get_v` each recomputed the identical
`_drag` fraction (density load, solver-pressure read,
resistance clamp, fraction). Now:
- New `accumulate_uv()` on the base handler (default:
  get_u + get_v, so subclasses overriding only get_u/get_v
  keep working). The solver calls it instead of
  `accumulate()` (which stays for API compat).
- Electrostatic override: `_drag` split into `_drag_frac`
  (the shared fraction) + `_drag`; `accumulate_uv`
  computes the fraction once and applies it to both
  components.
- ConstantVelocity override: the fused direction logic
  (exactly one of the four branches can fire).

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 58.79 | 2.72 |

**Neutral** (within run noise).

## Why ~0%: the drag is one small pass

The drag runs once per cell per step inside
update_momentum; halving its per-cell math saves a
fraction of one pass out of ~86 — below noise.

## Parity (vs `ff_ehd_test/perf_ref/`)

PASS — worst rel diff 1.8e-14 (tol 1e-12). Not bitwise:
the fused path compiles the fraction computation in a
slightly different arrangement (last-bit codegen), but
the math is identical.

## Verdict

Neutral — correct, but no measurable speedup. Keep as a
cleanup (one call, one fraction computation), not as a
speedup.
