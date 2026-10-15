# Idea 14: Fuse diffusion + pressure passes (ion handler)

Branch: `perf/fuse-diffusion-pressure` (off main @ 826e795).
File changed: `src/fast_flow/core/electrostaticforcehandler.pyx`
(`_advance_ion_state`, passes 4+5).

## Change

The diffusion pass (density stencil into the other buffer)
and the pressure pass (ion pressure = coupling * d * |E|^2)
ran as two full-domain passes. Fused into one: the pressure
at a cell needs only the DIFFUSED density at that same cell
(no stencil), so it uses the just-computed value. Also
saves the pressure pass's own `_solid_at` per cell.

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 60.43 | 2.65 |

**Neutral** (within run noise).

## Why ~0%: one pass out of ~86

The ion advance does ~82 subcycled transport passes plus
velocity, generation, diffusion, and pressure passes.
Fusing diffusion+pressure removes 1 of ~86 passes (~1%
of the ion advance, ~1% overall) — below the run-to-run
noise floor (~5%), so no gain is measurable.

## Parity (vs `ff_ehd_test/perf_ref/`)

Bitwise identical (worst rel diff 0.0) — the pressure
input is the exact value diffusion wrote.

## Verdict

Neutral — exact, but no measurable speedup. Keep as a
cleanup (one less pass, one less solid check per cell),
not as a speedup.
