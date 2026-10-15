# Idea 4: idx -> running base+offset in hot loops

Branch: `perf/idx-base` (off main @ 826e795).
File changed: `src/fast_flow/core/flow.pyx`.

## Change

`pressure_poisson_step` called `idx()` 8x per cell (center
+ neighbor solid checks and reads) and
`build_up_pressure_step` 9x; the mean-pinning loops called
`idx()` for the solid check while already computing
`base + ii` for the pressure read. Now all three use one
`idx()` per cell plus `idx +/- 1` / `idx +/- nx` offsets
(pinning: `solid[base + ii]`). `update_momentum` already
used offsets; the BC loops (O(perimeter)) and
`add_obstacle` (not hot) are untouched.

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 57.73 | 2.77 |

**Small positive (+2.6%)**, at the top of the run-noise
band (baseline runs span ~2.5–2.8). The compiler had
apparently not fully CSE'd the repeated `y*nx + x`
multiplies, so the manual offsets help a little.

## Parity (vs `ff_ehd_test/perf_ref/`)

Bitwise identical (worst rel diff 0.0) — the offsets
compute the same indices.

## Verdict

Keep. Strictly fewer multiplies for identical results;
the gain is small but real (and the code is simpler to
read — one index derivation per cell).
