# Idea 5: Fuse mean-pinning sum into last Jacobi sweep

Branch: `perf/mean-in-sweep` (off main @ 826e795).
File changed: `src/fast_flow/core/flow.pyx`
(`pressure_poisson`).

## Change

Mean-pinning was two passes (sum, then subtract). The sum
is now fused into the last Jacobi sweep (throughflow only,
where pinning runs): interior cells accumulate as the sweep
writes them (BCs only touch boundary cells, so interior
values are final at sweep time); the perimeter is summed
after the BC pass, in row-major order. The subtract pass is
unchanged. `nit <= 1` (no q-loop) falls back to the full
sum pass.

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 59.21 | 2.70 |

**Neutral** — no measurable gain.

## Why ~0%: the sum pass was cache-hot

The removed pass is one memory-light pass (150k cells,
~1.2 MB of pressure + 150 KB of mask — cache-resident),
~0.05% of step time. Fusing it saves nothing measurable.

## Parity (vs `ff_ehd_test/perf_ref/`)

PASS — worst rel diff 1.5e-14 (tol 1e-12). Not bitwise:
the fusion reorders the sum (interior during the sweep,
then perimeter, vs the original fully row-major
interleave), and floating-point addition is order
dependent, so the mean differs by ~1 ulp. The physics is
identical; the rounding is not.

## Verdict

Neutral, and not recommended: it buys no speed and trades
bitwise-exact pinning for a reordered sum. The experiment
confirms the sum pass was never a bottleneck.
