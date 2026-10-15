# Idea 8b: Ion transport — exact empty-cell early-out

Branch: `perf/ion-empty-earlyout` (off main @ 826e795).
File changed: `src/fast_flow/core/electrostaticforcehandler.pyx`
(`_advance_ion_state`, pass 2 — the subcycled flux transport).

## Change

Before the face-flux work, if the cell and all four neighbours
hold exactly zero density, every face flux is velocity x 0 = 0
and the update is 0, so write `dbuf = 0` and skip the cell.
Conservative: any nonzero neighbour takes the full path.
(NaN would fail the `== 0.0` tests and also take the full path.)

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After | 53.16 | **3.01** |

**1.11x speedup.**

## Parity (vs `ff_ehd_test/perf_ref/`)

Bitwise identical (worst rel diff 0.0) — the early-out is
exactly equivalent, not an approximation.

## Verdict

Keep. A modest, exactly-equivalent win. The gain is small
because after 0.05 s the ion density is nonzero (if tiny)
over most of the domain — diffusion + transport spread it —
so the early-out fires for only part of the cells. The
sparse-transport idea (13) attacks the same observation
structurally instead of per-cell.
