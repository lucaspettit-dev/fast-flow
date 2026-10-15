# Idea 13: Sparse ion transport — FAILED (field is dense)

Branch: `perf/sparse-ion-transport` (off main @ 826e795).
**Do not merge.** The idea is exact but slower.
This file documents why, so the experiment is not repeated.

## Change

Transport (the ~82-subcycle flux loop, ~86% of step time)
went block-sparse: the domain is tiled into 16x16 blocks;
a block is active when any cell holds ions (or is in the
ionization zone). Each subcycle processes active blocks
plus a 1-cell halo (CFL <= 0.8 moves density at most one
cell per subcycle, so the halo captures all inflow exactly
and mass is conserved). Halo cells that gain density
activate their block for the next subcycle. A per-subcycle
`seen` stamp (monotonic across advances) ensures each cell
is done at most once despite overlapping halos. The active
set is rebuilt each advance (fused into the velocity pass:
start from zone blocks, OR in blocks holding density).
Per-cell update extracted verbatim into `_transport_cell`
(single call site, `cdef inline`).

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After, run 1 | 62.88 | 2.54 |
| After, run 2 | 62.48 | 2.56 |

**Slower (~5%).** Confirmed on two runs.

## Why it fails: the density field is numerically dense

After 1 step, density is nonzero in 0.9% of cells (8% of
blocks) — sparse would win big. But diffusion + transport
spread tiny-but-nonzero density everywhere within ~10
steps: after 10 steps, density is nonzero in **75% of
cells / 79% of blocks**, and it stays there (steady
state). So the sparse loop processes ~79% of blocks plus
a ~12% halo, plus the block-loop / seen-stamp /
activation overhead — more work than the dense loop.
The sparsity the idea assumed does not exist: the cloud
is physically concentrated but numerically dense (the
tiny tail values are nonzero, so `d > 0` is true almost
everywhere). Same phenomenon that limited Idea 8b's
empty-cell early-out.

## Parity (vs `ff_ehd_test/perf_ref/`)

Bitwise identical (worst rel diff 0.0) on both runs —
the sparse logic is exactly correct (mass conserved,
same results). The failure is performance only.

## Verdict

Abandon. Exact but slower, because the field is dense.
A thresholded variant (treat `d < eps` as empty) could
recover sparsity, but it is approximate (drops mass) and
would need its own parity analysis against the 1e-12 bar
— noted here as a possible follow-up, not pursued on
this branch.
