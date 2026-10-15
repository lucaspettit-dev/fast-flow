# New idea: transport CFL 0.8 -> 1.0 — +19% but CHANGES results

Branch: `perf/cfl-1.0` (off main @ 826e795).
**Do not merge without an explicit numerical decision.**
This is a numerical change, not a pure optimization.

## Change

One line: the transport subcycle count went from
`nsub = dt * rate / 0.8 + 1` to `dt * rate / 1.0 + 1`.
Donor-cell upwind is stable up to CFL = 1, so 1.0
cuts ~20% of the subcycles (82 -> ~66).

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After (CFL 1.0) | 49.68 | 3.22 |

**+19%.**

## Parity (vs `ff_ehd_test/perf_ref/`)

**FAIL** (worst rel diff 4.2e-02, tol 1e-12):
u 3.8e-03, v 4.2e-02, p 1.5e-03. A 4% change in v
is not functional parity.

## Why the results change

First-order donor-cell upwind has numerical diffusion
proportional to (1 - CFL) * v * dx / 2. At CFL = 0.8
the scheme is diffusive; at CFL = 1.0 it is an exact
shift (zero numerical diffusion). So CFL = 1.0 is
arguably *more* accurate (closer to the true PDE) —
but it is a different discretization, and the ion
distribution (hence the ion-wind velocity) shifts by
percent-level amounts. The 1e-12 parity bar cannot
hold across a discretization change.

## Verdict

Needs a user decision. If the 4% shift is acceptable
(CFL = 1.0 is the less-diffusive, more-accurate
scheme), this is a real +19% win that stacks with the
other winners. If the baseline discretization must be
preserved exactly, this idea is abandoned.
