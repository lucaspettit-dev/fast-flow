# Idea 3: Native C + thread pool — threading kept (+13%), rewrite moot

Branch: `perf/openmp-threads` (off main @ 826e795).

## Change

**Native C rewrite: NOT done — moot.** Cython already
compiles the hot loops to C (nogil, typed, boundscheck
off). The measured bottleneck is arithmetic / memory
traffic, not Cython overhead, so a hand-written C
version would not be meaningfully faster. Rewriting
~2,000 lines of Cython into C buys nothing and risks
correctness.

**Thread pool: done — correct row-based OpenMP.**
`prange` over rows in: Poisson sweeps (first + q
sweeps), update_momentum, ion transport subcycles,
ion generation, ion diffusion, ion pressure. Rows are
independent (Jacobi / per-cell: each thread writes
disjoint rows, reads shared source buffers), so there
is no shared write. This is NOT the old 826e795 prange:
that parallelized over the sweep index, and threads
raced flipping the shared buffer index — its "2.8
it/s parallel" number was invalid. Row-parallelism has
no such race.

Cython notes: `cdef` is not allowed inside a prange
body or a `parallel()` block, and neither takes a
`private` clause in this Cython version — per-cell
temporaries are auto-privatized by Cython when assigned
with `=` inside the loop. The choking `fx *= ...` had
to become `fx = fx * (...)` because Cython infers `*=`
as a reduction (and forbids reading a reduction
variable in the loop body). setup.py re-adds
`-fopenmp` (compile + link; Darwin via `-lomp`).

The velocity pass stays serial (it computes the max
transport rate — a reduction). BCs, pinning, and the
handler-call passes stay serial.

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After, OMP_NUM_THREADS=2, run 1 | 52.67 | 3.04 |
| After, OMP_NUM_THREADS=2, run 2 | 52.11 | 3.07 |
| After, OMP_NUM_THREADS=1 | 62.96 | 2.54 |

**+13% with 2 threads.** With 1 thread the prange
overhead (parallel-region setup per subcycle / sweep,
~140 regions per step) costs ~6% vs serial.

**Caveat:** with `OMP_NUM_THREADS` unset, this VM
behaves like 1 thread (2.58 it/s, ≈ the 1-thread
number) — the gain needs `OMP_NUM_THREADS=2` here.
On the user's Mac, OpenMP's default is the machine's
core count, so no setting is needed there. Do not
hardcode the thread count (it varies by machine).

## Parity (vs `ff_ehd_test/perf_ref/`)

Bitwise identical (worst rel diff 0.0) on all runs —
the parallelism is correct, no race. (Parallel
floating-point is bitwise-identical here because each
cell's update is computed by exactly one thread from
the same source buffer; no reduction reordering.)

## Verdict

Keep — a real, if modest, +13%. The gain is limited
by 2 CPUs, the parallel-region overhead per subcycle /
sweep, the serial passes (velocity reduction, BCs,
pinning, handler-call passes), and the shared-L3
working set. Threading alone does not close the gap
to 60 it/s; it composes with the other winners (8a,
8b, 4) toward ~8-9 it/s combined.
