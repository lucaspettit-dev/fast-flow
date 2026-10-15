# Native-C solver + thread pool — 2.70 -> ~2.9 it/s, bitwise

Branch: `perf/native-c-threadpool` (off main @ 826e795).

## What was built

- `src/fast_flow/core/native/solver_c.c` (+ `solver_c.h`):
  the FlowSolverCore computation in plain C — pressure
  Poisson (build-up + Jacobi sweeps + BCs + pinning),
  momentum (upwind + pressure + viscosity), clamp,
  enforce — driven by a persistent pthread pool.
- The pool fetches the max available threads on the
  system (`sysconf(_SC_NPROCESSORS_ONLN)`, 2 on this VM).
  Workers persist (no per-region fork/join); the calling
  thread participates as one worker; a generation counter
  + condvars form the barrier.
- The solver's `step()` runs entirely nogil (dt selection
  + the C sub-steps + the handler passes). The GIL is
  reacquired only afterwards: the renderer hook (needs
  Python) and the read-buffer sync (pk/uvk). The read
  buffer is live — the C code writes the same memory the
  Python accessors view (no copy), and the GIL
  reacquire is the synchronization point.

## Split (the key design decision)

The C code is pure-fluid. The handler passes stay in
Cython nogil (typed method calls on `self.force`), with
the wrapper orchestrating:
Poisson -> momentum -> handler interior -> flip uvk ->
clamp -> handler boundary -> handler pressure (serial,
exactly-once) -> enforce. The momentum does not flip
uvk; the wrapper flips it after the handler interior
pass, so the handler still reads the source buffer.

An earlier attempt passed the handler to C via function
pointers (`cdef api` shims) — abandoned: `cdef api`
cannot take a private extension type, and making the
class public needs an object-struct name. The split
avoids all of it (no void* -> object cast nogil) at the
cost of one extra interior pass (~1%).

## Benchmark (demo config, 0.05 s / 160 steps)

| | wall (s) | it/s |
|---|---|---|
| Before (main @ 826e795) | 59.34 | 2.70 |
| After, parity run 1 | 54.67 | 2.93 |
| After, parity run 2 | 55.44 | 2.89 |
| After, demo | 55.14 | 2.9 |

**~2.9 it/s (+8%).** The gain is small because the
solver is only ~14% of step time (the handler's ion
advance is 86% and stays Cython); the pool parallelizes
the 14%.

## Parity (vs `ff_ehd_test/perf_ref/`)

**PASS, bitwise identical (0.0 diff)** — the C port is
exact. No `-ffast-math`; expressions are written in the
same form as the .pyx source.

## Verdict

Keep as the native-C + thread-pool architecture. The
next step for a bigger gain is porting the handler's
ion advance to C (the 86%) — a larger port, not done
here. The C solver is the foundation for it.
