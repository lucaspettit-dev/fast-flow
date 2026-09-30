# fast-flow

Fast 2D incompressible flow solver for rapid design iteration.

## Status

v0.2.0 is a work-in-progress Cython rewrite of the solver core.
Uniform (obstacle-free) flow is stable, but **flow around obstacles is
currently numerically unstable** — velocities blow up after a few dozen
steps. Not recommended for real use yet.

## How it works

Jos Stam's **"Stable Fluids"** (1999) operator splitting, compiled with
Cython so every timestep runs as native C loops. Per step:

1. implicit velocity diffusion (Jacobi),
2. pressure projection (iterative Poisson solve → divergence-free),
3. semi-Lagrangian self-advection of velocity,
4. projection again, then per dye source: inject → diffuse → advect.

Boundary conditions: uniform inflow on the left, zero-gradient outflow on
the right, no-slip walls top/bottom, no-slip inside obstacle polygons.

All fields stay in C-typed arrays; Python only orchestrates. Rendering is
**lazy** — `run()` never pays for pixels; the `render_*` methods compute
their image on demand (cached until the next step) and return **read-only**
NumPy arrays:

- `solver.speed` → `(ny, nx)` float64 speed magnitude
- `solver.render_density()` → `(ny, nx, 3)` uint8 RGB composite of all dye
  sources, each tinted with its own color
- `solver.render_direction()` → `(ny, nx, 3)` uint8 RGB flow-direction
  color wheel (red = up, cyan = down; brightness scales with speed)

## Install

```bash
pip install -e .   # compiles the Cython core
```

Requires a C compiler, Cython ≥ 3, and NumPy.

## Quickstart

```python
from fast_flow import FlowSolver

solver = FlowSolver(nx=256, ny=128)
solver.add_obstacle([(0.5, 0.4), (0.6, 0.4), (0.6, 0.6), (0.5, 0.6)])
solver.add_density_source(0.3, 0.5, radius=0.05, color=(1.0, 0.0, 0.0))
solver.run(500)
speed = solver.speed            # read-only (128, 256) float64
dye = solver.render_density()   # read-only (128, 256, 3) uint8
```

## Development

```bash
pip install pytest
pytest
```

## License

MIT — see [LICENSE](LICENSE).
