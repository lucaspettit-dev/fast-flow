# fast-flow

Fast 2D incompressible flow solver for rapid design iteration.

> **Status: pre-alpha scaffold.** The package layout, metadata, CLI entry
> point, and CI are in place. The solver numerics are not implemented yet.

## Why

Existing Python flow solvers (e.g. phiFlow) are great for correctness but
too slow for interactive design loops. fast-flow aims to solve the same
incompressible Navier-Stokes equations -- semi-Lagrangian advection,
explicit diffusion, pressure projection on a staggered grid -- with
compiled hot loops so a screening run finishes in seconds.

## Install

Not on PyPI yet. For development:

```bash
pip install -e .
```

## Planned quickstart

```python
from fast_flow import FlowSolver

solver = FlowSolver(nx=256, ny=128)
solver.add_obstacle([(0.5, 0.4), (0.6, 0.4), (0.6, 0.6), (0.5, 0.6)])
solver.run(500)
speed = solver.speed  # (ny, nx) array
```

## Development

```bash
pip install pytest
pytest
```

## License

MIT -- see [LICENSE](LICENSE).
