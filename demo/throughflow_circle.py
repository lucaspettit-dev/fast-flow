"""Throughflow past a circular obstacle (fast-flow demo).

The fast-flow analogue of the ehd-flow "infinite flow" setup:
uniform inflow on the left, zero-gradient outflow on the right,
no-slip top/bottom walls, and a solid circular obstacle as a
no-slip polygon.  Renders the speed field after a set amount of
simulated time (dynamic dt -- see FlowSolver.run).

The plot extent and figure shape are derived from the solver's
lx/ly, so changing --lx/--ly (or nx/ny resolution) is actually
reflected in the graph.  nx/ny are grid *resolution*; lx/ly are
the physical domain size the axes show.

Defaults are the 8x2 channel scenario: ny=128, nx derived as
int(lx/ly) * ny (= 512), inflow 5.0, nominal dt 0.0001, run for
1.0 s of simulated time (the equivalent of 10000 steps x 0.0001).

Examples:
    python demo/throughflow_circle.py
    python demo/throughflow_circle.py --nx 1000 --inflow 3.0
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from fast_flow import FlowSolver


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=None,
                   help="default: int(lx/ly) * ny")
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--lx", type=float, default=8.0)
    p.add_argument("--ly", type=float, default=2.0)
    p.add_argument("--inflow", type=float, default=5.0)
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.0001,
                   help="nominal dt; actual steps scale it by "
                        "powers of two to stay stable")
    p.add_argument("--total-time", type=float, default=1.0,
                   help="simulated seconds to advance "
                        "(default 1.0 = 10000 steps x 0.0001)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.nx is None:
        args.nx = int(args.lx / args.ly) * args.ny
    return args


if __name__ == '__main__':
    args = parse_args()
    solver = FlowSolver(nx=args.nx, ny=args.ny, nit=50, nu=args.nu,
                        dt=args.dt, boundary="throughflow",
                        inflow_velocity=args.inflow,
                        lx=args.lx, ly=args.ly)

    # circular obstacle: centre (1.0, 1.0), radius 0.15, as a polygon
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    cx, cy, r = 1.0, args.ly / 2.0, 0.15
    circle = [(float(cx + r * np.cos(t)),
               float(cy + r * np.sin(t))) for t in theta]
    solver.add_obstacle(circle)

    steps = solver.run(args.total_time)

    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    speed = np.hypot(u, v)
    solid = solver.solid.astype(bool)
    print(f"nx={args.nx} ny={args.ny} lx={args.lx} ly={args.ly} "
          f"inflow={args.inflow} total_time={args.total_time}")
    print(f"steps taken: {steps}, simulated time: {solver.time:.6f}s, "
          f"last dt: {solver.last_dt:.3g}, "
          f"stable dt limit now: {solver.stable_dt():.3g}")
    print(f"finite: {np.isfinite(u).all() and np.isfinite(v).all()}, "
          f"max speed outside obstacle: "
          f"{speed[~solid].max():.3f}, "
          f"max speed inside obstacle: "
          f"{speed[solid].max() if solid.any() else 0.0:.3g}")

    # --- Visualization (extent and figure shape follow lx/ly) ---
    cmap = plt.cm.jet.copy()
    cmap.set_bad('0.35')
    speed = np.ma.masked_where(solid, speed)

    height = 4.5
    width = (args.lx / args.ly) * height
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    im = ax.imshow(speed, origin='lower',
                   extent=[0, args.lx, 0, args.ly],
                   cmap=cmap, aspect='equal')
    ax.set_xlim(0, args.lx)
    ax.set_ylim(0, args.ly)
    fig.colorbar(im, ax=ax, label='Speed')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Throughflow past a circular obstacle '
                 f'({args.total_time:g}s, {steps} adaptive steps, '
                 f'nx={args.nx}, inflow={args.inflow:g})')
    plt.tight_layout()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'throughflow_circle.png')
    fig.savefig(out)
    print(f"saved {out}")
