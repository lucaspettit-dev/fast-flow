"""Throughflow past a circular obstacle (fast-flow demo).

The fast-flow analogue of the ehd-flow "infinite flow" setup:
uniform inflow on the left, zero-gradient outflow on the right,
no-slip top/bottom walls, and a solid circular obstacle as a
no-slip polygon.  Renders the speed field after 1000 steps.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from fast_flow import FlowSolver

if __name__ == '__main__':
    nx, ny = 256, 64
    solver = FlowSolver(nx=nx, ny=ny, nit=50, nu=0.05, dt=0.001,
                        boundary="throughflow", inflow_velocity=1.0,
                        lx=4.0, ly=2.0)

    # circular obstacle: centre (1.0, 1.0), radius 0.15, as a polygon
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    circle = [(float(1.0 + 0.15 * np.cos(t)),
               float(1.0 + 0.15 * np.sin(t))) for t in theta]
    solver.add_obstacle(circle)

    for _ in tqdm(range(1000)):
        solver.step()

    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    speed = np.hypot(u, v)
    solid = solver.solid.astype(bool)
    print(f"finite: {np.isfinite(u).all() and np.isfinite(v).all()}, "
          f"max speed outside obstacle: "
          f"{speed[~solid].max():.3f}")

    # --- Visualization ---
    cmap = plt.cm.jet.copy()
    cmap.set_bad('0.35')
    speed = np.ma.masked_where(solid, speed)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=100)
    im = ax.imshow(speed, origin='lower', extent=[0, 4, 0, 2],
                   cmap=cmap, aspect='equal')
    fig.colorbar(im, ax=ax, label='Speed')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Throughflow past a circular obstacle (1000 steps)')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'throughflow_circle.png')
    fig.savefig(out)
    print(f"saved {out}")
    plt.show()
