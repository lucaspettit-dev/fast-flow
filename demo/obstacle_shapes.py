"""Throughflow past assorted polygonal obstacles (fast-flow demo).

Same channel as throughflow_circle.py (4x2, U = 1, nu = 0.05,
1000 steps at dt = 0.001), but swaps the circle for a gallery of
shapes -- square, diamond, triangle, pentagon, star, and a
concave plus -- to exercise the polygon rasterizer.  Saves one
combined speed figure next to this script.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from fast_flow import FlowSolver

CX, CY = 1.0, 1.0
NX, NY = 256, 64
LX, LY = 4.0, 2.0
TOTAL_TIME = 1.0  # simulated seconds (= old 1000 steps x dt=0.001)


def regular(n, r, rot=0.0):
    a = rot + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(CX + r * np.cos(t)), float(CY + r * np.sin(t)))
            for t in a]


def star(points=5, r_out=0.22, r_in=0.09, rot=np.pi):
    verts = []
    for k in range(2 * points):
        r = r_out if k % 2 == 0 else r_in
        t = rot + k * np.pi / points
        verts.append((float(CX + r * np.cos(t)),
                      float(CY + r * np.sin(t))))
    return verts


def plus(arm=0.21, half=0.07):
    c = [(arm, half), (half, half), (half, arm),
         (-half, arm), (-half, half), (-arm, half),
         (-arm, -half), (-half, -half), (-half, -arm),
         (half, -arm), (half, -half), (arm, -half)]
    return [(float(CX + x), float(CY + y)) for x, y in c]


SHAPES = {
    "Square":   [(CX - 0.15, CY - 0.15), (CX + 0.15, CY - 0.15),
                 (CX + 0.15, CY + 0.15), (CX - 0.15, CY + 0.15)],
    "Diamond":  regular(4, 0.20),
    "Triangle": regular(3, 0.22, rot=np.pi),
    "Pentagon": regular(5, 0.20, rot=np.pi),
    "Star":     star(),
    "Plus":     plus(),
}


def run(poly):
    solver = FlowSolver(nx=NX, ny=NY, nit=50, nu=0.05, dt=0.001,
                        boundary="throughflow", inflow_velocity=1.0,
                        lx=LX, ly=LY)
    solver.add_obstacle(poly)
    solver.run(TOTAL_TIME)
    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    p = solver.pressure
    ok = bool(np.isfinite(u).all() and np.isfinite(v).all()
              and np.isfinite(p).all())
    return solver, u, v, p, ok


def pressure_gradient_magnitude(p, solid, dx, dy):
    """|grad p| with the solver's own wall rule: a solid neighbour
    contributes the cell's own pressure (zero normal gradient)."""
    pp = np.pad(p, 1, mode='edge')
    ss = np.pad(solid, 1, constant_values=True)
    c = pp[1:-1, 1:-1]
    e = np.where(ss[1:-1, 2:], c, pp[1:-1, 2:])
    w = np.where(ss[1:-1, :-2], c, pp[1:-1, :-2])
    n = np.where(ss[2:, 1:-1], c, pp[2:, 1:-1])
    s = np.where(ss[:-2, 1:-1], c, pp[:-2, 1:-1])
    return np.hypot((e - w) / (2 * dx), (n - s) / (2 * dy))


if __name__ == '__main__':
    cmap = plt.cm.jet.copy()
    cmap.set_bad('0.35')

    panels = []
    for name, poly in tqdm(SHAPES.items(), desc="shapes"):
        solver, u, v, p, ok = run(poly)
        solid = solver.solid.astype(bool)
        speed = np.hypot(u, v)
        inside = float(np.abs(speed[solid]).max()) if solid.any() else 0.0
        grad = pressure_gradient_magnitude(p, solid, LX / (NX - 1),
                                           LY / (NY - 1))
        print(f"{name:9s} finite={ok} solid cells={int(solid.sum()):4d} "
              f"max|u| inside={inside:.3g} "
              f"max |grad p|={grad[~solid].max():.1f}")
        panels.append((name, grad, solid))

    # shared colour scale so the panels stay comparable
    allg = np.concatenate([g[~s].ravel() for _, g, s in panels])
    vmax = float(np.percentile(allg, 99.5))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 4.6), dpi=100)
    for ax, (name, grad, solid) in zip(axes.ravel(), panels):
        ax.imshow(np.ma.masked_where(solid, grad), origin='lower',
                  extent=[0, LX, 0, LY], cmap=cmap, aspect='equal',
                  vmin=0.0, vmax=vmax)
        ax.set_xlim(0, LX)
        ax.set_ylim(0, LY)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Throughflow past polygonal obstacles: '
                 f'|pressure gradient| ({TOTAL_TIME:g}s each)')
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'obstacle_shapes.png')
    fig.savefig(out)
    print(f"saved {out}")
    plt.show()
