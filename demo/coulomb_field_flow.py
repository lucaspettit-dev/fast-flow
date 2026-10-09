#!/usr/bin/env python3
"""Coulomb field flow test: positive terminals -> negative one.

Builds an image with two positively charged circles (channel
0, +20000) and one negatively charged circle (channel 1,
-20000), attaches an ElectrostaticForceHandler, and renders
the static Coulomb field (force at the centre of each grid
cell) as streamlines.  The field lines should leave the
positive terminals (red) and terminate on the negative
terminal (blue): the force on a positive test charge points
away from + and toward -.

Numeric checks are printed (and fail the script if wrong):
midway between the terminals the force points from + to -,
next to the negative terminal it points toward the terminal
from all four sides, and next to a positive terminal it
points away from it.

For the plot only, the field inside the electrodes is set to
zero (inside a conductor the field vanishes); the computed
field itself is untouched.

Run:  python demo/coulomb_field_flow.py
Out:  demo/coulomb_field_flow.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fast_flow import FlowSolver, ElectrostaticForceHandler

NY, NX = 100, 200                    # image rows (y), cols (x)
POSITIVES = [(35, 25), (35, 75)]     # (x, y) centres, pixels
NEGATIVE = (165, 50)
R = 12                               # terminal radius, pixels
CHARGE = 20000.0


def build_image():
    yy, xx = np.mgrid[0:NY, 0:NX]
    img = np.zeros((NY, NX, 3), dtype=np.uint8)
    for cx, cy in POSITIVES:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2, 0] = 255
    cx, cy = NEGATIVE
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2, 1] = 255
    return img


def main():
    img = build_image()

    # lx/ly chosen so grid units are pixels (dx = dy = 1)
    solver = FlowSolver(nx=NX, ny=NY, nit=10,
                        lx=float(NX - 1), ly=float(NY - 1))
    handler = ElectrostaticForceHandler(
        image=img, layermap={0: CHARGE, 1: -CHARGE})
    solver.add_force_handler(handler)
    core = handler._handler

    print(f"structs: {core.num_shapes()}, "
          f"charges: {[core.shape_charge(i) for i in range(core.num_shapes())]}, "
          f"edges: {[len(core.shape_edges(i)) for i in range(core.num_shapes())]}")
    assert core.num_shapes() == 2
    assert core.shape_charge(0) == CHARGE and core.shape_charge(1) == -CHARGE

    Fu, Fv = handler.coulomb_field()
    print(f"field: shape {Fu.shape}, |F| max {np.hypot(Fu, Fv).max():.1f}")

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    # Midway between the + pair and the - terminal: force on a
    # positive test charge points +x (away from +, toward -)
    check("midpoint (100, 37): Fu > 0", Fu[37, 100] > 0)
    check("midpoint (100, 62): Fu > 0", Fu[62, 100] > 0)

    # Around the negative terminal the force points TOWARD it
    nx_, ny_ = NEGATIVE
    d = R + 4
    check("left of - terminal: Fu > 0 (toward)", Fu[ny_, nx_ - d] > 0)
    check("right of - terminal: Fu < 0 (toward)", Fu[ny_, nx_ + d] < 0)
    check("above - terminal: Fv > 0 (toward)", Fv[ny_ - d, nx_] > 0)
    check("below - terminal: Fv < 0 (toward)", Fv[ny_ + d, nx_] < 0)

    # Around a positive terminal the force points AWAY from it
    px, py = POSITIVES[0]
    check("right of + terminal: Fu > 0 (away)", Fu[py, px + d] > 0)
    check("left of + terminal: Fu < 0 (away)", Fu[py, px - d] < 0)
    check("above + terminal: Fv < 0 (away)", Fv[py - d, px] < 0)
    check("below + terminal: Fv > 0 (away)", Fv[py + d, px] > 0)

    if not ok:
        raise SystemExit("field direction checks FAILED")

    # --- render ---
    # solver grid units are pixels here, so cell centres sit at
    # half-integer coordinates; y increases downward (image rows)
    xs = (np.arange(NX) + 0.5) * solver.dx
    ys = (np.arange(NY) + 0.5) * solver.dy
    mask = core.solids_mask.astype(bool)
    Pu = np.where(mask, 0.0, Fu)
    Pv = np.where(mask, 0.0, Fv)
    speed = np.hypot(Pu, Pv)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.streamplot(xs, ys, Pu, Pv, density=1.4, linewidth=0.7,
                  color=np.log10(speed + 1.0), cmap="viridis")
    for cx, cy in POSITIVES:
        ax.add_patch(Circle((cx, cy), R, facecolor="red",
                            edgecolor="black", zorder=5))
        ax.text(cx, cy, "+", ha="center", va="center",
                fontsize=20, fontweight="bold", zorder=6)
    cx, cy = NEGATIVE
    ax.add_patch(Circle((cx, cy), R, facecolor="royalblue",
                        edgecolor="black", zorder=5))
    ax.text(cx, cy, "−", ha="center", va="center", color="white",
            fontsize=20, fontweight="bold", zorder=6)
    ax.set_xlim(0, NX - 1)
    ax.set_ylim(NY - 1, 0)  # y down, image orientation
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y (image rows)")
    ax.set_title("Coulomb field: positive terminals → negative terminal\n"
                 "(force at each grid cell centre, static)")

    out = "demo/coulomb_field_flow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
