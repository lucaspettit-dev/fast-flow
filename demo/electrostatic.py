import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fast_flow import FlowSolver, ElectrostaticForceHandler
import cv2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img", type=Path, default=Path('./res/cow-cock.png'))
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
    return args


def load_image(path: Path) -> np.ndarray:
    return cv2.imread(path)


if __name__ == '__main__':
    args = parse_args()
    img = load_image(args.img)
    if img.ndim != 3:
        raise ValueError('Need a 3d image for demo')
    ny, nx, nz = img.shape

    handler = ElectrostaticForceHandler(img, layermap={0: 20000, 2: 20000})
    solver = FlowSolver(nx=nx, ny=ny, nit=50, nu=args.nu,
                        dt=args.dt, boundary="throughflow",
                        inflow_velocity=args.inflow,
                        lx=args.lx, ly=args.ly, force_handlers=[handler])

    start_time = time.time_ns()
    steps = solver.run(args.total_time)
    end_time = time.time_ns()

    fps = steps / ((end_time - start_time) / 100000000.0)
    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    speed = np.hypot(u, v)
    solid = solver.solid.astype(bool)
    print('----------- RESULTS ------------')
    print(f"Params Used:\n\t(nx, ny) = ({nx}, {ny})\n\t(lx, ly) = ({args.lx}, {args.ly})"
          f"\n\tinflow = {args.inflow}\n\ttotal_time = {args.total_time}\n\tFPS = {fps:.2f}")
    print(f"steps taken: {steps}\n\tsimulated time: {solver.time:.6f}s\n\t"
          f"last dt: {solver.last_dt:.3g}\n\t"
          f"stable dt limit now: {solver.stable_dt():.3g}")
    print(f"finite: {np.isfinite(u).all() and np.isfinite(v).all()}\n\t"
          f"max speed outside obstacle: "
          f"{speed[~solid].max():.3f}\n\t"
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
                 f'nx={nx}, inflow={args.inflow:g})')
    plt.tight_layout()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'throughflow_circle.png')
    fig.savefig(out)
    print(f"saved {out}")
