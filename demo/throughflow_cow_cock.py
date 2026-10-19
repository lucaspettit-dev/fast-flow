import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2

from fast_flow import FlowSolver, Boundry
from fast_flow.core.renderer import Renderer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lx", type=float, default=5.0)
    p.add_argument("--ly", type=float, default=3.0)
    p.add_argument("--inflow", type=float, default=20.0)
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--fps", type=int, default=120*4,
                   help="renderer frames per simulated second")
    p.add_argument("--total-time", type=float, default=1.0,
                   help="simulated seconds to advance "
                        "(default 1.0 = 10000 steps x 0.0001)")
    p.add_argument("--file", type=str, default="./res/cow.png")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    img = cv2.imread(args.file)
    img = cv2.flip(img, 0)

    print(f'Loaded image: dims=({img.shape[0]}, {img.shape[1]})')
    renderer = Renderer("ehd_cow_cock", args.fps, img.shape[0], img.shape[1], record=True)
    solver = FlowSolver(img=img, nit=50, nu=args.nu,
                        boundary=Boundry.THROUGHFLOW,
                        inflow_velocity=args.inflow,
                        lx=args.lx, ly=args.ly,
                        renderer = renderer)

    renderer.run()
    start_time = time.time_ns()
    steps = solver.run(args.total_time)
    end_time = time.time_ns()

    fps = steps / ((end_time - start_time) / 100000000.0)
    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    speed = np.hypot(u, v)
    solid = solver.solid.astype(bool)
    print('----------- RESULTS ------------')
    print(f"Params Used:(lx, ly) = ({args.lx}, {args.ly})"
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

    height = (1.5 * img.shape[0]) // 100
    width = (1.5 * img.shape[1]) // 100
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(speed, origin='lower',
                   extent=[0, args.lx, 0, args.ly],
                   cmap=cmap, aspect='equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'Throughflow past a cow and a large cock\n'
                 f'({args.total_time:g}s, {steps} adaptive steps, '
                 f'nx={img.shape[1]}, inflow={args.inflow:g})')
    plt.tight_layout()

    filename = args.file.rsplit('/', 1)[-1]
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'out/throughflow_{filename}')
    renderer.save_vidio(f'out/throughflow_{filename.split(".", 1)[0]}.mp4', 30)
    fig.savefig(out)
    print(f"saved {out}")
