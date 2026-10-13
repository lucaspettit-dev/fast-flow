"""EHD ion wind on the cow-cock test image
(demo/res/cow-cock.png): with zero inflow, all of the air
motion comes from the ions.

The cow (red channel, 2) is charged +20000 and the rooster
(blue channel, 0) -20000 through the
ElectrostaticForceHandler's layermap.  The handler
generates +ions in the air cells at the cow's surface,
drifts them along the Coulomb field (plus the air's own
motion) toward the rooster, and the moving ions drag the
air with them -- the ion wind.  The domain starts at
rest and inflow is 0, so every bit of velocity in the
result was imparted by the ions.

Prints the usual results block plus the ion density
totals, and writes two renders next to the other demo
outputs (out/): the air velocity (streamlines coloured
by speed over the two electrodes) and the +ion density.
Both figures are the same shape as the input image, with
no ticks, labels or colour bars.
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from fast_flow import ElectrostaticForceHandler, FlowSolver


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img", default="./res/cow-cock.png")
    p.add_argument("--lx", type=float, default=8.0)
    p.add_argument("--ly", type=float, default=2.0)
    p.add_argument("--inflow", type=float, default=0.0)
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--total-time", type=float, default=1.0,
                   help="simulated seconds to advance")
    p.add_argument("--charge", type=float, default=20000.0,
                   help="cow gets +charge, rooster gets -charge")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img = cv2.imread(args.img)
    img = cv2.flip(img, 0)
    ny, nx = img.shape[0], img.shape[1]
    print(f'Loaded image: dims=({ny}, {nx})')

    # channel 2 (red) = cow -> +charge; channel 0 (blue) =
    # rooster -> -charge
    handler = ElectrostaticForceHandler(
        img, layermap={2: args.charge, 0: -args.charge})
    solver = FlowSolver(img=img, nit=50, rho=1.0, nu=args.nu,
                        boundary="throughflow",
                        inflow_velocity=args.inflow,
                        lx=args.lx, ly=args.ly,
                        force_handlers=[handler])

    steps = solver.run(args.total_time)

    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    speed = np.hypot(u, v)
    solid = solver.solid.astype(bool)
    density = handler.ion_density()
    print('----------- RESULTS ------------')
    print(f"steps taken: {steps}\n\tsimulated time: {solver.time:.6f}s")
    print(f"finite: {np.isfinite(u).all() and np.isfinite(v).all()}\n\t"
          f"max speed outside obstacle: {speed[~solid].max():.3f}\n\t"
          f"max speed inside obstacle: {speed[solid].max():.3g}")
    print(f"ion density: total {density.sum():.1f}, "
          f"max {density.max():.3f}")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(out_dir, exist_ok=True)
    dpi = 100

    # --- Air velocity: streamlines over the electrodes ---
    fig, ax = plt.subplots(figsize=(nx / dpi, ny / dpi), dpi=dpi)
    overlay = np.zeros((ny, nx, 4))
    overlay[img[:, :, 2] > 127] = (0.85, 0.25, 0.20, 1.0)   # cow, red
    overlay[img[:, :, 0] > 127] = (0.20, 0.35, 0.90, 1.0)   # rooster, blue
    ax.imshow(overlay, origin='lower',
              extent=[0, args.lx, 0, args.ly], aspect='auto', zorder=1)
    xs = np.linspace(0.0, args.lx, nx)
    ys = np.linspace(0.0, args.ly, ny)
    ax.streamplot(xs, ys, u, v, color=speed, cmap='jet',
                  norm=plt.matplotlib.colors.PowerNorm(
                      gamma=0.5, vmin=0.0, vmax=max(speed.max(), 1e-12)),
                  density=1.6, linewidth=0.7, arrowsize=0.7, zorder=2)
    ax.set_xlim(0, args.lx)
    ax.set_ylim(0, args.ly)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_position([0, 0, 1, 1])
    out = os.path.join(out_dir, 'ehd_cow_cock_velocity.png')
    fig.savefig(out)
    print(f"saved {out}")

    # --- +ion density ---
    cmap = plt.cm.Blues.copy()
    cmap.set_bad('0.35')
    density_m = np.ma.masked_where(solid, density)
    fig, ax = plt.subplots(figsize=(nx / dpi, ny / dpi), dpi=dpi)
    ax.imshow(density_m, origin='lower',
              extent=[0, args.lx, 0, args.ly], cmap=cmap,
              aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_xlim(0, args.lx)
    ax.set_ylim(0, args.ly)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_position([0, 0, 1, 1])
    out = os.path.join(out_dir, 'ehd_cow_cock_density.png')
    fig.savefig(out)
    print(f"saved {out}")
