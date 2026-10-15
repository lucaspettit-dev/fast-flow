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

The air velocity is rendered live by the new Renderer
(fast_flow.core.renderer): after every solver step the
core hands the renderer the (u, v) fields, and the
renderer draws a frame -- direction as hue on a colour
wheel, magnitude as brightness, scaled by that frame's
max magnitude -- whenever the run time crosses the next
1/fps boundary.  At the end of the run the demo saves
the renderer's last frame (out/ehd_cow_cock_velocity.png),
flipped back to the input image's orientation.  The
figure is the same shape as the input image (500x300).

Prints the usual results block plus the ion density
totals.  The +ion density is no longer plotted (the
renderer draws one vector field); its totals are still
printed.
"""
import argparse
import os
import time

import cv2
import numpy as np

from fast_flow import ElectrostaticForceHandler, FlowSolver
from fast_flow.core.renderer import Renderer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img", default="./res/cow-cock.png")
    p.add_argument("--lx", type=float, default=8.0)
    p.add_argument("--ly", type=float, default=2.0)
    p.add_argument("--inflow", type=float, default=0.0)
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--total-time", type=float, default=1.0,
                   help="simulated seconds to advance")
    p.add_argument("--charge", type=float, default=40000.0,
                   help="cow gets +charge, rooster gets -charge")
    p.add_argument("--fps", type=int, default=120,
                   help="renderer frames per simulated second")
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
    renderer = Renderer("ehd_cow_cock", args.fps, ny, nx)
    solver = FlowSolver(img=img, nit=50, rho=1.0, nu=args.nu,
                        boundary="throughflow",
                        inflow_velocity=args.inflow,
                        lx=args.lx, ly=args.ly,
                        force_handlers=[handler],
                        renderer=renderer)

    renderer.run()
    t0 = time.perf_counter()
    steps = solver.run(args.total_time)
    wall = time.perf_counter() - t0

    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    speed = np.hypot(u, v)
    solid = solver.solid.astype(bool)
    density = handler.ion_density()
    print('----------- RESULTS ------------')
    print(f"steps taken: {steps}\n\tsimulated time: {solver.time:.6f}s")
    print(f"wall-clock runtime: {wall:.2f}s\n\t"
          f"iterations per second: {steps / wall:.1f}")
    print(f"finite: {np.isfinite(u).all() and np.isfinite(v).all()}\n\t"
          f"max speed outside obstacle: {speed[~solid].max():.3f}\n\t"
          f"max speed inside obstacle: {speed[solid].max():.3g}")
    print(f"ion density: total {density.sum():.1f}, "
          f"max {density.max():.3f}")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'ehd_cow_cock_velocity.png')


    # The renderer's buffer is in solver order (row 0 = the
    # domain's bottom, y-up), while cv2 writes row 0 at the
    # top of the image -- so flip the saved frame back to
    # the input image's orientation, mirroring the
    # cv2.flip applied to the input above.
    renderer.save(out)
    print(f"saved {out}")
