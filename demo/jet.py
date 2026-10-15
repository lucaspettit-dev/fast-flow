import argparse
import os
import time

import cv2
import numpy as np

from fast_flow import ElectrostaticForceHandler, FlowSolver
from fast_flow.core.renderer import Renderer

TOTAL_TIME = 5.0

if __name__ == '__main__':
    img = cv2.imread('./res/jet.png')
    img = cv2.flip(img, 0)
    ny, nx = img.shape[0], img.shape[1]
    print(f'Loaded image: dims=({ny}, {nx})')

    # channel 2 (red) = cow -> +charge; channel 0 (blue) =
    # rooster -> -charge
    handler = ElectrostaticForceHandler(
        img, layermap={2: 20000, 0: -20000})
    renderer = Renderer("jet", 120, ny, nx)
    solver = FlowSolver(img=img, nit=50, rho=1.0, nu=0.05,
                        boundary="throughflow",
                        inflow_velocity=0.001,
                        lx=8.0, ly=5.0,
                        force_handlers=[handler],
                        renderer=renderer)

    renderer.run()
    t0 = time.perf_counter()
    steps = solver.run(TOTAL_TIME)
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

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'jet_velocity.png')
    renderer.save(out)
    print(f"saved {out}")
