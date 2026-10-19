import argparse
import os
import time

import cv2
import numpy as np

from fast_flow import ElectrostaticForceHandler, FlowSolver, Boundry
from fast_flow.core.renderer import Renderer

TOTAL_TIME = 1
P_CHARGE = 40000
N_ChARGE_MULTIPLIER = -1

if __name__ == '__main__':

    for jet_id in range(1, 11):
        img = cv2.imread(f'./res/jet{jet_id}.png')
        img = cv2.flip(img, 0)
        ny, nx = img.shape[0], img.shape[1]
        print(f'Loaded image: dims=({ny}, {nx})')

        # channel 2 (red) = cow -> +charge; channel 0 (blue) =
        # rooster -> -charge
        handler = ElectrostaticForceHandler(
            img, layermap={0: P_CHARGE, 2: P_CHARGE * N_ChARGE_MULTIPLIER})
        renderer = Renderer("jet", 120*2, ny, nx, record=True)
        solver = FlowSolver(img=img, nit=50, rho=1.0, nu=0.05,
                            boundary=Boundry.NONE,
                            lx=8.0, ly=5.0,
                            force_handlers=[handler],
                            renderer=renderer)

        renderer.run()
        t0 = time.perf_counter()
        steps = solver.run(TOTAL_TIME)
        wall = time.perf_counter() - t0

        u = solver.horizontal_velocity
        v = solver.vertical_velocity
        speed = np.hypot(u[:, -1], v[:, -1])
        solid = solver.solid.astype(bool)[:, -1]
        density = handler.ion_density()
        force = handler.solid_forces()
        print('----------- RESULTS ------------')
        print(f"steps taken: {steps}\n\tsimulated time: {solver.time:.6f}s")
        print(f"wall-clock runtime: {wall:.2f}s\n\t"
              f"iterations per second: {steps / wall:.1f}")
        print(f"finite: {np.isfinite(u).all() and np.isfinite(v).all()}\n\t"
              f"avg RHS exit speed: {speed[~solid].mean():.10f}\n\t")
        print(f"ion density: total {density.sum():.1f}, "
              f"max {density.max():.3f}")
        print(f"Net forces (x,y) = {force['net']}")

        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'out/jet')
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f'jet_velocity_{jet_id}.png')
        renderer.save(out)
        renderer.save_video(os.path.join(out_dir, f'/jet_velocity_{jet_id}.mp4'), 30)
        print(f"saved {out}")
