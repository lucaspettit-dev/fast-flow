import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from fast_flow import ElectrostaticForceHandler, FlowSolver, Boundry
from fast_flow.core.renderer import Renderer

JET_ID = 1
TOTAL_TIME = 1
P_CHARGE = 40000
N_ChARGE_MULTIPLIER = -1.0


def mirror_stack_pad(img: np.ndarray) -> np.ndarray:
    # Flip vertically and stack top to bottom
    #img = img[::2, ::2]
    flipped = np.flip(img, axis=0)
    stacked = np.concatenate((img, flipped), axis=0)

    # Add padding equal to 50% of the stacked dimensions on each side
    h, w = stacked.shape[:2]
    pad_y = h
    pad_x = w

    # Preserve additional dimensions (e.g., RGB/RGBA channels)
    padding = [(pad_y, pad_y), (pad_x, pad_x)]
    padding += [(0, 0)] * (stacked.ndim - 2)

    return np.pad(stacked, padding, mode="constant", constant_values=0)


def to_heatmap(a: np.ndarray, solid: np.ndarray, label: str, title: str, out: str, lx: int, ly: int):
    # --- Visualization (extent and figure shape follow lx/ly) ---
    cmap = plt.cm.jet.copy()
    cmap.set_bad('0.35')
    a = np.ma.masked_where(solid, a)

    height = 4.5
    width = (a.shape[1] / a.shape[0]) * height
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    im = ax.imshow(a, origin='lower',
                   extent=[0, lx, 0, ly],
                   cmap=cmap, aspect='equal')
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} ({TOTAL_TIME:g}s, {steps} adaptive steps)')
    plt.tight_layout()
    fig.savefig(out)


if __name__ == '__main__':

    img = cv2.imread(f'./res/jet{JET_ID}.png')
    img = mirror_stack_pad(img)
    ny, nx = img.shape[0], img.shape[1]
    ly = ny // 100
    lx = nx // 100
    print(f'Loaded image: dims=({ny}, {nx})')

    # channel 2 (red) = cow -> +charge; channel 0 (blue) =
    # rooster -> -charge
    handler = ElectrostaticForceHandler(
        img, layermap={0: P_CHARGE, 2: P_CHARGE * N_ChARGE_MULTIPLIER})

    renderer = Renderer("jet", 120*2, ny, nx, record=True, legend=True)
    solver = FlowSolver(img=img, nit=50, rho=1.0, nu=0.05,
                        boundary=Boundry.NONE,
                        lx=lx, ly=ly,
                        force_handlers=[handler],
                        renderer=renderer)

    renderer.run()
    t0 = time.perf_counter()
    steps = solver.run(TOTAL_TIME)
    wall = time.perf_counter() - t0

    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    speed = np.sqrt(np.square(u) + np.square(v))
    solid = solver.solid.astype(bool)
    density = handler.ion_density()
    force = handler.solid_forces()
    print('----------- RESULTS ------------')
    print(f"steps taken: {steps}\n\tsimulated time: {solver.time:.6f}s")
    print(f"wall-clock runtime: {wall:.2f}s\n\t"
          f"iterations per second: {steps / wall:.1f}")
    print(f"ion density: total {density.sum():.1f}, "
          f"max {density.max():.3f}")
    print(f"Net forces (x,y) = {force['net']}")

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'out/jet')
    os.makedirs(out_dir, exist_ok=True)

    to_heatmap(speed, solid, 'velocity', 'Air Velocity', os.path.join(out_dir, f'{JET_ID}_velocity_hm.png'), lx, ly)
    to_heatmap(density, solid, 'density', 'Air Density', os.path.join(out_dir, f'{JET_ID}_density_hm.png'), lx, ly)

    u, v = handler.ion_velocity()
    speed = np.sqrt(np.square(u) + np.square(v))
    to_heatmap(speed, solid, 'velocity', 'Ion Velocity', os.path.join(out_dir,f'{JET_ID}_ion_velocity.png'), lx, ly)
    to_heatmap(handler.ion_density(), solid, 'Ion Density', 'density', os.path.join(out_dir,f'{JET_ID}_ion_density.png'), lx, ly)

    u, v = handler.o2_velocity()
    speed = np.sqrt(np.square(u) + np.square(v))
    to_heatmap(speed, solid, 'velocity', 'O2 Velocity', os.path.join(out_dir,f'{JET_ID}_o2_velocity.png'), lx, ly)
    to_heatmap(handler.o2_density(), solid, 'density', 'O2 Density', os.path.join(out_dir,f'{JET_ID}_o2_density.png'), lx, ly)

    out = os.path.join(out_dir, f'{JET_ID}_velocity_directional.png')
    renderer.save(out)
    renderer.save_video(os.path.join(out_dir, f'{JET_ID}.mp4'), 30)
    print(f"saved {out}")
