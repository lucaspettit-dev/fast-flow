"""ElectrostaticForceHandler shape-mask / edge overlay test.

Builds a 100x200 image (ny=100 rows, nx=200 cols) with a solid
circle of radius 20 centred at image coordinates x=20, y=20 in
channel 0, attaches an ElectrostaticForceHandler with
layermap {0: 10000}, then renders:

  - the handler's solids mask: solid = red, non-solid = white
  - every coordinate stored in the Shape struct, plotted
    directly as a pixel (row = y, col = x): green

The solver is built with lx = nx - 1 and ly = ny - 1 so the
handler's stored coordinates are numerically on the pixel
grid (one unit per pixel) and can be compared 1:1 with the
mask.  If the stored coordinates were pure image coordinates,
the green pixels would land exactly on the red circle's
boundary ring.

Run:  python demo/electrostatic_mask_test.py
Writes demo/electrostatic_mask_test.png (4x nearest-neighbour
upscale of the 100x200 render).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fast_flow import FlowSolver, ElectrostaticForceHandler

NY, NX = 100, 200          # image rows (y), cols (x)
CX, CY, R = 20, 20, 20     # circle centre (x, y) and radius, pixels
LX, LY = NX - 1, NY - 1    # solver domain: 1 unit per pixel

# --- the image: uint8 (ny, nx, 3), circle in channel 0 ---
yy, xx = np.mgrid[0:NY, 0:NX]
image = np.zeros((NY, NX, 3), dtype=np.uint8)
image[:, :, 0] = np.where((xx - CX) ** 2 + (yy - CY) ** 2 <= R ** 2,
                          255, 0).astype(np.uint8)

# --- solver + handler (lx/ly chosen so coords sit on pixels) ---
solver = FlowSolver(nx=NX, ny=NY, nit=10, lx=float(LX), ly=float(LY))
handler = ElectrostaticForceHandler(image=image, layermap={0: 10000})
solver.add_force_handler(handler)
core = handler._handler  # compiled ElectrostaticForceHandler

print(f"structs: {core.num_shapes()}, charge: {core.shape_charge(0)}")
print(f"solids in mask: {int(core.solids_mask.sum())}")

# --- render: mask red on white, struct coordinates green ---
render = np.full((NY, NX, 3), 255, dtype=np.uint8)
render[core.solids_mask == 1] = (255, 0, 0)

edges = core.shape_edges(0)
print(f"edge coordinates in struct: {len(edges)}")
for (x, y) in edges:
    render[int(round(y)), int(round(x))] = (0, 200, 0)

xs = [p[0] for p in edges]
ys = [p[1] for p in edges]
print(f"edge x range: {min(xs):.0f}..{max(xs):.0f}  (circle spans x 0..40)")
print(f"edge y range: {min(ys):.0f}..{max(ys):.0f}  (circle spans y 0..40)")

# --- save (4x nearest-neighbour upscale so pixels are visible) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

big = np.repeat(np.repeat(render, 4, axis=0), 4, axis=1)
out = os.path.join(os.path.dirname(__file__),
                   "electrostatic_mask_test.png")
plt.imsave(out, big)
print(f"wrote {out}")
