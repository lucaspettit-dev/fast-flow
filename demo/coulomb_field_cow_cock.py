"""Starting Coulomb field of the ElectrostaticForceHandler on
the cow-cock test image (demo/res/cow-cock.png).

The image carries the two animals on separate colour
channels: the cow is the red channel (2), the rooster is the
blue channel (0).  This demo charges the cow +20000 and the
rooster -20000 (layermap {2: 20000, 0: -20000}), binds the
handler to a solver on the same image, and renders the
static Coulomb field the handler computes at construction
(handler.coulomb_field()): field lines run from the positive
cow to the negative rooster, coloured by log10 |F|.

The image is flipped vertically before use, exactly as in
throughflow_cow_cock.py, so this picture is in the same
orientation as the flow renders.  The field inside the
electrodes is masked out of the plot (it is not meaningful
there); the electrodes themselves are drawn in their
channel colours.

Writes coulomb_field_cow_cock.png next to this script.
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
    p.add_argument("--charge", type=float, default=20000.0,
                   help="cow gets +charge, rooster gets -charge")
    p.add_argument("--out", default=None)
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
    solver = FlowSolver(img=img, lx=args.lx, ly=args.ly,
                        force_handlers=[handler])

    charges = handler.shape_charges
    print(f'structs: {handler.num_shapes}, charges: {charges}')
    for i in range(handler.num_shapes):
        print(f'  struct {i}: charge {charges[i]:+g}, '
              f'{len(handler.shape_edges(i))} edge points')

    fu, fv = handler.coulomb_field()

    # mask the field inside the electrodes for the plot
    solid = solver.solid.astype(bool)
    fu = np.where(solid, 0.0, fu)
    fv = np.where(solid, 0.0, fv)
    mag = np.hypot(fu, fv)
    print(f'field: max |F| outside the electrodes = {mag.max():.1f}')

    # --- Visualization: the output figure has the same shape
    # as the input image -- ny pixels tall, nx pixels wide ---
    xs = np.linspace(0.0, args.lx, nx)
    ys = np.linspace(0.0, args.ly, ny)
    logmag = np.log10(mag + 1.0)

    dpi = 100
    fig, ax = plt.subplots(figsize=(nx / dpi, ny / dpi), dpi=dpi)
    # electrodes in their channel colours
    overlay = np.zeros((ny, nx, 4))
    overlay[img[:, :, 2] > 127] = (0.85, 0.25, 0.20, 1.0)   # cow, red
    overlay[img[:, :, 0] > 127] = (0.20, 0.35, 0.90, 1.0)   # rooster, blue
    ax.imshow(overlay, origin='lower',
              extent=[0, args.lx, 0, args.ly], aspect='auto', zorder=1)
    ax.streamplot(xs, ys, fu, fv, color=logmag,
                  cmap='inferno', density=1.6, linewidth=0.7,
                  arrowsize=0.7, zorder=2)
    # + / - markers at the two animals' centroids
    for channel, sign in ((2, '+'), (0, '-')):
        ys_i, xs_i = np.nonzero(img[:, :, channel] > 127)
        ax.text(xs_i.mean() / (nx - 1) * args.lx,
                ys_i.mean() / (ny - 1) * args.ly, sign,
                color='white', fontsize=11, fontweight='bold',
                ha='center', va='center', zorder=3)
    ax.set_xlim(0, args.lx)
    ax.set_ylim(0, args.ly)
    # no colour bar, ticks, labels or title: the axes fill the
    # whole figure, so the render has exactly the input's shape
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_position([0, 0, 1, 1])

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'coulomb_field_cow_cock.png')
    fig.savefig(out)
    print(f"saved {out}")
