"""Behavioral tests for the compiled solver core."""

import numpy as np

from fast_flow import FlowSolver


def test_inflow_maintained():
    solver = FlowSolver(nx=48, ny=24, dt=0.02)
    solver.run(10)
    speed = solver.speed
    assert speed.shape == (24, 48)
    assert speed.dtype == np.float64
    assert not speed.flags.writeable
    # left edge mid-height keeps the inflow speed (cell-centered average
    # of the prescribed inlet face and its interior neighbor)
    assert abs(speed[12, 0] - 1.0) < 0.05


def test_obstacle_blocks_flow():
    solver = FlowSolver(nx=48, ny=24, dt=0.02)
    solver.add_obstacle([(0.9, 0.4), (1.1, 0.4), (1.1, 0.6), (0.9, 0.6)])
    solver.run(10)
    speed = solver.speed
    # cells safely inside the plate never move
    assert speed[10:14, 22:26].max() == 0.0
    # ...but flow exists around it
    assert speed.max() > 0.5


def test_density_source_and_render():
    solver = FlowSolver(nx=48, ny=24, dt=0.02)
    solver.add_density_source(0.3, 0.5, radius=0.1,
                              color=(1.0, 0.0, 0.0), rate=20.0)
    solver.run(5)
    img = solver.render_density()
    assert img.shape == (24, 48, 3)
    assert img.dtype == np.uint8
    assert not img.flags.writeable
    assert img[..., 0].sum() > 0      # red dye was injected
    assert img[..., 1].sum() == 0    # no green/blue from this source
    assert img[..., 2].sum() == 0


def test_direction_render_and_cache():
    solver = FlowSolver(nx=48, ny=24, dt=0.02)
    solver.run(5)
    first = solver.render_direction()
    assert first.shape == (24, 48, 3)
    assert first.dtype == np.uint8
    assert not first.flags.writeable
    assert solver.render_direction() is first  # cached, no recompute
    solver.step()
    assert solver.render_direction() is not first  # recomputed after step
    # uniform rightward inflow -> hue 90 deg -> greenish, non-black
    assert first[12, 5].sum() > 0
