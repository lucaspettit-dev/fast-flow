"""Smoke tests for the fast-flow package framework."""

import numpy as np

from fast_flow import FlowSolver, __version__


def test_version():
    assert __version__ == "0.2.0"


def test_solver_configuration():
    solver = FlowSolver(nx=64, ny=32, lx=2.0, ly=1.0)
    solver.add_obstacle([(0.5, 0.4), (0.6, 0.4), (0.6, 0.6), (0.5, 0.6)])
    assert (solver.nx, solver.ny) == (64, 32)
    assert len(solver.obstacles) == 1
    assert solver.step_count == 0
