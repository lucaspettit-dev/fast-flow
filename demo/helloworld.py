from fast_flow import FlowSolver

if __name__ == '__main__':
    solver = FlowSolver(nx=256, ny=128)
    solver.add_obstacle([(0.5, 0.4), (0.6, 0.4), (0.6, 0.6), (0.5, 0.6)])
    solver.add_density_source(0.3, 0.5, radius=0.05, color=(1.0, 0.0, 0.0))
    solver.run(500)
    speed = solver.speed            # read-only (128, 256) float64
    dye = solver.render_density()   # read-only (128, 256, 3) uint8
