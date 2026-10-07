from fast_flow import FlowSolver, Direction, ConstantVelocityForceHandler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    nx, ny = 50, 50
    solver = FlowSolver(nx=50, ny=50, nit=50, dt=0.001)
    handler = ConstantVelocityForceHandler(solver, Direction.DOWN, 3.0)

    for i in tqdm(range(1000)):
        solver.step()

    u = solver.horizontal_velocity
    v = solver.vertical_velocity
    p = solver.pressure

    x = np.linspace(0, solver.lx, nx)
    y = np.linspace(0, solver.ly, ny)
    X, Y = np.meshgrid(x, y)

    # --- Visualization ---
    fig = plt.figure(figsize=(11, 7), dpi=100)
    # Plotting the pressure field as a contour
    plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.viridis)
    plt.colorbar(label='Pressure')
    # Plotting velocity streamlines
    plt.streamplot(X, Y, u, v, color=u, cmap=plt.cm.jet)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Lid-Driven Cavity Flow (Navier-Stokes)')
    plt.show()

