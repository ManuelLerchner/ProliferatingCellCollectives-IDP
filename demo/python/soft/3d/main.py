import numpy as np
from Simulation import Simulation
from Spherocylinder import Spherocylinder

# Example usage
if __name__ == "__main__":
    # Create and run simulation with drag
    l0 = 1.0
    lambbda = 1e-4
    friction_factor = 0.1

    tao_growtscale = 30
    lambda_dimensionless = (tao_growtscale / friction_factor) * l0**2 * lambbda

    sim = Simulation(
        tao_growthrate=tao_growtscale,
        lambda_sensitivity=lambda_dimensionless,
    )

    angle = 0

    # orientation quaternion
    p1 = Spherocylinder(
        position=[0, 1, 0],
        orientation=np.array([np.cos(angle / 2),  0, 0, np.sin(angle / 2)]),
        linear_velocity=0.0 * np.array([np.cos(angle), np.sin(angle), 0]),
        angular_velocity=np.array([0, 0, 0]),
        l0=1,
    )

    p2 = Spherocylinder(
        position=[0, 1, 0],
        orientation=np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
        linear_velocity=-0.0 * np.array([np.cos(angle), np.sin(angle), 0]),
        angular_velocity=np.array([0, 0, 0]),
        l0=1,
    )
    sim.particles.append(p1)
    # sim.particles.append(p2)

    # control position of particle 2 using arrow keys
    # rotate particle 2 using q and e

    # fix a top down view. We are working in 2D
    sim.ax.azim = -90
    sim.ax.elev = 90

    sim.ax.disable_mouse_rotation()

    sim.run_simulation(num_frames=500, dt=0.1, interval=50, show_ghosts=True)
