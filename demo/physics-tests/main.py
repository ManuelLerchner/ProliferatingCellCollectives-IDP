import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from tqdm import tqdm
from Simulation import Simulation
from Spherocylinder import Spherocylinder


# Example usage
if __name__ == "__main__":
    # Create and run simulation with drag
    l0 = 1.0
    lambbda = 1e-2

    tao_growtscale = 30
    lambda_dimensionless = (tao_growtscale) * l0**2 * lambbda

    sim = Simulation(
        box_size=10,
        tao_growthrate=tao_growtscale,
        lambda_sensitivity=lambda_dimensionless,
    )

    angle = np.pi / 18

    p1 = Spherocylinder(
        position=[5.01, 3],
        orientation=angle,
        linear_velocity=0.0 * np.array([np.cos(angle), np.sin(angle)]),
        angular_velocity=0.01,
        l0=1,
    )

    p2 = Spherocylinder(
        position=[3, 3],
        orientation=angle,
        linear_velocity=0.0 * np.array([np.cos(angle), np.sin(angle)]),
        angular_velocity=-0.01,
        l0=1,
    )
    sim.particles.append(p1)
    sim.particles.append(p2)

    bounding_radius = 1.5
    sim.ax.set_xlim(-bounding_radius, sim.box_size + bounding_radius)
    sim.ax.set_ylim(-bounding_radius, sim.box_size + bounding_radius)

    frames = 1000

    ani = sim.run_simulation(frames=frames, base_dt=0.5, scaling_factor=0.4,
                             interval=100, show_ghosts=True)

    plt.show(block=False)

    def update_func(i, n):
        plt.pause(.01)
        progress_bar.update(1)

    with tqdm(total=frames, bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]") as progress_bar:

        ani.save("simulation.mp4", writer="ffmpeg",
                 dpi=100, progress_callback=update_func, fps=60)
