import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from tqdm import tqdm
from Simulation import Simulation
from Spherocylinder import Spherocylinder
import cProfile

# Example usage


def main():
    # Create and run simulation with drag
    l0 = 1.0
    lambbda = 1e-3

    tao_growtscale = 30
    lambda_dimensionless = (tao_growtscale) * l0**2 * lambbda

    sim = Simulation(
        box_size=20,
        tao_growthrate=tao_growtscale,
        lambda_sensitivity=lambda_dimensionless,
    )

    angle = np.pi / 18

    p1 = Spherocylinder(
        position=[sim.box_size / 2, sim.box_size / 2],
        orientation=angle,
        linear_velocity=0.0 * np.array([np.cos(angle), np.sin(angle)]),
        angular_velocity=0.00,
        l0=1,
    )

    sim.particles.append(p1)

    bounding_radius = 1.5
    sim.ax.set_xlim(-bounding_radius, sim.box_size + bounding_radius)
    sim.ax.set_ylim(-bounding_radius, sim.box_size + bounding_radius)

    frames = 250

    ani = sim.run_simulation(frames=frames, base_dt=0.1, scaling_factor=0.0,
                             interval=10, show_ghosts=True, show_grid=True)

    plt.show(block=False)

    # move second speher with arrow keys. rotate with q and e keys

    def update_func(i, n):
        plt.pause(.01)
        progress_bar.update(1)

    with tqdm(total=frames, bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]") as progress_bar:

        ani.save("simulation.mp4", writer="ffmpeg",
                 dpi=100, progress_callback=update_func, fps=60)


if __name__ == "__main__":
    cProfile.run('main()', sort='time')
    # main()
