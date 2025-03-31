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
        box_size=60,
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

    end_time = 250

    ani = sim.run_simulation(end_time=end_time, base_dt=0.01, scaling_factor=0.0,
                             interval=10, show_ghosts=True, show_grid=True)

    plt.show(block=False)

    # move second speher with arrow keys. rotate with q and e keys

    def update_func(i, n):
        plt.pause(.01)
        progress_bar.n = sim.total_time
        progress_bar.refresh()

    with tqdm(total=end_time, bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]") as progress_bar:

        ani.save("simulation.mp4", writer="ffmpeg",
                 dpi=100, progress_callback=update_func, fps=30)


if __name__ == "__main__":
    cProfile.run('main()', sort='time')
    # main()
