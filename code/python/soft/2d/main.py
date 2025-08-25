
import time

import numpy as np
from matplotlib import pyplot as plt
from Simulation import Simulation
from Spherocylinder import Spherocylinder
from tqdm import tqdm

# Example usage


with open("log.csv", "w") as f:
    f.write("time_game,time_total,particles,ups\n")
    t_start = time.time()

    def main():
        # Create and run simulation with drag
        l0 = 1.0
        lambbda = 1e-3

        tao_growtscale = 30
        lambda_dimensionless = (tao_growtscale) * l0**2 * lambbda

        sim = Simulation(
            tao_growthrate=tao_growtscale,
            lambda_sensitivity=lambda_dimensionless,
        )

        p1 = Spherocylinder(
            position=np.array([0.0, 0.0]),
            orientation=0,
            linear_velocity=np.array([0.0, 0.0]),
            angular_velocity=0.00,
            l0=1,
        )

        sim.particles.append(p1)

        end_time = 300

        ani = sim.run_simulation(end_time=end_time, base_dt=2**(-5), scaling_factor=0.0,
                                 interval=10, show_ghosts=True, show_grid=True)

        plt.show(block=False)

        # move second speher with arrow keys. rotate with q and e keys

        def update_func(i, n):

            def expected_time_min(gt):
                return (np.exp(-0.31 + 0.032 * gt) + 12.01) / 1.6 / 60

            remaining_time = expected_time_min(
                end_time) - expected_time_min(sim.total_time)

            progress_bar.set_description(
                f'Δt: {sim.dt:.2f}, Particles: {len(sim.particles)}')

            progress_bar.set_postfix({'time_left': f"{remaining_time:.2f} min",
                                      'ups': sim.ups})
            progress_bar.n = float(f"{sim.total_time:.2f}")
            progress_bar.refresh()

            plt.gca().relim()
            plt.gca().autoscale_view()
            plt.pause(0.01)

            f.write(
                f"{sim.total_time},{time.time()-t_start},{len(sim.particles)},{sim.ups}\n")

        with tqdm(total=end_time, bar_format="{l_bar}{bar}{r_bar}") as progress_bar:
            ani.save("simulation.mp4", writer="ffmpeg",
                     dpi=100, progress_callback=update_func, fps=30)

    if __name__ == "__main__":
        # cProfile.run('main()', sort='time')
        main()
