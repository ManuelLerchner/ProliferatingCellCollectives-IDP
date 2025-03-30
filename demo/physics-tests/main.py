import numpy as np
from Simulation import Simulation
from Spherocylinder import Spherocylinder

# Example usage
if __name__ == "__main__":
    # Create and run simulation with drag
    l0 = 1.0
    lambbda = 1e-3
    friction_factor = 0.1

    tao_growtscale = 30
    lambda_dimensionless = (tao_growtscale / friction_factor) * l0**2 * lambbda

    sim = Simulation(
        box_size=5,
        tao_growthrate=tao_growtscale,
        lambda_sensitivity=lambda_dimensionless,
    )

    angle = np.pi / 18

    p1 = Spherocylinder(
        position=[2.01, 3],
        orientation=angle,
        linear_velocity=0.0 * np.array([np.cos(angle), np.sin(angle)]),
        angular_velocity=0,
        l0=1,
    )

    p2 = Spherocylinder(
        position=[3, 3],
        orientation=angle,
        linear_velocity=0.0 * np.array([np.cos(angle), np.sin(angle)]),
        angular_velocity=0,
        l0=1,
    )
    sim.particles.append(p1)
    sim.particles.append(p2)

    bounding_radius = 2.5
    sim.ax.set_xlim(-bounding_radius, sim.box_size + bounding_radius)
    sim.ax.set_ylim(-bounding_radius, sim.box_size + bounding_radius)

    # control position of particle 2 using arrow keys
    # rotate particle 2 using q and e

    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            p2.position = np.array([event.xdata, event.ydata])
            sim.ax.set_title(f"Particle 2 position: {p2.position}")

    def on_key(event):
        if event.key == 'up':
            p2.position[1] += 0.01
        elif event.key == 'down':
            p2.position[1] -= 0.01
        elif event.key == 'left':
            p2.position[0] -= 0.01
        elif event.key == 'right':
            p2.position[0] += 0.01
        elif event.key == 'x':
            p2.orientation += np.pi / 100
        elif event.key == 'y':
            p2.orientation -= np.pi / 100
    sim.fig.canvas.mpl_connect('button_press_event', on_click)
    sim.fig.canvas.mpl_connect('key_press_event', on_key)

    sim.run_simulation(num_frames=500, dt=0.01, interval=50, show_ghosts=True)
