import matplotlib.pyplot as plt
import numpy as np
from forces import (calc_friction_forces, calc_herzian_collision_forces)
from generator import (make_particle_length, make_particle_position,
                       normalize_quaternions_vectorized)
from matplotlib.animation import FuncAnimation
from physics import make_map, make_particle_mobility_matrix
from visualization import render_particles


def init_particles():
    """Initialize particle positions, lengths, and velocities"""
    p1 = make_particle_position([0, 0, 0], 1.0, [0, 0, 0])
    p2 = make_particle_position([0.4, 0.3, 0], 1.0, [0, 0, 0])

    l1 = make_particle_length(1.0)
    l2 = make_particle_length(1.0)

    L = np.stack((l1, l2))
    C = np.concatenate((p1, p2), axis=0)
    U = np.zeros((6*len(L), ))

    return C, L, U


def calc_forces(C, U, L):
    """Calculate forces on particles"""
    F_fric = calc_friction_forces(U, L)
    F_coll = calc_herzian_collision_forces(C, L)
    # F_gravity = calc_gravity_forces(L)
    return F_fric + F_coll  # + F_gravity


def simulation_step(C, L, U, dt):
    """Perform one simulation step and return updated configuration"""
    # Calculate forces
    F = calc_forces(C, U, L)

    # Update positions
    M = make_particle_mobility_matrix(L)
    G = make_map(C)

    U = M @ F

    Cdot = G @ U

    # Update configuration
    C_new = normalize_quaternions_vectorized(C + dt * Cdot)

    return C_new, U


def main():
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=90, azim=-90)
    ax.set_title(f"3D Particle Visualization")

    # Initialize particles
    C0, L0, U0 = init_particles()

    # Number of simulation steps per frame
    steps_per_frame = 1

    # Time step
    dt = 0.01

    # Store configurations for animation
    C = C0
    U = U0

    def update(frame):
        nonlocal C, U

        # Run multiple simulation steps per frame for smoother/faster animation
        for _ in range(steps_per_frame):
            C, U = simulation_step(C, L0, U, dt)

        # Render the updated particles
        render_particles(ax, C, L0)

        # Return the artists that were modified
        return ax,

    # Create the animation
    anim = FuncAnimation(
        fig,
        update,
        frames=1000,  # Number of frames to generate
        interval=50,  # Delay between frames in milliseconds
        blit=False,   # Redraw the full figure (needed for 3D)
        repeat=False  # Don't loop the animation
    )

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
