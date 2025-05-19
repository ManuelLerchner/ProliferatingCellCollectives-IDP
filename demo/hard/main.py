import matplotlib.pyplot as plt
import numpy as np
from forces import (calc_constraint_collision_forces, calc_friction_forces)
from generator import (make_particle_length, make_particle_position,
                       normalize_quaternions_vectorized)
from matplotlib.animation import FuncAnimation
from physics import make_map, make_particle_mobility_matrix
from visualization import render_particles


def init_particles():
    """Initialize particle positions, lengths, and velocities"""

    # seed
    np.random.seed(42)

    ps = []
    ls = []

    for i in range(5):
        pos = np.random.rand(3)  # Random position in [-1, 1]^3
        pos[2] = 0.0  # Set z-coordinate to 0 for a flat plane

        q = np.random.rand(4)  # Random quaternion in x, y plane
        q[1] = 0.0  # Set y-component to 0 for a flat plane
        q[2] = 0.0  # Set z-component to 0 for a flat plane
        q /= np.linalg.norm(q)
        p = make_particle_position(pos, q[0], [q[1], q[2], q[3]])

        l = make_particle_length(1.0)

        ps.append(p)
        ls.append(l)

    L = np.stack(ls)
    C = np.concatenate(ps, axis=0)
    U = np.zeros((6*len(L), ))

    return C, L, U


def calc_forces(C, U, L, M, dt):
    """Calculate forces on particles"""
    F_fric = calc_friction_forces(U, L)
    # F_herz = calc_herzian_collision_forces(C, L)
    F_con = calc_constraint_collision_forces(C, L, M, dt)
    # F_gravity = calc_gravity_forces(L)
    return F_fric + F_con


def simulation_step(C, L, U, dt):
    """Perform one simulation step and return updated configuration"""
    # Calculate forces

    # Update positions
    M = make_particle_mobility_matrix(L)

    F = calc_forces(C, U, L, M, dt)

    G = make_map(C)

    U = M @ F

    Cdot = G @ U

    # Update configuration
    C_new = C + dt * Cdot

    # Normalize quaternions (if applicable)
    C_new = normalize_quaternions_vectorized(C_new)
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
