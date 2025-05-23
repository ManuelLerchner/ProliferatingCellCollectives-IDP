
import cProfile
import matplotlib.pyplot as plt
import numpy as np
from quaternion import getDirectionVector
from forces import calc_constraint_collision_forces_recursive
from generator import (make_particle_length, make_particle_position,
                       normalize_quaternions_vectorized)
from matplotlib.animation import FuncAnimation
from physics import make_map, make_particle_mobility_matrix
from visualization import render_particles

import matplotlib.pyplot as plt

np.seterr(all='warn')


def init_particles():
    """Initialize particle positions, lengths, and velocities"""

    # seed
    np.random.seed(42)

    ps = []
    ls = []

    # for i in range(2):
    #     pos = np.random.rand(3)  # Random position in [-1, 1]^3
    #     pos[2] = 0.0  # Set z-coordinate to 0 for a flat plane

    #     q = np.random.rand(4)  # Random quaternion in x, y plane
    #     q[1] = 0.0  # Set y-component to 0 for a flat plane
    #     q[2] = 0.0  # Set z-component to 0 for a flat plane
    #     q /= np.linalg.norm(q)
    #     p = make_particle_position(pos, q[0], [q[1], q[2], q[3]])

    #     l = make_particle_length(1.0)

    #     ps.append(p)
    #     ls.append(l)

    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1.0, 0, 0.0])

    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([1.0, 0.0, 0.0, 0.0])

    p1 = make_particle_position(pos1, q1[0], [q1[1], q1[2], q1[3]])
    p2 = make_particle_position(pos2, q2[0], [q2[1], q2[2], q2[3]])

    l1 = make_particle_length(1.0)
    l2 = make_particle_length(1.0)

    ps.append(p1)
    ls.append(l1)
    ps.append(p2)
    ls.append(l2)

    L = np.concatenate(ls)
    C = np.concatenate(ps, axis=0)

    return C, L


def apply_force(C, M, F, dt):
    G = make_map(C)

    U = M @ F

    Cdot = G @ U

    # Update configuration
    C_new = C + dt * Cdot

    # Normalize quaternions (if applicable)
    C_new = normalize_quaternions_vectorized(C_new)

    return C_new


def qMul(q1, q2):
    """Quaternion multiplication"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def grow(L, dt, tao=0.1, lamb=0.1):
    growth = (L / tao) * np.exp(-lamb * 1)
    L += growth * dt

    L[L > 2] = 2
    return L


def divideCells(C, L):
    for i in range(len(L)):
        if L[i] >= 2:
            xi = C[7*i:7*i+3]
            q = C[7*i+3:7*i+7]

            dir = getDirectionVector(q)

            newCenterLeft = xi - dir * (0.25 * L[i])
            newCenterRight = xi + dir * (0.25 * L[i])

            angle = np.random.uniform(-np.pi / 32.0,
                                      np.pi / 32.0) / (1 + L[i])

            dqLeft = np.array([np.cos(angle), 0.0, 0.0, np.sin(angle)])
            dqRight = np.array([np.cos(-angle), 0.0, 0.0, np.sin(-angle)])

            newOrientationLeft = qMul(q, dqLeft)
            newOrientationRight = qMul(q, dqRight)

            # Create new particles
            newParticleLeft = np.concatenate(
                (newCenterLeft, newOrientationLeft))
            newParticleRight = np.concatenate(
                (newCenterRight, newOrientationRight))

            C[7*i:7*i+7] = newParticleLeft
            L[i] = 1

            C = np.concatenate((C, newParticleRight))
            L = np.concatenate((L, [1]))

    return C, L


def simulation_step(C, L, dt):
    """Perform one simulation step and return updated configuration"""
    # Calculate forces

    # grow the particles
    # L = grow(L, dt, 1, 0.05)
    C, L = divideCells(C, L)

    # Update positions

    M = make_particle_mobility_matrix(L)

    C_new, L_new = calc_constraint_collision_forces_recursive(
        C, L, M, dt, eps=0.001)

    # apply_force(C, M, calc_herzian_collision_forces(C, L), dt)

    return C_new, L_new


def main():
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=90, azim=-90)
    ax.set_title(f"3D Particle Visualization")

    BOX = 6

    ax.set_xlim((-BOX, BOX))
    ax.set_ylim((-BOX, BOX))
    ax.set_zlim((-BOX, BOX))

    # Initialize particles
    C0, L0 = init_particles()

    # Time step
    dt = 0.01

    # Store configurations for animation
    C = C0
    L = L0

    def update(frame):
        nonlocal C, L
        print(f"\rFrame: {frame}, Particles: {len(L)}", end="")

        if frame == 300:
            pass

        C, L = simulation_step(C, L, dt)

        # Render the updated particles
        render_particles(ax, C, L)

        # Return the artists that were modified
        return ax,

    # Create the animation
    anim = FuncAnimation(
        fig,
        update,
        frames=600,  # Number of frames to generate
        interval=50,  # Delay between frames in milliseconds
        blit=False,   # Redraw the full figure (needed for 3D)
        repeat=False  # Don't loop the animation
    )

    # Show the plot

    # save the animation as a GIF
    # anim.save('particles.mp4', fps=30, dpi=100)
    plt.show()


if __name__ == "__main__":
    # cProfile.run('main()', sort='time')
    main()
