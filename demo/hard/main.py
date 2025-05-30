
import cProfile
import matplotlib.pyplot as plt
import numpy as np
from forces import calc_constraint_collision_forces_recursive
from generator import (make_particle_length, make_particle_position,
                       normalize_quaternions_vectorized)
from physics import make_map, make_particle_mobility_matrix
from quaternion import getDirectionVector
from scipy.sparse import csr_matrix, vstack
from verletlist import LinkedCellList
from vtk import SimulationState, VTKLogger, VTKSimulationLogger

np.seterr(all='warn')


def init_particles(l0):
    """Initialize particle positions, lengths, and velocities"""

    # seed
    np.random.seed(42)

    ps = []
    ls = []

    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([1.0, 0, 0.0])

    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([1.0, 0.0, 0.0, 0.0])

    p1 = make_particle_position(pos1, q1[0], [q1[1], q1[2], q1[3]])
    p2 = make_particle_position(pos2, q2[0], [q2[1], q2[2], q2[3]])

    l1 = make_particle_length(l0)
    l2 = make_particle_length(l0)

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


def divideCells(C, l, L, l0):
    for i in range(len(l)):
        if l[i] >= 2 * l0:
            xi = C[7*i:7*i+3]
            q = C[7*i+3:7*i+7]

            dir = getDirectionVector(q)

            newCenterLeft = xi - dir * (0.25 * l[i])
            newCenterRight = xi + dir * (0.25 * l[i])

            angle = np.random.uniform(-np.pi / 32.0,
                                      np.pi / 32.0) / (1 + l[i])

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
            l[i] = l0

            C = np.concatenate((C, newParticleRight))
            l = np.concatenate((l, [l0]))

    return C, l, L


t_last_saved = 0
t_last_saved_frame = 0


def simulation_step(state, linked_cells, dt, timestep=0, logger=None):
    """Perform one simulation step and return updated configuration"""
    # Calculate forces

    # grow the particles
    # L = grow(L, dt, 1, 0.05)
    C, l, L = divideCells(state.C, state.l, state.L, state.l0)
    state.C = C
    state.l = l
    state.L = L

    # Update positions

    final_state = calc_constraint_collision_forces_recursive(
        state, dt, linked_cells, eps=state.l0/1000)

    # every 10 seconds log
    global t_last_saved
    global t_last_saved_frame

    if (timestep - t_last_saved) * dt >= 60:
        t_last_saved = timestep

        if logger:
            logger.log_timestep_complete(t_last_saved_frame, dt, final_state)
        t_last_saved_frame += 1

    return final_state


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
    l0 = 1
    C, l = init_particles(l0)

    # Time step
    dt = 30

    # Store configurations for animation

    linked_cells = LinkedCellList(cutoff_distance=2.5 * l0)

    # Set up VTK logging
    vtk_logger = VTKLogger("vtk_output", prefix="Bacteria_Simulation_")
    simulation_logger = VTKSimulationLogger(vtk_logger)

    timestep = 0

    state = SimulationState(C=C, l=l, L=csr_matrix(np.ones((len(l), 1))), max_overlap=0.0, forces=np.zeros((len(l), 3)), torques=np.zeros(
        (len(l), 3)), stresses=np.zeros((len(l), 1)), constraint_iterations=0, avg_bbpgd_iterations=0, l0=l0)

    while timestep * dt < 60*60*60:
        print(f"\rMinutes: {timestep * dt / 60:.2f}, Particles: {len(state.l)}",
              end="", flush=True)
        final_state = simulation_step(state, linked_cells, dt,
                                      timestep=timestep, logger=simulation_logger)
        state = final_state

        timestep += 1


if __name__ == "__main__":
    main()
