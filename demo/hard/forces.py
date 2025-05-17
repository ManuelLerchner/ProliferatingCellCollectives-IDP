import numpy as np
from capsula import signed_distance_capsule


def calc_friction_forces(U, L):
    """
    Calculate the friction force on a particle.
    Parameters:
    U (array-like): Velocity vector
    L (array-like): Length vector
    Returns:
    np.ndarray: The friction force vector
    """
    n = len(L)

    Un = np.array(np.split(U, 2*n)[::2])
    Omegan = np.array(np.split(U, 2*n)[1::2])

    # Calculate the friction force
    XI = 0.1
    Fn = -XI * L * Un

    # Calculate the torque
    Tn = -XI * L**3/12 * Omegan

    # interleave the forces and torques
    F = np.zeros((6*n, ))
    for i in range(n):
        F[6*i:6*i+3] = Fn[i]
        F[6*i+3:6*i+6] = Tn[i]

    return F


def calc_gravity_forces(L):
    """
    Calculate the gravitational forces on particles.
    Parameters:
    C (array-like): Particle positions and orientations
    L (array-like): Particle lengths
    Returns:
    np.ndarray: The gravitational force vector
    """
    n = len(L)

    direction = np.array([0, 0, -1])
    g = 9.81

    F = np.zeros((6*n, ))
    for i in range(n):
        # Extract the quaternion

        F[6*i:6*i+3] = direction * g * L[i]

    return F


def calc_herzian_collision_forces(C, L):

    n = len(L)

    F = np.zeros((6*n, ))

    for i in range(n):
        for j in range(i+1, n):
            xi = C[6*i:6*i+3]
            xj = C[6*j:6*j+3]

            distMin, yi, yj, _, _ = signed_distance_capsule(C, L, i, j)

            diameter = 0.5

            sep = distMin - (diameter + diameter) / 2
            if sep < 0:

                norm = (yj - yi) / np.linalg.norm(yj - yi)

                # Calculate the force and torque
                F_n = norm * sep

                T_n = np.cross((yi - xi), F_n)
                T_m = np.cross((yj - xj), F_n)

                # Update the forces and torques
                F[6*i:6*i+3] += F_n
                F[6*i+3:6*i+6] += T_n

                F[6*j:6*j+3] -= F_n
                F[6*j+3:6*j+6] += T_m

    return F
