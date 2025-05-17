import numpy as np


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


def calc_collision_forces(gamma):

    F = np.array([], dtype=float)
    T = np.array([], dtype=float)

    phi = []
    for n in range(len(C)//7):
        for m in range(n+1, len(C)//7):
            # todo n should be surface normal

            s, yN, yM, norm = signed_distance_capsule(C, L, n, m)
            phi.append(s)

            f = gamma[n] * norm

            t = np.cross(yN, norm * gamma[n])

            F = np.concatenate((F, f))
            T = np.concatenate((T, t))

    return F, T
