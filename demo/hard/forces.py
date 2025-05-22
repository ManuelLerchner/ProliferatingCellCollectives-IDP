import numpy as np
from capsula import getDirectionVector, signed_distance_capsule
from constraint import (BBPGD, SingleConstraintDeepestPoint,
                        calculate_D_matrix)
from stress import collide_stress


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
    Fn = -XI * np.reshape(L, (n, 1)) * Un

    # Calculate the torque
    Tn = -XI * np.reshape(L, (n, 1))**3/12 * Omegan

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
    S = np.zeros((n, ))

    for i in range(n):
        for j in range(i+1, n):
            xi = C[7*i:7*i+3]
            xj = C[7*j:7*j+3]

            qi = C[7*i+3:7*i+7]
            qj = C[7*j+3:7*j+7]

            orientationI = getDirectionVector(qi)
            orientationJ = getDirectionVector(qj)

            distMin, Ploc, Qloc, _, _ = signed_distance_capsule(C, L, i, j)

            diameter = 0.5

            sep = distMin - (diameter + diameter) / 2
            if sep < 0:

                norm = (Ploc - Qloc) / np.linalg.norm(Ploc - Qloc)

                # Calculate the force and torque
                F_n = norm * (-sep)

                posI = Ploc - xi
                posJ = Qloc - xj

                T_n = np.cross(posI, F_n)
                T_m = np.cross(posJ, F_n)

                # Update the forces and torques
                F[6*i:6*i+3] += F_n
                F[6*j:6*j+3] -= F_n

                F[6*i+3:6*i+6] += T_n
                F[6*j+3:6*j+6] -= T_m

                stress = collide_stress(
                    orientationI,
                    orientationJ,
                    xi,
                    xj,
                    L[i],
                    L[j],
                    0.5*diameter,
                    0.5*diameter,
                    1.0,
                    Ploc,
                    Qloc
                )

                compression = stress[0, 0] + stress[1, 1] + stress[2, 2]
                S[i] += compression
                S[j] += compression
    return F


def calc_constraint_collision_forces(C, L, M, dt, U_known=None):

    constraints, S = SingleConstraintDeepestPoint(C, L)

    if not constraints:
        return np.zeros((6 * len(L), )), np.zeros(len(L))

    D = calculate_D_matrix(constraints, len(L))

    phi = np.array([c.delta0 for c in constraints])
    gamma0 = np.array([c.gamma for c in constraints])

    def gradient(gamma):
        return phi+dt * D.T @ M @ D @ gamma

    def project(gradient, gamma):
        return np.where(gamma > 0, gradient, np.minimum(gradient, 0))

    def residual(gradient, gamma):
        return np.linalg.norm(project(gradient, gamma), ord=np.inf)

    gamma = BBPGD(gradient, residual, gamma0)

    F = D @ gamma

    return F
