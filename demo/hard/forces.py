import numpy as np
from capsula import getDirectionVector, signed_distance_capsule
from constraint import (BBPGD, calculate_D_matrix,
                        create_deepest_point_constraints)
from generator import normalize_quaternions_vectorized
from physics import make_map
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

    constraints = set(create_deepest_point_constraints(
        C, L, overlap_tolerance=1e-6))

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


def calc_constraint_collision_forces_recursive(C, L, M, dt, U_known=None,
                                               max_iterations=10, eps=1e-5):
    # Initialize state variables
    C_prev = C
    gamma_prev = np.empty((0,))  # Previous gamma values
    phi_prev = np.empty((0,))  # Previous phi values
    D_prev = np.empty((len(L)*6, 0))  # Previous D matrix

    for iteration in range(max_iterations):
        # Generate new constraints from deepest penetrations
        new_constraints = set(create_deepest_point_constraints(C_prev, L))

        # Check convergence - if no significant overlaps, we're done
        max_overlap = max([-c.delta0 for c in new_constraints])
        if max_overlap < eps:
            if iteration == 0:
                return np.zeros((6 * len(L), ))
            return D_prev @ gamma_prev

        # Build constraint vectors and matrices
        phi_new = np.array([c.delta0 for c in new_constraints])
        gamma_new_init = np.array([c.gamma for c in new_constraints])
        D_new = calculate_D_matrix(new_constraints, len(L))

        # Concatenate with previous iteration data
        phi_current = np.concatenate([phi_prev, phi_new])
        gamma_current = np.concatenate([gamma_prev, gamma_new_init])
        D_current = np.concatenate([D_prev, D_new], axis=1)

        # Define LCP problem functions
        def gradient(gamma):
            """Gradient function for the LCP solver"""
            return phi_current + dt * D_current.T @ M @ D_current @ (gamma - gamma_current)

        def project(gradient_val, gamma):
            """Projection function for complementarity constraints"""
            return np.where(gamma > 0, gradient_val, np.minimum(gradient_val, 0))

        def residual(gradient_val, gamma):
            """Residual function for convergence check"""
            return np.linalg.norm(project(gradient_val, gamma), ord=np.inf)

        # Solve the Linear Complementarity Problem (LCP)
        gamma_next = BBPGD(gradient, residual, gamma_current, eps=eps)

        # Calculate force increment
        gamma_prev_padded = np.pad(
            gamma_prev, (0, len(gamma_current) - len(gamma_prev)), 'constant')
        gamma_diff = gamma_next - gamma_prev_padded

        # Update total forces
        df = D_current @ gamma_diff

        # Update configuration
        dU = M @ df
        G = make_map(C_prev)  # Configuration mapping
        dC = G @ dU
        C_prev = C_prev + dt * dC

        # Normalize quaternions to maintain unit length
        C_prev = normalize_quaternions_vectorized(C_prev)

        # Update separation distances
        phi_current = phi_current + dt * D_current.T @ dU

        # Prepare for next iteration
        gamma_prev = gamma_next
        phi_prev = phi_current
        D_prev = D_current

    print("Max iterations reached without convergence.")
    return D_prev @ gamma_prev
