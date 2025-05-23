import numpy as np
from distance import    signed_distance_capsule
from constraint import (calculate_D_matrix, calculate_growth_rates,
                        create_deepest_point_constraints)
from optimization import BBPGD
from generator import normalize_quaternions_vectorized
from physics import make_map


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


def calc_constraint_collision_forces_recursive(C, l, M, dt, eps,
                                               max_iterations=100):
    # Initialize state variables
    C_prev = C
    gamma_prev = np.empty((0,))  # Previous gamma values
    phi_prev = np.empty((0,))  # Previous phi values
    ldot_prev = np.empty((len(l),))  # Previous growth rates

    D_prev = np.empty((len(l)*6, 0))  # Previous D matrix
    L_prev = np.empty((len(l), 0))  # Previous L matrix

    for iteration in range(max_iterations):
        # Generate new constraints from deepest penetrations
        new_constraints = create_deepest_point_constraints(C_prev, l)

        # Build constraint vectors and matrices
        phi_new = np.array([c.delta0 for c in new_constraints])
        gamma_new_init = -phi_new

        D_new = calculate_D_matrix(new_constraints, len(l))
        L_new, ldot_new = calculate_growth_rates(
            new_constraints,  l, gamma_new_init, 1, 10000)

        # Check convergence - if no significant overlaps, we're done
        max_overlap = max([-c.delta0 for c in new_constraints]
                          ) if new_constraints else 0

        if max_overlap < eps:
            l_next = l + dt * ldot_new
            return C_prev, l_next

        # Concatenate with previous iteration data
        gamma_current = np.concatenate([gamma_prev, gamma_new_init])
        phi_current = np.concatenate([phi_prev, phi_new])

        D_current = np.concatenate([D_prev, D_new], axis=1)
        L_current = np.concatenate([L_prev, L_new], axis=1)

        # Define LCP problem functions

        def gradient(gamma):
            """Gradient function for the LCP solver"""
            phi_movement = D_current.T @ M @ D_current @ (
                gamma - gamma_current)
            phi_growth = - L_current.T @ (ldot_new - ldot_prev)
            return phi_current + dt * (phi_movement + phi_growth)

        def project(gradient_val, gamma):
            """Projection function for complementarity constraints"""
            return np.where(gamma > 0, gradient_val, np.minimum(gradient_val, 0))

        def residual(gradient_val, gamma):
            """Residual function for convergence check"""
            return np.linalg.norm(project(gradient_val, gamma), ord=np.inf)

        # Solve the Linear Complementarity Problem (LCP)
        gamma_next = BBPGD(gradient, residual, gamma_current, eps=eps)

        # if nan
        if np.isnan(gamma_next).any():
            print("NaN detected in gamma_next. Exiting.")
            BBPGD(gradient, residual, gamma_current, eps=eps)
            return C_prev, l

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
        C_next = normalize_quaternions_vectorized(C_prev)

        # Update separation distances
        phi_current = phi_current + dt * D_current.T @ dU

        # Prepare for next iteration
        C_prev = C_next
        gamma_prev = gamma_next
        phi_prev = phi_current
        D_prev = D_current
        L_prev = L_current

        if np.isnan(ldot_new).any():
            print("NaN detected in ldot_new. Exiting.")
            return C_next, l

        ldot_prev = ldot_new

    print("Max iterations reached without convergence.")
    print(f"Max overlap: {max_overlap}")
    l_next = l + dt * ldot_new
    return C_next, l_next
