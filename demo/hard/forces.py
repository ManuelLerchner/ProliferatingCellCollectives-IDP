
import numpy as np
from constraint import (calculate_D_matrix, calculate_growth_rates,
                        create_deepest_point_constraints)
from generator import normalize_quaternions_vectorized
from optimization import BBPGD
from physics import make_map
from scipy.sparse import csr_matrix, hstack
from vtk import SimulationState


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


def calc_constraint_collision_forces_recursive(C, l, M, dt, linked_cells, eps,
                                               max_iterations=100):
    """
    Calculate collision forces with optional logging support.

    Args:
        C: Configuration vector (positions + quaternions)
        l: Particle lengths
        M: Mobility matrix
        dt: Time step
        linked_cells: Spatial data structure for neighbor finding
        eps: Convergence tolerance
        max_iterations: Maximum solver iterations
        timestep: Current simulation timestep (for logging)
        logger: Optional logger for simulation data

    Returns:
        Tuple of (updated_C, updated_l)
    """
    # Initialize state variables
    C_prev = C
    gamma_prev = np.zeros((0,))  # Previous gamma values
    phi_prev = np.zeros((0,))  # Previous phi values
    ldot_prev = np.zeros((len(l),))  # Previous growth rates

    D_prev = csr_matrix((len(l)*6, 0))  # Previous D matrix
    L_prev = csr_matrix((len(l), 0))  # Previous L matrix

    # Initialize accumulating force and torque variables
    n_particles = len(l)
    total_forces = np.zeros((n_particles, 3))
    total_torques = np.zeros((n_particles, 3))
    total_stresses = np.zeros((n_particles, 1))
    total_bbpgd_iterations = 0

    # Convergence tracking
    converged = False
    max_overlap = 0.0
    constraint_iterations = 0

    # Main solver loop
    while constraint_iterations < max_iterations and not converged:
        # Generate new constraints from deepest penetrations
        new_constraints = create_deepest_point_constraints(
            C_prev, l, linked_cells)

        # Build constraint vectors and matrices
        phi_new = np.array([c.delta0 for c in new_constraints])

        D_new = calculate_D_matrix(new_constraints, len(l))
        S, L_new, ldot_new = calculate_growth_rates(
            new_constraints,  l, -phi_new, 1, 10000)

        # Check convergence - if no significant overlaps, we're done
        max_overlap = max([-c.delta0 for c in new_constraints]
                          ) if new_constraints else 0

        converged = (max_overlap < eps)

        # If converged, prepare final length update and exit loop
        if converged:
            l_next = l + dt * ldot_new
            break

        # Continue with LCP solving if not converged
        # Concatenate with previous iteration data
        gamma_new_init = np.zeros((len(phi_new),))
        gamma_current = np.concatenate([gamma_prev, gamma_new_init])
        phi_current = np.concatenate([phi_prev, phi_new])

        D_current = hstack([D_prev, D_new])
        L_current = hstack([L_prev, L_new])

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
        gamma_next, iters = BBPGD(gradient, residual, gamma_current, eps=eps)

        # Calculate force increment
        gamma_prev_padded = np.pad(
            gamma_prev, (0, len(gamma_new_init)), 'constant')
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
        phi_next = phi_current + dt * D_current.T @ dU

        # Accumulate forces, torques, and velocities for this iteration
        for i in range(n_particles):
            total_forces[i] += df[6*i:6*i+3]
            total_torques[i] += df[6*i+3:6*i+6]
            total_stresses[i] += S[i]
        total_bbpgd_iterations += iters

        # Prepare for next iteration
        C_prev = C_next
        gamma_prev = gamma_next
        D_prev = D_current
        L_prev = L_current
        phi_prev = phi_next
        ldot_prev = ldot_new
        constraint_iterations += 1

    # Handle case where we didn't converge in the loop
    if not converged:
        print("Max iterations reached without convergence.")
        print(f"Max overlap: {max_overlap}")
        l_next = l + dt * ldot_new
        C_final = C_next if constraint_iterations > 0 else C_prev
    else:
        C_final = C_prev

    # Log final state with accumulated forces and torques

    final_state = SimulationState(
        C=C_final.copy(),
        l=l_next.copy(),
        max_overlap=max_overlap,
        forces=total_forces.copy(),
        torques=total_torques.copy(),
        stresses=total_stresses.copy(),
        constraint_iterations=constraint_iterations,
        avg_bbpgd_iterations=total_bbpgd_iterations /
        constraint_iterations if constraint_iterations > 0 else 0
    )
    return final_state
