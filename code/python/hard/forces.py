import numpy as np
from constraint import (calculate_D_matrix, calculate_growth_rates,
                        calculate_stress_matrix,
                        create_deepest_point_constraints)
from generator import normalize_quaternions_vectorized
from optimization import BBPGD
from physics import make_map, make_particle_mobility_matrix
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


def calc_constraint_collision_forces_recursive(state: SimulationState, dt, linked_cells, eps,
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
    C_prev = state.C
    l = state.l
    gamma_prev = np.zeros((0,))  # Previous gamma values
    phi_prev = np.zeros((0,))  # Previous phi values
    ldot_prev = np.zeros((len(l),))  # Previous growth rates

    D_prev = csr_matrix((len(l)*6, 0))  # Previous D matrix
    L_prev = csr_matrix((len(l), 0))  # Previous L matrix

    # Initialize accumulating force and torque variables
    n_particles = len(l)
    total_bbpgd_iterations = 0

    # Convergence tracking
    converged = False
    max_overlap = 0.0
    constraint_iterations = 0

    # ecoli
    xi = 200 * 3600
    TAU = (54*60)

    # tolerance agains growth
    #  LAMBDA_ecoli = 2.44e-1
    #  LAMBDA = 0 -> pressure has no effect on growth -> exponential growth

    LAMBDA = 2.44e-1

    LAMBDA_DIMENSIONLESS = (TAU / (xi * state.l0**2)) * LAMBDA

    M = make_particle_mobility_matrix(state.l, xi)
    G = make_map(C_prev)  # Configuration mapping

    total_constraints = []

    # Main solver loop
    while constraint_iterations < max_iterations and not converged:
        # Generate new constraints from deepest penetrations
        new_constraints = create_deepest_point_constraints(
            C_prev, l, 0.5*state.l0, linked_cells, phase=constraint_iterations)
        total_constraints.extend(new_constraints)

        # Build constraint vectors and matrices
        phi_new = np.array([c.delta0 for c in new_constraints])
        phi_current = np.concatenate([phi_prev, phi_new])

        D_new = calculate_D_matrix(new_constraints, len(l))
        D_current = hstack([D_prev, D_new])

        gamma_new = np.zeros((len(new_constraints),))

        gamma_current = np.concatenate([gamma_prev, gamma_new])

        L_new = calculate_stress_matrix(new_constraints, len(l))
        L_current = hstack([L_prev, L_new])

        max_overlap = np.max(
            np.where(phi_current < 0, -phi_current, 0)) if len(phi_current) > 0 else 0.0
        converged = (max_overlap <= eps) and constraint_iterations > 0

        gamma_prev_padded = np.pad(
            gamma_prev, (0, len(gamma_current) - len(gamma_prev)), 'constant')

        def calculate_ldot(gamma):
            if constraint_iterations == 0:
                l_current = np.zeros((len(l),))
            else:
                l_current = l

            sigma = L_current @ gamma
            I, ldot_new = calculate_growth_rates(
                l_current, sigma, LAMBDA_DIMENSIONLESS, TAU)
            return I, ldot_new

        def gradient(gamma):
            """Gradient function for the LCP solver"""
            if len(gamma) == 0:
                return np.zeros((len(gamma),))

            phi_movement = D_current.T @ M @ D_current @ (
                gamma - gamma_prev_padded)

            _, ldot = calculate_ldot(gamma)

            phi_growth = - L_current.T @ (ldot - ldot_prev)

            return phi_current + dt * (phi_movement + phi_growth)

        def residual(gradient_val, gamma):
            """Residual function for convergence check"""
            projected = np.where(gamma > 0, gradient_val,
                                 np.minimum(gradient_val, 0))
            return np.linalg.norm(projected, ord=np.inf) if len(projected) > 0 else 0

        # Solve the Linear Complementarity Problem (LCP)
        gamma_next, iters = BBPGD(gradient, residual, gamma_current, eps=eps)

        # Calculate force increment

        gamma_diff = gamma_next - gamma_prev_padded

        impedance, ldot_new = calculate_ldot(gamma_next)

        # Update total forces
        df = D_current @ gamma_diff

        # Update configuration
        dU = M @ df

        dC = G @ dU
        C_prev = C_prev + dt * dC

        # Normalize quaternions to maintain unit length
        C_next = normalize_quaternions_vectorized(C_prev)

        # Update separation distances
        phi_next = gradient(gamma_next)

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

    l_next = l + dt * ldot_prev

    # Log final state with accumulated forces and torques

    U = D_current @ gamma_next
    U_2d = U.reshape(n_particles, 6)
    total_forces = U_2d[:, :3]
    total_torques = U_2d[:, 3:]
    impedance = impedance

    final_state = SimulationState(
        C=C_prev.copy(),
        l=l_next.copy(),
        L=L_prev.copy(),
        max_overlap=max_overlap,
        forces=total_forces.copy(),
        torques=total_torques.copy(),
        impedance=impedance.copy(),
        constraint_iterations=constraint_iterations,
        avg_bbpgd_iterations=total_bbpgd_iterations /
        constraint_iterations if constraint_iterations > 0 else 0,
        l0=state.l0,
        constraints=total_constraints
    )
    return final_state
