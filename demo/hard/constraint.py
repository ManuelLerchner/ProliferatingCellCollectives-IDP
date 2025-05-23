
import numpy as np
from distance import signed_distance_capsule
from quaternion import getDirectionVector
from matplotlib import pyplot as plt


class ConstraintBlock:

    def __init__(self, delta0_, gidI_, gidJ_, normI_, normJ_, posI_, posJ_, labI_,
                 labJ_, orientationI_, orientationJ_, ):
        self.delta0 = delta0_
        """Constraint initial value"""
        self.gidI = gidI_
        """Unique global ID of particle I"""
        self.gidJ = gidJ_
        """Unique global ID of particle J"""
        self.normI = normI_
        """Surface normal vector at the location of constraint (minimal separation) for particle I"""
        self.normJ = normJ_
        """Surface normal vector at the location of constraint (minimal separation) for particle J"""
        self.posI = posI_
        """Relative constraint position on particle I"""
        self.posJ = posJ_
        """Relative constraint position on particle J"""
        self.labI = labI_
        """Lab frame location of collision point on particle I"""
        self.labJ = labJ_
        """Lab frame location of collision point on particle J"""
        self.orientationI = orientationI_
        """Orientation of particle I"""
        self.orientationJ = orientationJ_
        """Orientation of particle J"""

    def __repr__(self):
        return f"ConstraintBlock(gidI={self.gidI}, gidJ={self.gidJ}, delta0={self.delta0}, " \
            f"normI={self.normI}, normJ={self.normJ}, posI={self.posI}, posJ={self.posJ})"


def create_deepest_point_constraints(C, L):

    n = len(L)

    constraints = []

    for i in range(n):
        for j in range(i+1, n):
            xi = C[7*i:7*i+3]
            xj = C[7*j:7*j+3]

            qi = C[7*i+3:7*i+7]
            qj = C[7*j+3:7*j+7]

            orientationI = getDirectionVector(qi)
            orientationJ = getDirectionVector(qj)

            distMin, Ploc, Qloc, s, t = signed_distance_capsule(C, L, i, j)

            diameter = 0.5

            sep = distMin - (diameter + diameter) / 2

            # continue if the distance is too small

            if sep < 0:

                norm = np.linalg.norm(Ploc - Qloc)
                if norm == 0:
                    # pick a default normal if the distance is zero
                    norm = np.array([1.0, 0.0, 0.0])
                else:
                    norm = (Ploc - Qloc) / norm
                normI = (Ploc - Qloc) / np.linalg.norm(Ploc - Qloc)

                normJ = -normI

                posI = Ploc - xi
                posJ = Qloc - xj

                conBlock = ConstraintBlock(
                    sep,
                    i,
                    j,
                    normI,
                    normJ,
                    posI,
                    posJ,
                    Ploc,
                    Qloc,
                    orientationI,
                    orientationJ,
                )

                constraints.append(conBlock)

    return constraints


def calculate_D_matrix(constraints, num_bodies):
    """
    Calculate the D matrix that maps constraint forces to body forces/torques

    Args:
        constraints: List of constraint blocks
        num_bodies: Number of rigid bodies in the system

    Returns:
        D matrix (6*num_bodies × len(constraints))
    """
    num_constraints = len(constraints)
    D = np.zeros((6*num_bodies, num_constraints))

    for c_idx, constraint in enumerate(constraints):
        # Get the bodies involved in this constraint
        body_i = constraint.gidI
        body_j = constraint.gidJ

        # Get the contact normal (from body i to body j)
        n = constraint.normI  # Using normI as the contact normal

        # Get the contact positions in world coordinates
        r_i = constraint.posI  # Position on body i
        r_j = constraint.posJ  # Position on body j

        # Force is in the direction of the normal
        D[6*body_i:6*body_i+3, c_idx] = n  # Linear force (negative normal)
        D[6*body_j:6*body_j+3, c_idx] = -n  # Linear force (positive normal)

        # Torque contributions
        D[6*body_i+3:6*body_i+6, c_idx] = np.cross(r_i, n)
        D[6*body_j+3:6*body_j+6, c_idx] = -np.cross(r_j, n)

    return D


def calculate_stress_matrix(constraints, num_bodies, gamma):
    """
    Calculate the stress matrix for the system
    Args:
        constraints: List of constraint blocks
        num_bodies: Number of rigid bodies in the system
        gamma: Constraint forces
    Returns:
        Stress matrix (num_bodies × len(constraints))
    """

    # Matrix mapping constraint forces to stress (sigma)
    # sigma_n = sum_alpha (1/2 |t_n · n_alpha| * gamma_alpha)

    num_constraints = len(constraints)
    S = np.zeros((num_bodies, num_constraints))

    for c_idx, constraint in enumerate(constraints):
        body_i = constraint.gidI
        body_j = constraint.gidJ

        # Get the contact normal
        n_alpha = constraint.normI  # Contact normal for constraint alpha

        # Get tangent vectors for both particles
        # Main axis direction for particle i
        t_i = constraint.orientationI
        # Main axis direction for particle j
        t_j = constraint.orientationJ

        # Calculate the stress contribution for this constraint
        S[body_i, c_idx] = 0.5 * abs(np.dot(t_i, n_alpha))
        S[body_j, c_idx] = 0.5 * abs(np.dot(t_j, n_alpha))

    sigma = S * gamma

    return S, sigma


def calculate_impedance_matrix(stress_matrix, lamb):
    # Calculate the impedance matrix
    I = np.exp(-lamb * np.sum(stress_matrix, axis=1))

    return I


def calculate_growth_rates(constraints, L, gamma, tau, lamb):
    """
    Calculate actual growth rates with impedance: l_dot_n = (l_n / tau) * I_n(gamma)

    Args:
        particles: List of particle objects with length properties
        impedance: Impedance values for each particle
        tau: Growth timescale

    Returns:
        Growth rates for each particle
    """
    num_bodies = len(L)

    S, sigma = calculate_stress_matrix(constraints, num_bodies, gamma)

    I = calculate_impedance_matrix(sigma, lamb)

    growth_rates = L / tau * I

    if np.isnan(sigma).any():
        raise ValueError("Growth rates contain NaN values.")
    if np.isnan(growth_rates).any():
        raise ValueError("Growth rates contain infinite values.")

    return sigma, growth_rates
