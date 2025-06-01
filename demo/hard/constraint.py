import numpy as np
from distance import signed_distance_capsule
from scipy.sparse import lil_matrix


class ConstraintBlock:

    def __init__(self, delta0_, gidI_, gidJ_, normI_, normJ_, posI_, posJ_, labI_,
                 labJ_, orientationI_, orientationJ_, phase_):
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
        self.phase = phase_
        """Phase of constraint"""

    def __repr__(self):
        return f"ConstraintBlock(gidI={self.gidI}, gidJ={self.gidJ}, delta0={self.delta0}, " \
            f"normI={self.normI}, normJ={self.normJ}, posI={self.posI}, posJ={self.posJ})"


def create_deepest_point_constraints(C, L, diameter, linked_cells, phase=0):
    """
    Optimized constraint creation using Verlet list
    """
    n = len(L)

    # Extract positions for Verlet list
    positions = np.array([C[7*i:7*i+3] for i in range(n)])

    # Update Verlet list if needed
    if linked_cells.needs_update(positions, L):
        linked_cells.build(positions, L)

    # Get close pairs (this replaces your nested loops!)
    cutoff = diameter * 2  # Adjust as needed
    close_pairs = linked_cells.get_close_pairs(positions, L, cutoff)

    constraints = []

    cache = {}

    for i, j in close_pairs:
        x1 = C[7*i:7*i + 3]
        q1 = C[7*i + 3:7*i + 7]

        x2 = C[7*j:7*j + 3]
        q2 = C[7*j + 3:7*j + 7]

        l1 = L[i]
        l2 = L[j]

        (distMin, Ploc, Qloc, s, t), orientationI, orientationJ = signed_distance_capsule(i,
                                                                                          x1, q1, l1, j, x2, q2, l2, diameter, cache=cache)

        sep = distMin - (diameter + diameter) / 2

        if sep < 0:
            norm = np.linalg.norm(Ploc - Qloc)
            if norm == 0:
                norm = np.array([1.0, 0.0, 0.0])
            else:
                norm = (Ploc - Qloc) / norm
            normI = norm
            normJ = -normI

            posI = Ploc - x1
            posJ = Qloc - x2

            conBlock = ConstraintBlock(
                sep, i, j, normI, normJ, posI, posJ, Ploc, Qloc,
                orientationI, orientationJ, phase)

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
    D = lil_matrix((6*num_bodies, num_constraints))

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

    return D.tocsr()


def calculate_stress_matrix(constraints, num_bodies):
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
    S = lil_matrix((num_bodies, num_constraints))

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

    # Convert to sparse matrix format
    S = S.tocsr()

    return S


def calculate_impedance_matrix(stress_matrix, lamb):
    # Calculate the impedance matrix
    I = np.exp(-lamb * stress_matrix)

    return I


def calculate_growth_rates(L, sigma,  lamb, tau):
    """
    Calculate actual growth rates with impedance: l_dot_n = (l_n / tau) * I_n(gamma)

    Args:
        particles: List of particle objects with length properties
        impedance: Impedance values for each particle
        tau: Growth timescale

    Returns:
        Growth rates for each particle
    """

    I = calculate_impedance_matrix(sigma, lamb)

    growth_rates = L / tau * I

    return growth_rates
