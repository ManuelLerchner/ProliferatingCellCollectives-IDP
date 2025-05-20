
from scipy import optimize as opt
import numpy as np

from capsula import getDirectionVector, signed_distance_capsule
from stress import collide_stress


class ConstraintBlock:

    def __init__(self, delta0_, gamma_, gidI_, gidJ_, normI_, normJ_, posI_, posJ_, labI_,
                 labJ_, oneSide_, bilateral_, kappa_, gammaLB_):
        self.delta0 = delta0_
        """Constraint initial value"""
        self.gamma = gamma_
        """Force magnitude, could be an initial guess"""
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
        self.oneSide = oneSide_
        """Flag for one-sided constraint. Particle J does not appear in mobility matrix"""
        self.bilateral = bilateral_
        """If this is a bilateral constraint or not"""
        self.kappa = kappa_
        """Spring constant. 0 means no spring"""
        self.gammaLB = gammaLB_
        """Lower bound of gamma for unilateral constraints"""
        self.stress = None
        """Stress tensor for this constraint, if applicable"""

    def __repr__(self):
        return (f"ConstraintBlock(delta0={self.delta0}, gamma={self.gamma}, "
                f"gidI={self.gidI}, gidJ={self.gidJ}, normI={self.normI}, "
                f"normJ={self.normJ}, posI={self.posI}, posJ={self.posJ}, "
                f"labI={self.labI}, labJ={self.labJ}, oneSide={self.oneSide}, "
                f"bilateral={self.bilateral}, kappa={self.kappa}, gammaLB={self.gammaLB})")


def getConstraints(C, L):

    n = len(L)

    constraints = []

    S = np.zeros(n)

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
            buffer = 0.1
            if sep < buffer:

                gamma = -sep if sep < 0 else 0

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
                    gamma,
                    i,
                    j,
                    normI,
                    normJ,
                    posI,
                    posJ,
                    Ploc,
                    Qloc,
                    False,
                    True,
                    0.0,
                    0.0
                )

                constraints.append(conBlock)

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

    return constraints, S


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


def solve_rigid_contact_problem(M, q, gamma0=None):
    """
    Solve the LCP: 0 ≤ Mγ + q ⊥ γ ≥ 0
    Using SciPy's optimization for simplicity and stability
    """

    n = len(q)

    # Define objective function: min 0.5*γᵀMγ + qᵀγ
    def objective(gamma):
        return 0.5 * np.dot(gamma, np.dot(M, gamma)) + np.dot(q, gamma)

    # Gradient of the objective
    def gradient(gamma):
        return np.dot(M, gamma) + q

    # Initial guess
    if gamma0 is not None:
        gamma_init = gamma0
    else:
        gamma_init = np.zeros(n)

    # Bounds: γ ≥ 0
    bounds = [(0, None) for _ in range(n)]

    # Solve the optimization problem
    result = opt.minimize(
        objective,
        gamma_init,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'ftol': 1e-8, 'gtol': 1e-8}
    )

    return result.x
