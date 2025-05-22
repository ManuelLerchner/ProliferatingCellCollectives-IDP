
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


def SingleConstraintDeepestPoint(C, L):

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


def BBPGD(gradient, residual, gamma, eps=0.5/5000, max_iter=100):
    grad = gradient(gamma)
    res = residual(grad, gamma)
    alpha = 1 / res

    for _ in range(max_iter):
        gamma_new = np.maximum(0, gamma - alpha * grad)
        grad_new = gradient(gamma_new)
        res_new = residual(grad_new, gamma_new)

        if res <= eps:
            break

        alpha = ((gamma_new - gamma).T@(gamma_new - gamma)) / \
            ((gamma_new - gamma).T @ (grad_new - grad))
        gamma = gamma_new
        grad = grad_new
        res = res_new

    if res > eps:
        print("Warning: BBPGD did not converge within the maximum number of iterations.")
        print(f"Final residual: {res}")
        print(f"Final gamma: {gamma}")
    return gamma
