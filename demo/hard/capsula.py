import numpy as np

from quaternion import rotateVectorOfPositions


def signed_seperation_points(y, alpha, n, m):
    yn = y[n][alpha]
    ym = y[m][alpha]

    # Calculate the signed separation
    distance = np.linalg.norm(yn - ym)
    signed_distance = distance * \
        np.sign(np.dot(yn - ym, y[n][alpha] - y[m][alpha]))
    return signed_distance


def getDirectionVector(q):
    defaultDirection = [np.array([1.0, 0.0, 0.0])]
    return rotateVectorOfPositions(q, defaultDirection)[0]


def signed_distance_capsule(C, L, n, m):
    x1 = C[7*n:7*n + 3]  # position of capsule
    s1 = C[7*n + 3:7*n + 4]  # scalar of quaternion
    w1 = C[7*n + 4:7*n + 7]  # vector of quaternion

    x2 = C[7*m:7*m + 3]  # position of capsule
    s2 = C[7*m + 3:7*m + 4]  # scalar of quaternion
    w2 = C[7*m + 4:7*m + 7]  # vector of quaternion

    # P0 and P1 are the endpoints of the capsule
    dir1 = getDirectionVector([s1, w1[0], w1[1], w1[2]])
    dir2 = getDirectionVector([s2, w2[0], w2[1], w2[2]])

    # TODO: Radius of the capsule?
    P0 = x1 - (L[n] / 2) * dir1
    P1 = x1 + (L[n] / 2) * dir1

    Q0 = x2 - (L[m] / 2) * dir2
    Q1 = x2 + (L[m] / 2) * dir2

    def R(s, t):
        p = np.array([s*t])
        M = np.array([[np.dot(P1-P0, P1-P0), -np.dot(P1-P0, Q1-Q0)],
                      [-np.dot(P1-P0, Q1-Q0), np.dot(Q1-Q0, Q1-Q0)]])

        K = np.array([(np.dot(P1-P0, P0-Q0)), (np.dot(Q1-Q0, P0-Q0))])

        f = np.dot(P0-Q0, P0-Q0)

        return p@M@p + 2*K@p + f

    # constrained conjugate gradient approach

    def f(s, t):
        p = np.array([s, t])
        M = np.array([[np.dot(P1-P0, P1-P0), -np.dot(P1-P0, Q1-Q0)],
                      [-np.dot(P1-P0, Q1-Q0), np.dot(Q1-Q0, Q1-Q0)]])

        K = np.array([(np.dot(P1-P0, P0-Q0)), (np.dot(Q1-Q0, P0-Q0))])

        f = np.dot(P0-Q0, P0-Q0)

        return p@M@p + 2*K@p + f

    def f_prime(s, t):
        p = np.array([s, t])
        M = np.array([[np.dot(P1-P0, P1-P0), -np.dot(P1-P0, Q1-Q0)],
                      [-np.dot(P1-P0, Q1-Q0), np.dot(Q1-Q0, Q1-Q0)]])

        K = np.array([(np.dot(P1-P0, P0-Q0)), (np.dot(Q1-Q0, P0-Q0))])

        f = np.dot(P0-Q0, P0-Q0)

        return 2*M@p + 2*K

    # Initial guess
    s0 = 0.5
    t0 = 0.5

    # Gradient descent
    alpha = 0.01
    for _ in range(100):
        grad = f_prime(s0, t0)
        s0 -= alpha * grad[0]
        t0 -= alpha * grad[1]
        if np.linalg.norm(grad) < 1e-6:
            break

    # Calculate intersection point
    s = s0
    t = t0

    P = P0 + s * (P1 - P0)
    Q = Q0 + t * (Q1 - Q0)

    # radius

    normal = (P - Q) / np.linalg.norm(P - Q)

    # points at surface where collision occurs
    radius = 1  # TODO
    P_surface = P + normal * radius
    Q_surface = Q - normal * radius

    # Calculate the signed distance
    signed_distance = np.linalg.norm(P_surface - Q_surface)
    return signed_distance, P_surface, Q_surface, normal
