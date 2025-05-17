deltaT = 1


def g_fun(gamma):

    # find argmin gamma >=0

    def y_dot(P, n):
        # caluclate speed of poiint P on rod n
        x = C[7*n:7*n + 3]

        v_linear = U[6*n:6*n + 3]
        w_angular = U[6*n + 3:6*n + 6]

        # Calculate the velocity of point P
        v_point = v_linear + np.cross(w_angular, P - x)
        return v_point

    def calc_distances():
        phi = []
        phi_dot = []
        for n in range(len(C)//7):
            for m in range(n+1, len(C)//7):
                # todo n should be surface normal

                s, yN, yM, norm = signed_distance_capsule(C, L, n, m)

                phi.append(s)

                phi_dot_alpha = np.dot(y_dot(yN, n), norm) + \
                    np.dot(y_dot(yM, n), norm)
                phi_dot.append(phi_dot_alpha)

        return np.array(phi), np.array(phi_dot)

    phi, phi_dot = calc_distances()

    return phi+deltaT * phi_dot


def residual(gamma):
    def projectedGradientDescent(X, gamma):
        X[gamma > 0] = np.minimum(0, X[gamma > 0])
        X[gamma <= 0] = np.maximum(0, X[gamma <= 0])
        return X

    return np.linalg.norm(projectedGradientDescent(g_fun(gamma), gamma), ord=np.inf)


def barzilaiBorweinProjectedGradientDescent(steps=100, eps=1e-6):
    gamma = np.zeros((len(C)//7 * (len(C)//7 - 1))//2)
    g = g_fun(gamma)
    res = residual(gamma)
    alpha = 1/res

    for i in range(steps):
        gamma_old = gamma
        gamma = np.maximum(gamma - alpha * g, 0)
        g_old = g
        g = g_fun(gamma)
        res = residual(gamma)
        if res < eps:
            break
        alpha = (gamma - gamma_old).T @ (gamma - gamma_old) / \
            ((gamma - gamma_old).T @ (g - g_old))
        if np.isnan(alpha) or alpha < 0:
            alpha = 0.1

    return gamma


gamma = barzilaiBorweinProjectedGradientDescent()
print("Gamma:", gamma)