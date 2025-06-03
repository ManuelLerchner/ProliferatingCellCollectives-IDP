
import numpy as np


def BBPGD(gradient, residual, gamma, eps, max_iter=100000):
    grad = gradient(gamma)
    res = residual(grad, gamma)
    alpha = 1 / res if res != 0 else np.inf

    for iters in range(max_iter):
        gamma_new = np.maximum(0, gamma - alpha * grad)
        grad_new = gradient(gamma_new)
        res = residual(grad_new, gamma_new)

        if res <= eps:
            return gamma_new, iters

        alpha = ((gamma_new - gamma).T@(gamma_new - gamma)) / \
            ((gamma_new - gamma).T @ (grad_new - grad))
        gamma = gamma_new
        grad = grad_new

    if res > eps:
        print("Warning: BBPGD did not converge within the maximum number of iterations.")
        print(f"Final residual: {res}")

    return gamma, iters
