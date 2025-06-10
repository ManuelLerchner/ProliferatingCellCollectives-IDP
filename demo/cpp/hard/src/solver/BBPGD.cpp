#include "BBPGD.h"

#include <iostream>

VecWrapper BBPGD(
    std::function<VecWrapper(const VecWrapper&)> gradient,
    std::function<double(const VecWrapper&, const VecWrapper&)> residual,
    const VecWrapper& gamma0,
    const SolverConfig& config) {
  // Initialize gamma with gamma0
  VecWrapper gamma;
  VecDuplicate(gamma0.get(), gamma.get_ref());
  VecCopy(gamma0.get(), gamma.get());

  // Compute initial gradient and residual
  VecWrapper g = gradient(gamma);

  double res = residual(g, gamma);

  // Initial step size
  double alpha = 1.0 / res;

  // Storage for previous values
  VecWrapper gamma_prev;
  VecDuplicate(gamma.get(), gamma_prev.get_ref());

  VecWrapper g_prev;
  VecDuplicate(g.get(), g_prev.get_ref());

  // Create zero vector for projection (reuse in loop)
  VecWrapper zero_vec;
  VecDuplicate(gamma.get(), zero_vec.get_ref());
  VecZeroEntries(zero_vec.get());

  int i = 0;

  for (i = 0; i < config.max_iterations; i++) {
    // Store previous values
    VecCopy(gamma.get(), gamma_prev.get());
    VecCopy(g.get(), g_prev.get());

    // Step 6: γi := max(γi−1 − αi−1*gi−1, 0)
    // First compute gamma - alpha * g
    VecAXPY(gamma.get(), -alpha, g.get());  // gamma = gamma - alpha * g

    // Project onto non-negative orthant using PETSc's built-in function
    VecPointwiseMax(gamma.get(), gamma.get(), zero_vec.get());  // gamma = max(gamma, 0)

    // Step 7: gi := g(γi)
    g = gradient(gamma);

    // Step 8: resi := res(γi)
    res = residual(g, gamma);

    // Step 9-11: Check convergence
    if (res <= config.tolerance) {
      break;
    }

    // Step 12: Compute new step size using Barzilai-Borwein formula
    // αi := (γi − γi−1)^T (γi − γi−1) / (γi − γi−1)^T (gi − gi−1)

    VecWrapper delta_gamma;
    VecDuplicate(gamma.get(), delta_gamma.get_ref());
    VecCopy(gamma.get(), delta_gamma.get());
    VecAXPY(delta_gamma.get(), -1.0, gamma_prev.get());  // delta_gamma = gamma - gamma_prev

    VecWrapper delta_g;
    VecDuplicate(g.get(), delta_g.get_ref());
    VecCopy(g.get(), delta_g.get());
    VecAXPY(delta_g.get(), -1.0, g_prev.get());  // delta_g = g - g_prev

    PetscScalar numerator, denominator;
    VecDot(delta_gamma.get(), delta_gamma.get(), &numerator);  // ||δγ||²
    VecDot(delta_gamma.get(), delta_g.get(), &denominator);    // δγ^T δg

    // Avoid division by zero
    alpha = PetscRealPart(numerator) / PetscRealPart(denominator);
    // Ensure alpha is positive and reasonable
    alpha = std::max(1e-12, std::min(alpha, 1e12));
  }

  if (i == config.max_iterations) {
    PetscPrintf(PETSC_COMM_WORLD, "BBPGD did not converge after %d iterations, residual: %e\n", i + 1, res);
  }

  return std::move(gamma);
}