#include "BBPGD.h"

#include <algorithm>  // For std::min/max
#include <functional>
#include <iostream>
#include <utility>  // For std::move

// Assuming VecWrapper and SolverConfig are defined elsewhere
// #include "VecWrapper.h"
// #include "SolverConfig.h"

BBPGDResult BBPGD(
    std::function<void(const VecWrapper& input, VecWrapper& output)> gradient,
    std::function<double(const VecWrapper&, const VecWrapper&)> residual,
    const VecWrapper& gamma0,
    const SolverConfig& config) {
  // Initialize gamma with gamma0
  VecWrapper gamma;
  VecDuplicate(gamma0.get(), gamma.get_ref());
  VecCopy(gamma0.get(), gamma.get());

  VecWrapper g;
  VecDuplicate(gamma0.get(), g.get_ref());

  // Compute initial gradient and residual
  gradient(gamma, g);
  double res = residual(g, gamma);

  if (res <= config.tolerance) {
    return {.gamma = std::move(gamma), .bbpgd_iterations = 0, .residual = res};
  }

  // Initial step size
  double alpha = 1.0 / res;

  // Storage for previous values
  VecWrapper gamma_prev;
  VecDuplicate(gamma.get(), gamma_prev.get_ref());

  VecWrapper g_prev;
  VecDuplicate(g.get(), g_prev.get_ref());

  // Workspace for differences: s = gamma - gamma_prev and y = g - g_prev
  VecWrapper delta_gamma;
  VecDuplicate(gamma.get(), delta_gamma.get_ref());

  VecWrapper delta_g;
  VecDuplicate(g.get(), delta_g.get_ref());

  // Constant zero vector for projection (reused in loop)
  VecWrapper zero_vec;
  VecDuplicate(gamma.get(), zero_vec.get_ref());
  VecZeroEntries(zero_vec.get());

  long long iteration = 0;

  for (iteration = 0; iteration < config.max_bbpgd_iterations; iteration++) {
    // Store previous values
    VecCopy(gamma.get(), gamma_prev.get());
    VecCopy(g.get(), g_prev.get());

    // Step 6: γi := max(γi−1 − αi−1*gi−1, 0)
    // First compute gamma - alpha * g
    VecAXPY(gamma.get(), -alpha, g.get());  // gamma = gamma - alpha * g

    // Project onto non-negative orthant using PETSc's built-in function
    VecPointwiseMax(gamma.get(), gamma.get(), zero_vec.get());  // gamma = max(gamma, 0)

    // Step 7: gi := g(γi)
    gradient(gamma, g);

    // Step 8: resi := res(γi)
    res = residual(g, gamma);

    // Step 9-11: Check convergence
    if (res <= config.tolerance) {
      break;
    }

    // Step 12: Compute new step size using Barzilai-Borwein formula
    // αi := (γi − γi−1)^T (γi − γi−1) / (γi − γi−1)^T (gi − gi−1)

    // Compute delta_gamma = gamma - gamma_prev
    VecWAXPY(delta_gamma.get(), -1.0, gamma_prev.get(), gamma.get());

    // Compute delta_g = g - g_prev
    VecWAXPY(delta_g.get(), -1.0, g_prev.get(), g.get());

    PetscScalar numerator, denominator;
    VecDot(delta_gamma.get(), delta_gamma.get(), &numerator);  // ||δγ||²
    VecDot(delta_gamma.get(), delta_g.get(), &denominator);    // δγ^T δg

    // Safeguard against division by zero or very small denominator

    alpha = PetscRealPart(numerator) / PetscRealPart(denominator);
  }

  if (iteration == config.max_bbpgd_iterations) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  BBPGD did not converge after %lld iterations. Residual: %f", iteration, res);
  }

  // The destructors of the VecWrapper objects will automatically free the PETSc vectors.
  return {.gamma = std::move(gamma),
          .bbpgd_iterations = iteration,
          .residual = res};
}