#include "BBPGD.h"

#include <algorithm>  // For std::min/max
#include <cmath>      // For std::abs
#include <functional>
#include <iostream>
#include <limits>
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

    // Step 12: Compute new step size using alternating Barzilai-Borwein formula
    // αi_1 := (si−1^T si−1) / (si−1^T yi−1)
    // αi_2 := (si−1^T yi−1) / (yi−1^T yi−1)

    // Compute delta_gamma = gamma - gamma_prev
    VecWAXPY(delta_gamma.get(), -1.0, gamma_prev.get(), gamma.get());

    // Compute delta_g = g - g_prev
    VecWAXPY(delta_g.get(), -1.0, g_prev.get(), g.get());

    PetscScalar s_dot_y;
    VecDot(delta_gamma.get(), delta_g.get(), &s_dot_y);

    double numerator_val;
    double denominator_val;

    // Alternating BB1 and BB2 methods
    if (iteration % 2 == 0) {
      // Barzilai-Borwein step size Choice 1 (BB1)
      PetscScalar s_dot_s;
      VecDot(delta_gamma.get(), delta_gamma.get(), &s_dot_s);
      numerator_val = PetscRealPart(s_dot_s);
      denominator_val = PetscRealPart(s_dot_y);
    } else {
      // Barzilai-Borwein step size Choice 2 (BB2)
      PetscScalar y_dot_y;
      VecDot(delta_g.get(), delta_g.get(), &y_dot_y);
      numerator_val = PetscRealPart(s_dot_y);
      denominator_val = PetscRealPart(y_dot_y);
    }

    if (std::abs(denominator_val) < 10 * std::numeric_limits<double>::epsilon()) {
      denominator_val += 10 * std::numeric_limits<double>::epsilon();  // prevent div 0 error
    }

    alpha = numerator_val / denominator_val;

    if (alpha < 10 * std::numeric_limits<double>::epsilon()) {
      PetscPrintf(PETSC_COMM_WORLD,
                  "\n  BBPGD step size is too small on iteration %lld. Stagnating.",
                  iteration);
      break;
    }
  }

  if (iteration == config.max_bbpgd_iterations) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  BBPGD did not converge after %lld iterations. Residual: %f", iteration, res);
  }

  // The destructors of the VecWrapper objects will automatically free the PETSc vectors.
  return {.gamma = std::move(gamma),
          .bbpgd_iterations = iteration,
          .residual = res};
}