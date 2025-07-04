#pragma once

#include <petsc.h>

#include <functional>

#include "util/Config.h"
#include "util/PetscRaii.h"

struct BBPGDResult {
  long long bbpgd_iterations;
  double residual;
};

template <typename GradientFunc, typename ResidualFunc>
BBPGDResult BBPGD(
    GradientFunc&& gradient,
    ResidualFunc&& residual,
    VecWrapper& gamma,
    const SolverConfig& solver_config);

template <typename GradientFunc, typename ResidualFunc>
inline BBPGDResult BBPGD(
    GradientFunc&& gradient,
    ResidualFunc&& residual,
    VecWrapper& gamma,  // in-out parameter
    const SolverConfig& config) {
  auto g_curr = VecWrapper::Like(gamma);

  // Compute initial gradient and residual
  gradient(gamma, g_curr);
  double res = residual(g_curr, gamma);

  if (res <= config.tolerance) {
    return {.bbpgd_iterations = 0, .residual = res};
  }

  // Initial step size
  double alpha = 1.0 / res;

  // Storage for next gradient
  auto g_next = VecWrapper::Like(g_curr);

  // Workspace for differences: s = gamma_next - gamma_curr and y = g_next - g_curr
  auto delta_gamma = VecWrapper::Like(gamma);
  auto delta_g = VecWrapper::Like(g_curr);

  // Constant zero vector for projection (reused in loop)
  auto zero_vec = VecWrapper::Like(gamma);
  VecZeroEntries(zero_vec);

  long long iteration = 0;

  for (iteration = 0; iteration < config.max_bbpgd_iterations; iteration++) {
    // We compute the next gamma into a temporary, then swap.
    auto gamma_next = VecWrapper::Like(gamma);

    // Step 6: γ_next := max(γ_curr − α*g_curr, 0)
    VecWAXPY(gamma_next, -alpha, g_curr, gamma);
    VecPointwiseMax(gamma_next, gamma_next, zero_vec);

    // Step 7: g_next := g(γ_next)
    gradient(gamma_next, g_next);

    // Step 8: res_next := res(γ_next)
    res = residual(g_next, gamma_next);

    // Step 9-11: Check convergence
    if (res <= config.tolerance) {
      gamma = std::move(gamma_next);
      break;
    }

    // Step 12: Compute new step size
    VecWAXPY(delta_gamma, -1.0, gamma, gamma_next);
    VecWAXPY(delta_g, -1.0, g_curr, g_next);

    PetscScalar s_dot_y;
    VecDot(delta_gamma, delta_g, &s_dot_y);

    double numerator_val;
    double denominator_val;

    if (iteration % 2 == 0) {
      PetscScalar s_dot_s;
      VecDot(delta_gamma, delta_gamma, &s_dot_s);
      numerator_val = PetscRealPart(s_dot_s);
      denominator_val = PetscRealPart(s_dot_y);
    } else {
      PetscScalar y_dot_y;
      VecDot(delta_g, delta_g, &y_dot_y);
      numerator_val = PetscRealPart(s_dot_y);
      denominator_val = PetscRealPart(y_dot_y);
    }

    if (std::abs(denominator_val) < 10 * std::numeric_limits<double>::epsilon()) {
      denominator_val += 10 * std::numeric_limits<double>::epsilon();
    }

    alpha = numerator_val / denominator_val;

    if (alpha < 10 * std::numeric_limits<double>::epsilon()) {
      PetscPrintf(PETSC_COMM_WORLD,
                  "\n  BBPGD step size is too small on iteration %lld. Stagnating.",
                  iteration);
      gamma = std::move(gamma_next);
      break;
    }

    // Prepare for next iteration by swapping buffers
    gamma = std::move(gamma_next);
    std::swap(g_curr, g_next);
  }

  if (iteration == config.max_bbpgd_iterations) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  BBPGD did not converge after %lld iterations. Residual: %f", iteration, res);
  }

  return {.bbpgd_iterations = iteration, .residual = res};
}