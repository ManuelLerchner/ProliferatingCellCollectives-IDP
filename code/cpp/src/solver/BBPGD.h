#pragma once

#include <math.h>
#include <petsc.h>

#include <functional>
#include <optional>

#include "logger/BBPGDLogger.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

void NormAdotAB(const VecWrapper& a, const VecWrapper& b, double& a_norm, double& ab, int N) {
  const PetscScalar *a_array, *b_array;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(a, &a_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(b, &b_array));

  double a_norm_local = 0.0;
  double ab_local = 0.0;

#pragma omp parallel for reduction(+ : ab_local, a_norm_local)
  for (PetscInt i = 0; i < N; ++i) {
    a_norm_local += (a_array[i]) * (a_array[i]);
    ab_local += (a_array[i]) * (b_array[i]);
  }

  // Restore arrays
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(a, &a_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(b, &b_array));

  // Combine both values into a single reduction
  std::array<double, 2> local_values = {a_norm_local, ab_local};
  std::array<double, 2> global_values;
  globalReduce_v(local_values.data(), global_values.data(), 2, MPI_SUM);
  a_norm = global_values[0];
  ab = global_values[1];
}

struct BBPGDResult {
  size_t bbpgd_iterations;
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
    double allowed_residual,
    size_t max_bbpgd_iterations,
    size_t iter,
    int N) {
  std::optional<vtk::BBPGDLogger> bbpgd_logger;

  if (iter % 100 == 0) {
    bbpgd_logger = vtk::BBPGDLogger("logs", "bbpgdtrace", true);
  }

  auto phi_curr = VecWrapper::Like(gamma);

  // Compute initial gradient and residual

  gradient(gamma, phi_curr);
  double res = residual(phi_curr, gamma);

  if (res <= allowed_residual) {
    return {.bbpgd_iterations = 0, .residual = res};
  }

  // Initial step size
  double alpha = 1.0 / res;

  // Storage for next gradient
  auto g_next = VecWrapper::Like(phi_curr);

  auto delta_gamma = VecWrapper::Like(gamma);
  auto delta_phi = VecWrapper::Like(phi_curr);

  // Constant zero vector for projection (reused in loop)
  auto zero_vec = VecWrapper::Like(gamma);
  VecZeroEntries(zero_vec);

  size_t iteration = 0;

  for (iteration = 0; iteration < max_bbpgd_iterations; iteration++) {
    // We compute the next gamma into a temporary, then swap.
    auto gamma_next = VecWrapper::Like(gamma);

    // Step 6: γ_next := max(γ_curr − α*phi_curr, 0)
    VecWAXPY(gamma_next, -alpha, phi_curr, gamma);
    VecPointwiseMax(gamma_next, gamma_next, zero_vec);

    // Step 7: g_next := g(γ_next)
    gradient(gamma_next, g_next);

    // if (iteration % 10 == 0) {
    // Step 8: res_next := res(γ_next)
    res = residual(g_next, gamma_next);

    if (bbpgd_logger) {
      double grad_norm, dummy;
      NormAdotAB(g_next, g_next, grad_norm, dummy, N);
      double gamma_norm, dummy2;
      NormAdotAB(gamma_next, gamma_next, gamma_norm, dummy2, N);

      bbpgd_logger->collect({.iteration = iteration,
                             .residual = res,
                             .step_size = alpha,
                             .grad_norm = std::sqrt(grad_norm),
                             .gamma_norm = std::sqrt(gamma_norm)});
      bbpgd_logger->log();
    }

    // Step 9-11: Check convergence
    if (res <= allowed_residual) {
      gamma = std::move(gamma_next);
      break;
    }
    // }

    // Step 12: Compute new step size
    VecWAXPY(delta_gamma, -1.0, gamma, gamma_next);
    VecWAXPY(delta_phi, -1.0, phi_curr, g_next);

    double numerator_val;
    double denominator_val;

    // BB1 step: (d^T d)/(d^T g)
    NormAdotAB(delta_gamma, delta_phi, numerator_val, denominator_val, N);

    alpha = numerator_val / denominator_val;

    // Prepare for next iteration by swapping buffers
    gamma = std::move(gamma_next);
    std::swap(phi_curr, g_next);
  }

  if (iteration == max_bbpgd_iterations) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  BBPGD did not converge after %ld iterations. Residual: %f", iteration, res);
  }

  return {.bbpgd_iterations = iteration, .residual = res};
}
