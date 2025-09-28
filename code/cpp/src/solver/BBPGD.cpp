#include "BBPGD.h"

void NormAdotAB(const VecWrapper& a, const VecWrapper& b, double& a_norm, double& ab) {
  const PetscScalar *a_array, *b_array;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(a, &a_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(b, &b_array));

  double a_norm_local = 0.0;
  double ab_local = 0.0;

  PetscInt N;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(a, &N));

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

BBPGDResult BBPGD(
    Gradient& G,
    VecWrapper& g0,  // in-out parameter
    double allowed_residual,
    size_t max_bbpgd_iterations,
    size_t iter) {
  std::optional<vtk::BBPGDLogger> bbpgd_logger;

  if (iter % 100 == 0) {
    bbpgd_logger = vtk::BBPGDLogger("logs", "bbpgdtrace", true);
  }

  auto& gamma_prev = g0;
  auto gamma_i = VecWrapper::Like(gamma_prev);

  auto g_prev = VecWrapper::Like(gamma_prev);
  auto g_i = VecWrapper::Like(gamma_prev);

  auto delta_gamma = VecWrapper::Like(gamma_prev);
  auto delta_phi = VecWrapper::Like(gamma_prev);

  auto zero_vec = VecWrapper::Like(gamma_prev);

  // Compute initial gradient and residual
  VecCopy(G.gradient(gamma_prev), g_prev);

  double res = G.residual(g_prev, gamma_prev);
  double alpha = 1.0 / res;

  size_t iteration = 1;
  for (; iteration < max_bbpgd_iterations; iteration++) {
    // Step 6: γ_next := max(γ_curr − α*g_prev, 0)
    VecWAXPY(gamma_i, -alpha, g_prev, gamma_prev);
    VecPointwiseMax(gamma_i, gamma_i, zero_vec);

    // Step 7: g_i := g(γ_next)
    auto& g_i = G.gradient(gamma_i);

    // Step 8: res_next := res(γ_next)

    // Step 9-11: Check convergence every 10 iterations
    // We do not check every iteration to save computation time
    if (iteration % 10 == 0) {
      res = G.residual(g_i, gamma_i);

      if (bbpgd_logger) {
        double grad_norm, dot;
        NormAdotAB(g_i, gamma_i, grad_norm, dot);

        bbpgd_logger->collect({.iteration = iteration,
                               .residual = res,
                               .step_size = alpha,
                               .grad_norm = std::sqrt(grad_norm),
                               .gamma_norm = dot});
        bbpgd_logger->log();
      }

      if (res <= allowed_residual) {
        break;
      }
    }

    // Step 12: Compute new step size
    VecWAXPY(delta_gamma, -1.0, gamma_prev, gamma_i);
    VecWAXPY(delta_phi, -1.0, g_prev, g_i);

    double numerator_val;
    double denominator_val;

    // BB1 step: (d^T d)/(d^T g)

    // alternating bb1 and bb2 methods
    if (iteration % 2 == 0) {
      NormAdotAB(delta_gamma, delta_phi, numerator_val, denominator_val);
    } else {
      NormAdotAB(delta_phi, delta_gamma, denominator_val, numerator_val);
    }

    if (fabs(denominator_val) < 10 * std::numeric_limits<double>::epsilon()) {
      denominator_val += 10 * std::numeric_limits<double>::epsilon();  // prevent div 0 error
    }

    alpha = numerator_val / denominator_val;

    // if alpha is nan or inf or negative, reset it
    if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0) {
      alpha = 1.0 / res;
    }

    // Prepare for next iteration by swapping buffers
    std::swap(gamma_prev, gamma_i);
    std::swap(g_prev, g_i);
  }

  if (iteration == max_bbpgd_iterations) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  BBPGD did not converge after %ld iterations. Residual: %f", iteration, res);
  }

  // Return the result in the input parameter
  g0 = std::move(gamma_prev);

  return {.bbpgd_iterations = iteration, .residual = res};
}
