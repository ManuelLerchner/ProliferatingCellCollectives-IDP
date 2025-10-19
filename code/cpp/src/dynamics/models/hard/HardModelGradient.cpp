#include "HardModelGradient.h"

void estimate_phi_dot_movement_inplace(
    const MatWrapper& D,
    const MatWrapper& M,
    const VecWrapper& U_known,
    const VecWrapper& gamma,
    VecWrapper& F_g,
    VecWrapper& U_c,
    VecWrapper& U_total,
    VecWrapper& phi_dot_out) {
  // phi_dot = D^T (U_known + M * D * gamma)

  // Step 1: F_g = D * gamma
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(D, gamma, F_g));

  // Step 2: U_c = M * F_g
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F_g, U_c));

  // Step 3: U_total = U_known + U_c
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(U_total, 1.0, U_known, U_c));

  // Step 4: phi_dot_out = D^T * U_total
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(D, U_total, phi_dot_out));
}

void estimate_phi_dot_growth_inplace(
    const MatWrapper& Sigma,
    const VecWrapper& ldot,
    VecWrapper& phi_dot_growth_result) {
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(Sigma, ldot, phi_dot_growth_result));
}

void calculate_impedance_inplace(VecWrapper& stresses_and_impedance_out, double lambda) {
  // I is a (num_bodies, 1) vector
  // I = np.exp(-lamb * stress_matrix)
  // This function modifies the input vector in-place.

  VecScale(stresses_and_impedance_out, -lambda);
  VecExp(stresses_and_impedance_out);
}

void calculate_growth_rate_vector(const VecWrapper& l, VecWrapper& sigma_and_impedance_out, double lambda, double tau, VecWrapper& growth_rates_out) {
  // growth rate is a (num_bodies, 1) vector
  // growth_rate = l / tau * I

  calculate_impedance_inplace(sigma_and_impedance_out, lambda);

  // sigma_and_impedance_out now contains the impedance
  VecPointwiseMult(growth_rates_out, l, sigma_and_impedance_out);
  VecScale(growth_rates_out, 1 / tau);
}

void calculate_ldot_inplace(const MatWrapper& L, const VecWrapper& l, const VecWrapper& gamma, double lambda, double tau, VecWrapper& ldot_curr_out, VecWrapper& stress_curr_out, VecWrapper& impedance_curr_out) {
  // Use impedance_curr_out as a temporary vector to store stresses (sigma)
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(L, gamma, stress_curr_out));

  // We now have stresses in impedance_curr_out.
  // We need to calculate the growth rate, which will overwrite ldot_curr_out,
  // and the final impedance, which will overwrite the stresses in impedance_curr_out.

  VecCopy(stress_curr_out, impedance_curr_out);
  calculate_growth_rate_vector(l, impedance_curr_out, lambda, tau, ldot_curr_out);
};

HardModelGradient::HardModelGradient(
    const MatWrapper& D_PREV,
    const MatWrapper& M,
    const MatWrapper& L_PREV,
    const VecWrapper& U_ext,
    const VecWrapper& PHI,
    const VecWrapper& gamma_old,
    const VecWrapper& l,
    const VecWrapper& ldot_prev,
    const SimulationParameters& params,
    double dt)
    : D_PREV_(D_PREV),
      M_(M),
      L_PREV_(L_PREV),
      U_ext_(U_ext),
      PHI_(PHI),
      gamma_old_(gamma_old),
      l_(l),
      ldot_prev_(ldot_prev),
      workspaces_(Workspace(D_PREV, M, L_PREV, gamma_old, PHI, l)),
      params_(params),
      dt_(dt) {}

VecWrapper& HardModelGradient::gradient(const VecWrapper& gamma_curr) {
  // --- MOVEMENT PART ---
  // gamma_diff = gamma_curr - gamma_old
  VecWAXPY(workspaces_.gamma_diff_workspace, -1.0, gamma_old_, gamma_curr);

  // phi_dot_movement = D * M * D^T * gamma_diff
  estimate_phi_dot_movement_inplace(
      D_PREV_, M_, U_ext_, workspaces_.gamma_diff_workspace,
      workspaces_.F_g_workspace, workspaces_.U_c_workspace,
      workspaces_.U_total_workspace, workspaces_.phi_dot_movement_workspace);

  // --- GROWTH PART ---
  // ldot_curr = growth_rate(gamma_curr)
  calculate_ldot_inplace(
      L_PREV_, l_, gamma_curr, params_.physics_config.getLambdaDimensionless(),
      params_.physics_config.TAU, workspaces_.ldot_curr_workspace,
      workspaces_.stress_curr_workspace, workspaces_.impedance_curr_workspace);

  // ldot_diff = ldot_curr - ldot_prev
  VecWAXPY(workspaces_.ldot_diff_workspace, -1.0, ldot_prev_, workspaces_.ldot_curr_workspace);

  // phi_dot_growth = -L^T * ldot_diff
  estimate_phi_dot_growth_inplace(L_PREV_, workspaces_.ldot_diff_workspace, workspaces_.phi_dot_growth_result);

  // Start with the base violation: phi_next_out = phi
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(PHI_, workspaces_.phi_next_out));

  Vec vecs_to_add[] = {workspaces_.phi_dot_movement_workspace, workspaces_.phi_dot_growth_result};
  PetscScalar scales[] = {dt_, -dt_};
  PetscCallAbort(PETSC_COMM_WORLD, VecMAXPY(workspaces_.phi_next_out, 2, scales, vecs_to_add));

  return workspaces_.phi_next_out;
}

double HardModelGradient::residual(const VecWrapper& gradient_val, const VecWrapper& gamma) {
  // This function computes the infinity norm of the projected gradient.
  // The projection is defined as:
  //   projected_gradient_i = gradient_val_i, if gamma_i > 0
  //   projected_gradient_i = min(0, gradient_val_i), if gamma_i <= 0

  // Get local size and pointers to vector data
  PetscInt n_local;
  const PetscScalar *grad_array, *gamma_array;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(gradient_val, &n_local));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(gradient_val, &grad_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(gamma, &gamma_array));

  // Compute projected gradient component-wise
  double res = 0.0;  // Initialize to 0 since we're taking max of absolute values

  for (PetscInt i = 0; i < n_local; ++i) {
    double v;
    if (PetscRealPart(gamma_array[i]) > 0) {
      v = grad_array[i];
    } else {
      v = std::min(0.0, PetscRealPart(grad_array[i]));
    }

    res = std::max(res, std::abs(v));  // Take absolute value for infinity norm
  }

  // Restore arrays
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(gradient_val, &grad_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(gamma, &gamma_array));

  // Compute the infinity norm of the projected gradient: max(abs(projected_gradient_i))
  auto res_global = globalReduce(res, MPI_MAX);

  return res_global;
};

std::tuple<double, double, double, double> HardModelGradient::energy(const VecWrapper& gamma) {
  // \begin{equation}
  //     \small
  //     \begin{aligned}
  //         E(\boldsymbol{\gamma}) =
  //         \boldsymbol{\gamma}^\top\mathbf{\Phi}^k
  //          & + \frac{\Delta t}{2} \boldsymbol{\gamma}^\top \mathbfcal{D}^\top \mathbfcal{M} \mathbfcal{D} \boldsymbol{\gamma} \\
  //          & + \mathbf{1}^\top \frac{\Delta t}{\lambda}
  //         \left( \frac{\boldsymbol{\ell}}{\tau} e^{-\lambda \mathbfcal{L} \boldsymbol{\gamma}} \right).
  //     \end{aligned}
  // \end{equation}

  // Term 1: gamma^T * phi
  double term1 = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecDot(gamma, PHI_, &term1));

  // Term 2: (dt / 2) * gamma^T * D^T * M * D * gamma
  VecWrapper D_gamma = VecWrapper::FromMatRows(D_PREV_);
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(D_PREV_, gamma, D_gamma));

  VecWrapper M_D_gamma = VecWrapper::FromMat(M_);
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M_, D_gamma, M_D_gamma));

  double term2 = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecDot(D_gamma, M_D_gamma, &term2));
  term2 *= (dt_ / 2.0);

  // Term 3: (dt / lambda) * sum( (l / tau) * exp(-lambda * L * gamma) )
  VecWrapper L_gamma = VecWrapper::FromMatRows(L_PREV_);
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(L_PREV_, gamma, L_gamma));

  VecScale(L_gamma, -params_.physics_config.getLambdaDimensionless());
  PetscCallAbort(PETSC_COMM_WORLD, VecExp(L_gamma));

  VecPointwiseMult(L_gamma, l_, L_gamma);

  double term3 = 0.0;
  PetscCallAbort(PETSC_COMM_WORLD, VecSum(L_gamma, &term3));

  term3 *= (dt_ / (params_.physics_config.getLambdaDimensionless() * params_.physics_config.TAU));

  double energy = term1 + term2 + term3;

  return std::make_tuple(term1, term2, term3, energy);
}