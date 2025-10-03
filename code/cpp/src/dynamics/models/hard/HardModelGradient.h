#pragma once

#include "Workspace.h"
#include "solver/BBPGD.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

void calculate_growth_rate_vector(const VecWrapper& l, VecWrapper& sigma_and_impedance_out, double lambda, double tau, VecWrapper& growth_rates_out);
class HardModelGradient : public Gradient {
 public:
  HardModelGradient(
      const MatWrapper& D_PREV,
      const MatWrapper& M,
      const MatWrapper& L_PREV,
      const VecWrapper& U_ext,
      const VecWrapper& PHI,
      const VecWrapper& gamma_old,
      const VecWrapper& l,
      const VecWrapper& ldot_prev,
      const SimulationParameters& params,
      double dt);

  VecWrapper& gradient(const VecWrapper& gamma_curr) override;
  double residual(const VecWrapper& gradient_val, const VecWrapper& gamma) override;

  std::tuple<double, double, double, double> energy(const VecWrapper& gamma) override;

  const MatWrapper& D_PREV_;
  const MatWrapper& M_;
  const MatWrapper& L_PREV_;
  const VecWrapper& U_ext_;
  const VecWrapper& PHI_;
  const VecWrapper& gamma_old_;
  const VecWrapper& l_;
  const VecWrapper& ldot_prev_;
  Workspace workspaces_;
  const SimulationParameters& params_;
  double dt_;
};