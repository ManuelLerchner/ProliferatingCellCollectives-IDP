#pragma once

#include "Workspace.h"
#include "solver/BBPGD.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

class HardModelGradient : public Gradient {
 public:
  HardModelGradient(
      const MatWrapper& D_PREV,
      const MatWrapper& M,
      const MatWrapper& G,
      const MatWrapper& L_PREV,
      const VecWrapper& PHI,
      const VecWrapper& gamma_old,
      const VecWrapper& l,
      Workspace& workspace,
      const SimulationParameters& params,
      double dt);

  void gradient(const VecWrapper& gamma_curr, VecWrapper& phi_next_out) override;
  double residual(const VecWrapper& gradient_val, const VecWrapper& gamma) override;

 private:
  const MatWrapper& D_PREV_;
  const MatWrapper& M_;
  const MatWrapper& G_;
  const MatWrapper& L_PREV_;
  const VecWrapper& PHI_;
  const VecWrapper& gamma_old_;
  const VecWrapper& l_;
  Workspace& workspaces_;
  const SimulationParameters& params_;
  double dt_;
};