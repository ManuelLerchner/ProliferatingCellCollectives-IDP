
#include "Workspace.h"

Workspace::Workspace(const MatWrapper& D_PREV, const MatWrapper& M, const MatWrapper& G, const MatWrapper& L_PREV, const VecWrapper& GAMMA_PREV, const VecWrapper& PHI_PREV, const VecWrapper& l) {
  // movement
  gamma_diff_workspace = VecWrapper::Like(GAMMA_PREV);
  VecCopy(GAMMA_PREV, gamma_diff_workspace);
  phi_dot_movement_workspace = VecWrapper::Like(PHI_PREV);
  F_g_workspace = VecWrapper::FromMatRows(D_PREV);
  U_c_workspace = VecWrapper::FromMat(M);
  U_total_workspace = VecWrapper::FromMatRows(D_PREV);

  // growth
  phi_dot_growth_result = VecWrapper::FromMat(L_PREV);

  ldot_curr_workspace = VecWrapper::Like(l);
  ldot_diff_workspace = VecWrapper::Like(l);

  impedance_curr_workspace = VecWrapper::Like(l);
  stress_curr_workspace = VecWrapper::Like(l);

  U_ext = VecWrapper::FromMat(M);
  F_ext_workspace = VecWrapper::FromMat(M);

  // force
  df = VecWrapper::FromMatRows(D_PREV);
  du = VecWrapper::FromMat(M);
  dC = VecWrapper::FromMat(G);
}
