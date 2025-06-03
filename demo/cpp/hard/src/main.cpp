#include <mpi.h>
#include <petsc.h>

// Macro for concise PETSc error checking
#define PetscCall(expr)                  \
  do {                                   \
    PetscErrorCode ierr_macro_ = (expr); \
    CHKERRQ(ierr_macro_);                \
  } while (0)

// Function to compute gradient g(γ) = A*γ + b for LCP
PetscErrorCode ComputeGradient(Mat A, Vec gamma, Vec b, Vec g) {
  PetscFunctionBeginUser;
  // g = A * gamma + b
  PetscCall(MatMult(A, gamma, g));
  PetscCall(VecAXPY(g, 1.0, b));
  PetscFunctionReturn(0);
}

// Function to compute residual res(γ) = ||min(γ, g(γ))||
PetscErrorCode ComputeResidual(Vec gamma, Vec g, Vec temp, PetscReal *residual) {
  PetscInt i, n_local;
  const PetscScalar *gamma_array, *g_array;
  PetscScalar *temp_array;

  PetscFunctionBeginUser;
  // Get local array sizes and pointers
  PetscCall(VecGetLocalSize(gamma, &n_local));
  PetscCall(VecGetArrayRead(gamma, &gamma_array));
  PetscCall(VecGetArrayRead(g, &g_array));
  PetscCall(VecGetArray(temp, &temp_array));

  // Compute min(γ, g) element-wise
  for (i = 0; i < n_local; i++) {
    PetscReal gamma_val = PetscRealPart(gamma_array[i]);
    PetscReal g_val = PetscRealPart(g_array[i]);
    temp_array[i] = PetscMin(gamma_val, g_val);
  }

  // Restore arrays
  PetscCall(VecRestoreArrayRead(gamma, &gamma_array));
  PetscCall(VecRestoreArrayRead(g, &g_array));
  PetscCall(VecRestoreArray(temp, &temp_array));

  // Compute norm
  PetscCall(VecNorm(temp, NORM_2, residual));
  PetscFunctionReturn(0);
}

// Function to apply projection: max(x, 0)
PetscErrorCode ApplyProjection(Vec x, Vec result) {
  PetscInt i, n_local;
  const PetscScalar *x_array;
  PetscScalar *result_array;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n_local));
  PetscCall(VecGetArrayRead(x, &x_array));
  PetscCall(VecGetArray(result, &result_array));

  // Apply max(x, 0) element-wise
  for (i = 0; i < n_local; i++) {
    PetscReal val = PetscRealPart(x_array[i]);
    result_array[i] = PetscMax(val, 0.0);
  }

  PetscCall(VecRestoreArrayRead(x, &x_array));
  PetscCall(VecRestoreArray(result, &result_array));
  PetscFunctionReturn(0);
}

// BBPGD Algorithm for LCP: 0 <= gamma \perp (A*gamma + b) >= 0
PetscErrorCode SolveBBPGD(Mat A, Vec b, Vec gamma, PetscInt max_iter, PetscReal tol,
                          PetscMPIInt rank,
                          PetscReal *final_residual, PetscBool *converged_status, PetscInt *iterations_taken) {
  char version[PETSC_VERSION_MAJOR];
  PetscErrorCode PetscGetVersion(char version[], size_t lengthofversion);

  Vec gamma_old,
      g, g_old, work_vec_projection, work_vec_residual;
  PetscReal current_residual, alpha = 1.0;
  PetscInt iter;
  PetscBool local_converged = PETSC_FALSE;

  PetscFunctionBeginUser;

  // Create work vectors
  PetscCall(VecDuplicate(gamma, &gamma_old));
  PetscCall(VecDuplicate(gamma, &g));
  PetscCall(VecDuplicate(gamma, &g_old));
  PetscCall(VecDuplicate(gamma, &work_vec_projection));  // For gamma_new = P(gamma_old - alpha*g_old) step
  PetscCall(VecDuplicate(gamma, &work_vec_residual));    // For ComputeResidual

  // Initial computations:
  // g0 = g(γ0) (gamma is the initial guess, e.g. pre-set to zero by caller)
  PetscCall(ComputeGradient(A, gamma, b, g));

  // res0 = res(γ0)
  PetscCall(ComputeResidual(gamma, g, work_vec_residual, &current_residual));

  // α0 = 1/res0 (or a default value)
  if (current_residual > PETSC_MACHINE_EPSILON) {  // Avoid division by zero or tiny number
    alpha = 1.0 / current_residual;
  } else {
    alpha = 1.0;  // Default alpha
  }
  // Safeguard initial alpha
  alpha = PetscMax(1e-10, PetscMin(alpha, 1e10));

  if (rank == 0) {
    PetscPrintf(PETSC_COMM_WORLD, "BBPGD: Initial residual = %g, Initial alpha = %g\n", (double)current_residual, (double)alpha);
  }

  // Main BBPGD loop
  for (iter = 1; iter <= max_iter; iter++) {
    // Save previous values
    PetscCall(VecCopy(gamma, gamma_old));
    PetscCall(VecCopy(g, g_old));

    // Step 6: γi = max(γi-1 - αi-1 * gi-1, 0)
    PetscCall(VecWAXPY(work_vec_projection, -alpha, g_old, gamma_old));  // work_vec_projection = gamma_old - alpha*g_old
    PetscCall(ApplyProjection(work_vec_projection, gamma));              // gamma = max(work_vec_projection, 0)

    // Step 7: gi = g(γi)
    PetscCall(ComputeGradient(A, gamma, b, g));

    // Step 8: resi = res(γi)
    PetscCall(ComputeResidual(gamma, g, work_vec_residual, &current_residual));

    // Step 9: Check convergence
    if (current_residual <= tol) {
      local_converged = PETSC_TRUE;
      break;
    }

    // Step 12: Compute new step size using Barzilai-Borwein formula
    // αi = (γi - γi-1)^T (γi - γi-1) / (γi - γi-1)^T (gi - gi-1)
    Vec delta_gamma, delta_g;
    PetscCall(VecDuplicate(gamma, &delta_gamma));
    PetscCall(VecDuplicate(gamma, &delta_g));

    PetscCall(VecWAXPY(delta_gamma, -1.0, gamma_old, gamma));  // delta_gamma = gamma - gamma_old
    PetscCall(VecWAXPY(delta_g, -1.0, g_old, g));              // delta_g = g - g_old

    PetscScalar numerator, denominator;
    PetscCall(VecDot(delta_gamma, delta_gamma, &numerator));
    PetscCall(VecDot(delta_gamma, delta_g, &denominator));

    if (PetscAbsScalar(denominator) > 1e-12) {  // Avoid division by zero
      alpha = PetscRealPart(numerator / denominator);
      alpha = PetscMax(1e-10, PetscMin(alpha, 1e10));  // Safeguard step size
    } else {
      // Reset alpha or use a predefined sequence if denominator is too small
      // For simplicity, retain previous alpha or reset to a moderate value if stuck
      // This part can be made more sophisticated (e.g. using alpha_max, alpha_min from literature)
      if (rank == 0 && iter % 10 == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Iter %D: Denominator too small for BB step, alpha maintained or reset.\n", iter);
      }
      // Keep previous alpha, or set to a fallback if it became extreme
      alpha = PetscMax(1e-5, PetscMin(alpha, 1e5));
    }

    PetscCall(VecDestroy(&delta_gamma));
    PetscCall(VecDestroy(&delta_g));

    // Print progress
    if (rank == 0 && iter % 10 == 0) {
      PetscPrintf(PETSC_COMM_WORLD, "Iter %D: residual = %g, alpha = %g\n",
                  iter, (double)current_residual, (double)alpha);
    }
  }

  // Set output parameters
  *final_residual = current_residual;
  *converged_status = local_converged;
  *iterations_taken = (local_converged == PETSC_TRUE) ? iter : max_iter;
  // If loop finished due to max_iter, iter will be max_iter + 1.
  // The print logic in main handles this correctly.
  // For consistency, if not converged, iterations_taken is max_iter.
  // If converged, it is the iteration number it converged on.
  if (local_converged == PETSC_TRUE) {
    *iterations_taken = iter;
  } else {
    *iterations_taken = max_iter;  // iter would be max_iter + 1 here if loop completed
  }

  // Clean up work vectors
  PetscCall(VecDestroy(&gamma_old));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&g_old));
  PetscCall(VecDestroy(&work_vec_projection));
  PetscCall(VecDestroy(&work_vec_residual));

  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  // At the beginning of main(), after PetscInitialize()
  PetscCall(PetscLogDefaultBegin());

  Mat A;
  Vec gamma, b, temp_main;  // temp_main is for LCP check in main
  PetscInt n = 200000000, max_iter = 1000, iter_taken;
  PetscReal tol = 1e-6, final_residual;
  PetscMPIInt rank, size;
  PetscInt i, Istart, Iend;
  PetscBool converged = PETSC_FALSE;

  PetscFunctionBeginUser;
  // Initialize PETSc
  PetscCall(PetscInitialize(&argc, &argv, NULL,
                            "BBPGD solver for Linear Complementarity Problem\\n"));

  // Get MPI info
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  // Get command line options
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_iter", &max_iter, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));

  if (rank == 0) {
    PetscPrintf(PETSC_COMM_WORLD, "BBPGD LCP Solver: n=%D, max_iter=%D, tol=%g\n",
                n, max_iter, (double)tol);
    PetscPrintf(PETSC_COMM_WORLD, "Running on %d MPI ranks\n", size);
  }

  // Create vectors
  PetscCall(VecCreate(PETSC_COMM_WORLD, &gamma));
  PetscCall(VecSetSizes(gamma, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(gamma));
  PetscCall(VecDuplicate(gamma, &b));
  PetscCall(VecDuplicate(gamma, &temp_main));  // For LCP check

  // Create matrix A (tridiagonal with diagonal dominance)
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  // Get local ownership range
  PetscCall(VecGetOwnershipRange(gamma, &Istart, &Iend));

  // Fill matrix A
  for (i = Istart; i < Iend; i++) {
    PetscScalar v = 4.0;  // Diagonal
    PetscCall(MatSetValue(A, i, i, v, INSERT_VALUES));
    if (i > 0) {
      v = -1.0;
      PetscCall(MatSetValue(A, i, i - 1, v, INSERT_VALUES));
    }
    if (i < n - 1) {
      v = -1.0;
      PetscCall(MatSetValue(A, i, i + 1, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Fill vector b
  for (i = Istart; i < Iend; i++) {
    PetscScalar v = -1.0 + 2.0 * PetscSinReal(2.0 * PETSC_PI * i / n);
    PetscCall(VecSetValue(b, i, v, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  // BBPGD Algorithm Implementation
  // Step 1: γ0 = initial guess (all zeros)
  PetscCall(VecSet(gamma, 0.0));

  PetscLogDouble mem;
  PetscCall(PetscMemoryGetCurrentUsage(&mem));
  if (rank == 0) {
    printf("Current memory usage: %f MB\n", mem / (1024.0 * 1024.0));
  }

  PetscLogDouble start_time, end_time;
  PetscCall(PetscTime(&start_time));

  // Call the BBPGD solver
  PetscCall(SolveBBPGD(A, b, gamma, max_iter, tol, rank, &final_residual, &converged, &iter_taken));

  PetscCall(PetscTime(&end_time));
  if (rank == 0) {
    PetscPrintf(PETSC_COMM_WORLD, "Total solver time: %f seconds\n", end_time - start_time);
  }

  // Print final results
  if (rank == 0) {
    if (converged) {
      PetscPrintf(PETSC_COMM_WORLD, "BBPGD converged in %D iterations\n", iter_taken);
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "BBPGD reached max iterations (%D)\n", max_iter);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Final residual: %g\n", (double)final_residual);
  }

  // Verify LCP conditions
  // temp_main = A*gamma + b
  PetscCall(MatMult(A, gamma, temp_main));
  PetscCall(VecAXPY(temp_main, 1.0, b));

  PetscScalar complementarity;
  PetscCall(VecDot(gamma, temp_main, &complementarity));

  if (rank == 0) {
    PetscPrintf(PETSC_COMM_WORLD, "Complementarity γ^T(Aγ+b) = %g\n",
                (double)PetscRealPart(complementarity));
  }

  // Optionally view solution
  PetscBool view_solution = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_solution", &view_solution, NULL));

  if (view_solution) {
    if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Solution γ:\n");  // Print header only once
    PetscCall(VecView(gamma, PETSC_VIEWER_STDOUT_WORLD));
  }

  // Clean up
  PetscCall(VecDestroy(&gamma));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&temp_main));
  PetscCall(MatDestroy(&A));

  // At the end of main(), before PetscFinalize()
  PetscCall(PetscLogView(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(0);
}
