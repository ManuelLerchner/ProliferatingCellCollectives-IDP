#pragma once

#include "PetscRaii.h"

VecWrapper concatenateVectors(const VecWrapper& vec1, const VecWrapper& vec2) {
  PetscInt vec1_size, vec2_size;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(vec1, &vec1_size));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(vec2, &vec2_size));
  PetscInt total_size = vec1_size + vec2_size;

  // Create result vector with proper parallel layout

  VecWrapper result;
  PetscCallAbort(PETSC_COMM_WORLD, VecCreate(PETSC_COMM_WORLD, result.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetSizes(result, PETSC_DECIDE, total_size));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetType(result, VECSTANDARD));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetFromOptions(result));

  // Copy first vector using global indices
  if (vec1_size > 0) {
    VecScatter scatter;
    IS is_from, is_to;

    // Create global index sets
    PetscCallAbort(PETSC_COMM_WORLD, ISCreateStride(PETSC_COMM_WORLD, vec1_size, 0, 1, &is_from));
    PetscCallAbort(PETSC_COMM_WORLD, ISCreateStride(PETSC_COMM_WORLD, vec1_size, 0, 1, &is_to));

    PetscCallAbort(PETSC_COMM_WORLD, VecScatterCreate(vec1, is_from, result, is_to, &scatter));
    PetscCallAbort(PETSC_COMM_WORLD, VecScatterBegin(scatter, vec1, result, INSERT_VALUES, SCATTER_FORWARD));
    PetscCallAbort(PETSC_COMM_WORLD, VecScatterEnd(scatter, vec1, result, INSERT_VALUES, SCATTER_FORWARD));

    PetscCallAbort(PETSC_COMM_WORLD, VecScatterDestroy(&scatter));
    PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_from));
    PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_to));
  }

  // Append second vector using global indices
  if (vec2_size > 0) {
    VecScatter scatter;
    IS is_from, is_to;

    // Create global index sets with offset
    PetscCallAbort(PETSC_COMM_WORLD, ISCreateStride(PETSC_COMM_WORLD, vec2_size, 0, 1, &is_from));
    PetscCallAbort(PETSC_COMM_WORLD, ISCreateStride(PETSC_COMM_WORLD, vec2_size, vec1_size, 1, &is_to));

    PetscCallAbort(PETSC_COMM_WORLD, VecScatterCreate(vec2, is_from, result, is_to, &scatter));
    PetscCallAbort(PETSC_COMM_WORLD, VecScatterBegin(scatter, vec2, result, INSERT_VALUES, SCATTER_FORWARD));
    PetscCallAbort(PETSC_COMM_WORLD, VecScatterEnd(scatter, vec2, result, INSERT_VALUES, SCATTER_FORWARD));

    PetscCallAbort(PETSC_COMM_WORLD, VecScatterDestroy(&scatter));
    PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_from));
    PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_to));
  }

  return result;
}

MatWrapper horizontallyStackMatrices(const MatWrapper& matLeft, const MatWrapper& matRight) {
  PetscInt m1, m2, n_left, n_right;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetLocalSize(matLeft, &m1, NULL));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetLocalSize(matRight, &m2, NULL));

  if (m1 != m2) {
    throw std::runtime_error("Matrices must have the same number of local rows");
  }

  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(matLeft, NULL, &n_left));
  PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(matRight, NULL, &n_right));

  if (n_left == -1) {
    n_left = 0;
  }

  PetscInt n_total = n_left + n_right;

  // Get ownership ranges for rows (should be the same for both matrices)
  PetscInt row_start, row_end;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRange(matLeft, &row_start, &row_end));

  // Verify that both matrices have the same row distribution
  PetscInt right_start, right_end;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRange(matRight, &right_start, &right_end));

  if (row_start != right_start || row_end != right_end) {
    throw std::runtime_error("Matrices must have the same row distribution across processes");
  }

  // Create output matrix
  MatWrapper matResult;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, matResult.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(matResult, m1, PETSC_DETERMINE, PETSC_DETERMINE, n_total));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(matResult, MATMPIAIJ));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetFromOptions(matResult));

  // Get column ownership range for the result matrix
  PetscInt col_start, col_end;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRangeColumn(matResult, &col_start, &col_end));

  // Estimate preallocation
  std::vector<PetscInt> d_nnz(m1, 0), o_nnz(m1, 0);

  // Count nonzeros from left matrix
  for (PetscInt i = row_start; i < row_end; ++i) {
    PetscInt ncols;
    const PetscInt* cols;
    const PetscScalar* vals;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(matLeft, i, &ncols, &cols, &vals));

    PetscInt local_row = i - row_start;
    for (PetscInt j = 0; j < ncols; ++j) {
      PetscInt global_col = cols[j];  // No offset for left matrix
      if (global_col >= col_start && global_col < col_end) {
        d_nnz[local_row]++;
      } else {
        o_nnz[local_row]++;
      }
    }

    PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(matLeft, i, &ncols, &cols, &vals));
  }

  // Count nonzeros from right matrix
  for (PetscInt i = row_start; i < row_end; ++i) {
    PetscInt ncols;
    const PetscInt* cols;
    const PetscScalar* vals;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(matRight, i, &ncols, &cols, &vals));

    PetscInt local_row = i - row_start;
    for (PetscInt j = 0; j < ncols; ++j) {
      PetscInt global_col = cols[j] + n_left;  // Add offset for right matrix
      if (global_col >= col_start && global_col < col_end) {
        d_nnz[local_row]++;
      } else {
        o_nnz[local_row]++;
      }
    }

    PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(matRight, i, &ncols, &cols, &vals));
  }

  // Preallocate
  PetscCallAbort(PETSC_COMM_WORLD, MatMPIAIJSetPreallocation(matResult, 0, d_nnz.data(), 0, o_nnz.data()));

  // Ensure input matrices are assembled
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(matLeft, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(matLeft, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(matRight, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(matRight, MAT_FINAL_ASSEMBLY));

  // Copy left matrix (no column offset)
  for (PetscInt i = row_start; i < row_end; i++) {
    PetscInt ncols;
    const PetscInt* cols;
    const PetscScalar* vals;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(matLeft, i, &ncols, &cols, &vals));
    PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(matResult, 1, &i, ncols, cols, vals, INSERT_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(matLeft, i, &ncols, &cols, &vals));
  }

  // Copy right matrix with column offset
  for (PetscInt i = row_start; i < row_end; i++) {
    PetscInt ncols;
    const PetscInt* cols;
    const PetscScalar* vals;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(matRight, i, &ncols, &cols, &vals));

    // Create offset columns array
    std::vector<PetscInt> offset_cols(ncols);
    for (PetscInt j = 0; j < ncols; j++) {
      offset_cols[j] = cols[j] + n_left;  // Add offset to column indices
    }

    PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(matResult, 1, &i, ncols, offset_cols.data(), vals, INSERT_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(matRight, i, &ncols, &cols, &vals));
  }

  // Final assembly
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(matResult, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(matResult, MAT_FINAL_ASSEMBLY));

  return matResult;
}

size_t vecSize(const VecWrapper& vec) {
  PetscInt size;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(vec, &size));
  return size;
}