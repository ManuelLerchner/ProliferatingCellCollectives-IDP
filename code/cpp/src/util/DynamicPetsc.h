#pragma once

#include <stdexcept>
#include <vector>

#include "Config.h"
#include "PetscRaii.h"

class DynamicVecWrapper {
 private:
  VecWrapper vec;
  PetscInt local_capacity;
  int size;

  double growth_factor;

 public:
  DynamicVecWrapper(int capacity, double growth_factor_) {
    growth_factor = growth_factor_;
    local_capacity = capacity;
    size = 0;
    vec = VecWrapper::Create(capacity);
  }

  bool ensureCapacity(PetscInt additional_items) {
    PetscInt required_local_capacity = size + additional_items;

    auto max_required_capacity = globalReduce(required_local_capacity, MPI_MAX);

    if (local_capacity < max_required_capacity) {
      PetscInt new_local_capacity = max_required_capacity * growth_factor;

      PetscPrintf(PETSC_COMM_WORLD, "Resizing vec from %d to %d\n", local_capacity, new_local_capacity);

      VecWrapper new_vec = VecWrapper::Create(new_local_capacity);

      vec = std::move(new_vec);
      local_capacity = new_local_capacity;
      size = 0;
      return true;
    }
    return false;
  }

  operator VecWrapper&() { return vec; }
  operator const VecWrapper&() const { return vec; }
  operator Vec() const { return vec; }

  PetscInt getSize() const { return size; }

  PetscInt getCapacity() const { return local_capacity; }

  void incrementSize(PetscInt size_delta) {
    size += size_delta;
  }
};

class DynamicMatWrapper {
 private:
  MatWrapper mat;
  PetscInt local_cols;      // Fixed number of columns
  PetscInt local_capacity;  // Current row capacity
  PetscInt size;            // Current number of rows used

  double growth_factor;

 public:
  // For transposed storage:
  // local_cols_ = number of columns (fixed)
  // capacity_ = initial row capacity (will grow)
  DynamicMatWrapper(PetscInt local_cols_, int capacity_, double growth_factor_) {
    growth_factor = growth_factor_;
    local_capacity = capacity_;
    local_cols = local_cols_;
    size = 0;

    // Create matrix with dimensions (capacity x cols)
    mat = MatWrapper::CreateAIJ(capacity_, local_cols_);
  }

  bool ensureCapacity(PetscInt additional_rows) {
    PetscInt required_local_capacity = size + additional_rows;

    auto max_required_capacity = globalReduce(required_local_capacity, MPI_MAX);

    if (local_capacity < max_required_capacity) {
      PetscInt new_local_capacity = max_required_capacity * growth_factor;

      // Create new matrix with dimensions (new_capacity x cols)
      MatWrapper new_mat = MatWrapper::CreateAIJ(new_local_capacity, local_cols);

      mat = std::move(new_mat);
      local_capacity = new_local_capacity;
      size = 0;
      return true;
    }
    return false;
  }

  operator MatWrapper&() { return mat; }
  operator const MatWrapper&() const { return mat; }
  operator Mat() const { return mat; }

  PetscInt getSize() const { return size; }

  PetscInt getCapacity() const { return local_capacity; }

  void incrementSize(PetscInt size_delta) {
    size += size_delta;
  }
};
