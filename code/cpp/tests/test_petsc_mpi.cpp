#include <gtest/gtest.h>
#include <petsc.h>
#include <mpi.h>
#include <array>
#include <cmath>
#include "util/MPIUtil.h"
#include "util/PetscRaii.h"

// Global PETSc initialization state
static bool petsc_initialized_globally = false;

// Helper macro to ensure PETSc is initialized
#define ENSURE_PETSC_INIT() \
    if (!petsc_initialized_globally) { \
        int argc = 0; \
        char** argv = nullptr; \
        PetscInitialize(&argc, &argv, nullptr, nullptr); \
        petsc_initialized_globally = true; \
    }

// Test MPI rank and size
TEST(MPITest, MPIInitialized) {
    ENSURE_PETSC_INIT();
    
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    EXPECT_GE(rank, 0);
    EXPECT_GE(size, 1);
    EXPECT_LT(rank, size);
}

// Test MPI communication
TEST(MPITest, MPIBasicCommunication) {
    ENSURE_PETSC_INIT();
    
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    // Each rank sends its rank number
    int send_data = rank;
    int recv_data = 0;
    
    MPI_Allreduce(&send_data, &recv_data, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    
    // Sum of ranks should be 0 + 1 + 2 + ... + (size-1) = size*(size-1)/2
    int expected_sum = size * (size - 1) / 2;
    EXPECT_EQ(recv_data, expected_sum);
}

// Test global reduce helper for integers
TEST(MPITest, GlobalReduceInteger) {
    ENSURE_PETSC_INIT();
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    int local_val = rank + 1; // Each rank contributes rank+1
    int global_sum = globalReduce(local_val, MPI_SUM);
    
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    // Sum should be 1 + 2 + ... + size = size*(size+1)/2
    int expected_sum = size * (size + 1) / 2;
    EXPECT_EQ(global_sum, expected_sum);
}

// Test global reduce helper for doubles
TEST(MPITest, GlobalReduceDouble) {
    ENSURE_PETSC_INIT();
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    double local_val = static_cast<double>(rank) * 2.5;
    double global_sum = globalReduce(local_val, MPI_SUM);
    
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    // Sum should be 0*2.5 + 1*2.5 + 2*2.5 + ... = 2.5 * (0+1+...+(size-1))
    double expected_sum = 2.5 * size * (size - 1) / 2;
    EXPECT_NEAR(global_sum, expected_sum, 1e-10);
}

// Test global reduce for max operation
TEST(MPITest, GlobalReduceMax) {
    ENSURE_PETSC_INIT();
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    double local_val = static_cast<double>(rank);
    double global_max = globalReduce(local_val, MPI_MAX);
    
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    // Max should be the highest rank number
    EXPECT_DOUBLE_EQ(global_max, static_cast<double>(size - 1));
}

// Test global reduce for min operation
TEST(MPITest, GlobalReduceMin) {
    ENSURE_PETSC_INIT();
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    double local_val = static_cast<double>(rank) + 10.0;
    double global_min = globalReduce(local_val, MPI_MIN);
    
    // Min should be 10.0 (from rank 0)
    EXPECT_DOUBLE_EQ(global_min, 10.0);
}

// Test global reduce on array
TEST(MPITest, GlobalReduceVector) {
    ENSURE_PETSC_INIT();
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    // Each rank contributes an array
    double local_vals[3] = {
        static_cast<double>(rank),
        static_cast<double>(rank) * 2.0,
        static_cast<double>(rank) * 3.0
    };
    double global_vals[3];
    
    globalReduce_v(local_vals, global_vals, 3, MPI_SUM);
    
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    // Expected sums for each element
    double expected_sum = size * (size - 1) / 2.0;
    EXPECT_NEAR(global_vals[0], expected_sum, 1e-10);
    EXPECT_NEAR(global_vals[1], expected_sum * 2.0, 1e-10);
    EXPECT_NEAR(global_vals[2], expected_sum * 3.0, 1e-10);
}

// Test getGlobalMinMax helper
TEST(MPITest, GetGlobalMinMax) {
    ENSURE_PETSC_INIT();
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    double local_min = static_cast<double>(rank) * 10.0;
    double local_max = static_cast<double>(rank) * 10.0 + 5.0;
    
    double global_min, global_max;
    getGlobalMinMax(local_min, local_max, global_min, global_max);
    
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    // Global min should be from rank 0
    EXPECT_DOUBLE_EQ(global_min, 0.0);
    
    // Global max should be from highest rank
    EXPECT_DOUBLE_EQ(global_max, (size - 1) * 10.0 + 5.0);
}

// Test VecWrapper creation
TEST(PetscTest, VecWrapperCreate) {
    ENSURE_PETSC_INIT();
    
    auto vec = VecWrapper::Create(10);
    
    PetscInt size;
    VecGetLocalSize(vec, &size);
    
    EXPECT_EQ(size, 10);
}

// Test VecWrapper set and get
TEST(PetscTest, VecWrapperSetGet) {
    ENSURE_PETSC_INIT();
    
    auto vec = VecWrapper::Create(5);
    
    // Set all values to 3.14
    VecSet(vec, 3.14);
    
    // Get values
    const PetscScalar* array;
    VecGetArrayRead(vec, &array);
    
    for (int i = 0; i < 5; i++) {
        EXPECT_DOUBLE_EQ(array[i], 3.14);
    }
    
    VecRestoreArrayRead(vec, &array);
}

// Test VecWrapper operations
TEST(PetscTest, VecWrapperOperations) {
    ENSURE_PETSC_INIT();
    
    auto vec1 = VecWrapper::Create(10);
    auto vec2 = VecWrapper::Create(10);
    
    VecSet(vec1, 2.0);
    VecSet(vec2, 3.0);
    
    // vec1 = vec1 + vec2 (should be 5.0)
    VecAXPY(vec1, 1.0, vec2);
    
    // Check result
    const PetscScalar* array;
    VecGetArrayRead(vec1, &array);
    
    for (int i = 0; i < 10; i++) {
        EXPECT_DOUBLE_EQ(array[i], 5.0);
    }
    
    VecRestoreArrayRead(vec1, &array);
}

// Test VecWrapper norm
TEST(PetscTest, VecWrapperNorm) {
    ENSURE_PETSC_INIT();
    
    auto vec = VecWrapper::Create(4);
    
    // Set values [1, 2, 3, 4]
    PetscScalar values[] = {1.0, 2.0, 3.0, 4.0};
    PetscInt indices[] = {0, 1, 2, 3};
    
    VecSetValues(vec, 4, indices, values, INSERT_VALUES);
    VecAssemblyBegin(vec);
    VecAssemblyEnd(vec);
    
    // Calculate L2 norm: sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
    double norm;
    VecNorm(vec, NORM_2, &norm);
    
    EXPECT_NEAR(norm, std::sqrt(30.0), 1e-10);
}

// Test VecWrapper Like (duplicate)
TEST(PetscTest, VecWrapperLike) {
    ENSURE_PETSC_INIT();
    
    auto vec1 = VecWrapper::Create(8);
    VecSet(vec1, 1.5);
    
    auto vec2 = VecWrapper::Like(vec1);
    
    // vec2 should have same size but different values
    PetscInt size1, size2;
    VecGetLocalSize(vec1, &size1);
    VecGetLocalSize(vec2, &size2);
    
    EXPECT_EQ(size1, size2);
}

// Test MatWrapper creation
TEST(PetscTest, MatWrapperCreate) {
    ENSURE_PETSC_INIT();
    
    MatWrapper mat;
    
    MatCreate(PETSC_COMM_WORLD, mat.get_ref());
    MatSetSizes(mat, 5, 5, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetType(mat, MATDENSE);
    MatSetUp(mat);
    
    PetscInt rows, cols;
    MatGetLocalSize(mat, &rows, &cols);
    
    EXPECT_EQ(rows, 5);
    EXPECT_EQ(cols, 5);
}

// Test MPI data type helper
TEST(MPITest, GetMpiDataTypeInt) {
    ENSURE_PETSC_INIT();
    
    auto dtype = getMpiDataType<int>();
    EXPECT_EQ(dtype, MPI_INT);
}

TEST(MPITest, GetMpiDataTypeDouble) {
    ENSURE_PETSC_INIT();
    
    auto dtype = getMpiDataType<double>();
    EXPECT_EQ(dtype, MPI_DOUBLE);
}

// Test PETSc error handling with PetscCallAbort
TEST(PetscTest, PetscCallAbortSuccess) {
    ENSURE_PETSC_INIT();
    
    auto vec = VecWrapper::Create(5);
    
    // This should succeed
    EXPECT_NO_THROW({
        PetscCallAbort(PETSC_COMM_WORLD, VecSet(vec, 1.0));
    });
}

// Test VecWrapper move semantics
TEST(PetscTest, VecWrapperMove) {
    ENSURE_PETSC_INIT();
    
    auto vec1 = VecWrapper::Create(10);
    VecSet(vec1, 7.0);
    
    // Move construct
    VecWrapper vec2 = std::move(vec1);
    
    // vec2 should have the values
    const PetscScalar* array;
    VecGetArrayRead(vec2, &array);
    EXPECT_DOUBLE_EQ(array[0], 7.0);
    VecRestoreArrayRead(vec2, &array);
}

// Test that all MPI ranks see the same test results
TEST(MPITest, AllRanksAgree) {
    ENSURE_PETSC_INIT();
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    // All ranks compute the same value
    int local_result = 42;
    int global_min = globalReduce(local_result, MPI_MIN);
    int global_max = globalReduce(local_result, MPI_MAX);
    
    // Min and max should be the same since all ranks have same value
    EXPECT_EQ(global_min, 42);
    EXPECT_EQ(global_max, 42);
    EXPECT_EQ(global_min, global_max);
}
