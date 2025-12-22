#include <gtest/gtest.h>
#include <petsc.h>
#include <mpi.h>

int main(int argc, char **argv) {
    // Initialize PETSc (which initializes MPI)
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Run all tests
    int result = RUN_ALL_TESTS();
    
    // Finalize PETSc (which finalizes MPI)
    PetscFinalize();
    
    return result;
}
