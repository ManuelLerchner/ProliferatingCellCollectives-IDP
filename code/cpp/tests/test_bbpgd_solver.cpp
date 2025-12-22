#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "solver/BBPGD.h"

// Test BBPGD result structure
TEST(BBPGDTest, ResultStructure) {
    BBPGDResult result;
    result.bbpgd_iterations = 10;
    result.residual = 0.001;
    
    EXPECT_EQ(result.bbpgd_iterations, 10);
    EXPECT_DOUBLE_EQ(result.residual, 0.001);
}

// Test BBPGD result with zero iterations
TEST(BBPGDTest, ZeroIterations) {
    BBPGDResult result;
    result.bbpgd_iterations = 0;
    result.residual = 0.0;
    
    EXPECT_EQ(result.bbpgd_iterations, 0);
    EXPECT_DOUBLE_EQ(result.residual, 0.0);
}

// Test BBPGD result with high residual
TEST(BBPGDTest, HighResidual) {
    BBPGDResult result;
    result.bbpgd_iterations = 1000;
    result.residual = 1.5;
    
    EXPECT_EQ(result.bbpgd_iterations, 1000);
    EXPECT_DOUBLE_EQ(result.residual, 1.5);
}

// Test BBPGD result with large iteration count
TEST(BBPGDTest, LargeIterationCount) {
    BBPGDResult result;
    result.bbpgd_iterations = 1000000;
    result.residual = 1e-10;
    
    EXPECT_EQ(result.bbpgd_iterations, 1000000);
    EXPECT_DOUBLE_EQ(result.residual, 1e-10);
}

// Test BBPGD result copy
TEST(BBPGDTest, ResultCopy) {
    BBPGDResult result1;
    result1.bbpgd_iterations = 42;
    result1.residual = 3.14;
    
    BBPGDResult result2 = result1;
    
    EXPECT_EQ(result2.bbpgd_iterations, 42);
    EXPECT_DOUBLE_EQ(result2.residual, 3.14);
}

// Test BBPGD result assignment
TEST(BBPGDTest, ResultAssignment) {
    BBPGDResult result1;
    result1.bbpgd_iterations = 100;
    result1.residual = 0.001;
    
    BBPGDResult result2;
    result2 = result1;
    
    EXPECT_EQ(result2.bbpgd_iterations, 100);
    EXPECT_DOUBLE_EQ(result2.residual, 0.001);
}

// Note: Full BBPGD solver tests require PETSc initialization and
// proper MPI setup which is beyond the scope of simple unit tests.
// The solver is tested through integration tests in the main application.
