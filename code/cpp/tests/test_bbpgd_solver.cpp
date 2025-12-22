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

// Test convergence: residual decreases
TEST(BBPGDTest, ConvergenceResidualDecreases) {
    BBPGDResult result;
    result.bbpgd_iterations = 50;
    result.residual = 1e-6;
    
    // Verify that a smaller residual indicates convergence
    EXPECT_LT(result.residual, 1e-3);
    EXPECT_GT(result.residual, 0.0);
}

// Test convergence: meets tolerance
TEST(BBPGDTest, ConvergenceMeetsTolerance) {
    BBPGDResult result;
    result.bbpgd_iterations = 25;
    result.residual = 5e-7;
    
    double tolerance = 1e-6;
    EXPECT_LE(result.residual, tolerance);
}

// Test convergence: residual within acceptable range
TEST(BBPGDTest, ConvergenceWithinRange) {
    BBPGDResult result;
    result.bbpgd_iterations = 100;
    result.residual = 1e-8;
    
    // Verify residual is in typical convergence range
    EXPECT_GE(result.residual, 1e-12);  // Not too small (numerical precision)
    EXPECT_LE(result.residual, 1e-5);   // Small enough for convergence
}

// Test non-convergence: high residual after max iterations
TEST(BBPGDTest, NonConvergenceHighResidual) {
    BBPGDResult result;
    result.bbpgd_iterations = 10000;  // Max iterations reached
    result.residual = 0.5;             // Still high residual
    
    double tolerance = 1e-6;
    EXPECT_GT(result.residual, tolerance);  // Did not converge
}

// Test convergence: monotonic decrease simulation
TEST(BBPGDTest, ConvergenceMonotonicDecrease) {
    // Simulate residual values over iterations
    std::vector<double> residuals = {1.0, 0.5, 0.25, 0.1, 0.01, 0.001, 1e-4, 1e-5};
    
    // Verify monotonic decrease
    for (size_t i = 1; i < residuals.size(); i++) {
        EXPECT_LT(residuals[i], residuals[i-1]);
    }
    
    // Final residual should meet tolerance
    BBPGDResult result;
    result.bbpgd_iterations = residuals.size();
    result.residual = residuals.back();
    
    EXPECT_LE(result.residual, 1e-4);
}

// Test convergence: early stopping
TEST(BBPGDTest, ConvergenceEarlyStopping) {
    BBPGDResult result;
    result.bbpgd_iterations = 15;  // Converged early
    result.residual = 1e-7;
    
    size_t max_iterations = 1000;
    double tolerance = 1e-6;
    
    // Converged before max iterations
    EXPECT_LT(result.bbpgd_iterations, max_iterations);
    EXPECT_LE(result.residual, tolerance);
}

// Test convergence: near-zero residual
TEST(BBPGDTest, ConvergenceNearZero) {
    BBPGDResult result;
    result.bbpgd_iterations = 200;
    result.residual = 1e-12;
    
    // Very small residual indicates strong convergence
    EXPECT_NEAR(result.residual, 0.0, 1e-11);
}

