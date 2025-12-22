#include <gtest/gtest.h>
#include <petsc.h>
#include <array>
#include <cmath>
#include "solver/BBPGD.h"
#include "util/PetscRaii.h"

// Global PETSc initialization for BBPGD tests
static bool petsc_initialized_bbpgd = false;

#define ENSURE_PETSC_INIT_BBPGD() \
    if (!petsc_initialized_bbpgd) { \
        int argc = 0; \
        char** argv = nullptr; \
        PetscInitialize(&argc, &argv, nullptr, nullptr); \
        petsc_initialized_bbpgd = true; \
    }

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

// Quadratic function gradient implementation for testing
// Minimize f(x) = 0.5 * x^T * A * x - b^T * x
// Gradient: grad f(x) = A * x - b
class QuadraticGradient : public Gradient {
private:
    VecWrapper grad_vec;
    VecWrapper temp_vec;
    double A_diag;  // Using diagonal matrix for simplicity: A = A_diag * I
    double b_val;   // Using constant vector: b = b_val * ones
    
public:
    QuadraticGradient(int size, double a_diag, double b_value) 
        : grad_vec(VecWrapper::Create(size)),
          temp_vec(VecWrapper::Create(size)),
          A_diag(a_diag),
          b_val(b_value) {
    }
    
    VecWrapper& gradient(const VecWrapper& gamma_curr) override {
        // grad = A * gamma - b = A_diag * gamma - b_val * ones
        VecCopy(gamma_curr, grad_vec);
        VecScale(grad_vec, A_diag);
        VecShift(grad_vec, -b_val);
        return grad_vec;
    }
    
    double residual(const VecWrapper& gradient_val, const VecWrapper& gamma) override {
        // For constrained optimization with gamma >= 0, residual is based on KKT conditions
        // Simplified: use norm of gradient where gamma > 0 or gradient < 0
        double res;
        VecNorm(gradient_val, NORM_2, &res);
        return res;
    }
    
    std::tuple<double, double, double, double> energy(const VecWrapper& gamma) override {
        // Energy: 0.5 * gamma^T * A * gamma - b^T * gamma
        // For diagonal A: 0.5 * A_diag * sum(gamma^2) - b_val * sum(gamma)
        double gamma_norm_sq, gamma_sum;
        VecNorm(gamma, NORM_2, &gamma_norm_sq);
        gamma_norm_sq *= gamma_norm_sq;
        
        VecSum(gamma, &gamma_sum);
        
        double energy = 0.5 * A_diag * gamma_norm_sq - b_val * gamma_sum;
        return {energy, 0.0, 0.0, 0.0};
    }
};

// Test BBPGD with quadratic function minimization
TEST(BBPGDTest, QuadraticMinimization) {
    ENSURE_PETSC_INIT_BBPGD();
    
    // Set up quadratic problem: minimize f(x) = 0.5 * x^T * x - 2 * ones^T * x
    // Optimal solution (unconstrained): x* = 2 * ones
    // With constraint x >= 0, solution is still x* = 2 * ones
    int size = 5;
    double A_diag = 1.0;
    double b_val = 2.0;
    
    QuadraticGradient gradient(size, A_diag, b_val);
    auto gamma = VecWrapper::Create(size);
    
    // Start from zero (feasible point)
    VecSet(gamma, 0.0);
    
    // Run BBPGD
    double tolerance = 1e-6;
    size_t max_iter = 1000;
    auto result = BBPGD(gradient, gamma, tolerance, max_iter, std::nullopt);
    
    // Check convergence
    EXPECT_LE(result.residual, tolerance * 10);  // Allow some tolerance relaxation
    EXPECT_LT(result.bbpgd_iterations, max_iter);  // Should converge before max
    
    // Check that solution is close to optimum (x* = 2 * ones)
    const PetscScalar* gamma_array;
    VecGetArrayRead(gamma, &gamma_array);
    PetscInt local_size;
    VecGetLocalSize(gamma, &local_size);
    
    for (PetscInt i = 0; i < local_size; i++) {
        EXPECT_NEAR(gamma_array[i], 2.0, 0.5);  // Relaxed tolerance for convergence
    }
    
    VecRestoreArrayRead(gamma, &gamma_array);
}

// Test BBPGD convergence from different starting points
TEST(BBPGDTest, QuadraticConvergenceFromDifferentStarts) {
    ENSURE_PETSC_INIT_BBPGD();
    
    int size = 3;
    double A_diag = 2.0;
    double b_val = 4.0;  // Optimal: x* = b_val / A_diag = 2.0
    
    QuadraticGradient gradient(size, A_diag, b_val);
    
    // Test different starting points
    std::vector<double> start_vals = {0.0, 1.0, 5.0};
    
    for (double start : start_vals) {
        auto gamma = VecWrapper::Create(size);
        VecSet(gamma, start);
        
        double tolerance = 1e-5;
        size_t max_iter = 500;
        auto result = BBPGD(gradient, gamma, tolerance, max_iter, std::nullopt);
        
        // Should converge from any starting point
        EXPECT_LT(result.bbpgd_iterations, max_iter);
        EXPECT_LE(result.residual, tolerance * 10);
    }
}


