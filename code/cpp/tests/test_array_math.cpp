#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "util/ArrayMath.h"

using namespace utils::ArrayMath;

// Test addition
TEST(ArrayMathTest, AddTwoArrays) {
    std::array<double, 3> a = {1.0, 2.0, 3.0};
    std::array<double, 3> b = {4.0, 5.0, 6.0};
    auto result = add(a, b);
    
    EXPECT_DOUBLE_EQ(result[0], 5.0);
    EXPECT_DOUBLE_EQ(result[1], 7.0);
    EXPECT_DOUBLE_EQ(result[2], 9.0);
}

TEST(ArrayMathTest, AddOperator) {
    std::array<double, 3> a = {1.0, 2.0, 3.0};
    std::array<double, 3> b = {4.0, 5.0, 6.0};
    auto result = a + b;
    
    EXPECT_DOUBLE_EQ(result[0], 5.0);
    EXPECT_DOUBLE_EQ(result[1], 7.0);
    EXPECT_DOUBLE_EQ(result[2], 9.0);
}

// Test subtraction
TEST(ArrayMathTest, SubtractTwoArrays) {
    std::array<double, 3> a = {5.0, 7.0, 9.0};
    std::array<double, 3> b = {1.0, 2.0, 3.0};
    auto result = subtract(a, b);
    
    EXPECT_DOUBLE_EQ(result[0], 4.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 6.0);
}

TEST(ArrayMathTest, SubtractOperator) {
    std::array<double, 3> a = {5.0, 7.0, 9.0};
    std::array<double, 3> b = {1.0, 2.0, 3.0};
    auto result = a - b;
    
    EXPECT_DOUBLE_EQ(result[0], 4.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 6.0);
}

// Test element-wise multiplication
TEST(ArrayMathTest, MultiplyTwoArrays) {
    std::array<double, 3> a = {2.0, 3.0, 4.0};
    std::array<double, 3> b = {5.0, 6.0, 7.0};
    auto result = multiply(a, b);
    
    EXPECT_DOUBLE_EQ(result[0], 10.0);
    EXPECT_DOUBLE_EQ(result[1], 18.0);
    EXPECT_DOUBLE_EQ(result[2], 28.0);
}

TEST(ArrayMathTest, MultiplyArrayByScalar) {
    std::array<double, 3> a = {2.0, 3.0, 4.0};
    double scalar = 2.5;
    auto result = multiply(a, scalar);
    
    EXPECT_DOUBLE_EQ(result[0], 5.0);
    EXPECT_DOUBLE_EQ(result[1], 7.5);
    EXPECT_DOUBLE_EQ(result[2], 10.0);
}

TEST(ArrayMathTest, MultiplyScalarByArray) {
    std::array<double, 3> a = {2.0, 3.0, 4.0};
    double scalar = 2.5;
    auto result = scalar * a;
    
    EXPECT_DOUBLE_EQ(result[0], 5.0);
    EXPECT_DOUBLE_EQ(result[1], 7.5);
    EXPECT_DOUBLE_EQ(result[2], 10.0);
}

// Test negation
TEST(ArrayMathTest, NegateArray) {
    std::array<double, 3> a = {1.0, -2.0, 3.0};
    auto result = negate(a);
    
    EXPECT_DOUBLE_EQ(result[0], -1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], -3.0);
}

TEST(ArrayMathTest, NegateOperator) {
    std::array<double, 3> a = {1.0, -2.0, 3.0};
    auto result = -a;
    
    EXPECT_DOUBLE_EQ(result[0], -1.0);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], -3.0);
}

// Test cross product
TEST(ArrayMathTest, CrossProduct) {
    std::array<double, 3> a = {1.0, 0.0, 0.0};
    std::array<double, 3> b = {0.0, 1.0, 0.0};
    auto result = cross_product(a, b);
    
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 1.0);
}

TEST(ArrayMathTest, CrossProductParallel) {
    std::array<double, 3> a = {2.0, 0.0, 0.0};
    std::array<double, 3> b = {4.0, 0.0, 0.0};
    auto result = cross_product(a, b);
    
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}

// Test dot product
TEST(ArrayMathTest, DotProduct) {
    std::array<double, 3> a = {1.0, 2.0, 3.0};
    std::array<double, 3> b = {4.0, 5.0, 6.0};
    auto result = dot(a, b);
    
    EXPECT_DOUBLE_EQ(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

TEST(ArrayMathTest, DotProductOrthogonal) {
    std::array<double, 3> a = {1.0, 0.0, 0.0};
    std::array<double, 3> b = {0.0, 1.0, 0.0};
    auto result = dot(a, b);
    
    EXPECT_DOUBLE_EQ(result, 0.0);
}

// Test magnitude squared
TEST(ArrayMathTest, MagnitudeSquared) {
    std::array<double, 3> a = {3.0, 4.0, 0.0};
    auto result = magnitude_squared(a);
    
    EXPECT_DOUBLE_EQ(result, 25.0); // 3^2 + 4^2 = 9 + 16 = 25
}

// Test magnitude
TEST(ArrayMathTest, Magnitude) {
    std::array<double, 3> a = {3.0, 4.0, 0.0};
    auto result = magnitude(a);
    
    EXPECT_DOUBLE_EQ(result, 5.0);
}

TEST(ArrayMathTest, MagnitudeUnitVector) {
    std::array<double, 3> a = {1.0, 0.0, 0.0};
    auto result = magnitude(a);
    
    EXPECT_DOUBLE_EQ(result, 1.0);
}

// Test distance
TEST(ArrayMathTest, Distance) {
    std::array<double, 3> a = {0.0, 0.0, 0.0};
    std::array<double, 3> b = {3.0, 4.0, 0.0};
    auto result = distance(a, b);
    
    EXPECT_DOUBLE_EQ(result, 5.0);
}

TEST(ArrayMathTest, DistanceSamePoint) {
    std::array<double, 3> a = {1.0, 2.0, 3.0};
    std::array<double, 3> b = {1.0, 2.0, 3.0};
    auto result = distance(a, b);
    
    EXPECT_DOUBLE_EQ(result, 0.0);
}

// Test distance squared
TEST(ArrayMathTest, DistanceSquared) {
    std::array<double, 3> a = {0.0, 0.0, 0.0};
    std::array<double, 3> b = {3.0, 4.0, 0.0};
    auto result = distance_squared(a, b);
    
    EXPECT_DOUBLE_EQ(result, 25.0);
}

// Test normalization
TEST(ArrayMathTest, Normalize) {
    std::array<double, 3> a = {3.0, 4.0, 0.0};
    auto result = normalize(a);
    
    EXPECT_DOUBLE_EQ(result[0], 0.6);
    EXPECT_DOUBLE_EQ(result[1], 0.8);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
    
    // Check that it's a unit vector
    EXPECT_NEAR(magnitude(result), 1.0, 1e-10);
}

TEST(ArrayMathTest, NormalizeZeroVector) {
    std::array<double, 3> a = {0.0, 0.0, 0.0};
    auto result = normalize(a);
    
    // Should return a default unit vector (1, 0, 0)
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}

TEST(ArrayMathTest, NormalizeAlreadyUnit) {
    std::array<double, 3> a = {0.0, 1.0, 0.0};
    auto result = normalize(a);
    
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 1.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}

// Test infinity norm
TEST(ArrayMathTest, InfinityNorm) {
    std::array<double, 3> a = {1.0, -5.0, 3.0};
    auto result = infinity_norm(a);
    
    EXPECT_DOUBLE_EQ(result, 5.0);
}

TEST(ArrayMathTest, InfinityNormPositive) {
    std::array<double, 3> a = {1.0, 2.0, 3.0};
    auto result = infinity_norm(a);
    
    EXPECT_DOUBLE_EQ(result, 3.0);
}

TEST(ArrayMathTest, InfinityNormZero) {
    std::array<double, 3> a = {0.0, 0.0, 0.0};
    auto result = infinity_norm(a);
    
    EXPECT_DOUBLE_EQ(result, 0.0);
}

// Test with different array sizes
TEST(ArrayMathTest, WorksWithDifferentSizes) {
    std::array<double, 2> a2 = {1.0, 2.0};
    std::array<double, 2> b2 = {3.0, 4.0};
    auto result2 = add(a2, b2);
    
    EXPECT_DOUBLE_EQ(result2[0], 4.0);
    EXPECT_DOUBLE_EQ(result2[1], 6.0);
    
    std::array<double, 4> a4 = {1.0, 2.0, 3.0, 4.0};
    std::array<double, 4> b4 = {5.0, 6.0, 7.0, 8.0};
    auto result4 = add(a4, b4);
    
    EXPECT_DOUBLE_EQ(result4[0], 6.0);
    EXPECT_DOUBLE_EQ(result4[1], 8.0);
    EXPECT_DOUBLE_EQ(result4[2], 10.0);
    EXPECT_DOUBLE_EQ(result4[3], 12.0);
}
