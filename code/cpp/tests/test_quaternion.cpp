#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "util/Quaternion.h"

using namespace utils::Quaternion;

constexpr double PI = 3.14159265358979323846;

// Test quaternion multiplication
TEST(QuaternionTest, IdentityMultiplication) {
    std::array<double, 4> identity = {1.0, 0.0, 0.0, 0.0};
    std::array<double, 4> q = {0.707, 0.0, 0.707, 0.0};
    
    auto result = qmul(identity, q);
    
    EXPECT_NEAR(result[0], q[0], 1e-6);
    EXPECT_NEAR(result[1], q[1], 1e-6);
    EXPECT_NEAR(result[2], q[2], 1e-6);
    EXPECT_NEAR(result[3], q[3], 1e-6);
}

TEST(QuaternionTest, MultiplicationNonCommutative) {
    std::array<double, 4> q1 = {0.5, 0.5, 0.5, 0.5};
    std::array<double, 4> q2 = {0.707, 0.707, 0.0, 0.0};
    
    auto result1 = qmul(q1, q2);
    auto result2 = qmul(q2, q1);
    
    // Quaternion multiplication is not commutative
    bool different = false;
    for (int i = 0; i < 4; ++i) {
        if (std::abs(result1[i] - result2[i]) > 1e-6) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

TEST(QuaternionTest, MultiplicationBasic) {
    // i * j = k (in quaternion terms: [0,1,0,0] * [0,0,1,0] = [0,0,0,1])
    std::array<double, 4> qi = {0.0, 1.0, 0.0, 0.0};
    std::array<double, 4> qj = {0.0, 0.0, 1.0, 0.0};
    
    auto result = qmul(qi, qj);
    
    EXPECT_NEAR(result[0], 0.0, 1e-10);
    EXPECT_NEAR(result[1], 0.0, 1e-10);
    EXPECT_NEAR(result[2], 0.0, 1e-10);
    EXPECT_NEAR(result[3], 1.0, 1e-10);
}

// Test quaternion from Euler angles
TEST(QuaternionTest, QuaternionFromEulerZeroAngles) {
    std::array<double, 3> euler = {0.0, 0.0, 0.0};
    auto result = quaternionFromEuler(euler);
    
    // Zero Euler angles should give identity quaternion
    EXPECT_NEAR(result[0], 1.0, 1e-10);
    EXPECT_NEAR(result[1], 0.0, 1e-10);
    EXPECT_NEAR(result[2], 0.0, 1e-10);
    EXPECT_NEAR(result[3], 0.0, 1e-10);
}

TEST(QuaternionTest, QuaternionFromEulerPitchOnly) {
    std::array<double, 3> euler = {0.0, PI / 2, 0.0}; // 90 degree pitch
    auto result = quaternionFromEuler(euler);
    
    // Should produce a valid quaternion
    double magnitude = std::sqrt(result[0]*result[0] + result[1]*result[1] + 
                                  result[2]*result[2] + result[3]*result[3]);
    EXPECT_NEAR(magnitude, 1.0, 1e-6);
}

TEST(QuaternionTest, QuaternionFromEulerRollOnly) {
    std::array<double, 3> euler = {PI / 2, 0.0, 0.0}; // 90 degree roll
    auto result = quaternionFromEuler(euler);
    
    // Should produce a valid unit quaternion
    double magnitude = std::sqrt(result[0]*result[0] + result[1]*result[1] + 
                                  result[2]*result[2] + result[3]*result[3]);
    EXPECT_NEAR(magnitude, 1.0, 1e-6);
}

TEST(QuaternionTest, QuaternionFromEulerYawOnly) {
    std::array<double, 3> euler = {0.0, 0.0, PI / 2}; // 90 degree yaw
    auto result = quaternionFromEuler(euler);
    
    // Should produce a valid unit quaternion
    double magnitude = std::sqrt(result[0]*result[0] + result[1]*result[1] + 
                                  result[2]*result[2] + result[3]*result[3]);
    EXPECT_NEAR(magnitude, 1.0, 1e-6);
}

// Test vector rotation
TEST(QuaternionTest, RotateVectorIdentity) {
    std::array<double, 4> identity = {1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> v = {1.0, 2.0, 3.0};
    
    auto result = rotateVectorOfPositions(identity, v);
    
    // Identity rotation should not change the vector
    EXPECT_NEAR(result[0], v[0], 1e-10);
    EXPECT_NEAR(result[1], v[1], 1e-10);
    EXPECT_NEAR(result[2], v[2], 1e-10);
}

TEST(QuaternionTest, RotateVector90DegreesAroundZ) {
    // Quaternion for 90 degree rotation around Z axis
    double angle = PI / 2;
    std::array<double, 4> q = {std::cos(angle/2), 0.0, 0.0, std::sin(angle/2)};
    std::array<double, 3> v = {1.0, 0.0, 0.0};
    
    auto result = rotateVectorOfPositions(q, v);
    
    // (1,0,0) rotated 90 degrees around Z should give approximately (0,1,0)
    EXPECT_NEAR(result[0], 0.0, 1e-10);
    EXPECT_NEAR(result[1], 1.0, 1e-10);
    EXPECT_NEAR(result[2], 0.0, 1e-10);
}

TEST(QuaternionTest, RotateVector180DegreesAroundZ) {
    // Quaternion for 180 degree rotation around Z axis
    double angle = PI;
    std::array<double, 4> q = {std::cos(angle/2), 0.0, 0.0, std::sin(angle/2)};
    std::array<double, 3> v = {1.0, 0.0, 0.0};
    
    auto result = rotateVectorOfPositions(q, v);
    
    // (1,0,0) rotated 180 degrees around Z should give (-1,0,0)
    EXPECT_NEAR(result[0], -1.0, 1e-10);
    EXPECT_NEAR(result[1], 0.0, 1e-10);
    EXPECT_NEAR(result[2], 0.0, 1e-10);
}

TEST(QuaternionTest, RotateVectorPreservesMagnitude) {
    // Any rotation should preserve vector magnitude
    double angle = PI / 3;
    std::array<double, 4> q = {std::cos(angle/2), 0.0, std::sin(angle/2), 0.0};
    std::array<double, 3> v = {3.0, 4.0, 5.0};
    
    auto result = rotateVectorOfPositions(q, v);
    
    double original_mag = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    double rotated_mag = std::sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]);
    
    EXPECT_NEAR(original_mag, rotated_mag, 1e-10);
}

// Test get direction vector
TEST(QuaternionTest, GetDirectionVectorIdentity) {
    std::array<double, 4> identity = {1.0, 0.0, 0.0, 0.0};
    
    auto result = getDirectionVector(identity);
    
    // Identity should give the standard direction (1,0,0)
    EXPECT_NEAR(result[0], 1.0, 1e-10);
    EXPECT_NEAR(result[1], 0.0, 1e-10);
    EXPECT_NEAR(result[2], 0.0, 1e-10);
}

TEST(QuaternionTest, GetDirectionVector90DegreesAroundZ) {
    // Quaternion for 90 degree rotation around Z axis
    double angle = PI / 2;
    std::array<double, 4> q = {std::cos(angle/2), 0.0, 0.0, std::sin(angle/2)};
    
    auto result = getDirectionVector(q);
    
    // Direction should be rotated to (0,1,0)
    EXPECT_NEAR(result[0], 0.0, 1e-10);
    EXPECT_NEAR(result[1], 1.0, 1e-10);
    EXPECT_NEAR(result[2], 0.0, 1e-10);
}

TEST(QuaternionTest, GetDirectionVectorIsUnitLength) {
    std::array<double, 3> euler = {0.5, 0.3, 0.7};
    auto q = quaternionFromEuler(euler);
    
    auto result = getDirectionVector(q);
    
    double magnitude = std::sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]);
    EXPECT_NEAR(magnitude, 1.0, 1e-10);
}

// Test combined operations
TEST(QuaternionTest, EulerToQuaternionToDirection) {
    // Create quaternion from Euler angles, then get direction
    std::array<double, 3> euler = {0.0, 0.0, 0.0};
    auto q = quaternionFromEuler(euler);
    auto dir = getDirectionVector(q);
    
    EXPECT_NEAR(dir[0], 1.0, 1e-10);
    EXPECT_NEAR(dir[1], 0.0, 1e-10);
    EXPECT_NEAR(dir[2], 0.0, 1e-10);
}

TEST(QuaternionTest, QuaternionComposition) {
    // Test that composing two rotations works correctly
    double angle = PI / 4; // 45 degrees
    std::array<double, 4> q1 = {std::cos(angle/2), 0.0, 0.0, std::sin(angle/2)}; // 45° around Z
    std::array<double, 4> q2 = {std::cos(angle/2), 0.0, 0.0, std::sin(angle/2)}; // 45° around Z
    
    // Two 45° rotations = one 90° rotation
    auto q_combined = qmul(q1, q2);
    
    std::array<double, 3> v = {1.0, 0.0, 0.0};
    auto result = rotateVectorOfPositions(q_combined, v);
    
    // Should be approximately (0,1,0) after 90° rotation
    EXPECT_NEAR(result[0], 0.0, 1e-6);
    EXPECT_NEAR(result[1], 1.0, 1e-6);
    EXPECT_NEAR(result[2], 0.0, 1e-6);
}
