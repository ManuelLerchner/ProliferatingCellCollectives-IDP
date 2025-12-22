#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "dynamics/Constraint.h"

// Test Constraint default construction
TEST(ConstraintTest, DefaultConstruction) {
    Constraint c;
    
    // Default constructor exists and creates a constraint
    // Values are uninitialized, so we just test that gamma defaults to 0
    EXPECT_DOUBLE_EQ(c.gamma, 0.0);
}

// Test Constraint with parameters
TEST(ConstraintTest, ParameterizedConstruction) {
    double signed_distance = -0.5;
    int gidI = 10;
    int gidJ = 20;
    std::array<double, 3> normI = {1.0, 0.0, 0.0};
    std::array<double, 3> posI = {1.0, 2.0, 3.0};
    std::array<double, 3> posJ = {4.0, 5.0, 6.0};
    std::array<double, 3> contactPoint = {2.5, 3.5, 4.5};
    double stressI = 100.0;
    double stressJ = 200.0;
    int gid = 5;
    int iteration = 3;
    bool localI = true;
    bool localJ = false;
    int localIdxI = 0;
    int localIdxJ = 1;
    
    Constraint c(signed_distance, gidI, gidJ, normI, posI, posJ, contactPoint,
                 stressI, stressJ, gid, iteration, localI, localJ, localIdxI, localIdxJ);
    
    EXPECT_DOUBLE_EQ(c.signed_distance, -0.5);
    EXPECT_EQ(c.gidI, 10);
    EXPECT_EQ(c.gidJ, 20);
    EXPECT_EQ(c.normI[0], 1.0);
    EXPECT_EQ(c.normI[1], 0.0);
    EXPECT_EQ(c.normI[2], 0.0);
    EXPECT_EQ(c.contactPoint[0], 2.5);
    EXPECT_EQ(c.contactPoint[1], 3.5);
    EXPECT_EQ(c.contactPoint[2], 4.5);
    EXPECT_DOUBLE_EQ(c.stressI, 100.0);
    EXPECT_DOUBLE_EQ(c.stressJ, 200.0);
    EXPECT_EQ(c.gid, 5);
    EXPECT_EQ(c.iteration, 3);
    EXPECT_TRUE(c.localI);
    EXPECT_FALSE(c.localJ);
    EXPECT_EQ(c.localIdxI, 0);
    EXPECT_EQ(c.localIdxJ, 1);
}

// Test Constraint gamma modification
TEST(ConstraintTest, GammaModification) {
    Constraint c;
    
    EXPECT_DOUBLE_EQ(c.gamma, 0.0);
    
    c.gamma = 5.5;
    EXPECT_DOUBLE_EQ(c.gamma, 5.5);
    
    c.gamma = -2.3;
    EXPECT_DOUBLE_EQ(c.gamma, -2.3);
}

// Test Constraint with negative signed distance
TEST(ConstraintTest, NegativeSignedDistance) {
    Constraint c(-1.5, 1, 2, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                 {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, true, 0, 1);
    
    EXPECT_LT(c.signed_distance, 0.0);
    EXPECT_DOUBLE_EQ(c.signed_distance, -1.5);
}

// Test Constraint with zero signed distance
TEST(ConstraintTest, ZeroSignedDistance) {
    Constraint c(0.0, 1, 2, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                 {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, true, 0, 1);
    
    EXPECT_DOUBLE_EQ(c.signed_distance, 0.0);
}

// Test Constraint with positive signed distance
TEST(ConstraintTest, PositiveSignedDistance) {
    Constraint c(2.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                 {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, true, 0, 1);
    
    EXPECT_GT(c.signed_distance, 0.0);
    EXPECT_DOUBLE_EQ(c.signed_distance, 2.0);
}

// Test Constraint normal vector
TEST(ConstraintTest, NormalVector) {
    std::array<double, 3> normI = {0.707, 0.707, 0.0};
    Constraint c(0.0, 1, 2, normI, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                 {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, true, 0, 1);
    
    EXPECT_NEAR(c.normI[0], 0.707, 0.001);
    EXPECT_NEAR(c.normI[1], 0.707, 0.001);
    EXPECT_DOUBLE_EQ(c.normI[2], 0.0);
}

// Test Constraint with unit normal vectors
TEST(ConstraintTest, UnitNormals) {
    std::array<double, 3> normals[] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {-1.0, 0.0, 0.0}
    };
    
    for (const auto& norm : normals) {
        Constraint c(0.0, 1, 2, norm, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                     {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, true, 0, 1);
        
        // Check magnitude is approximately 1
        double mag = std::sqrt(c.normI[0]*c.normI[0] + c.normI[1]*c.normI[1] + c.normI[2]*c.normI[2]);
        EXPECT_NEAR(mag, 1.0, 0.001);
    }
}

// Test Constraint with different particle IDs
TEST(ConstraintTest, DifferentParticleIDs) {
    Constraint c1(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                  {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, true, 0, 1);
    Constraint c2(0.0, 5, 10, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                  {0.5, 0.5, 0.5}, 50.0, 75.0, 2, 1, true, true, 0, 1);
    
    EXPECT_NE(c1.gidI, c2.gidI);
    EXPECT_NE(c1.gidJ, c2.gidJ);
    EXPECT_NE(c1.gid, c2.gid);
}

// Test Constraint stress values
TEST(ConstraintTest, StressValues) {
    Constraint c(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                 {0.5, 0.5, 0.5}, 123.45, 678.90, 1, 1, true, true, 0, 1);
    
    EXPECT_DOUBLE_EQ(c.stressI, 123.45);
    EXPECT_DOUBLE_EQ(c.stressJ, 678.90);
}

// Test Constraint iteration number
TEST(ConstraintTest, IterationNumber) {
    Constraint c1(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                  {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 0, true, true, 0, 1);
    Constraint c2(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                  {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 5, true, true, 0, 1);
    
    EXPECT_EQ(c1.iteration, 0);
    EXPECT_EQ(c2.iteration, 5);
}

// Test Constraint locality flags
TEST(ConstraintTest, LocalityFlags) {
    // Both local
    Constraint c1(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                  {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, true, 0, 1);
    EXPECT_TRUE(c1.localI);
    EXPECT_TRUE(c1.localJ);
    
    // I local, J ghost
    Constraint c2(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                  {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, false, 0, 1);
    EXPECT_TRUE(c2.localI);
    EXPECT_FALSE(c2.localJ);
    
    // I ghost, J local
    Constraint c3(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                  {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, false, true, 0, 1);
    EXPECT_FALSE(c3.localI);
    EXPECT_TRUE(c3.localJ);
    
    // Both ghost
    Constraint c4(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                  {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, false, false, 0, 1);
    EXPECT_FALSE(c4.localI);
    EXPECT_FALSE(c4.localJ);
}

// Test Constraint local indices
TEST(ConstraintTest, LocalIndices) {
    Constraint c(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                 {0.5, 0.5, 0.5}, 50.0, 75.0, 1, 1, true, true, 42, 99);
    
    EXPECT_EQ(c.localIdxI, 42);
    EXPECT_EQ(c.localIdxJ, 99);
}

// Test Constraint contact point
TEST(ConstraintTest, ContactPoint) {
    std::array<double, 3> contact = {10.5, 20.3, 30.7};
    Constraint c(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                 contact, 50.0, 75.0, 1, 1, true, true, 0, 1);
    
    EXPECT_DOUBLE_EQ(c.contactPoint[0], 10.5);
    EXPECT_DOUBLE_EQ(c.contactPoint[1], 20.3);
    EXPECT_DOUBLE_EQ(c.contactPoint[2], 30.7);
}

// Test Constraint with zero stress
TEST(ConstraintTest, ZeroStress) {
    Constraint c(0.0, 1, 2, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
                 {0.5, 0.5, 0.5}, 0.0, 0.0, 1, 1, true, true, 0, 1);
    
    EXPECT_DOUBLE_EQ(c.stressI, 0.0);
    EXPECT_DOUBLE_EQ(c.stressJ, 0.0);
}

// Test Constraint print method exists
TEST(ConstraintTest, PrintMethodExists) {
    Constraint c;
    
    // Just verify that print method exists and doesn't crash
    EXPECT_NO_THROW({
        c.print();
    });
}
