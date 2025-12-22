#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "simulation/Particle.h"

// Test Particle constructor
TEST(ParticleTest, ConstructorBasic) {
    std::array<double, 3> position = {1.0, 2.0, 3.0};
    std::array<double, 4> quaternion = {1.0, 0.0, 0.0, 0.0};
    double length = 5.0;
    double l0 = 4.0;
    double diameter = 1.0;
    
    Particle p(42, position, quaternion, length, l0, diameter);
    
    EXPECT_EQ(p.getGID(), 42);
    EXPECT_EQ(p.getLength(), 5.0);
    EXPECT_EQ(p.getDiameter(), 1.0);
    
    auto pos = p.getPosition();
    EXPECT_EQ(pos[0], 1.0);
    EXPECT_EQ(pos[1], 2.0);
    EXPECT_EQ(pos[2], 3.0);
}

TEST(ParticleTest, ConstructorFromParticleData) {
    ParticleData data;
    data.gID = 123;
    data.position = {4.0, 5.0, 6.0};
    data.quaternion = {1.0, 0.0, 0.0, 0.0};
    data.length = 3.0;
    data.l0 = 2.5;
    data.ldot = 0.0;
    data.diameter = 0.8;
    data.age = 10;
    
    Particle p(data);
    
    EXPECT_EQ(p.getGID(), 123);
    EXPECT_EQ(p.getLength(), 3.0);
    EXPECT_EQ(p.getDiameter(), 0.8);
    EXPECT_EQ(p.getAge(), 10);
    
    auto pos = p.getPosition();
    EXPECT_EQ(pos[0], 4.0);
    EXPECT_EQ(pos[1], 5.0);
    EXPECT_EQ(pos[2], 6.0);
}

// Test getters and setters
TEST(ParticleTest, SetPosition) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    std::array<double, 3> newPos = {10.0, 20.0, 30.0};
    p.setPosition(newPos);
    
    auto pos = p.getPosition();
    EXPECT_EQ(pos[0], 10.0);
    EXPECT_EQ(pos[1], 20.0);
    EXPECT_EQ(pos[2], 30.0);
}

TEST(ParticleTest, SetGID) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    p.setGID(999);
    EXPECT_EQ(p.getGID(), 999);
}

TEST(ParticleTest, SetLdot) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    p.setLdot(2.5);
    EXPECT_EQ(p.getLdot(), 2.5);
}

TEST(ParticleTest, SetImpedance) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    p.setImpedance(100.0);
    EXPECT_EQ(p.getImpedance(), 100.0);
}

TEST(ParticleTest, SetStress) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    p.setStress(50.0);
    EXPECT_EQ(p.getStress(), 50.0);
}

// Test volume calculation
TEST(ParticleTest, GetVolume) {
    double diameter = 2.0;
    double length = 6.0;
    double radius = diameter / 2.0;
    
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, length, length, diameter);
    
    double volume = p.getVolume();
    
    // Volume of spherocylinder: π*r^2*h + (4/3)*π*r^3
    // where h is the cylindrical part length (length - diameter)
    double h = length - diameter;
    double expected_volume = M_PI * radius * radius * h + (4.0/3.0) * M_PI * radius * radius * radius;
    
    EXPECT_NEAR(volume, expected_volume, 1e-10);
}

TEST(ParticleTest, GetVolumeMinimalLength) {
    // When length equals diameter, it's just a sphere
    double diameter = 2.0;
    double radius = diameter / 2.0;
    
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, diameter, diameter, diameter);
    
    double volume = p.getVolume();
    
    // Volume of sphere: (4/3)*π*r^3
    double expected_volume = (4.0/3.0) * M_PI * radius * radius * radius;
    
    EXPECT_NEAR(volume, expected_volume, 1e-10);
}

// Test age management
TEST(ParticleTest, IncrementAge) {
    ParticleData data;
    data.gID = 1;
    data.position = {0.0, 0.0, 0.0};
    data.quaternion = {1.0, 0.0, 0.0, 0.0};
    data.length = 1.0;
    data.l0 = 1.0;
    data.ldot = 0.0;
    data.diameter = 0.5;
    data.age = 5;
    
    Particle p(data);
    
    EXPECT_EQ(p.getAge(), 5);
    
    p.incrementAge();
    EXPECT_EQ(p.getAge(), 6);
    
    p.incrementAge();
    EXPECT_EQ(p.getAge(), 7);
}

// Test constraint management
TEST(ParticleTest, IncrementNumConstraints) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    EXPECT_EQ(p.getNumConstraints(), 0);
    
    p.incrementNumConstraints();
    EXPECT_EQ(p.getNumConstraints(), 1);
    
    p.incrementNumConstraints();
    EXPECT_EQ(p.getNumConstraints(), 2);
}

TEST(ParticleTest, Reset) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    p.setLdot(5.0);
    p.incrementNumConstraints();
    p.incrementNumConstraints();
    
    EXPECT_EQ(p.getNumConstraints(), 2);
    EXPECT_EQ(p.getLdot(), 5.0);
    
    p.reset();
    
    EXPECT_EQ(p.getNumConstraints(), 0);
    EXPECT_EQ(p.getLdot(), 0.0);
}

// Test gravitational force calculation
TEST(ParticleTest, CalculateGravitationalForce) {
    double diameter = 1.0;
    double length = 2.0;
    
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, length, length, diameter);
    
    std::array<double, 3> gravity = {0.0, 0.0, -9.81};
    auto force = p.calculateGravitationalForce(gravity);
    
    // Force should be in the direction of gravity
    EXPECT_EQ(force[0], 0.0);
    EXPECT_EQ(force[1], 0.0);
    EXPECT_LT(force[2], 0.0); // Negative z direction
}

TEST(ParticleTest, CalculateGravitationalForceZeroGravity) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    std::array<double, 3> gravity = {0.0, 0.0, 0.0};
    auto force = p.calculateGravitationalForce(gravity);
    
    EXPECT_EQ(force[0], 0.0);
    EXPECT_EQ(force[1], 0.0);
    EXPECT_EQ(force[2], 0.0);
}

// Test state size
TEST(ParticleTest, GetStateSize) {
    EXPECT_EQ(Particle::getStateSize(), 7); // 3 position + 4 quaternion
}

// Test quaternion remains normalized after construction
TEST(ParticleTest, QuaternionNormalized) {
    std::array<double, 4> quaternion = {0.5, 0.5, 0.5, 0.5};
    
    Particle p(1, {0.0, 0.0, 0.0}, quaternion, 1.0, 1.0, 0.5);
    
    auto q = p.getQuaternion();
    double magnitude = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    
    EXPECT_NEAR(magnitude, 1.0, 1e-10);
}

// Test data access
TEST(ParticleTest, GetDataConst) {
    ParticleData data;
    data.gID = 42;
    data.position = {1.0, 2.0, 3.0};
    data.quaternion = {1.0, 0.0, 0.0, 0.0};
    data.length = 5.0;
    data.l0 = 4.0;
    data.ldot = 0.1;
    data.diameter = 1.0;
    data.age = 3;
    
    const Particle p(data);
    
    const auto& retrieved_data = p.getData();
    EXPECT_EQ(retrieved_data.gID, 42);
    EXPECT_EQ(retrieved_data.age, 3);
}

TEST(ParticleTest, GetDataNonConst) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    auto& data = p.getData();
    data.gID = 999;
    
    EXPECT_EQ(p.getGID(), 999);
}

// Test force and torque accessors
TEST(ParticleTest, ForceAndTorqueAccessors) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    // Initially should be zero
    auto force = p.getForce();
    auto torque = p.getTorque();
    
    EXPECT_EQ(force[0], 0.0);
    EXPECT_EQ(force[1], 0.0);
    EXPECT_EQ(force[2], 0.0);
    
    EXPECT_EQ(torque[0], 0.0);
    EXPECT_EQ(torque[1], 0.0);
    EXPECT_EQ(torque[2], 0.0);
}

// Test velocity accessors
TEST(ParticleTest, VelocityAccessors) {
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0, 0.5);
    
    auto vlin = p.getVelocityLinear();
    auto vang = p.getVelocityAngular();
    
    EXPECT_EQ(vlin[0], 0.0);
    EXPECT_EQ(vlin[1], 0.0);
    EXPECT_EQ(vlin[2], 0.0);
    
    EXPECT_EQ(vang[0], 0.0);
    EXPECT_EQ(vang[1], 0.0);
    EXPECT_EQ(vang[2], 0.0);
}
