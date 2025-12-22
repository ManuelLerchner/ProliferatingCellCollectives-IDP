#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "spatial/CollisionDetector.h"
#include "simulation/Particle.h"
#include "util/Quaternion.h"

// Test CollisionDetector construction
TEST(CollisionDetectorTest, Construction) {
    double collision_tolerance = 0.01;
    double cell_size = 1.0;
    
    CollisionDetector detector(collision_tolerance, cell_size);
    
    // Should construct without errors
    EXPECT_NO_THROW({
        auto grid = detector.getSpatialGrid();
    });
}

// Test updateBounds
TEST(CollisionDetectorTest, UpdateBounds) {
    CollisionDetector detector(0.01, 1.0);
    
    std::array<double, 3> min_bounds = {-5.0, -5.0, -5.0};
    std::array<double, 3> max_bounds = {5.0, 5.0, 5.0};
    
    EXPECT_NO_THROW({
        detector.updateBounds(min_bounds, max_bounds);
    });
}

// Test reset functionality
TEST(CollisionDetectorTest, Reset) {
    CollisionDetector detector(0.01, 1.0);
    
    EXPECT_NO_THROW({
        detector.reset();
    });
}

// Test getParticleEndpoints
TEST(CollisionDetectorTest, GetParticleEndpoints) {
    CollisionDetector detector(0.01, 1.0);
    
    // Create a particle aligned with x-axis
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 4.0, 4.0, 1.0);
    
    auto [start, end] = detector.getParticleEndpoints(p);
    
    // For a particle of length 4.0 and diameter 1.0, centered at origin
    // Core length = 4.0 - 1.0 = 3.0
    // Half core length = 1.5
    // Start should be at (-1.5, 0, 0), end at (1.5, 0, 0)
    EXPECT_NEAR(start[0], -1.5, 0.01);
    EXPECT_NEAR(start[1], 0.0, 0.01);
    EXPECT_NEAR(start[2], 0.0, 0.01);
    
    EXPECT_NEAR(end[0], 1.5, 0.01);
    EXPECT_NEAR(end[1], 0.0, 0.01);
    EXPECT_NEAR(end[2], 0.0, 0.01);
}

// Test getParticleEndpoints with different orientation
TEST(CollisionDetectorTest, GetParticleEndpointsRotated) {
    CollisionDetector detector(0.01, 1.0);
    
    // Create a particle rotated 90 degrees around Z axis
    double angle = M_PI / 2;
    std::array<double, 4> q = {std::cos(angle/2), 0.0, 0.0, std::sin(angle/2)};
    Particle p(1, {0.0, 0.0, 0.0}, q, 4.0, 4.0, 1.0);
    
    auto [start, end] = detector.getParticleEndpoints(p);
    
    // Should now be along y-axis
    EXPECT_NEAR(start[0], 0.0, 0.01);
    EXPECT_NEAR(start[1], -1.5, 0.01);
    EXPECT_NEAR(start[2], 0.0, 0.01);
    
    EXPECT_NEAR(end[0], 0.0, 0.01);
    EXPECT_NEAR(end[1], 1.5, 0.01);
    EXPECT_NEAR(end[2], 0.0, 0.01);
}

// Test getParticleEndpoints with offset position
TEST(CollisionDetectorTest, GetParticleEndpointsOffset) {
    CollisionDetector detector(0.01, 1.0);
    
    // Particle at position (10, 20, 30)
    Particle p(1, {10.0, 20.0, 30.0}, {1.0, 0.0, 0.0, 0.0}, 4.0, 4.0, 1.0);
    
    auto [start, end] = detector.getParticleEndpoints(p);
    
    // Endpoints should be offset by particle position
    EXPECT_NEAR(start[0], 10.0 - 1.5, 0.01);
    EXPECT_NEAR(start[1], 20.0, 0.01);
    EXPECT_NEAR(start[2], 30.0, 0.01);
    
    EXPECT_NEAR(end[0], 10.0 + 1.5, 0.01);
    EXPECT_NEAR(end[1], 20.0, 0.01);
    EXPECT_NEAR(end[2], 30.0, 0.01);
}

// Test getParticleEndpoints with minimal length (sphere)
TEST(CollisionDetectorTest, GetParticleEndpointsSphere) {
    CollisionDetector detector(0.01, 1.0);
    
    // When length equals diameter, it's a sphere
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 2.0, 2.0, 2.0);
    
    auto [start, end] = detector.getParticleEndpoints(p);
    
    // Core length = 0, so both endpoints should be at center
    EXPECT_NEAR(start[0], 0.0, 0.01);
    EXPECT_NEAR(start[1], 0.0, 0.01);
    EXPECT_NEAR(start[2], 0.0, 0.01);
    
    EXPECT_NEAR(end[0], 0.0, 0.01);
    EXPECT_NEAR(end[1], 0.0, 0.01);
    EXPECT_NEAR(end[2], 0.0, 0.01);
}

// Test getSpatialGrid
TEST(CollisionDetectorTest, GetSpatialGrid) {
    double cell_size = 2.5;
    CollisionDetector detector(0.01, cell_size);
    
    auto grid = detector.getSpatialGrid();
    
    EXPECT_DOUBLE_EQ(grid.getCellSize(), cell_size);
}

// Test with different collision tolerances
TEST(CollisionDetectorTest, DifferentTolerances) {
    CollisionDetector detector1(0.001, 1.0);
    CollisionDetector detector2(0.1, 1.0);
    CollisionDetector detector3(1.0, 1.0);
    
    // All should construct successfully
    EXPECT_NO_THROW({
        auto grid1 = detector1.getSpatialGrid();
        auto grid2 = detector2.getSpatialGrid();
        auto grid3 = detector3.getSpatialGrid();
    });
}

// Test with different cell sizes
TEST(CollisionDetectorTest, DifferentCellSizes) {
    CollisionDetector detector1(0.01, 0.5);
    CollisionDetector detector2(0.01, 1.0);
    CollisionDetector detector3(0.01, 5.0);
    
    auto grid1 = detector1.getSpatialGrid();
    auto grid2 = detector2.getSpatialGrid();
    auto grid3 = detector3.getSpatialGrid();
    
    EXPECT_DOUBLE_EQ(grid1.getCellSize(), 0.5);
    EXPECT_DOUBLE_EQ(grid2.getCellSize(), 1.0);
    EXPECT_DOUBLE_EQ(grid3.getCellSize(), 5.0);
}

// Test particle endpoints distance
TEST(CollisionDetectorTest, EndpointsDistance) {
    CollisionDetector detector(0.01, 1.0);
    
    double length = 6.0;
    double diameter = 2.0;
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, length, length, diameter);
    
    auto [start, end] = detector.getParticleEndpoints(p);
    
    // Calculate distance between endpoints
    double dx = end[0] - start[0];
    double dy = end[1] - start[1];
    double dz = end[2] - start[2];
    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    // Distance should equal core length (length - diameter)
    double expected_distance = length - diameter;
    EXPECT_NEAR(distance, expected_distance, 0.01);
}

// Test multiple resets
TEST(CollisionDetectorTest, MultipleResets) {
    CollisionDetector detector(0.01, 1.0);
    
    for (int i = 0; i < 10; i++) {
        EXPECT_NO_THROW({
            detector.reset();
        });
    }
}

// Test bounds update with different domains
TEST(CollisionDetectorTest, UpdateBoundsVariousDomains) {
    CollisionDetector detector(0.01, 1.0);
    
    // Small domain
    detector.updateBounds({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    
    // Medium domain
    detector.updateBounds({-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    
    // Large domain
    detector.updateBounds({-100.0, -100.0, -100.0}, {100.0, 100.0, 100.0});
    
    // Each should work without errors
    auto grid = detector.getSpatialGrid();
    EXPECT_DOUBLE_EQ(grid.getCellSize(), 1.0);
}

// Test particle endpoints with various lengths
TEST(CollisionDetectorTest, EndpointsVariousLengths) {
    CollisionDetector detector(0.01, 1.0);
    
    double diameter = 1.0;
    std::vector<double> lengths = {2.0, 3.0, 5.0, 10.0};
    
    for (double length : lengths) {
        Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, length, length, diameter);
        auto [start, end] = detector.getParticleEndpoints(p);
        
        // Calculate distance
        double dx = end[0] - start[0];
        double dist = std::abs(dx);
        
        // Should match core length
        EXPECT_NEAR(dist, length - diameter, 0.01);
    }
}

// Test endpoints symmetry
TEST(CollisionDetectorTest, EndpointsSymmetry) {
    CollisionDetector detector(0.01, 1.0);
    
    Particle p(1, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, 4.0, 4.0, 1.0);
    
    auto [start, end] = detector.getParticleEndpoints(p);
    
    // For a particle centered at origin, start and end should be symmetric
    EXPECT_NEAR(start[0], -end[0], 0.01);
    EXPECT_NEAR(start[1], -end[1], 0.01);
    EXPECT_NEAR(start[2], -end[2], 0.01);
}

// Test with 3D rotation
TEST(CollisionDetectorTest, GetParticleEndpoints3DRotation) {
    CollisionDetector detector(0.01, 1.0);
    
    // Create a particle rotated in 3D space
    std::array<double, 3> euler = {0.5, 0.3, 0.7};
    auto q = utils::Quaternion::quaternionFromEuler(euler);
    
    Particle p(1, {5.0, 5.0, 5.0}, q, 4.0, 4.0, 1.0);
    
    auto [start, end] = detector.getParticleEndpoints(p);
    
    // Calculate distance between endpoints
    double dx = end[0] - start[0];
    double dy = end[1] - start[1];
    double dz = end[2] - start[2];
    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    // Distance should still equal core length regardless of rotation
    EXPECT_NEAR(distance, 3.0, 0.01);
    
    // Midpoint should be at particle position
    double mid_x = (start[0] + end[0]) / 2.0;
    double mid_y = (start[1] + end[1]) / 2.0;
    double mid_z = (start[2] + end[2]) / 2.0;
    
    EXPECT_NEAR(mid_x, 5.0, 0.01);
    EXPECT_NEAR(mid_y, 5.0, 0.01);
    EXPECT_NEAR(mid_z, 5.0, 0.01);
}
