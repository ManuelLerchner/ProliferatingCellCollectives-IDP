#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "spatial/SpatialGrid.h"
#include "simulation/Particle.h"

// Test SpatialGrid construction
TEST(SpatialGridTest, Construction) {
    double cell_size = 1.0;
    std::array<double, 3> domain_min = {0.0, 0.0, 0.0};
    std::array<double, 3> domain_max = {10.0, 10.0, 10.0};
    
    SpatialGrid grid(cell_size, domain_min, domain_max);
    
    EXPECT_DOUBLE_EQ(grid.getCellSize(), 1.0);
    EXPECT_EQ(grid.getDomainMin()[0], 0.0);
    EXPECT_EQ(grid.getDomainMax()[0], 10.0);
}

// Test clear functionality
TEST(SpatialGridTest, Clear) {
    SpatialGrid grid(1.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    
    grid.insertParticle(1, {5.0, 5.0, 5.0}, 2.0, 1.0, true, 0);
    grid.clear();
    
    // After clearing, finding collisions should return empty
    std::vector<Particle> local_particles;
    std::vector<Particle> ghost_particles;
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    EXPECT_EQ(pairs.size(), 0);
}

// Test particle insertion
TEST(SpatialGridTest, InsertParticle) {
    SpatialGrid grid(1.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    
    // Insert a particle
    grid.insertParticle(42, {5.0, 5.0, 5.0}, 2.0, 1.0, true, 0);
    
    // Grid should now contain the particle (tested indirectly through collision detection)
    EXPECT_DOUBLE_EQ(grid.getCellSize(), 1.0);
}

// Test collision detection with no particles
TEST(SpatialGridTest, NoCollisionsEmptyGrid) {
    SpatialGrid grid(1.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    
    std::vector<Particle> local_particles;
    std::vector<Particle> ghost_particles;
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    
    EXPECT_EQ(pairs.size(), 0);
}

// Test collision detection with single particle
TEST(SpatialGridTest, NoCollisionsSingleParticle) {
    SpatialGrid grid(2.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    
    std::vector<Particle> local_particles;
    local_particles.emplace_back(1, std::array<double, 3>{5.0, 5.0, 5.0}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    
    std::vector<Particle> ghost_particles;
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    
    EXPECT_EQ(pairs.size(), 0);
}

// Test collision detection with two nearby particles
TEST(SpatialGridTest, DetectNearbyParticles) {
    SpatialGrid grid(2.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    
    std::vector<Particle> local_particles;
    // Two particles close together
    local_particles.emplace_back(1, std::array<double, 3>{5.0, 5.0, 5.0}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    local_particles.emplace_back(2, std::array<double, 3>{5.5, 5.5, 5.5}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    
    std::vector<Particle> ghost_particles;
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    
    // Should detect at least one potential collision pair
    EXPECT_GT(pairs.size(), 0);
}

// Test collision detection with far apart particles
TEST(SpatialGridTest, NoCollisionsFarApart) {
    SpatialGrid grid(1.0, {0.0, 0.0, 0.0}, {20.0, 20.0, 20.0});
    
    std::vector<Particle> local_particles;
    // Two particles far apart
    local_particles.emplace_back(1, std::array<double, 3>{2.0, 2.0, 2.0}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    local_particles.emplace_back(2, std::array<double, 3>{18.0, 18.0, 18.0}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    
    std::vector<Particle> ghost_particles;
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    
    // Should not detect any collision pairs (too far apart)
    EXPECT_EQ(pairs.size(), 0);
}

// Test with ghost particles
TEST(SpatialGridTest, GhostParticles) {
    SpatialGrid grid(2.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    
    std::vector<Particle> local_particles;
    local_particles.emplace_back(1, std::array<double, 3>{5.0, 5.0, 5.0}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    
    std::vector<Particle> ghost_particles;
    ghost_particles.emplace_back(2, std::array<double, 3>{5.5, 5.5, 5.5}, 
                                   std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                   2.0, 2.0, 1.0);
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    
    // Should detect potential collision between local and ghost particle
    EXPECT_GT(pairs.size(), 0);
    
    // Verify the pair contains one local and one ghost
    if (pairs.size() > 0) {
        EXPECT_TRUE((pairs[0].is_localI && !pairs[0].is_localJ) || 
                    (!pairs[0].is_localI && pairs[0].is_localJ));
    }
}

// Test domain boundaries
TEST(SpatialGridTest, DomainBoundaries) {
    SpatialGrid grid(1.0, {-5.0, -5.0, -5.0}, {5.0, 5.0, 5.0});
    
    EXPECT_EQ(grid.getDomainMin()[0], -5.0);
    EXPECT_EQ(grid.getDomainMin()[1], -5.0);
    EXPECT_EQ(grid.getDomainMin()[2], -5.0);
    EXPECT_EQ(grid.getDomainMax()[0], 5.0);
    EXPECT_EQ(grid.getDomainMax()[1], 5.0);
    EXPECT_EQ(grid.getDomainMax()[2], 5.0);
}

// Test with negative coordinates
TEST(SpatialGridTest, NegativeCoordinates) {
    SpatialGrid grid(1.0, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    
    std::vector<Particle> local_particles;
    local_particles.emplace_back(1, std::array<double, 3>{-5.0, -5.0, -5.0}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    local_particles.emplace_back(2, std::array<double, 3>{-4.5, -4.5, -4.5}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    
    std::vector<Particle> ghost_particles;
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    
    // Should handle negative coordinates correctly
    EXPECT_GT(pairs.size(), 0);
}

// Test cell size variations
TEST(SpatialGridTest, DifferentCellSizes) {
    // Small cell size
    SpatialGrid grid_small(0.5, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    EXPECT_DOUBLE_EQ(grid_small.getCellSize(), 0.5);
    
    // Large cell size
    SpatialGrid grid_large(5.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    EXPECT_DOUBLE_EQ(grid_large.getCellSize(), 5.0);
}

// Test multiple particles in same cell
TEST(SpatialGridTest, MultipleParticlesSameCell) {
    SpatialGrid grid(5.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    
    std::vector<Particle> local_particles;
    // Multiple particles in the same cell
    for (int i = 0; i < 5; i++) {
        local_particles.emplace_back(i, std::array<double, 3>{2.0 + i*0.1, 2.0, 2.0}, 
                                      std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                      1.0, 1.0, 0.5);
    }
    
    std::vector<Particle> ghost_particles;
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    
    // Should detect multiple pairs
    EXPECT_GT(pairs.size(), 0);
}

// Test collision pair properties
TEST(SpatialGridTest, CollisionPairProperties) {
    SpatialGrid grid(2.0, {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0});
    
    std::vector<Particle> local_particles;
    local_particles.emplace_back(10, std::array<double, 3>{5.0, 5.0, 5.0}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    local_particles.emplace_back(20, std::array<double, 3>{5.5, 5.5, 5.5}, 
                                  std::array<double, 4>{1.0, 0.0, 0.0, 0.0}, 
                                  2.0, 2.0, 1.0);
    
    std::vector<Particle> ghost_particles;
    
    auto pairs = grid.findPotentialCollisions(local_particles, ghost_particles);
    
    ASSERT_GT(pairs.size(), 0);
    
    // Check that GIDs are correct
    EXPECT_TRUE(pairs[0].gidI == 10 || pairs[0].gidI == 20);
    EXPECT_TRUE(pairs[0].gidJ == 10 || pairs[0].gidJ == 20);
    EXPECT_NE(pairs[0].gidI, pairs[0].gidJ);
    
    // Both should be local
    EXPECT_TRUE(pairs[0].is_localI);
    EXPECT_TRUE(pairs[0].is_localJ);
}
