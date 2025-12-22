#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "util/SpherocylinderCell.h"

using namespace utils::geometry;

// Test parallel segments
TEST(DCPQueryTest, ParallelSegmentsSameLine) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {2.0, 0.0, 0.0};
    std::array<double, 3> Q1 = {3.0, 0.0, 0.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Segments are on same line but separated
    EXPECT_NEAR(result.distance, 1.0, 1e-10);
    EXPECT_NEAR(result.sqrDistance, 1.0, 1e-10);
}

TEST(DCPQueryTest, ParallelSegmentsOffset) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {0.0, 1.0, 0.0};
    std::array<double, 3> Q1 = {1.0, 1.0, 0.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Parallel segments offset by 1 unit
    EXPECT_NEAR(result.distance, 1.0, 1e-10);
}

// Test perpendicular segments
TEST(DCPQueryTest, PerpendicularSegmentsIntersecting) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {-1.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {0.0, -1.0, 0.0};
    std::array<double, 3> Q1 = {0.0, 1.0, 0.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Segments intersect at origin
    EXPECT_NEAR(result.distance, 0.0, 1e-10);
    EXPECT_NEAR(result.parameter[0], 0.5, 1e-6); // Middle of first segment
    EXPECT_NEAR(result.parameter[1], 0.5, 1e-6); // Middle of second segment
}

TEST(DCPQueryTest, PerpendicularSegmentsNonIntersecting) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {-1.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {0.0, -1.0, 1.0}; // Elevated by 1 unit in Z
    std::array<double, 3> Q1 = {0.0, 1.0, 1.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Closest distance should be 1 (elevation in Z)
    EXPECT_NEAR(result.distance, 1.0, 1e-10);
}

// Test identical segments
TEST(DCPQueryTest, IdenticalSegments) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 1.0, 1.0};
    
    auto result = query(P0, P1, P0, P1);
    
    // Identical segments have zero distance
    EXPECT_NEAR(result.distance, 0.0, 1e-10);
}

// Test endpoint-to-endpoint cases
TEST(DCPQueryTest, TouchingEndpoints) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q1 = {2.0, 0.0, 0.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Segments touch at (1,0,0)
    EXPECT_NEAR(result.distance, 0.0, 1e-10);
    EXPECT_NEAR(result.parameter[0], 1.0, 1e-6); // End of first segment
    EXPECT_NEAR(result.parameter[1], 0.0, 1e-6); // Start of second segment
}

// Test degenerate segments (point segments)
TEST(DCPQueryTest, DegenerateSegmentPoint) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {0.0, 0.0, 0.0}; // Degenerate (point)
    std::array<double, 3> Q0 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q1 = {2.0, 0.0, 0.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Distance from point (0,0,0) to segment from (1,0,0) to (2,0,0)
    EXPECT_NEAR(result.distance, 1.0, 1e-10);
}

TEST(DCPQueryTest, BothDegenerateSegments) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {0.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {3.0, 4.0, 0.0};
    std::array<double, 3> Q1 = {3.0, 4.0, 0.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Distance between two points
    EXPECT_NEAR(result.distance, 5.0, 1e-10); // sqrt(3^2 + 4^2) = 5
}

// Test 3D cases
TEST(DCPQueryTest, SkewLinesIn3D) {
    DCPQuery<double, 3> query;
    
    // Two skew lines in 3D
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {0.0, 1.0, 1.0};
    std::array<double, 3> Q1 = {1.0, 1.0, 1.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Should find minimum distance between skew lines
    EXPECT_GT(result.distance, 0.0);
    EXPECT_LT(result.distance, 2.0); // Sanity check
}

// Test ComputeRobust variant
TEST(DCPQueryTest, ComputeRobustParallelSegments) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {0.0, 1.0, 0.0};
    std::array<double, 3> Q1 = {1.0, 1.0, 0.0};
    
    auto result = query.ComputeRobust(P0, P1, Q0, Q1);
    
    // Parallel segments offset by 1 unit
    EXPECT_NEAR(result.distance, 1.0, 1e-10);
}

TEST(DCPQueryTest, ComputeRobustIntersecting) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {-1.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {0.0, -1.0, 0.0};
    std::array<double, 3> Q1 = {0.0, 1.0, 0.0};
    
    auto result = query.ComputeRobust(P0, P1, Q0, Q1);
    
    // Segments intersect
    EXPECT_NEAR(result.distance, 0.0, 1e-10);
}

// Test 2D cases
TEST(DCPQueryTest, TwoDimensionalSegments) {
    DCPQuery<double, 2> query;
    
    std::array<double, 2> P0 = {0.0, 0.0};
    std::array<double, 2> P1 = {1.0, 0.0};
    std::array<double, 2> Q0 = {0.0, 1.0};
    std::array<double, 2> Q1 = {1.0, 1.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Parallel segments in 2D offset by 1
    EXPECT_NEAR(result.distance, 1.0, 1e-10);
}

// Test closest points are correct
TEST(DCPQueryTest, ClosestPointsOnSegments) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {2.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {1.0, 1.0, 0.0};
    std::array<double, 3> Q1 = {1.0, 3.0, 0.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Closest point on P should be (1,0,0)
    EXPECT_NEAR(result.closest[0][0], 1.0, 1e-6);
    EXPECT_NEAR(result.closest[0][1], 0.0, 1e-6);
    EXPECT_NEAR(result.closest[0][2], 0.0, 1e-6);
    
    // Closest point on Q should be (1,1,0)
    EXPECT_NEAR(result.closest[1][0], 1.0, 1e-6);
    EXPECT_NEAR(result.closest[1][1], 1.0, 1e-6);
    EXPECT_NEAR(result.closest[1][2], 0.0, 1e-6);
    
    // Distance should be 1
    EXPECT_NEAR(result.distance, 1.0, 1e-10);
}

// Test parameters are in valid range [0,1]
TEST(DCPQueryTest, ParametersInValidRange) {
    DCPQuery<double, 3> query;
    
    std::array<double, 3> P0 = {0.0, 0.0, 0.0};
    std::array<double, 3> P1 = {1.0, 0.0, 0.0};
    std::array<double, 3> Q0 = {0.5, 0.5, 0.0};
    std::array<double, 3> Q1 = {0.5, 1.5, 0.0};
    
    auto result = query(P0, P1, Q0, Q1);
    
    // Parameters should be in [0,1]
    EXPECT_GE(result.parameter[0], 0.0);
    EXPECT_LE(result.parameter[0], 1.0);
    EXPECT_GE(result.parameter[1], 0.0);
    EXPECT_LE(result.parameter[1], 1.0);
}
