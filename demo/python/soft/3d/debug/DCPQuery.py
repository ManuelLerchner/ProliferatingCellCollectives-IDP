"""
DCPQuery.py
Python implementation of geometric utilities for line segment collision detection
Converted from C++ implementation by wenyan4work (wenyan4work@gmail.com)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class ResultP3S3:
    """Result of point-segment distance in 3D"""
    distance: float = 0.0
    sqr_distance: float = 0.0
    segment_parameter: float = 0.0  # t in [0,1]
    segment_closest: np.ndarray = None  # (1-t)*p[0] + t*p[1]


@dataclass
class ResultS3S3:
    """Result of segment-segment distance in 3D"""
    distance: float = 0.0
    sqr_distance: float = 0.0
    segment_parameter: List[float] = None  # t in [0,1]
    segment_closest: List[np.ndarray] = None  # (1-t)*p[0] + t*p[1]


def norm(vec: np.ndarray) -> float:
    """Compute the norm of a 3D vector

    Args:
        vec: 3D vector

    Returns:
        Norm of the vector
    """
    return np.linalg.norm(vec)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector in place

    Args:
        vec: Vector to normalize

    Returns:
        Normalized vector
    """
    return vec / np.linalg.norm(vec)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot-product of two vectors

    Args:
        a: First vector
        b: Second vector

    Returns:
        Dot product result
    """
    return np.dot(a, b)


def dist_point_seg(point: np.ndarray, minus: np.ndarray, plus: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute minimal point-segment distance in 3D space

    Args:
        point: Point in 3D space
        minus: End point 0 of the segment
        plus: End point 1 of the segment

    Returns:
        Tuple of (minimal distance, point on segment with minimum distance)
    """
    result = ResultP3S3()

    # The direction vector is not unit length. The normalization is deferred
    # until it is needed.
    direction = plus - minus
    diff = point - plus
    t = dot(direction, diff)

    if t >= 0:
        result.segment_parameter = 1
        result.segment_closest = plus
    else:
        diff = point - minus
        t = dot(direction, diff)
        if t <= 0:
            result.segment_parameter = 0
            result.segment_closest = minus
        else:
            sqr_length = dot(direction, direction)
            if sqr_length > 0:
                t /= sqr_length
                result.segment_parameter = t
                result.segment_closest = minus + t * direction
            else:
                result.segment_parameter = 0
                result.segment_closest = minus

    diff = point - result.segment_closest
    result.sqr_distance = dot(diff, diff)
    result.distance = np.sqrt(result.sqr_distance)

    return result.distance, result.segment_closest


class DCPQuery:
    """Query for minimal segment-segment distance in 3D space

    This object must be thread private in multi-threading environment
    """

    @dataclass
    class Result:
        distance: float = 0.0
        sqr_distance: float = 0.0
        parameter: List[float] = None
        closest: List[np.ndarray] = None

    def __init__(self):
        # The coefficients of R(s,t), not including the constant term
        self.m_a = 0.0
        self.m_b = 0.0
        self.m_c = 0.0
        self.m_d = 0.0
        self.m_e = 0.0

        # dR/ds(i,j) at the four corners of the domain
        self.m_f00 = 0.0
        self.m_f10 = 0.0
        self.m_f01 = 0.0
        self.m_f11 = 0.0

        # dR/dt(i,j) at the four corners of the domain
        self.m_g00 = 0.0
        self.m_g10 = 0.0
        self.m_g01 = 0.0
        self.m_g11 = 0.0

    def __call__(self, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
        """Calculate minimum distance between two line segments

        Args:
            p0: End point 0 of segment P
            p1: End point 1 of segment P
            q0: End point 0 of segment Q
            q1: End point 1 of segment Q

        Returns:
            Tuple of (distance, point on P, point on Q, parameter s, parameter t)
        """
        result = self.Result()
        result.parameter = [0.0, 0.0]
        result.closest = [np.zeros(3), np.zeros(3)]

        # The code allows degenerate line segments; that is, P0 and P1 can be
        # the same point or Q0 and Q1'll be the same point. The quadratic
        # function for squared distance between the segment is
        #   R(s,t) = a*s^2 - 2*b*s*t + c*t^2 + 2*d*s - 2*e*t + f
        # for (s,t) in [0,1]^2 where
        #   a = Dot(P1-P0,P1-P0), b = Dot(P1-P0,Q1-Q0), c = Dot(Q1-Q0,Q1-Q0),
        #   d = Dot(P1-P0,P0-Q0), e = Dot(Q1-Q0,P0-Q0), f = Dot(P0-Q0,P0-Q0)
        p1m_p0 = p1 - p0
        q1m_q0 = q1 - q0
        p0m_q0 = p0 - q0

        self.m_a = dot(p1m_p0, p1m_p0)
        self.m_b = dot(p1m_p0, q1m_q0)
        self.m_c = dot(q1m_q0, q1m_q0)
        self.m_d = dot(p1m_p0, p0m_q0)
        self.m_e = dot(q1m_q0, p0m_q0)

        self.m_f00 = self.m_d
        self.m_f10 = self.m_f00 + self.m_a
        self.m_f01 = self.m_f00 - self.m_b
        self.m_f11 = self.m_f10 - self.m_b

        self.m_g00 = -self.m_e
        self.m_g10 = self.m_g00 - self.m_b
        self.m_g01 = self.m_g00 + self.m_c
        self.m_g11 = self.m_g10 + self.m_c

        if self.m_a > 0.0 and self.m_c > 0.0:
            # Compute the solutions to dR/ds(s0,0) = 0 and dR/ds(s1,1) = 0. The
            # location of sI on the s-axis is stored in classifyI (I = 0 or 1). If
            # sI <= 0, classifyI is -1. If sI >= 1, classifyI is 1. If 0 < sI < 1,
            # classifyI is 0. This information helps determine where to search for
            # the minimum point (s,t).

            s_value = [0.0, 0.0]
            s_value[0] = self._get_clamped_root(
                self.m_a, self.m_f00, self.m_f10)
            s_value[1] = self._get_clamped_root(
                self.m_a, self.m_f01, self.m_f11)

            classify = [0, 0]
            for i in range(2):
                if s_value[i] <= 0.0:
                    classify[i] = -1
                elif s_value[i] >= 1.0:
                    classify[i] = 1
                else:
                    classify[i] = 0

            if classify[0] == -1 and classify[1] == -1:
                # The minimum must occur on s = 0 for 0 <= t <= 1
                result.parameter[0] = 0.0
                result.parameter[1] = self._get_clamped_root(
                    self.m_c, self.m_g00, self.m_g01)

            elif classify[0] == 1 and classify[1] == 1:
                # The minimum must occur on s = 1 for 0 <= t <= 1
                result.parameter[0] = 1.0
                result.parameter[1] = self._get_clamped_root(
                    self.m_c, self.m_g10, self.m_g11)

            else:
                # The line dR/ds = 0 intersects the domain [0,1]^2 in a
                # nondegenerate segment. Compute the endpoints of that segment,
                # end[0] and end[1]. The edge[i] flag tells you on which domain
                # edge end[i] lives: 0 (s=0), 1 (s=1), 2 (t=0), 3 (t=1).
                edge = [0, 0]
                end = [[0.0, 0.0], [0.0, 0.0]]
                self._compute_intersection(s_value, classify, edge, end)

                # The directional derivative of R along the segment of
                # intersection is
                #   H(z) = (end[1][1]-end[1][0])*dR/dt((1-z)*end[0] + z*end[1])
                # for z in [0,1]. The formula uses the fact that dR/ds = 0 on
                # the segment. Compute the minimum of H on [0,1].
                self._compute_minimum_parameters(edge, end, result.parameter)

        else:
            if self.m_a > 0.0:
                # The Q-segment is degenerate (Q0 and Q1 are the same point) and
                # the quadratic is R(s,0) = a*s^2 + 2*d*s + f and has (half)
                # first derivative F(t) = a*s + d. The closest P-point is
                # interior to the P-segment when F(0) < 0 and F(1) > 0.
                result.parameter[0] = self._get_clamped_root(
                    self.m_a, self.m_f00, self.m_f10)
                result.parameter[1] = 0.0

            elif self.m_c > 0.0:
                # The P-segment is degenerate (P0 and P1 are the same point) and
                # the quadratic is R(0,t) = c*t^2 - 2*e*t + f and has (half)
                # first derivative G(t) = c*t - e. The closest Q-point is
                # interior to the Q-segment when G(0) < 0 and G(1) > 0.
                result.parameter[0] = 0.0
                result.parameter[1] = self._get_clamped_root(
                    self.m_c, self.m_g00, self.m_g01)

            else:
                # P-segment and Q-segment are degenerate
                result.parameter[0] = 0.0
                result.parameter[1] = 0.0

        result.closest[0] = (1.0 - result.parameter[0]) * \
            p0 + result.parameter[0] * p1
        result.closest[1] = (1.0 - result.parameter[1]) * \
            q0 + result.parameter[1] * q1
        diff = result.closest[0] - result.closest[1]
        result.sqr_distance = dot(diff, diff)
        result.distance = np.sqrt(result.sqr_distance)

        return result.distance, result.closest[0], result.closest[1], result.parameter[0], result.parameter[1]

    def _get_clamped_root(self, slope, h0, h1):
        """Compute the root of h(z) = h0 + slope*z and clamp it to [0,1].

        It is required that for h1 = h(1), either (h0 < 0 and h1 > 0)
        or (h0 > 0 and h1 < 0).

        Args:
            slope: Slope of the line
            h0: Value at z=0
            h1: Value at z=1

        Returns:
            Clamped root value
        """
        eps = np.finfo(float).eps
        assert abs(slope - (h1 - h0)) < 10 * eps * \
            max(abs(h1), abs(h0)), "Slope mismatch"

        if abs(h0) < eps and abs(h1) < eps:
            # Tiny slope, h0 ≈ h1, distance almost constant, choose mid point
            return 0.5
        elif h0 < 0:
            if h1 > 0:
                # Clamp r between [0,1]
                return min(max(-h0 / slope, 0.0), 1.0)
            else:
                return 1.0
        else:
            return 0.0

    def _compute_intersection(self, s_value, classify, edge, end):
        """Compute intersection of the line dR/ds = 0 with domain [0,1]^2

        Args:
            s_value: s-values at t=0 and t=1
            classify: Classification of s-values
            edge: Output edge flags
            end: Output endpoints
        """
        if classify[0] < 0:
            edge[0] = 0
            end[0][0] = 0.0
            end[0][1] = self.m_f00 / self.m_b
            if end[0][1] < 0.0 or end[0][1] > 1.0:
                end[0][1] = 0.5

            if classify[1] == 0:
                edge[1] = 3
                end[1][0] = s_value[1]
                end[1][1] = 1.0
            else:  # classify[1] > 0
                edge[1] = 1
                end[1][0] = 1.0
                end[1][1] = self.m_f10 / self.m_b
                if end[1][1] < 0.0 or end[1][1] > 1.0:
                    end[1][1] = 0.5

        elif classify[0] == 0:
            edge[0] = 2
            end[0][0] = s_value[0]
            end[0][1] = 0.0

            if classify[1] < 0:
                edge[1] = 0
                end[1][0] = 0.0
                end[1][1] = self.m_f00 / self.m_b
                if end[1][1] < 0.0 or end[1][1] > 1.0:
                    end[1][1] = 0.5
            elif classify[1] == 0:
                edge[1] = 3
                end[1][0] = s_value[1]
                end[1][1] = 1.0
            else:
                edge[1] = 1
                end[1][0] = 1.0
                end[1][1] = self.m_f10 / self.m_b
                if end[1][1] < 0.0 or end[1][1] > 1.0:
                    end[1][1] = 0.5

        else:  # classify[0] > 0
            edge[0] = 1
            end[0][0] = 1.0
            end[0][1] = self.m_f10 / self.m_b
            if end[0][1] < 0.0 or end[0][1] > 1.0:
                end[0][1] = 0.5

            if classify[1] == 0:
                edge[1] = 3
                end[1][0] = s_value[1]
                end[1][1] = 1.0
            else:
                edge[1] = 0
                end[1][0] = 0.0
                end[1][1] = self.m_f00 / self.m_b
                if end[1][1] < 0.0 or end[1][1] > 1.0:
                    end[1][1] = 0.5

    def _compute_minimum_parameters(self, edge, end, parameter):
        """Compute location of minimum of R on segment of intersection

        Args:
            edge: Edge flags
            end: Endpoints
            parameter: Output parameters
        """
        eps = np.finfo(float).eps
        delta = end[1][1] - end[0][1]
        h0 = delta * ((-self.m_b * end[0][0] -
                      self.m_e) + self.m_c * end[0][1])
        h1 = delta * ((-self.m_b * end[1][0] -
                      self.m_e) + self.m_c * end[1][1])

        if abs(h0) < abs(self.m_c) * eps and abs(h1) < abs(self.m_c) * eps:
            z = 0.5
            omz = 1.0 - z
            parameter[0] = omz * end[0][0] + z * end[1][0]
            parameter[1] = omz * end[0][1] + z * end[1][1]

        elif h0 >= 0.0:
            if edge[0] == 0:
                parameter[0] = 0.0
                parameter[1] = self._get_clamped_root(
                    self.m_c, self.m_g00, self.m_g01)
            elif edge[0] == 1:
                parameter[0] = 1.0
                parameter[1] = self._get_clamped_root(
                    self.m_c, self.m_g10, self.m_g11)
            else:
                parameter[0] = end[0][0]
                parameter[1] = end[0][1]

        else:
            if h1 <= 0.0:
                if edge[1] == 0:
                    parameter[0] = 0.0
                    parameter[1] = self._get_clamped_root(
                        self.m_c, self.m_g00, self.m_g01)
                elif edge[1] == 1:
                    parameter[0] = 1.0
                    parameter[1] = self._get_clamped_root(
                        self.m_c, self.m_g10, self.m_g11)
                else:
                    parameter[0] = end[1][0]
                    parameter[1] = end[1][1]
            else:  # h0 < 0 and h1 > 0
                z = self._get_clamped_root(h1 - h0, h0, h1)
                omz = 1.0 - z
                parameter[0] = omz * end[0][0] + z * end[1][0]
                parameter[1] = omz * end[0][1] + z * end[1][1]


# Convenience functions
def segment_segment_distance(p0, p1, q0, q1):
    """Calculate minimum distance between two line segments in 3D

    Args:
        p0: End point 0 of segment P
        p1: End point 1 of segment P
        q0: End point 0 of segment Q
        q1: End point 1 of segment Q

    Returns:
        Tuple of (distance, point on P, point on Q, parameter s, parameter t)
    """
    query = DCPQuery()
    return query(p0, p1, q0, q1)


# Example usage
if __name__ == "__main__":
    # Example: Two skew lines
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    q0 = np.array([0.0, 1.0, 0.0])
    q1 = np.array([0.0, 1.0, 1.0])

    distance, p_closest, q_closest, s, t = segment_segment_distance(
        p0, p1, q0, q1)

    print(f"Distance between segments: {distance}")
    print(f"Closest point on first segment: {p_closest}, parameter s={s}")
    print(f"Closest point on second segment: {q_closest}, parameter t={t}")
