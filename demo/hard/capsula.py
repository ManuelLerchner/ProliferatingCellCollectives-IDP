import numpy as np

from quaternion import rotateVectorOfPositions


def getDirectionVector(q):
    defaultDirection = [np.array([1.0, 0.0, 0.0])]
    return rotateVectorOfPositions(q, defaultDirection)[0]


"""
DCPQuery.py
Python implementation of geometric utilities for line segment collision detection
Ported from DCPQuery.hpp C++ code
"""


class ResultP3S3:
    """Result of point-segment distance in 3D"""

    def __init__(self):
        self.distance = 0.0
        self.sqr_distance = 0.0
        self.segment_parameter = 0.0  # t in [0,1]
        self.segment_closest = None   # (1-t)*p[0] + t*p[1]


class ResultS3S3:
    """Result of segment-segment distance in 3D"""

    def __init__(self):
        self.distance = 0.0
        self.sqr_distance = 0.0
        self.segment_parameter = [0.0, 0.0]  # t in [0,1]
        self.segment_closest = [None, None]  # (1-t)*p[0] + t*p[1]


def dist_point_seg(point, minus, plus):
    """Compute minimal point-segment distance in 3D space

    Args:
        point: numpy array representing the point
        minus: numpy array representing end point 0 of the segment
        plus: numpy array representing end point 1 of the segment

    Returns:
        distance: minimal distance
        point_perp: point on the segment with minimum distance
    """
    result = ResultP3S3()

    # The direction vector is not unit length. The normalization is deferred
    # until it is needed.
    direction = plus - minus
    diff = point - plus
    t = np.dot(direction, diff)

    if t >= 0:
        result.segment_parameter = 1
        result.segment_closest = plus
    else:
        diff = point - minus
        t = np.dot(direction, diff)
        if t <= 0:
            result.segment_parameter = 0
            result.segment_closest = minus
        else:
            sqr_length = np.dot(direction, direction)
            if sqr_length > 0:
                t /= sqr_length
                result.segment_parameter = t
                result.segment_closest = minus + t * direction
            else:
                result.segment_parameter = 0
                result.segment_closest = minus

    diff = point - result.segment_closest
    result.sqr_distance = np.dot(diff, diff)
    result.distance = np.sqrt(result.sqr_distance)

    return result.distance, result.segment_closest


class DCPQuery:
    """Functor for minimal segment-segment distance query in 3D space

    This object must be thread private in multi-threading environment
    """

    class Result:
        def __init__(self):
            self.distance = 0.0
            self.sqr_distance = 0.0
            self.parameter = [0.0, 0.0]
            self.closest = [None, None]

    def __init__(self):
        # The coefficients of R(s,t), not including the constant term
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0
        self.e = 0.0

        # dR/ds(i,j) at the four corners of the domain
        self.f00 = 0.0
        self.f10 = 0.0
        self.f01 = 0.0
        self.f11 = 0.0

        # dR/dt(i,j) at the four corners of the domain
        self.g00 = 0.0
        self.g10 = 0.0
        self.g01 = 0.0
        self.g11 = 0.0

    def __call__(self, p0, p1, q0, q1):
        """Functor for computing minimal segment-segment distance in 3D space

        Args:
            p0: End point 0 of segment P (numpy array)
            p1: End point 1 of segment P (numpy array)
            q0: End point 0 of segment Q (numpy array)
            q1: End point 1 of segment Q (numpy array)

        Returns:
            distance: Computed minimal distance
            p_loc: Result point on P
            q_loc: Result point on Q
            s: s∈[0,1] describing p_loc on P
            t: t∈[0,1] describing q_loc on Q
        """
        result = self.Result()

        # The code allows degenerate line segments; that is, P0 and P1 can be
        # the same point or Q0 and Q1 can be the same point. The quadratic
        # function for squared distance between the segment is
        #   R(s,t) = a*s^2 - 2*b*s*t + c*t^2 + 2*d*s - 2*e*t + f
        # for (s,t) in [0,1]^2 where
        #   a = Dot(P1-P0,P1-P0), b = Dot(P1-P0,Q1-Q0), c = Dot(Q1-Q0,Q1-Q0),
        #   d = Dot(P1-P0,P0-Q0), e = Dot(Q1-Q0,P0-Q0), f = Dot(P0-Q0,P0-Q0)
        p1m_p0 = p1 - p0
        q1m_q0 = q1 - q0
        p0m_q0 = p0 - q0

        self.a = np.dot(p1m_p0, p1m_p0)
        self.b = np.dot(p1m_p0, q1m_q0)
        self.c = np.dot(q1m_q0, q1m_q0)
        self.d = np.dot(p1m_p0, p0m_q0)
        self.e = np.dot(q1m_q0, p0m_q0)

        self.f00 = self.d
        self.f10 = self.f00 + self.a
        self.f01 = self.f00 - self.b
        self.f11 = self.f10 - self.b

        self.g00 = -self.e
        self.g10 = self.g00 - self.b
        self.g01 = self.g00 + self.c
        self.g11 = self.g10 + self.c

        if self.a > 0.0 and self.c > 0.0:
            # Compute the solutions to dR/ds(s0,0) = 0 and dR/ds(s1,1) = 0.  The
            # location of sI on the s-axis is stored in classifyI (I = 0 or 1).  If
            # sI <= 0, classifyI is -1.  If sI >= 1, classifyI is 1.  If 0 < sI < 1,
            # classifyI is 0.  This information helps determine where to search for
            # the minimum point (s,t).  The fij values are dR/ds(i,j) for i and j in
            # {0,1}.

            s_value = [0.0, 0.0]
            s_value[0] = self._get_clamped_root(self.a, self.f00, self.f10)
            s_value[1] = self._get_clamped_root(self.a, self.f01, self.f11)

            classify = [0, 0]
            for i in range(2):
                if s_value[i] <= 0.0:
                    classify[i] = -1
                elif s_value[i] >= 1.0:
                    classify[i] = 1
                else:
                    classify[i] = 0

            if classify[0] == -1 and classify[1] == -1:
                # The minimum must occur on s = 0 for 0 <= t <= 1.
                result.parameter[0] = 0.0
                result.parameter[1] = self._get_clamped_root(
                    self.c, self.g00, self.g01)
            elif classify[0] == 1 and classify[1] == 1:
                # The minimum must occur on s = 1 for 0 <= t <= 1.
                result.parameter[0] = 1.0
                result.parameter[1] = self._get_clamped_root(
                    self.c, self.g10, self.g11)
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
            if self.a > 0.0:
                # The Q-segment is degenerate (Q0 and Q1 are the same point) and
                # the quadratic is R(s,0) = a*s^2 + 2*d*s + f and has (half)
                # first derivative F(t) = a*s + d. The closest P-point is
                # interior to the P-segment when F(0) < 0 and F(1) > 0.
                result.parameter[0] = self._get_clamped_root(
                    self.a, self.f00, self.f10)
                result.parameter[1] = 0.0
            elif self.c > 0.0:
                # The P-segment is degenerate (P0 and P1 are the same point) and
                # the quadratic is R(0,t) = c*t^2 - 2*e*t + f and has (half)
                # first derivative G(t) = c*t - e. The closest Q-point is
                # interior to the Q-segment when G(0) < 0 and G(1) > 0.
                result.parameter[0] = 0.0
                result.parameter[1] = self._get_clamped_root(
                    self.c, self.g00, self.g01)
            else:
                # P-segment and Q-segment are degenerate.
                result.parameter[0] = 0.0
                result.parameter[1] = 0.0

        result.closest[0] = (1.0 - result.parameter[0]) * \
            p0 + result.parameter[0] * p1
        result.closest[1] = (1.0 - result.parameter[1]) * \
            q0 + result.parameter[1] * q1
        diff = result.closest[0] - result.closest[1]
        result.sqr_distance = np.dot(diff, diff)
        result.distance = np.sqrt(result.sqr_distance)

        return result.distance, result.closest[0], result.closest[1], result.parameter[0], result.parameter[1]

    def _get_clamped_root(self, slope, h0, h1):
        """Compute the root of h(z) = h0 + slope*z and clamp it to the interval [0,1].

        It is required that for h1 = h(1), either (h0 < 0 and h1 > 0)
        or (h0 > 0 and h1 < 0).
        """
        eps = np.finfo(float).eps

        # Verify that slope approximates h1-h0
        assert abs(slope - (h1 - h0)) < 10 * eps * max(abs(h1), abs(h0))

        if abs(h0) < eps and abs(h1) < eps:
            # Tiny slope, h0 ≈ h1, distance almost a constant, choose mid point
            r = 0.5
        elif h0 < 0:
            if h1 > 0:
                # Clamp r between [0,1]
                r = min(max(-h0 / slope, 0.0), 1.0)
            else:
                r = 1.0
        else:
            r = 0.0

        return r

    def _compute_intersection(self, s_value, classify, edge, end):
        """Compute the intersection of the line dR/ds = 0 with the domain [0,1]^2."""
        if classify[0] < 0:
            edge[0] = 0
            end[0][0] = 0.0
            end[0][1] = self.f00 / self.b
            if end[0][1] < 0.0 or end[0][1] > 1.0:
                end[0][1] = 0.5

            if classify[1] == 0:
                edge[1] = 3
                end[1][0] = s_value[1]
                end[1][1] = 1.0
            else:  # classify[1] > 0
                edge[1] = 1
                end[1][0] = 1.0
                end[1][1] = self.f10 / self.b
                if end[1][1] < 0.0 or end[1][1] > 1.0:
                    end[1][1] = 0.5

        elif classify[0] == 0:
            edge[0] = 2
            end[0][0] = s_value[0]
            end[0][1] = 0.0

            if classify[1] < 0:
                edge[1] = 0
                end[1][0] = 0.0
                end[1][1] = self.f00 / self.b
                if end[1][1] < 0.0 or end[1][1] > 1.0:
                    end[1][1] = 0.5
            elif classify[1] == 0:
                edge[1] = 3
                end[1][0] = s_value[1]
                end[1][1] = 1.0
            else:  # classify[1] > 0
                edge[1] = 1
                end[1][0] = 1.0
                end[1][1] = self.f10 / self.b
                if end[1][1] < 0.0 or end[1][1] > 1.0:
                    end[1][1] = 0.5

        else:  # classify[0] > 0
            edge[0] = 1
            end[0][0] = 1.0
            end[0][1] = self.f10 / self.b
            if end[0][1] < 0.0 or end[0][1] > 1.0:
                end[0][1] = 0.5

            if classify[1] == 0:
                edge[1] = 3
                end[1][0] = s_value[1]
                end[1][1] = 1.0
            else:  # classify[1] < 0 (implied)
                edge[1] = 0
                end[1][0] = 0.0
                end[1][1] = self.f00 / self.b
                if end[1][1] < 0.0 or end[1][1] > 1.0:
                    end[1][1] = 0.5

    def _compute_minimum_parameters(self, edge, end, parameter):
        """Compute the location of the minimum of R on the segment of intersection."""
        eps = np.finfo(float).eps
        delta = end[1][1] - end[0][1]
        # Source of rounding error
        h0 = delta * ((-self.b * end[0][0] - self.e) + self.c * end[0][1])
        h1 = delta * ((-self.b * end[1][0] - self.e) + self.c * end[1][1])

        if abs(h0) < abs(self.c) * eps and abs(h1) < abs(self.c) * eps:
            z = 0.5
            omz = 1.0 - z
            parameter[0] = omz * end[0][0] + z * end[1][0]
            parameter[1] = omz * end[0][1] + z * end[1][1]
        elif h0 >= 0.0:
            if edge[0] == 0:
                parameter[0] = 0.0
                parameter[1] = self._get_clamped_root(
                    self.c, self.g00, self.g01)
            elif edge[0] == 1:
                parameter[0] = 1.0
                parameter[1] = self._get_clamped_root(
                    self.c, self.g10, self.g11)
            else:  # edge[0] == 2 or edge[0] == 3
                parameter[0] = end[0][0]
                parameter[1] = end[0][1]
        else:  # h0 < 0
            if h1 <= 0.0:
                if edge[1] == 0:
                    parameter[0] = 0.0
                    parameter[1] = self._get_clamped_root(
                        self.c, self.g00, self.g01)
                elif edge[1] == 1:
                    parameter[0] = 1.0
                    parameter[1] = self._get_clamped_root(
                        self.c, self.g10, self.g11)
                else:  # edge[1] == 2 or edge[1] == 3
                    parameter[0] = end[1][0]
                    parameter[1] = end[1][1]
            else:  # h0 < 0 and h1 > 0
                z = self._get_clamped_root(h1 - h0, h0, h1)
                omz = 1.0 - z
                parameter[0] = omz * end[0][0] + z * end[1][0]
                parameter[1] = omz * end[0][1] + z * end[1][1]


DCPQuery = DCPQuery()


def signed_distance_capsule(C, L, n, m):
    """
    Compute the signed distance between two capsules in 3D space.
    Args:
        C: Configuration array containing position and quaternion data for particles
           Each particle has [x, y, z, s, wx, wy, wz]
        L: Length array containing lengths of the capsules
        n: Index of the first capsule
        m: Index of the second capsule
    Returns:
        distance: The signed distance between the two capsules
        closest_point_capsule1: Closest point on the first capsule
        closest_point_capsule2: Closest point on the second capsule
        parameter_capsule1: Parameter t for the first capsule
        parameter_capsule2: Parameter t for the second capsule
    """

    x1 = C[7*n:7*n + 3]  # position of capsule
    q1 = C[7*n + 3:7*n + 7]  # quaternion of capsule

    x2 = C[7*m:7*m + 3]  # position of capsule
    q2 = C[7*m + 3:7*m + 7]  # quaternion of capsule

    # P0 and P1 are the endpoints of the capsule
    dir1 = getDirectionVector(q1)  # Direction vector of the capsule
    dir2 = getDirectionVector(q2)  # Direction vector of the capsule

    # Length of the capsule
    l1 = L[n]  # Length of the capsule
    l2 = L[m]  # Length of the capsule

    diameter = 0.5  # Diameter of the capsule

    P1 = x1 - dir1 * (l1 / 2 - diameter / 2)
    P2 = x1 + dir1 * (l1 / 2 - diameter / 2)

    Q1 = x2 - dir2 * (l2 / 2 - diameter / 2)
    Q2 = x2 + dir2 * (l2 / 2 - diameter / 2)

    # Compute the signed distance
    return DCPQuery(P1, P2, Q1, Q2)
