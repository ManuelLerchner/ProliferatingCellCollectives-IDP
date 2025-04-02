from timeit import timeit
import numpy as np
from matplotlib import cm, transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arrow, Circle, Rectangle
from numba import jit


class Spherocylinder:
    """
    Class representing a single spherocylinder particle with growth dynamics, torque-based collision response,
    and background drag. Uses velocity Verlet integration for better numerical stability.
    """

    # Pre-compute color maps for reuse across all instances
    _length_colors = [(0.2, 0.5, 0.2), (0, 0.5, 0),
                      (0, 1, 0), (0.5, 1, 0.5), (0.5, 1, 0.5)]
    _length_cmap = LinearSegmentedColormap.from_list(
        'length_cmap', _length_colors, N=256)
    _stress_cmap = cm.get_cmap('Reds', 256)

    # Static method for distance calculation
    @staticmethod
    @jit(nopython=True)
    def minimum_distance_between_line_segments(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray,
                                               clampAll=False, clampA0=False, clampA1=False, clampB0=False, clampB1=False):
        # If clampAll=True, set all clamps to True
        if clampAll:
            clampA0 = clampA1 = clampB0 = clampB1 = True

        # Direction vectors
        A = a1 - a0
        B = b1 - b0

        # Magnitudes
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)

        # Normalized direction vectors
        _A = A / magA if magA > 0 else np.zeros_like(A)
        _B = B / magB if magB > 0 else np.zeros_like(B)

        # Cross product in 2D space (scalar)
        cross = _A[0] * _B[1] - _A[1] * _B[0]
        denom = cross * cross

        # If lines are parallel (denom=0)
        if denom < 1e-8:  # Use small epsilon instead of exactly 0
            d0 = np.dot(_A, (b0-a0))

            # Overlap only possible with clamping
            if clampA0 or clampA1 or clampB0 or clampB1:
                d1 = np.dot(_A, (b1-a0))

                # Is segment B before A?
                if d0 <= 0 >= d1:
                    if clampA0 and clampB1:
                        if np.absolute(d0) < np.absolute(d1):
                            return a0, b0, np.linalg.norm(a0-b0)
                        return a0, b1, np.linalg.norm(a0-b1)

                # Is segment B after A?
                elif d0 >= magA <= d1:
                    if clampA1 and clampB0:
                        if np.absolute(d0) < np.absolute(d1):
                            return a1, b0, np.linalg.norm(a1-b0)
                        return a1, b1, np.linalg.norm(a1-b1)

            # Segments overlap, return distance between parallel segments
            return None, None, np.linalg.norm(((d0*_A)+a0)-b0)

        # Lines criss-cross: Calculate the projected closest points
        # Use direct calculation instead of determinants
        t = b0 - a0
        t0 = (t[0] * _B[1] - t[1] * _B[0]) / cross
        t1 = (t[0] * _A[1] - t[1] * _A[0]) / cross

        pA = a0 + (_A * t0)  # Projected closest point on segment A
        pB = b0 + (_B * t1)  # Projected closest point on segment B

        # Clamp projections
        if clampA0 or clampA1 or clampB0 or clampB1:
            if clampA0 and t0 < 0:
                pA = a0
            elif clampA1 and t0 > magA:
                pA = a1

            if clampB0 and t1 < 0:
                pB = b0
            elif clampB1 and t1 > magB:
                pB = b1

            # Clamp projection A
            if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                dot = np.dot(_B, (pA-b0))
                if clampB0 and dot < 0:
                    dot = 0
                elif clampB1 and dot > magB:
                    dot = magB
                pB = b0 + (_B * dot)

            # Clamp projection B
            if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                dot = np.dot(_A, (pB-a0))
                if clampA0 and dot < 0:
                    dot = 0
                elif clampA1 and dot > magA:
                    dot = magA
                pA = a0 + (_A * dot)

        return pA, pB, np.linalg.norm(pA-pB)

    def __init__(self, position, linear_velocity, angular_velocity, orientation, l0):
        # Initialize basic properties
        self.position = np.array(position, dtype=float)
        self.orientation = float(orientation)
        self.l0 = float(l0)
        self.length = float(l0)
        self.max_length = float(2 * l0)
        self.diameter = 0.5 * l0
        self.stress = 0.0

        # Physical properties
        self.angular_velocity = angular_velocity
        self.linear_velocity = linear_velocity

        # Force and torque tracking
        self.torque = 0.0
        self.force = np.zeros(2)
        self.old_force = np.zeros(2)
        self.old_torque = 0.0

        # Visual representation (initialize as None)
        self.rod = None
        self.cap1 = None
        self.cap2 = None
        self.ax = None

        self.bounding_box_coords = None

    def grow(self, dt, tao, lamb):
        """Grow the particle based on stress and growth rate."""
        growth = (self.length/tao) * np.exp(-lamb * self.stress)
        self.length += growth * dt
        self.length = min(self.length, self.max_length)

    def move(self, dt):
        """Move the particle using velocity Verlet integration."""
        # Store current force as old force for next step
        self.old_force = self.force.copy()
        self.old_torque = self.torque

        # Update position and half-update velocity
        self.position += self.linear_velocity * dt + 0.5 * self.old_force * dt**2
        self.linear_velocity += 0.5 * self.old_force * dt

        # Update orientation and half-update angular velocity
        self.orientation += self.angular_velocity * dt + 0.5 * self.old_torque * dt**2
        self.angular_velocity += 0.5 * self.old_torque * dt

        # Normalize orientation to [0, 2*pi]
        self.orientation = self.orientation % (2 * np.pi)

        # Reset force and torque
        self.force = np.zeros(2)
        self.torque = 0.0

    def update_velocity(self, dt):
        """Second half of velocity Verlet - update velocities with new forces"""
        self.linear_velocity += 0.5 * self.force * dt
        self.angular_velocity += 0.5 * self.torque * dt

    def get_endpoints(self):
        """Get the coordinates of the two endpoints of the spherocylinder with caching."""
        half_length = self.length / 2
        orientation_vector = np.array(
            [np.cos(self.orientation), np.sin(self.orientation)])
        offset = (half_length - self.diameter / 2.0) * orientation_vector
        end1 = self.position + offset
        end2 = self.position - offset

        return end1, end2

    def initialize_visual_elements(self, ax):
        """Initialize the visual representation of the spherocylinder."""
        self.ax = ax

        # Create rod body and end caps
        self.rod = Rectangle((0, 0), 0, 0, fill=True, ec='black', zorder=3)
        self.cap1 = Circle((0, 0), self.diameter / 2,
                           fill=True, ec='black', zorder=2)
        self.cap2 = Circle((0, 0), self.diameter / 2,
                           fill=True, ec='black', zorder=2)

        # Add all elements to the axis
        ax.add_patch(self.rod)
        ax.add_patch(self.cap1)
        ax.add_patch(self.cap2)

    def update_visual_elements(self, ax):
        """Update the visual representation of the spherocylinder."""
        # Initialize visual elements if they don't exist
        if self.rod is None or self.cap1 is None or self.cap2 is None:
            self.initialize_visual_elements(ax)

        # Get endpoints
        end1, end2 = self.get_endpoints()

        # Update end caps positions
        self.cap1.set_center(end1)
        self.cap2.set_center(end2)

        # Update rod body
        rod_width = self.length - self.diameter
        self.rod.set_width(rod_width)
        self.rod.set_height(self.diameter)

        # Set color based on length
        color_index = int((self.length - self.l0) * 255 /
                          (self.max_length - self.l0))
        color_index = min(color_index, 255)
        color = self._length_cmap(color_index)

        # Set border color based on stress
        color_index_border = min(int(self.stress * 255), 255)
        border_color = self._stress_cmap(color_index_border)

        # Apply colors to all elements
        self.rod.set_facecolor(color)
        self.cap1.set_facecolor(color)
        self.cap2.set_facecolor(color)

        self.rod.set_edgecolor(border_color)
        self.cap1.set_edgecolor(border_color)
        self.cap2.set_edgecolor(border_color)

        t = transforms.Affine2D()

        t.translate(
            self.position[0] - rod_width / 2,
            self.position[1] - self.diameter / 2)
        t.rotate_around(
            self.position[0], self.position[1], self.orientation)
        self.rod.set_transform(t + ax.transData)

    def update_bounding_box(self):
        """Calculate the bounding box of the spherocylinder."""
        # end1, end2 = self.get_endpoints()

        min_x = self.position[0] - self.length / 2
        max_x = self.position[0] + self.length / 2
        min_y = self.position[1] - self.diameter / 2
        max_y = self.position[1] + self.diameter / 2

        self.bounding_box_coords = (min_x, max_x, min_y, max_y)

    def check_overlap(self, other):
        """Check if this spherocylinder overlaps with another one."""
        # Get line segments
        p1, p2 = self.get_endpoints()
        q1, q2 = other.get_endpoints()
        sum_radii = (self.diameter + other.diameter) / 2.0

        # Calculate minimum distance between line segments
        closest_p1, closest_p2, min_dist = self.minimum_distance_between_line_segments(
            p1, p2, q1, q2, clampA0=True, clampA1=True, clampB0=True, clampB1=True)

        # Check for overlap
        overlap = sum_radii - min_dist
        if overlap > 0:
            # Normal vector
            if closest_p1 is None or closest_p2 is None:
                # If closest points are None, use the center of the particles
                closest_p1 = self.position
                closest_p2 = other.position

            # Calculate normalized normal vector
            normal = closest_p1 - closest_p2
            normal_mag = np.linalg.norm(normal)
            if normal_mag > 0:
                normal = normal / normal_mag
            else:
                normal = np.array([1.0, 0.0])

            # Contact point is the midpoint between closest points
            contact_point = (closest_p1 + closest_p2) / 2
            return True, overlap, contact_point, normal

        return False, overlap, None, None

    def divide(self):
        """Divide particles"""
        if self.length >= self.max_length:
            # Calculate new positions for child particles
            orientation_vector = np.array(
                [np.cos(self.orientation), np.sin(self.orientation)])

            center_left = self.position - 1/4. * self.length * orientation_vector
            center_right = self.position + 1/4. * self.length * orientation_vector

            # Random orientation noise
            noise = np.random.uniform(-np.pi/32, np.pi/32)

            # self
            self.position = center_left
            self.orientation = self.orientation + noise
            self.length = self.l0

            # new
            new_particle_right = Spherocylinder(
                position=center_right,
                orientation=self.orientation - noise,
                linear_velocity=self.linear_velocity,
                angular_velocity=self.angular_velocity,
                l0=self.l0,
            )

            return new_particle_right
