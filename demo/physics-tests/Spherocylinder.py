import numpy as np
from matplotlib import cm, transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arrow, Circle, Rectangle


class Spherocylinder:
    """
    Class representing a single spherocylinder particle with growth dynamics, torque-based collision response,
    and background drag.
    """

    def __init__(self, position, linear_velocity, angular_velocity, orientation, l0):
        self.position = np.array(position, dtype=float)
        self.orientation = float(orientation)
        self.l0 = float(l0)
        self.length = float(l0)
        self.max_length = float(2 * l0)
        self.diameter = 0.5 * l0

        # Derived properties
        self.stress = 0.0

        # Physical properties for collision response
        self.angular_velocity = angular_velocity
        self.linear_velocity = linear_velocity
        self.mass = l0 * self.diameter  # Simple mass model
        # Simple moment of inertia for a rod
        self.moment_of_inertia = (l0**3 * self.diameter) / 12

        # Collision parameters
        self.restitution = 0.5  # Coefficient of restitution
        self.friction = 0.3     # Friction coefficient
        self.torque = 0.0       # Current torque applied to particle

        # Visual representation
        self.rod = None
        self.cap1 = None
        self.cap2 = None
        self.text = None
        self.ax = None
        self.to_delete = False

    def grow(self, dt, tao, lamb):
        """Grow the particle based on stress and growth rate."""
        growth = (self.length/tao) * np.exp(
            -lamb * self.stress)

        self.length += growth * dt
        self.length = min(self.length, self.max_length)

        # Update mass and moment of inertia as the particle grows
        self.mass = self.length * self.diameter
        self.moment_of_inertia = (self.length**3 * self.diameter) / 12

    def move(self, dt, box_size):
        """
        Move the particle with background drag effects.

        Parameters:
        -----------
        base_velocity : float
            Base self-propulsion speed
        dt : float
            Time step
        box_size : float
            Size of the simulation box for periodic boundaries
        linear_drag_coeff : float
            Coefficient for linear drag (resistance to linear motion)
        angular_drag_coeff : float
            Coefficient for angular drag (resistance to rotation)
        """

        # Combine self-propulsion with collision-induced velocity
        total_velocity = self.linear_velocity

        # Apply motion
        self.position += dt * total_velocity

        # Apply torque to update orientation
        self.apply_torque(dt)

        # Apply periodic boundary conditions
        self.position[0] = self.position[0] % box_size
        self.position[1] = self.position[1] % box_size

    def apply_torque(self, dt):
        """Apply torque to update angular velocity and orientation."""
        # Update angular velocity based on torque
        angular_acceleration = self.torque / self.moment_of_inertia
        self.angular_velocity += angular_acceleration * dt

        # Update orientation based on angular velocity
        self.orientation += self.angular_velocity * dt
        self.orientation = self.orientation % (2 * np.pi)

        # Reset torque
        self.torque = 0.0

    def get_endpoints(self):
        """Get the coordinates of the two endpoints of the spherocylinder."""
        half_length = self.length / 2
        orientation_vector = np.array(
            [np.cos(self.orientation), np.sin(self.orientation)])
        end1 = self.position + \
            (half_length - self.diameter / 2.0) * orientation_vector
        end2 = self.position - \
            (half_length - self.diameter / 2.0) * orientation_vector
        return end1, end2

    def initialize_visual_elements(self, ax):
        """Initialize the visual representation of the spherocylinder."""
        # Create rod body
        self.ax = ax

        self.rod = Rectangle((0, 0), 0, 0, fill=True, ec='black', zorder=3)
        ax.add_patch(self.rod)

        # Create end caps
        self.cap1 = Circle((0, 0), self.diameter / 2, fill=True,
                           ec='black', zorder=2)
        self.cap2 = Circle((0, 0), self.diameter / 2, fill=True,
                           ec='black', zorder=2)
        ax.add_patch(self.cap1)
        ax.add_patch(self.cap2)

        # add text with current length
        self.text = ax.text(0, 0, '', fontsize=8,
                            ha='center', va='center', color='black',
                            zorder=3)

    def update_visual_elements(self, ax, alpha=1.0, ghost=False):
        # Initialize visual elements if they don't exist
        if not hasattr(self, 'rod') or not all([self.rod, self.cap1, self.cap2, self.text]):
            self.initialize_visual_elements(ax)

        # Update end caps positions
        end1, end2 = self.get_endpoints()
        self.cap1.set_center(end1)
        self.cap2.set_center(end2)

        # Update text position
        if hasattr(self, 'text'):
            self.text.set_position(
                (self.position[0], self.position[1]))

            # Show velocity and other information
            lin_vel = np.linalg.norm(self.linear_velocity)
            if ghost:
                # If this is a ghost particle, use a different label
                self.text.set_text('Ghost')
                self.text.set_alpha(alpha)
            else:
                self.text.set_text(
                    f'L:{self.length:.1f}\nS:{self.stress:.1f}')

        # Update rod body
        self.rod.set_width(self.length - self.diameter)
        self.rod.set_height(self.diameter)

        # Set color based on length
        # length = black and l = white transition via green. Smooth transition
        colors = [(0.1, 0.3, 0), (0.6, 1, 0), (0.5, 1, 0.2)]
        cmap = LinearSegmentedColormap.from_list(
            'mycmap', colors, N=256)
        color_index = int((self.length / self.max_length) * 255)
        color_index = min(color_index, 255)
        color = cmap(color_index)

        cmap_border = cm.get_cmap('Reds', 256)
        color_index_border = int((self.stress) * 255)
        color_index_border = min(color_index_border, 255)

        # For ghost particles, use a different color or adjust transparency
        if ghost:
            # Option 1: Use different color for ghosts (light blue)
            color = (0.5, 0.5, 1.0, alpha)
            # Option 2: Just adjust transparency of normal color
            # color = (*color[:3], alpha)

        self.rod.set_facecolor(color)
        self.cap1.set_facecolor(color)
        self.cap2.set_facecolor(color)

        self.rod.set_edgecolor(cmap_border(color_index_border))
        self.cap1.set_edgecolor(cmap_border(color_index_border))
        self.cap2.set_edgecolor(cmap_border(color_index_border))

        # Set alpha for all elements
        self.rod.set_alpha(alpha)
        self.cap1.set_alpha(alpha)
        self.cap2.set_alpha(alpha)

        # Update rod position and rotation
        t = transforms.Affine2D().translate(
            self.position[0] - (self.length - self.diameter) / 2,
            self.position[1] - self.diameter / 2)
        t.rotate_around(
            self.position[0], self.position[1], self.orientation)
        self.rod.set_transform(t + ax.transData)

    def delete_visual_elements(self):
        """Delete the visual elements of the spherocylinder."""
        if self.rod is not None:
            self.rod.remove()
        if self.cap1 is not None:
            self.cap1.remove()
        if self.cap2 is not None:
            self.cap2.remove()
        if self.text is not None:
            self.text.remove()

    def check_overlap(self, other):
        """
        Check if two spherocylinders overlap.

        Parameters:
        other (Spherocylinder): Another spherocylinder to check against

        Returns:
        bool: True if the spherocylinders overlap, False otherwise
        distance: Minimum distance between line segments (negative if overlapping)
        contact_point: Point of contact between the two spherocylinders
        normal: Normal vector at the point of contact
        """
        # Get line segments
        p1, p2 = self.get_endpoints()
        q1, q2 = other.get_endpoints()
        diameter1 = self.diameter
        diameter2 = other.diameter

        # Calculate minimum distance between the two line segments
        closest_p1, closest_p2, min_dist = self.minimum_distance_between_line_segments(
            p1, p2, q1, q2, clampA0=True, clampA1=True, clampB0=True, clampB1=True)

        if closest_p1 is not None:
            closest_p1 = closest_p1[:2]
        if closest_p2 is not None:
            closest_p2 = closest_p2[:2]

        # Calculate sum of radii
        sum_radii = (diameter1 + diameter2) / 2.0
        overlap = sum_radii - min_dist

        # If they overlap, return contact information
        if overlap > 0:
            # Normal vector pointing from other to self at contact point
            normal = closest_p1 - closest_p2
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            else:
                # If points are exactly the same, use an arbitrary normal
                normal = np.array([1.0, 0.0])

            # Contact point is halfway between the closest points
            contact_point = (closest_p1 + closest_p2) / 2

            return True, overlap, contact_point, normal

        return False, overlap, None, None

    def minimum_distance_between_line_segments(self, a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False, clampB1=False):
        ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
            Return the closest points on each segment and their distance
        '''
        # Convert to 3d
        a0 = np.array([a0[0], a0[1], 0])
        a1 = np.array([a1[0], a1[1], 0])
        b0 = np.array([b0[0], b0[1], 0])
        b1 = np.array([b1[0], b1[1], 0])

        # If clampAll=True, set all clamps to True
        if clampAll:
            clampA0 = True
            clampA1 = True
            clampB0 = True
            clampB1 = True

        # Calculate denomitator
        A = a1 - a0
        B = b1 - b0
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)

        _A = A / magA
        _B = B / magB

        cross = np.cross(_A, _B)
        denom = np.linalg.norm(cross)**2

        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if not denom:
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
        t = (b0 - a0)
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])

        t0 = detA/denom
        t1 = detB/denom

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
