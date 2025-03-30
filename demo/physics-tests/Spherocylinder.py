import numpy as np
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from scipy.spatial.transform import Rotation


class Spherocylinder:
    """
    Class representing a single 3D spherocylinder particle with growth dynamics, torque-based collision response,
    and background drag. Uses quaternions for orientation in 3D space.
    """

    def __init__(self, position, linear_velocity, angular_velocity, orientation, l0):
        # Now a 3D vector [x, y, z]
        self.position = np.array(position, dtype=float)

        # Already a quaternion [w, x, y, z]
        self.orientation = np.array(orientation, dtype=float)
        # Normalize the quaternion
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

        self.l0 = float(l0)
        self.length = float(l0)
        self.max_length = float(2 * l0)
        self.diameter = 0.5 * l0

        # Derived properties
        self.stress = 0.0

        # Physical properties for collision response
        # Angular velocity is now a 3D vector [wx, wy, wz]
        self.angular_velocity = np.array(angular_velocity, dtype=float)
        # Linear velocity is now a 3D vector [vx, vy, vz]
        self.linear_velocity = np.array(linear_velocity, dtype=float)

        # Mass model for 3D
        # Simple cylinder mass model
        self.mass = l0 * np.pi * (self.diameter/2)**2

        # Moment of inertia for a rod in 3D (simplified)
        # For a cylinder along the z-axis
        radius = self.diameter / 2
        # Ixx = Iyy for a cylinder
        Ixx = Iyy = self.mass * (3*radius**2 + l0**2) / 12
        # Izz for a cylinder (rotation around its axis)
        Izz = self.mass * radius**2 / 2
        # Full inertia tensor
        self.inertia_tensor = np.diag([Ixx, Iyy, Izz])

        # Collision parameters
        self.restitution = 0.5  # Coefficient of restitution
        self.friction = 0.3     # Friction coefficient
        # Current torque applied to particle (3D vector)
        self.torque = np.zeros(3)

        # Visual representation
        self.rod = None
        self.cap1 = None
        self.cap2 = None
        self.text = None
        ax = None
        self.to_delete = False
        self.debug = []

    def grow(self, dt, tao, lamb):
        """Grow the particle based on stress and growth rate."""
        growth = (self.length/tao) * np.exp(-lamb * self.stress)

        self.length += growth * dt
        self.length = min(self.length, self.max_length)

        # Update mass and moment of inertia as the particle grows
        radius = self.diameter / 2
        self.mass = self.length * np.pi * radius**2

        # Update inertia tensor
        Ixx = Iyy = self.mass * (3*radius**2 + self.length**2) / 12
        Izz = self.mass * radius**2 / 2
        self.inertia_tensor = np.diag([Ixx, Iyy, Izz])

    def move(self, dt, box_size):
        """
        Move the particle with background drag effects in 3D.

        Parameters:
        -----------
        dt : float
            Time step
        box_size : list or tuple or numpy array
            Size of the simulation box for periodic boundaries [x_size, y_size, z_size]
        """
        # Apply motion
        self.position += dt * self.linear_velocity

        # Apply torque to update orientation
        self.apply_torque(dt)

        # Apply periodic boundary conditions
        if isinstance(box_size, (int, float)):
            # If box_size is a single value, assume cubic box
            box_size = [box_size, box_size, box_size]

        self.position[0] = self.position[0] % box_size[0]
        self.position[1] = self.position[1] % box_size[1]
        if len(self.position) > 2 and len(box_size) > 2:
            self.position[2] = self.position[2] % box_size[2]

    def apply_torque(self, dt):
        """Apply torque to update angular velocity and orientation using quaternions."""
        # Convert torque to body frame using current orientation
        quat_as_rot = Rotation.from_quat([self.orientation[1], self.orientation[2],
                                          self.orientation[3], self.orientation[0]])  # Convert to scipy format

        # Update angular velocity based on torque in body frame
        # Solve I·α = τ for α (angular acceleration)
        angular_acceleration = np.linalg.solve(
            self.inertia_tensor, self.torque)
        self.angular_velocity += angular_acceleration * dt

        # Update orientation quaternion using angular velocity
        # For small rotations: q_new = q_old + 0.5 * dt * q_old * [0, wx, wy, wz]
        # Create quaternion from angular velocity
        angle = np.linalg.norm(self.angular_velocity) * dt
        if angle > 0:
            axis = self.angular_velocity / \
                np.linalg.norm(self.angular_velocity)
            # Create rotation quaternion for this step
            rotation = Rotation.from_rotvec(angle * axis)
            quat_rotation = rotation.as_quat()  # [x, y, z, w] format
            quat_rotation = np.array([quat_rotation[3], quat_rotation[0],
                                      quat_rotation[1], quat_rotation[2]])  # Convert to [w, x, y, z]

            # Quaternion multiplication for new orientation
            new_orientation = self.quaternion_multiply(
                self.orientation, quat_rotation)
            self.orientation = new_orientation / \
                np.linalg.norm(new_orientation)  # Normalize

        # Reset torque
        self.torque = np.zeros(3)

    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions q1 and q2 (format: [w, x, y, z])
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def get_direction_vector(self):
        """Get the direction vector (unit vector along the cylinder axis) from quaternion."""
        # Default direction is along Z-axis [0, 0, 1]
        default_direction = np.array([1, 0, 0])

        # Convert orientation quaternion to rotation matrix
        quat_as_rot = Rotation.from_quat([self.orientation[1], self.orientation[2],
                                          self.orientation[3], self.orientation[0]])

        # Rotate the default direction
        return quat_as_rot.apply(default_direction)

    def get_endpoints(self):
        """Get the coordinates of the two endpoints of the 3D spherocylinder."""
        half_length = self.length / 2
        direction_vector = self.get_direction_vector()

        end1 = self.position + \
            (half_length - self.diameter / 2.0) * direction_vector
        end2 = self.position - \
            (half_length - self.diameter / 2.0) * direction_vector

        return end1, end2

    def initialize_visual_elements(self, ax):
        """Initialize the visual representation of the spherocylinder.

        Note: This method needs to be updated for 3D visualization with matplotlib's 3D plotting.
        """
        ax = ax

        # Fallback to 2D visualization (projection of 3D object onto XY plane)
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

    def draw(self, ax, n_phi=20, n_theta=10, n_height=10):
        """Update the visual representation of the 3D spherocylinder.

        This method needs significant changes for proper 3D visualization.
        """
        # Get endpoints and direction vector
        end1, end2 = self.get_endpoints()
        direction = self.get_direction_vector()

        # Calculate color based on length
        length_ratio = self.length / self.max_length
        # Green color with varying intensity
        color = (0, 0.5 + 0.5 * length_ratio, 0)

        # If ghost, use different color and transparency
        alpha = 1.0
        if hasattr(self, 'is_ghost') and self.is_ghost:
            color = (0.5, 0.5, 1.0)  # Light blue for ghosts
            alpha = 0.3

        # Particle properties
        radius = self.diameter / 2
        cylinder_length = np.linalg.norm(end2 - end1)

        # Create orthogonal vectors to the axis
        axis = direction
        if np.allclose(axis, [0, 0, 1]) or np.allclose(axis, [0, 0, -1]):
            ortho1 = np.array([1, 0, 0])
        else:
            ortho1 = np.cross(axis, [0, 0, 1])
            ortho1 = ortho1 / np.linalg.norm(ortho1)

        ortho2 = np.cross(axis, ortho1)
        ortho2 = ortho2 / np.linalg.norm(ortho2)

        # Generate points for the entire spherocylinder surface

        # 1. Generate cylindrical body
        phi = np.linspace(0, 2*np.pi, n_phi)
        height = np.linspace(0, cylinder_length, n_height)

        phi_grid, height_grid = np.meshgrid(phi, height)

        # Initialize cylindrical surface arrays
        x_cyl = np.zeros_like(phi_grid)
        y_cyl = np.zeros_like(phi_grid)
        z_cyl = np.zeros_like(phi_grid)

        # Calculate coordinates of cylindrical surface
        for i in range(len(height)):
            for j in range(len(phi)):
                # Point on the unit circle
                circle_point = radius * \
                    (np.cos(phi[j]) * ortho1 + np.sin(phi[j]) * ortho2)
                # Point on the cylinder with correct position
                pointL = end1 + height[i] * direction + circle_point
                x_cyl[i, j] = pointL[0]
                y_cyl[i, j] = pointL[1]
                z_cyl[i, j] = pointL[2]

        # 2. Generate hemispherical caps
        theta = np.linspace(0, np.pi/2, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

        # Initialize hemisphere arrays (one for each end)
        x_cap1 = np.zeros_like(theta_grid)
        y_cap1 = np.zeros_like(theta_grid)
        z_cap1 = np.zeros_like(theta_grid)

        x_cap2 = np.zeros_like(theta_grid)
        y_cap2 = np.zeros_like(theta_grid)
        z_cap2 = np.zeros_like(theta_grid)

            # Calculate coordinates for cap at end1
        for i in range(len(theta)):
            for j in range(len(phi)):
                # Point on the unit hemisphere
                cap_point = radius * (np.sin(theta[i]) * np.cos(phi[j]) * ortho1 +
                                      np.sin(theta[i]) * np.sin(phi[j]) * ortho2 -
                                      np.cos(theta[i]) * direction)
                pointL = end1 + cap_point
                x_cap1[i, j] = pointL[0]
                y_cap1[i, j] = pointL[1]
                z_cap1[i, j] = pointL[2]

                pointR = end1 + cylinder_length * direction + \
                    np.array([-1, -1, 1]) * cap_point
                x_cap2[i, j] = pointR[0]
                y_cap2[i, j] = pointR[1]
                z_cap2[i, j] = pointR[2]

        # Plot the surfaces
        # Cylindrical body
        ax.plot_surface(x_cyl, y_cyl, z_cyl,
                        color=color, alpha=alpha, shade=True)

        # Hemispherical caps
        ax.plot_surface(x_cap1, y_cap1, z_cap1,
                        color=color, alpha=alpha, shade=True)
        ax.plot_surface(x_cap2, y_cap2, z_cap2,
                        color=color, alpha=alpha, shade=True)

        # Optionally add velocity vector visualization
        if hasattr(self, 'linear_velocity') and np.linalg.norm(self.linear_velocity) > 0.1:
            vel_magnitude = np.linalg.norm(self.linear_velocity)
            vel_direction = self.linear_velocity / vel_magnitude

            # Scale arrow length with velocity
            arrow_length = min(vel_magnitude, 2.0)

            # Draw arrow for velocity
            ax.quiver(
                self.position[0], self.position[1], self.position[2],
                vel_direction[0], vel_direction[1], vel_direction[2],
                length=arrow_length, color='red', arrow_length_ratio=0.2, alpha=0.5
            )

    def check_overlap(self, other):
        """
        Check if two 3D spherocylinders overlap.

        Parameters:
        other (Spherocylinder): Another spherocylinder to check against

        Returns:
        bool: True if the spherocylinders overlap, False otherwise
        distance: Minimum distance between line segments (negative if overlapping)
        contact_point: Point of contact between the two spherocylinders
        normal: Normal vector at the point of contact
        """
        # Get line segments in 3D
        p1, p2 = self.get_endpoints()
        q1, q2 = other.get_endpoints()
        diameter1 = self.diameter
        diameter2 = other.diameter

        # Calculate minimum distance between the two line segments in 3D
        closest_p1, closest_p2, min_dist = self.minimum_distance_between_line_segments(
            p1, p2, q1, q2, clampA0=True, clampA1=True, clampB0=True, clampB1=True)

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
                normal = np.array([1.0, 0.0, 0.0])

            # Contact point is halfway between the closest points
            contact_point = (closest_p1 + closest_p2) / 2

            # Add debug visualization
            if len(self.position) <= 2 or len(contact_point) <= 2:
                # Only add debug Circle for 2D visualization
                contact_point_2d = contact_point[:2] if len(
                    contact_point) > 2 else contact_point
                c1 = Circle(contact_point_2d, 0.1, color='red', zorder=100)
                self.debug.append(c1)

            return True, overlap, contact_point, normal

        return False, 0, None, None

    def minimum_distance_between_line_segments(self, a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False, clampB1=False):
        ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
            Return the closest points on each segment and their distance
            Works for 3D vectors
        '''

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

        _A = A / magA if magA > 0 else A
        _B = B / magB if magB > 0 else B

        cross = np.cross(_A, _B)
        denom = np.linalg.norm(cross)**2

        # If lines are parallel (denom=0) test if lines overlap.
        # If they don't overlap then there is a closest point solution.
        # If they do overlap, there are infinite closest positions, but there is a closest distance
        if denom < 1e-10:  # Use small epsilon for floating-point comparison
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

        # For 3D vectors, use a different approach to find t0 and t1
        # Create matrices for solving the system
        c1 = np.dot(_A, _A)
        c2 = np.dot(_A, _B)
        c3 = np.dot(_B, _B)
        c4 = np.dot(_A, t)
        c5 = np.dot(_B, t)

        # Solve the system of equations
        den = c1*c3 - c2*c2
        if abs(den) < 1e-10:
            # Lines are nearly parallel
            # Use a default value
            t0 = 0
            t1 = np.dot(_B, (a0-b0)) / c3 if c3 > 1e-10 else 0
        else:
            t0 = (c2*c5 - c3*c4) / den
            t1 = (c1*c5 - c2*c4) / den

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
