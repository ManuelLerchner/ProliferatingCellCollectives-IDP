import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle


class Spherocylinder:
    """
    Class representing a single spherocylinder particle with growth dynamics, torque-based collision response,
    and background drag.
    """

    def __init__(self, position, orientation, length, diameter, max_length=5.0,
                 color='white', growth_rate=0.01):
        """
        Initialize a spherocylinder particle.

        Parameters:
        -----------
        position : numpy.ndarray
            2D position vector of the particle center
        orientation : float
            Orientation angle in radians
        length : float
            Initial length of the spherocylinder
        diameter : float
            Diameter of the spherocylinder (constant)
        max_length : float
            Maximum length the particle can grow to
        color : str
            Color of the particle ('green' or 'white')
        growth_rate : float
            Base rate at which the particle grows
        """
        self.position = np.array(position, dtype=float)
        self.orientation = float(orientation)
        self.initial_length = float(length)
        self.length = float(length)
        self.diameter = float(diameter)
        self.radius = diameter / 2
        self.max_length = float(max_length)
        self.color = color
        self.base_growth_rate = growth_rate

        # Derived properties
        self.orientation_vector = np.array(
            [np.cos(orientation), np.sin(orientation)])
        self.stress = 0.0

        # Physical properties for collision response
        self.angular_velocity = 0.0
        self.linear_velocity = np.zeros(2)
        self.mass = length * diameter  # Simple mass model
        # Simple moment of inertia for a rod
        self.moment_of_inertia = (length**3 * diameter) / 12

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

        self.debug = []

    def update_orientation_vector(self):
        """Update the orientation vector based on the current orientation angle."""
        self.orientation_vector = np.array(
            [np.cos(self.orientation), np.sin(self.orientation)])

    def grow(self, dt):
        """Grow the particle based on stress and growth rate."""
        growth = self.base_growth_rate * (1 - self.stress) * dt
        self.length = min(self.length + growth, self.max_length)

        # Update mass and moment of inertia as the particle grows
        self.mass = self.length * self.diameter
        self.moment_of_inertia = (self.length**3 * self.diameter) / 12

    def move(self, base_velocity, dt, box_size, linear_drag_coeff, angular_drag_coeff):
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
        # Calculate drag forces
        self.apply_drag(linear_drag_coeff, angular_drag_coeff, dt)

        # Combine self-propulsion with collision-induced velocity
        total_velocity = base_velocity * self.orientation_vector + self.linear_velocity

        # Apply motion
        self.position += dt * total_velocity

        # Apply torque to update orientation
        self.apply_torque(dt)

        # Apply periodic boundary conditions
        self.position[0] = self.position[0] % box_size
        self.position[1] = self.position[1] % box_size

    def apply_drag(self, linear_drag_coeff, angular_drag_coeff, dt):
        """
        Apply background drag forces to the particle.

        Parameters:
        -----------
        linear_drag_coeff : float
            Coefficient for linear drag (resistance to linear motion)
        angular_drag_coeff : float
            Coefficient for angular drag (resistance to rotation)
        dt : float
            Time step
        """
        # Linear drag (proportional to velocity and frontal area)
        # Calculate frontal area (approximation based on orientation and size)
        frontal_length = abs(
            np.dot(self.orientation_vector, self.linear_velocity))
        frontal_area = self.diameter * \
            (self.length - frontal_length) + np.pi * (self.diameter/2)**2
        drag_scale = linear_drag_coeff * frontal_area

        # Apply linear drag force (F = -kv)
        if np.linalg.norm(self.linear_velocity) > 0:
            drag_force = -drag_scale * self.linear_velocity
            self.linear_velocity += drag_force * dt / self.mass

        # Angular drag (proportional to angular velocity)
        # Apply angular drag torque (τ = -kω)
        angular_drag_torque = -angular_drag_coeff * self.angular_velocity
        self.torque += angular_drag_torque

    def apply_torque(self, dt):
        """Apply torque to update angular velocity and orientation."""
        # Update angular velocity based on torque
        angular_acceleration = self.torque / self.moment_of_inertia
        self.angular_velocity += angular_acceleration * dt

        # Update orientation based on angular velocity
        self.orientation += self.angular_velocity * dt
        self.orientation = self.orientation % (2 * np.pi)
        self.update_orientation_vector()

        # Reset torque
        self.torque = 0.0

    def rotate(self, noise_strength):
        """Add random rotation to the particle."""
        self.orientation += np.random.normal(0, noise_strength)
        self.orientation = self.orientation % (2 * np.pi)
        self.update_orientation_vector()

    def get_endpoints(self):
        """Get the coordinates of the two endpoints of the spherocylinder."""
        half_length = self.length / 2
        end1 = self.position - self.orientation_vector * half_length
        end2 = self.position + self.orientation_vector * half_length
        return end1, end2

    def initialize_visual_elements(self, ax):
        """Initialize the visual representation of the spherocylinder."""
        # Create rod body
        self.ax = ax

        self.rod = Rectangle((0, 0), 0, 0, fill=True, color=self.color,
                             ec='black', zorder=1)
        ax.add_patch(self.rod)

        # Create end caps
        self.cap1 = Circle((0, 0), self.radius, fill=True, color=self.color,
                           ec='black', zorder=2)
        self.cap2 = Circle((0, 0), self.radius, fill=True, color=self.color,
                           ec='black', zorder=2)
        ax.add_patch(self.cap1)
        ax.add_patch(self.cap2)

        # add text with current length
        self.text = ax.text(0, 0, '', fontsize=8,
                            ha='center', va='center', color='black',
                            zorder=3)

    def update_visual_elements(self, ax, show_velocity=True):
        """Update the visual representation to match the current state."""
        if not all([self.rod, self.cap1, self.cap2, self.text]):
            self.initialize_visual_elements(ax)

        # Get endpoints
        end1, end2 = self.get_endpoints()

        # Update end cap positions
        self.cap1.center = end1
        self.cap2.center = end2

        # Update text position
        self.text.set_position(
            (self.position[0], self.position[1] + self.diameter))

        # Show velocity information if requested
        if show_velocity:
            lin_vel = np.linalg.norm(self.linear_velocity)
            self.text.set_text(
                f'L:{self.length:.1f} S:{self.stress:.1f} V:{lin_vel:.1f} ω:{self.angular_velocity:.1f}')
        else:
            self.text.set_text(f'L:{self.length:.1f} S:{self.stress:.1f}')

        # Update rod body
        self.rod.set_width(self.length)
        self.rod.set_height(self.diameter)

        # set color based on length l0 = black and l = white transition via green. Smooth transition
        colors = [(0, 0.2, 0), (0, 1, 0), (0.8, 1, 0.8)]
        cmap = LinearSegmentedColormap.from_list(
            'mycmap', colors, N=256)
        color_index = int((self.length / self.max_length) * 255)
        color_index = min(color_index, 255)
        color = cmap(color_index)

        self.rod.set_facecolor(color)
        self.cap1.set_facecolor(color)
        self.cap2.set_facecolor(color)
        self.rod.set_edgecolor(color)
        self.cap1.set_edgecolor(color)
        self.cap2.set_edgecolor(color)

        # Move rectangle to correct position and orientation
        t = transforms.Affine2D().translate(-self.length/2, -self.diameter/2)
        t += transforms.Affine2D().rotate(self.orientation)
        t += transforms.Affine2D().translate(
            self.position[0], self.position[1])
        self.rod.set_transform(t + self.rod.axes.transData)

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

        # Calculate direction vectors
        v1 = p2 - p1
        v2 = q2 - q1

        # Calculate minimum distance between the two line segments
        min_dist, closest_p1, closest_p2 = self.minimum_distance_between_lines(
            p1, v1, q1, v2)

        # Calculate sum of radii
        sum_radii = (diameter1 + diameter2) / 2

        # Calculate overlap
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

            # Add debug visualization
            c1 = Circle(closest_p1, 0.2, color='orange', zorder=100)
            c2 = Circle(closest_p2, 0.2, color='orange', zorder=100)
            self.debug.append(c1)
            self.debug.append(c2)

            return True, overlap, contact_point, normal

        return False, 0, None, None

    def minimum_distance_between_lines(self, p1, v1, p2, v2):
        """
        Calculate the minimum distance between two line segments.

        Parameters:
        p1: Starting point of first line segment
        v1: Direction vector of first line segment
        p2: Starting point of second line segment
        v2: Direction vector of second line segment

        Returns:
        float: Minimum distance between the line segments
        numpy.ndarray: Closest point on first line segment
        numpy.ndarray: Closest point on second line segment
        """
        # Calculate parameters s and t for closest points
        w0 = p1 - p2
        a = np.dot(v1, v1)
        b = np.dot(v1, v2)
        c = np.dot(v2, v2)
        d = np.dot(v1, w0)
        e = np.dot(v2, w0)

        # Parameters for closest points
        denom = a*c - b*b
        if abs(denom) < 1e-10:
            # Lines are parallel, use endpoints for distance
            min_dist = np.inf
            closest_p1 = p1
            closest_p2 = p2
            for x in [p1, p1 + v1]:
                for y in [p2, p2 + v2]:
                    dist = np.linalg.norm(x - y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_p1 = x
                        closest_p2 = y
            return min_dist, closest_p1, closest_p2

        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom

        # Clamp parameters to line segments
        s = np.clip(s, 0, 1)
        t = np.clip(t, 0, 1)

        # Recalculate closest points with clamped parameters
        closest_p1 = p1 + s * v1
        closest_p2 = p2 + t * v2

        # Calculate distance between closest points
        return np.linalg.norm(closest_p1 - closest_p2), closest_p1, closest_p2


class SpherocylinderSimulation:
    """
    Simulation environment for a collection of spherocylinder particles 
    with torque-based collision response and background drag.
    """

    def __init__(self, num_particles=50, box_size=20,
                 particle_diameter=0.5, initial_length=2.0,
                 noise_strength=0.05, growth_rate=0.01,
                 max_length=5.0, velocity=0.2,
                 linear_drag=0.5, angular_drag=0.2):
        """
        Initialize the simulation of growing spherocylinder particles.

        Parameters:
        -----------
        num_particles : int
            Number of particles in the simulation
        box_size : float
            Size of the square box
        particle_diameter : float
            Diameter (b) of each spherocylinder
        initial_length : float
            Initial length of particles
        noise_strength : float
            Strength of the noise in the angular direction
        growth_rate : float
            Base rate at which particles grow in length
        max_length : float
            Maximum length a particle can grow to
        velocity : float
            Speed of the self-propelled particles
        linear_drag : float
            Coefficient for background linear drag
        angular_drag : float
            Coefficient for background angular drag
        """
        self.num_particles = num_particles
        self.box_size = box_size
        self.particle_diameter = particle_diameter
        self.noise_strength = noise_strength
        self.growth_rate = growth_rate
        self.max_length = max_length
        self.velocity = velocity

        # Drag coefficients
        self.linear_drag = linear_drag
        self.angular_drag = angular_drag

        # Collision parameters
        self.collision_stiffness = 2.0  # Spring constant for collision response
        self.damping = 0.5  # Damping coefficient

        # Create particles
        self.particles = []
        for i in range(num_particles):
            # Initialize position randomly in the box
            position = np.random.uniform(1, box_size-1, size=2)

            # Initialize orientation randomly
            orientation = np.random.uniform(0, 2*np.pi)

            # Create particle
            particle = Spherocylinder(
                position=position,
                orientation=orientation,
                length=initial_length,
                diameter=particle_diameter,
                max_length=max_length,
                growth_rate=growth_rate
            )

            self.particles.append(particle)

        # Initialize figure for animation
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, box_size)
        self.ax.set_ylim(0, box_size)
        self.ax.set_aspect('equal')
        self.ax.set_title('Growing Spherocylinder Simulation with Drag')

        # Add drag coefficient indicator
        self.drag_text = self.ax.text(
            0.02, 0.98, f'Linear Drag: {self.linear_drag}, Angular Drag: {self.angular_drag}',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

    def calculate_stresses(self):
        """Calculate compressive stress on each particle due to contacts."""
        # Reset all stresses
        for particle in self.particles:
            particle.stress = 0.0

        # Check all particle pairs for overlap
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles[i+1:], i+1):
                overlapping, _, _, _ = particle_i.check_overlap(particle_j)
                if overlapping:
                    # Simple model: each overlap adds to stress
                    particle_i.stress += 1
                    particle_j.stress += 1

        # Normalize stresses
        max_stress = 1.0
        for particle in self.particles:
            if particle.stress > max_stress:
                max_stress = particle.stress

        if max_stress > 0:
            for particle in self.particles:
                particle.stress /= max_stress

    def handle_collisions(self):
        """Apply torque-based collision response between particles."""
        # Check all particle pairs for collision
        for i, particle_i in enumerate(self.particles):
            for j, particle_j in enumerate(self.particles[i+1:], i+1):
                overlapping, overlap, contact_point, normal = particle_i.check_overlap(
                    particle_j)

                if overlapping:
                    # Calculate collision forces and torques
                    self.apply_collision_response(
                        particle_i, particle_j, overlap, contact_point, normal)

    def apply_collision_response(self, particle1, particle2, overlap, contact_point, normal):
        """
        Apply collision response between two particles with torque effects.

        Parameters:
        -----------
        particle1, particle2 : Spherocylinder
            The two colliding particles
        overlap : float
            Magnitude of overlap between particles
        contact_point : numpy.ndarray
            Point of contact between particles
        normal : numpy.ndarray
            Normal vector at contact point (pointing from particle2 to particle1)
        """
        # Calculate repulsive force magnitude based on overlap
        force_magnitude = self.collision_stiffness * overlap

        # Calculate force vector (from particle2 to particle1)
        force = force_magnitude * normal

        # Apply opposite forces to each particle
        linear_impulse = force / particle1.mass
        particle1.linear_velocity += linear_impulse

        linear_impulse = -force / particle2.mass
        particle2.linear_velocity += linear_impulse

        # Calculate torque for each particle
        # Torque = r × F where r is the vector from center of mass to contact point
        r1 = contact_point - particle1.position
        torque1 = np.cross(r1, force)  # Scalar in 2D

        r2 = contact_point - particle2.position
        torque2 = np.cross(r2, -force)  # Scalar in 2D

        # Apply torques
        particle1.torque += torque1
        particle2.torque += torque2

    def divide_particles(self):
        """Divide particles if they exceed a certain length."""
        for i in range(len(self.particles)):
            particle = self.particles[i]
            if particle.length >= particle.max_length:
                center_left = particle.position - \
                    particle.orientation_vector * (particle.length / 4)
                center_right = particle.position + \
                    particle.orientation_vector * (particle.length / 4)
                new_length = particle.length / 2
                new_particle_left = Spherocylinder(
                    position=center_left,
                    orientation=particle.orientation,
                    length=new_length,
                    diameter=particle.diameter,
                    max_length=particle.max_length,
                    growth_rate=particle.base_growth_rate
                )
                new_particle_right = Spherocylinder(
                    position=center_right,
                    orientation=particle.orientation,
                    length=new_length,
                    diameter=particle.diameter,
                    max_length=particle.max_length,
                    growth_rate=particle.base_growth_rate
                )
                # Transfer some momentum to child particles
                new_particle_left.angular_velocity = particle.angular_velocity
                new_particle_right.angular_velocity = particle.angular_velocity
                new_particle_left.linear_velocity = particle.linear_velocity
                new_particle_right.linear_velocity = particle.linear_velocity

                self.particles.append(new_particle_left)
                self.particles.append(new_particle_right)
                particle.to_delete = True  # Mark for deletion
                particle.length = 0  # Set length to zero to avoid further growth
        # Remove marked particles
        self.particles = [p for p in self.particles if not p.to_delete]

    def adjust_drag(self, linear_change=0, angular_change=0):
        """
        Adjust the drag coefficients.

        Parameters:
        -----------
        linear_change : float
            Amount to change linear drag coefficient
        angular_change : float
            Amount to change angular drag coefficient
        """
        self.linear_drag = max(0, self.linear_drag + linear_change)
        self.angular_drag = max(0, self.angular_drag + angular_change)

        # Update the drag text
        self.drag_text.set_text(
            f'Linear Drag: {self.linear_drag:.2f}, Angular Drag: {self.angular_drag:.2f}')

    def update(self, dt=0.1):
        """Update the simulation for one time step."""
        # Handle collisions and apply torques
        self.handle_collisions()

        # Calculate stresses
        self.calculate_stresses()

        # Divide particles if necessary
        self.divide_particles()

        # Update all particles
        for particle in self.particles:
            # Grow based on stress
            particle.grow(dt)

            # Add noise to orientation
            particle.rotate(self.noise_strength)

            # Move particle with collision effects and drag
            particle.move(self.velocity, dt, self.box_size,
                          self.linear_drag, self.angular_drag)

            # Update visual representation
            particle.update_visual_elements(self.ax, show_velocity=True)

    def animate(self, frame, dt=0.1):
        """Animation function for matplotlib."""
        # Update the simulation
        self.update(dt)

        # Return all visual elements that need to be redrawn
        visual_elements = [self.drag_text]  # Include drag text
        for particle in self.particles:
            visual_elements.extend(
                [particle.rod, particle.cap1, particle.cap2, particle.text])
        # draw all debug circles
        for particle in self.particles:
            for circle in particle.debug:
                visual_elements.append(circle)
                self.ax.add_patch(circle)
        # Clear debug circles after drawing
        for particle in self.particles:
            particle.debug = []

        return visual_elements

    def run_simulation(self, num_frames=500, dt=0.1, interval=50):
        """Run the animation for the specified number of frames."""
        # Initial update of visual elements
        for particle in self.particles:
            particle.update_visual_elements(self.ax)

        # Set up keyboard event handling for adjusting drag
        def on_key(event):
            if event.key == 'up':
                self.adjust_drag(linear_change=0.1)
            elif event.key == 'down':
                self.adjust_drag(linear_change=-0.1)
            elif event.key == 'right':
                self.adjust_drag(angular_change=0.1)
            elif event.key == 'left':
                self.adjust_drag(angular_change=-0.1)

        # Connect the key press event
        self.fig.canvas.mpl_connect('key_press_event', on_key)

        # Create and run animation
        anim = FuncAnimation(self.fig, self.animate, frames=num_frames,
                             interval=interval, blit=True, fargs=(dt,))
        plt.show()
        return anim


# Example usage
if __name__ == "__main__":
    # Create and run simulation with drag
    sim = SpherocylinderSimulation(
        num_particles=0,
        box_size=20,
        particle_diameter=0.5,
        initial_length=1,
        noise_strength=0.02,
        growth_rate=0.01,
        max_length=4.0,
        velocity=0.2,
        linear_drag=0.5,    # Initial linear drag coefficient
        angular_drag=0.2    # Initial angular drag coefficient
    )

    # Add two specific particles to observe collision behavior
    p1 = Spherocylinder(
        position=[5.0, 2],
        orientation=np.pi/2,
        length=3,
        diameter=0.5,
        max_length=5.0,
        growth_rate=0.05
    )

    p2 = Spherocylinder(
        position=[10.0, 5],
        orientation=np.pi,
        length=1,
        diameter=0.5,
        max_length=5.0,
        growth_rate=0.05
    )
    sim.particles.append(p1)
    sim.particles.append(p2)

    # Add instructions about drag adjustment
    print("Use arrow keys to adjust drag coefficients:")
    print("  Up/Down: Increase/decrease linear drag")
    print("  Right/Left: Increase/decrease angular drag")

    sim.run_simulation(num_frames=500, dt=0.1, interval=50)
