import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
from Spherocylinder import Spherocylinder
from scipy.spatial.transform import Rotation


class Simulation:
    """
    Simulation environment for a collection of 3D spherocylinder particles
    with torque-based collision response and background drag.
    Uses ghost particles to implement periodic boundary conditions.
    """

    def __init__(self, box_size=20, tao_growthrate=0.1, lambda_sensitivity=0.5):
        """
        Initialize the simulation of growing 3D spherocylinder particles.

        Parameters:
        -----------
        box_size : float or list/tuple
            Size of the simulation box. If float, creates a cubic box.
            If list/tuple, specifies [x_size, y_size, z_size]
        tao_growthrate : float
            Base growth rate parameter
        lambda_sensitivity : float
            Stress sensitivity parameter for growth
        """
        # Handle different box size specifications
        if isinstance(box_size, (int, float)):
            self.box_size = [box_size, box_size, box_size]  # Cubic box
        else:
            self.box_size = list(box_size)  # Custom dimensions

        self.tao_growthrate = tao_growthrate
        self.lambda_sensitivity = lambda_sensitivity

        # Collision parameters
        self.collision_stiffness = 2.0  # Spring constant for collision response
        self.damping = 0.5  # Damping coefficient

        # Create particles
        self.particles = []
        # Will store temporary ghost particles for boundary interactions
        self.ghost_particles = []

        # Initialize figure for 3D animation
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, self.box_size[0])
        self.ax.set_ylim(0, self.box_size[1])
        self.ax.set_zlim(0, self.box_size[2])
        self.ax.set_box_aspect(self.box_size)  # Equal aspect ratio for axes
        self.ax.set_title(
            '3D Growing Spherocylinder Simulation with Periodic Boundaries')

        # Add axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Draw the simulation box edges
        self.draw_simulation_box()

        # Add energy indicator
        self.energy_text = self.ax.text2D(
            0.02, 0.98, '',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

        # Store for tracking cylinders in 3D
        self.cylinder_actors = []

    def draw_simulation_box(self):
        """Draw the edges of the simulation box in 3D."""
        x_size, y_size, z_size = self.box_size

        # Define the 8 corners of the box
        points = np.array([
            [0, 0, 0],
            [x_size, 0, 0],
            [x_size, y_size, 0],
            [0, y_size, 0],
            [0, 0, z_size],
            [x_size, 0, z_size],
            [x_size, y_size, z_size],
            [0, y_size, z_size]
        ])

        # Define the 12 edges of the box
        edges = [
            # Bottom face
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Connecting edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        # Draw each edge
        for edge in edges:
            self.ax.plot3D(
                [points[edge[0]][0], points[edge[1]][0]],
                [points[edge[0]][1], points[edge[1]][1]],
                [points[edge[0]][2], points[edge[1]][2]],
                color='black', linestyle='-', linewidth=1
            )

    def create_ghost_particles(self):
        """
        Create ghost particles for periodic boundary conditions in 3D.
        Ghost particles are copies of real particles that are placed just outside
        the boundary to simulate periodic interactions.
        """
        self.ghost_particles = []  # Clear previous ghosts

        # Define offsets for ghost particles (26 surrounding regions in 3D)
        offsets = []
        for x_offset in [-self.box_size[0], 0, self.box_size[0]]:
            for y_offset in [-self.box_size[1], 0, self.box_size[1]]:
                for z_offset in [-self.box_size[2], 0, self.box_size[2]]:
                    # Skip the (0,0,0) offset which would duplicate the original particle
                    if x_offset == 0 and y_offset == 0 and z_offset == 0:
                        continue
                    offsets.append((x_offset, y_offset, z_offset))

        # Distance from boundary to create ghosts (based on max particle length)
        ghost_margin = max([p.length for p in self.particles]
                           ) if self.particles else 0

        for particle in self.particles:
            # Create ghost particles near boundaries
            pos_x, pos_y, pos_z = particle.position

            # Check if particle is near any boundary
            near_x_min = pos_x < ghost_margin
            near_x_max = pos_x > self.box_size[0] - ghost_margin
            near_y_min = pos_y < ghost_margin
            near_y_max = pos_y > self.box_size[1] - ghost_margin
            near_z_min = pos_z < ghost_margin
            near_z_max = pos_z > self.box_size[2] - ghost_margin

            # Create ghost particles as needed
            for offset_x, offset_y, offset_z in offsets:
                # Create ghost if the particle is near a relevant boundary
                if ((offset_x < 0 and near_x_max) or
                    (offset_x > 0 and near_x_min) or
                    (offset_x == 0 and (near_x_min or near_x_max)) or
                    (offset_y < 0 and near_y_max) or
                    (offset_y > 0 and near_y_min) or
                    (offset_y == 0 and (near_y_min or near_y_max)) or
                    (offset_z < 0 and near_z_max) or
                    (offset_z > 0 and near_z_min) or
                        (offset_z == 0 and (near_z_min or near_z_max))):

                    # Create ghost position
                    position_x = pos_x + offset_x
                    position_y = pos_y + offset_y
                    position_z = pos_z + offset_z

                    # Skip if the ghost particle is too far outside the box
                    if (position_x < -ghost_margin or position_x > self.box_size[0] + ghost_margin or
                        position_y < -ghost_margin or position_y > self.box_size[1] + ghost_margin or
                            position_z < -ghost_margin or position_z > self.box_size[2] + ghost_margin):
                        continue

                    # Create a ghost particle
                    ghost = Spherocylinder(
                        position=[position_x, position_y, position_z],
                        orientation=particle.orientation,
                        linear_velocity=particle.linear_velocity.copy(),
                        angular_velocity=particle.angular_velocity.copy(),
                        l0=particle.l0
                    )
                    ghost.length = particle.length
                    ghost.diameter = particle.diameter
                    ghost.real_particle = particle  # Reference to the real particle
                    ghost.is_ghost = True  # Mark as a ghost
                    self.ghost_particles.append(ghost)

    def calculate_stresses(self):
        """Calculate compressive stress on each particle due to contacts."""
        # Reset all stresses
        for particle in self.particles:
            particle.stress = 0.0

        # Create ghost particles
        self.create_ghost_particles()

        # Check all particle pairs for overlap (including ghosts)
        for i, particle_i in enumerate(self.particles):
            # Real-to-real particle interactions
            for j, particle_j in enumerate(self.particles[i+1:], i+1):
                overlapping, _, _, _ = particle_i.check_overlap(particle_j)
                if overlapping:
                    # Simple model: each overlap adds to stress
                    particle_i.stress += 1
                    particle_j.stress += 1

            # Real-to-ghost particle interactions
            for ghost_j in self.ghost_particles:
                # Skip if the ghost is a copy of the current particle
                if hasattr(ghost_j, 'real_particle') and ghost_j.real_particle == particle_i:
                    continue

                overlapping, _, _, _ = particle_i.check_overlap(ghost_j)
                if overlapping:
                    # Add stress to the real particle
                    particle_i.stress += 1
                    # Add stress to the real particle that the ghost represents
                    if hasattr(ghost_j, 'real_particle'):
                        ghost_j.real_particle.stress += 1

        # Normalize stresses
        max_stress = 1.0
        for particle in self.particles:
            if particle.stress > max_stress:
                max_stress = particle.stress

        if max_stress > 0:
            for particle in self.particles:
                particle.stress /= max_stress

    def handle_collisions(self):
        """Apply torque-based collision response between particles in 3D."""
        # Create ghost particles for boundary collisions
        self.create_ghost_particles()

        # Check all real particle pairs for collision
        for i, particle_i in enumerate(self.particles):
            # Real-to-real particle interactions
            for j, particle_j in enumerate(self.particles[i+1:], i+1):
                overlapping, overlap, contact_point, normal = particle_i.check_overlap(
                    particle_j)

                if overlapping:
                    # Calculate collision forces and torques
                    self.apply_collision_response(
                        particle_i, particle_j, overlap, contact_point, normal)

            # Real-to-ghost particle interactions
            for ghost_j in self.ghost_particles:
                # Skip if the ghost is a copy of the current particle
                if hasattr(ghost_j, 'real_particle') and ghost_j.real_particle == particle_i:
                    continue

                overlapping, overlap, contact_point, normal = particle_i.check_overlap(
                    ghost_j)

                if overlapping:
                    # For ghost collisions, apply force to the real particle and update the ghost's real particle
                    real_particle_j = ghost_j.real_particle if hasattr(
                        ghost_j, 'real_particle') else None
                    if real_particle_j:
                        self.apply_collision_response(
                            particle_i, real_particle_j, overlap, contact_point, normal)

    def apply_collision_response(self, particle1, particle2, overlap, contact_point, normal):
        """
        Apply collision response between two particles with torque effects in 3D.

        Parameters:
        -----------
        particle1, particle2 : Spherocylinder
            The two colliding particles
        overlap : float
            Magnitude of overlap between particles
        contact_point : numpy.ndarray
            Point of contact between particles (3D)
        normal : numpy.ndarray
            Normal vector at contact point (pointing from particle2 to particle1) (3D)
        """
        # Move particles apart to resolve overlap
        particle1.position += normal * overlap / 2
        particle2.position -= normal * overlap / 2

        # Calculate repulsive force magnitude based on overlap
        force_magnitude = self.collision_stiffness * overlap

        # Calculate force vector (from particle2 to particle1)
        force = force_magnitude * normal

        # Apply opposite forces to each particle
        linear_impulse = force / particle1.mass
        particle1.linear_velocity += linear_impulse

        linear_impulse = -force / particle2.mass
        particle2.linear_velocity += linear_impulse

        # Calculate torque for each particle in 3D
        # Torque = r × F where r is the vector from center of mass to contact point
        r1 = contact_point - particle1.position
        torque1 = np.cross(r1, force)  # 3D cross product

        r2 = contact_point - particle2.position
        torque2 = np.cross(r2, -force)  # 3D cross product

        # Apply torques
        particle1.torque += torque1
        particle2.torque += torque2

    def enforce_periodic_boundaries(self, particle):
        """
        Apply periodic boundary conditions to a particle in 3D.
        """
        # Apply periodic boundary conditions for all three dimensions
        for i in range(3):
            if particle.position[i] < 0:
                particle.position[i] += self.box_size[i]
            elif particle.position[i] > self.box_size[i]:
                particle.position[i] -= self.box_size[i]

    def divide_particles(self):
        """Divide particles if they exceed a certain length."""
        new_particles = []

        for particle in self.particles:
            if particle.length >= particle.max_length:
                # Get the cylinder axis direction
                direction_vector = particle.get_direction_vector()

                # Calculate new positions for child particles
                center_left = particle.position - 0.25 * particle.length * direction_vector
                center_right = particle.position + 0.25 * particle.length * direction_vector

                # Create new particles with small orientation noise
                # Create a random rotation matrix for small angular perturbation
                noise_angle = np.random.uniform(-np.pi/32, np.pi/32)
                noise_axis = np.random.randn(3)
                noise_axis = noise_axis / np.linalg.norm(noise_axis)
                noise_rot = Rotation.from_rotvec(noise_angle * noise_axis)

                # Apply noise to the orientation quaternion
                noise_quat = noise_rot.as_quat()  # [x, y, z, w] format
                noise_quat = np.array(
                    [noise_quat[3], noise_quat[0], noise_quat[1], noise_quat[2]])  # [w, x, y, z]

                # Create new particles with perturbed orientations
                new_particle_left = Spherocylinder(
                    position=center_left,
                    orientation=self.quaternion_multiply(
                        particle.orientation, noise_quat),
                    linear_velocity=particle.linear_velocity.copy(),
                    angular_velocity=particle.angular_velocity.copy(),
                    l0=particle.l0,
                )

                new_particle_right = Spherocylinder(
                    position=center_right,
                    orientation=self.quaternion_multiply(
                        particle.orientation, noise_quat),
                    linear_velocity=particle.linear_velocity.copy(),
                    angular_velocity=particle.angular_velocity.copy(),
                    l0=particle.l0,
                )

                # Set lengths to half of the parent
                new_particle_left.length = particle.length / 2
                new_particle_right.length = particle.length / 2

                # Apply periodic boundary conditions to new particles if needed
                self.enforce_periodic_boundaries(new_particle_left)
                self.enforce_periodic_boundaries(new_particle_right)

                new_particles.append(new_particle_left)
                new_particles.append(new_particle_right)

                particle.to_delete = True  # Mark for deletion

        # Add all new particles
        self.particles.extend(new_particles)

        # Remove marked particles
        self.particles = [p for p in self.particles if not p.to_delete]

    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions q1 and q2 (format: [w, x, y, z])
        This is a utility function to handle orientation changes in 3D
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def add_particle(self, position, orientation, linear_velocity=None, angular_velocity=None, l0=1.0):
        """
        Add a particle to the simulation.

        Parameters:
        -----------
        position : list or numpy.ndarray
            3D position [x, y, z]
        orientation : list or numpy.ndarray
            Orientation as quaternion [w, x, y, z] or Euler angles [roll, pitch, yaw]
        linear_velocity : list or numpy.ndarray, optional
            Linear velocity [vx, vy, vz]
        angular_velocity : list or numpy.ndarray, optional
            Angular velocity [wx, wy, wz]
        l0 : float, optional
            Initial length of the particle
        """
        if linear_velocity is None:
            linear_velocity = np.zeros(3)
        if angular_velocity is None:
            angular_velocity = np.zeros(3)

        particle = Spherocylinder(
            position=position,
            orientation=orientation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            l0=l0
        )
        self.particles.append(particle)
        return particle

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
            particle.grow(dt, self.tao_growthrate,
                          self.lambda_sensitivity)

            # Move particle with collision effects and drag
            particle.move(dt, self.box_size)

            # Apply periodic boundary conditions
            self.enforce_periodic_boundaries(particle)

        # Calculate total energy
        self.calculate_energy()

        # Update visual representation later in draw_particles

    def calculate_energy(self):
        """Calculate the total energy of the system."""
        total_energy = 0.0
        for particle in self.particles:
            # Kinetic energy
            translational_energy = 0.5 * particle.mass * \
                np.linalg.norm(particle.linear_velocity)**2

            # For rotational energy in 3D, use the full inertia tensor
            # E_rot = 0.5 * ω · (I · ω)
            # For a diagonal inertia tensor, this simplifies to:
            rotational_energy = 0.5 * np.sum(particle.inertia_tensor.diagonal() *
                                             particle.angular_velocity**2)

            total_energy += translational_energy + rotational_energy

        self.energy_text.set_text(
            f'Total Energy: {total_energy:.2f} | Particles: {len(self.particles)}')

        return total_energy

    def draw_particles(self, show_ghosts=False):
        """
        Draw all particles in 3D with proper spherocylinder representation.

        This method creates a complete surface representation of spherocylinders
        by generating the cylindrical body and hemispherical caps.

        Parameters:
        -----------
        show_ghosts : bool
            Whether to visualize ghost particles
        """
        # Clear all existing drawings
        for artist in self.ax.collections:
            artist.remove()

        # Create the combined list of particles to draw
        particles_to_draw = self.particles
        if show_ghosts:
            particles_to_draw = particles_to_draw + self.ghost_particles

        # Draw each particle
        for particle in particles_to_draw:
            particle.draw(self.ax)

    def animate(self, frame, dt=0.1, show_ghosts=False):
        """Animation function for matplotlib in 3D."""
        # Update the simulation
        self.update(dt)

        # Clear axis and redraw
        self.ax.cla()

        # Reset axis limits
        self.ax.set_xlim(0, self.box_size[0])
        self.ax.set_ylim(0, self.box_size[1])
        self.ax.set_zlim(0, self.box_size[2])

        # Redraw box
        self.draw_simulation_box()

        # Draw all particles
        self.draw_particles(show_ghosts)

        # Update energy text
        self.energy_text = self.ax.text2D(
            0.02, 0.98, f'Total Energy: {self.calculate_energy():.2f} | Particles: {len(self.particles)}',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

        # Add axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Growing Spherocylinder Simulation')

        # For 3D, we need to return an empty list since we're redrawing the whole scene
        return []

    def run_simulation(self, num_frames=500, dt=0.1, interval=50, show_ghosts=False):
        """
        Run the animation for the specified number of frames in 3D.

        Parameters:
        -----------
        num_frames : int
            Number of frames to simulate
        dt : float
            Time step for simulation
        interval : int
            Interval between animation frames in milliseconds
        show_ghosts : bool
            Whether to visualize ghost particles
        """
        # Create and run animation
        anim = FuncAnimation(self.fig, self.animate, frames=num_frames,
                             interval=interval, blit=False,
                             fargs=(dt, show_ghosts))
        plt.show()
        return anim
