import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from Spherocylinder import Spherocylinder


class Simulation:
    """
    Simulation environment for a collection of 3D spherocylinder particles
    with torque-based collision response and background drag.
    Uses ghost particles to implement periodic boundary conditions.
    """

    def __init__(self, tao_growthrate=0.1, lambda_sensitivity=0.5):
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

        self.tao_growthrate = tao_growthrate
        self.lambda_sensitivity = lambda_sensitivity

        # Collision parameters
        self.collision_stiffness = 2.0  # Spring constant for collision response
        self.damping = 0.5  # Damping coefficient
        self.friction = 0.5  # Friction parameter for force magnitude

        # Create particles
        self.particles = []
        # Will store temporary ghost particles for boundary interactions
        self.ghost_particles = []

        # Initialize figure for 3D animation
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_title(
            '3D Growing Spherocylinder Simulation with Periodic Boundaries')

        # Add axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Add energy indicator
        self.energy_text = self.ax.text2D(
            0.02, 0.98, '',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

        # Store for tracking cylinders in 3D
        self.cylinder_actors = []

    def calculate_stresses(self):
        """Calculate compressive stress on each particle due to contacts."""
        # Reset all stresses
        for particle in self.particles:
            particle.stress = 0.0

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
        Apply collision response between two particles with torque effects in 3D (matching 2D model).
        """
        # Move particles apart based on overlap
        move_distance = overlap / 2.0
        particle1.position += move_distance * normal
        particle2.position -= move_distance * normal

        # Calculate the force vector based on overlap and normal (use friction parameter)
        force_magnitude = self.friction * overlap
        force_vector = force_magnitude * normal

        # Apply forces to particles (accumulate, do not update velocity directly)
        particle1.force += force_vector
        particle2.force -= force_vector

        # Calculate the torque based on the contact point and normal
        r1 = contact_point - particle1.position
        r2 = contact_point - particle2.position

        torque1 = np.cross(r1, force_vector)
        torque2 = np.cross(r2, -force_vector)

        particle1.torque += torque1
        particle2.torque -= torque2

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
                noise_angle = 0
                noise_axis = np.random.randn(3)
                noise_axis = noise_axis / np.linalg.norm(noise_axis)
                noise_rot = Rotation.from_rotvec(noise_angle * noise_axis)

                # Apply noise to the orientation quaternion
                noise_quat = noise_rot.as_quat()  # [x, y, z, w] format
                noise_quat = np.array(
                    # [w, x, y, z]
                    [noise_quat[3], noise_quat[0], noise_quat[1], noise_quat[2]])

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
            particle.move(dt)

            print(particle)

        print("\n")

        # Update visual representation later in draw_particles

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

        # Draw all particles
        self.draw_particles(show_ghosts)

        # Update energy text
        self.energy_text = self.ax.text2D(
            0.02, 0.98, f'Particles: {len(self.particles)}',
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
