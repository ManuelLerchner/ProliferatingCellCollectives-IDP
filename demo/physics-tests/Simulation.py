import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from Spherocylinder import Spherocylinder
from tqdm import tqdm
import time

import nest_asyncio
nest_asyncio.apply()


class Simulation:
    """
    Simulation environment for a collection of spherocylinder particles
    with torque-based collision response and background drag.
    Uses ghost particles to implement periodic boundary conditions.
    """

    def __init__(self, box_size=20, tao_growthrate=0.1, lambda_sensitivity=0.5):
        """
        Initialize the simulation of growing spherocylinder particles.
        """
        self.box_size = box_size
        self.tao_growthrate = tao_growthrate
        self.lambda_sensitivity = lambda_sensitivity

        # Collision parameters
        self.collision_stiffness = 2.0  # Spring constant for collision response
        self.damping = 0.5  # Damping coefficient

        # Create particles
        self.particles = []
        # Will store temporary ghost particles for boundary interactions
        self.ghost_particles = []

        # Initialize figure for animation
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, box_size)
        self.ax.set_ylim(0, box_size)
        self.ax.set_aspect('equal')
        self.ax.set_title(
            'Growing Spherocylinder Simulation with Periodic Boundaries')

        border = plt.Rectangle(
            (0, 0), box_size, box_size, fill=False, color='black', lw=2)
        self.ax.add_patch(border)

        # Add energy indicator
        self.energy_text = self.ax.text(
            0.02, 0.98, f'',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        self.time_text = self.ax.text(
            0.02, 0.92, f'',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        self.total_time = 0.0
        self.time_last_frame = 0.0

    def create_ghost_particles(self):
        """
        Create ghost particles for periodic boundary conditions.
        Ghost particles are copies of real particles that are placed just outside
        the boundary to simulate periodic interactions.
        """
        for ghost in self.ghost_particles:
            ghost.delete_visual_elements()
        self.ghost_particles = []  # Clear previous ghosts

        # Define offsets for ghost particles (8 surrounding regions)
        offsets = [
            (-self.box_size, -self.box_size),  # bottom-left
            (0, -self.box_size),               # bottom
            (self.box_size, -self.box_size),   # bottom-right
            (-self.box_size, 0),               # left
            (self.box_size, 0),                # right
            (-self.box_size, self.box_size),   # top-left
            (0, self.box_size),                # top
            (self.box_size, self.box_size)     # top-right
        ]

        # Distance from boundary to create ghosts (based on max particle length)
        ghost_margin = max(
            [p.length for p in self.particles]) if self.particles else 0

        for particle in self.particles:
            # Create ghost particles near boundaries
            particle_x, particle_y = particle.position

            # Check if particle is near any boundary
            near_left = particle_x < ghost_margin
            near_right = particle_x > self.box_size - ghost_margin
            near_bottom = particle_y < ghost_margin
            near_top = particle_y > self.box_size - ghost_margin

            # Create ghost particles as needed
            for offset_x, offset_y in offsets:
                # Create ghost if the particle is near a relevant boundary
                if ((offset_x < 0 and near_right) or
                    (offset_x > 0 and near_left) or
                    (offset_x == 0 and (near_left or near_right)) or
                    (offset_y < 0 and near_top) or
                    (offset_y > 0 and near_bottom) or
                        (offset_y == 0 and (near_bottom or near_top))):

                    # Skip the no-offset case (would create duplicate of the original)
                    position_x = particle_x + offset_x
                    position_y = particle_y + offset_y

                    # Skip if the ghost particle is outside the box
                    if (position_x < -ghost_margin or
                            position_x > self.box_size + ghost_margin or
                            position_y < -ghost_margin or
                            position_y > self.box_size + ghost_margin):
                        continue

                    # Create a ghost particle
                    ghost = Spherocylinder(
                        position=[position_x, position_y],
                        orientation=particle.orientation,
                        linear_velocity=particle.linear_velocity,
                        angular_velocity=particle.angular_velocity,
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
                overlapping, overlap, _, _ = particle_i.check_overlap(
                    particle_j)
                penalty = np.exp(overlap) - np.exp(-0.5)
                if overlap > -0.5:
                    # Simple model: each overlap adds to stress
                    particle_i.stress += penalty
                    particle_j.stress += penalty

            # Real-to-ghost particle interactions
            for ghost_j in self.ghost_particles:
                # Skip if the ghost is a copy of the current particle
                if hasattr(ghost_j, 'real_particle') and ghost_j.real_particle == particle_i:
                    continue

                overlapping, overlap, _, _ = particle_i.check_overlap(ghost_j)
                penalty = np.exp(overlap) - np.exp(-0.5)
                if overlap > -0.5:
                    # Simple model: each overlap adds to stress
                    particle_i.stress += penalty

    def handle_collisions(self):
        """Apply torque-based collision response between particles."""
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

        # Calculate torque for each particle
        # Torque = r × F where r is the vector from center of mass to contact point
        r1 = contact_point - particle1.position
        torque1 = np.cross(r1, force)  # Scalar in 2D

        r2 = contact_point - particle2.position
        torque2 = np.cross(r2, -force)  # Scalar in 2D

        # Apply torques
        particle1.torque += torque1
        particle2.torque += torque2

    def enforce_periodic_boundaries(self, particle):
        """
        Apply periodic boundary conditions to a particle.
        """
        # Apply periodic boundary conditions
        if particle.position[0] < 0:
            particle.position[0] += self.box_size
        elif particle.position[0] > self.box_size:
            particle.position[0] -= self.box_size

        if particle.position[1] < 0:
            particle.position[1] += self.box_size
        elif particle.position[1] > self.box_size:
            particle.position[1] -= self.box_size

    def divide_particles(self):
        """Divide particles if they exceed a certain length."""
        for i in range(len(self.particles)):
            particle = self.particles[i]
            if particle.length >= particle.max_length:
                # Calculate new positions for child particles
                orientation_vector = np.array(
                    [np.cos(particle.orientation), np.sin(particle.orientation)])

                center_left = particle.position - 1/4. * particle.length * orientation_vector
                center_right = particle.position + 1/4. * particle.length * orientation_vector

                # Random orientation noise
                noise = np.random.uniform(-np.pi/32, np.pi/32)

                # Create new particles. Maintain total energy
                new_particle_left = Spherocylinder(
                    position=center_left,
                    orientation=particle.orientation + noise,
                    linear_velocity=particle.linear_velocity,
                    angular_velocity=particle.angular_velocity,
                    l0=particle.l0,
                )
                new_particle_right = Spherocylinder(
                    position=center_right,
                    orientation=particle.orientation + noise,
                    linear_velocity=particle.linear_velocity,
                    angular_velocity=particle.angular_velocity,
                    l0=particle.l0,
                )
                new_particle_left.length = particle.length / 2
                new_particle_right.length = particle.length / 2

                # Apply periodic boundary conditions to new particles if needed
                self.enforce_periodic_boundaries(new_particle_left)
                self.enforce_periodic_boundaries(new_particle_right)

                self.particles.append(new_particle_left)
                self.particles.append(new_particle_right)

                particle.to_delete = True  # Mark for deletion
                particle.delete_visual_elements()
        # Remove marked particles
        self.particles = [p for p in self.particles if not p.to_delete]

    def update(self, dt):
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

            # Update visual representation
            particle.update_visual_elements(self.ax)

        # Calculate total energy
        self.calculate_energy()
        self.total_time += dt

    def calculate_energy(self):
        """Calculate the total energy of the system."""
        total_energy = 0.0
        for particle in self.particles:
            # Kinetic energy
            translational_energy = 0.5 * particle.mass * \
                np.linalg.norm(particle.linear_velocity)**2
            rotational_energy = 0.5 * particle.moment_of_inertia * \
                np.linalg.norm(particle.angular_velocity)**2
            total_energy += translational_energy + rotational_energy

        self.energy_text.set_text(
            f'Total Energy: {total_energy:.2E}'
        )

    def animate(self, frame, base_dt, scaling_factor):
        """Animation function for matplotlib."""

        dt = base_dt / (1 + len(self.particles) * scaling_factor)

        # Update the simulation
        self.update(dt)

        # Return all visual elements that need to be redrawn
        visual_elements = [self.energy_text, self.time_text]
        self.time_text.set_text(
            f'Frame:{frame}, Time: {self.total_time:.2f}, Δt: {dt:.2f}')

        # Draw real particles
        for particle in self.particles:
            visual_elements.extend(
                [particle.rod, particle.cap1, particle.cap2, particle.text])

        # Optionally, draw ghost particles for visualization (with different color/transparency)
        if hasattr(self, 'show_ghosts') and self.show_ghosts:
            for ghost in self.ghost_particles:
                ghost.update_visual_elements(
                    self.ax, alpha=0.3, ghost=True)  # Lower alpha for ghosts
                visual_elements.extend([ghost.rod, ghost.cap1, ghost.cap2,ghost.text])

        return visual_elements

    def run_simulation(self, frames=500, base_dt=0.1, scaling_factor=0.5, interval=50, show_ghosts=False):

        self.show_ghosts = show_ghosts

        # Initial update of visual elements
        for particle in self.particles:
            particle.update_visual_elements(self.ax)

        # Create and run animation
        anim = FuncAnimation(self.fig, self.animate,
                             repeat=False,
                             frames=frames,
                             interval=interval, blit=True, fargs=(base_dt,
                                                                  scaling_factor))
        return anim
