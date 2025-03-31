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
    Uses linked cells for efficient collision detection.
    """

    def __init__(self, box_size=20, tao_growthrate=0.1, lambda_sensitivity=0.5, friction=100):
        """
        Initialize the simulation of growing spherocylinder particles.
        """
        self.box_size = box_size
        self.tao_growthrate = tao_growthrate
        self.lambda_sensitivity = lambda_sensitivity
        self.friction = friction  # Friction coefficient for drag

        # Collision parameters
        self.friction = 0.5  # Damping coefficient

        # Create particles
        self.particles = []
        # Will store temporary ghost particles for boundary interactions
        self.ghost_particles = []

        # Initialize linked cell grid
        self.cell_size = 0  # Will be updated based on particles
        self.grid_size = 0  # Number of cells per dimension
        self.cell_grid = {}  # Dictionary to store particles in each cell

        # Initialize figure for animation
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, box_size)
        self.ax.set_ylim(0, box_size)
        self.ax.set_aspect('equal')
        self.ax.set_title(
            'Growing Spherocylinder Simulation with Periodic Boundaries and Linked Cells')

        border = plt.Rectangle(
            (0, 0), box_size, box_size, fill=False, color='black', lw=2)
        self.ax.add_patch(border)

        # Add energy indicator
        self.energy_text = self.ax.text(
            0.02, 0.98, f'',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        self.time_text = self.ax.text(
            0.02, 0.995, f'',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        self.cell_text = self.ax.text(
            0.02, 0.86, f'',
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top')
        self.total_time = 0.0
        self.time_last_frame = 0.0

    def sweep_and_prune(self):
        """
        Broad-phase collision detection using Sweep and Prune.
        Sort particles along one axis and check for overlapping intervals.
        """

        def bounding_box_overlap(box1, box2):
            """Check if two bounding boxes overlap."""
            " box1: (x_min, x_max, y_min, y_max) "
            " box2: (x_min, x_max, y_min, y_max) "

            return (box1[0] < box2[1] and box1[1] > box2[0] and
                    box1[2] < box2[3] and box1[3] > box2[2])

        for particle in self.particles:
            particle.update_bounding_box()

        # Sort particles by their min x-coordinate
        self.particles.sort(key=lambda p: p.bounding_box_coords[0])

        for i in range(len(self.particles)):
            particle = self.particles[i]

            for j in range(i + 1, len(self.particles)):
                other_particle = self.particles[j]

                # Check if the bounding boxes overlap
                if particle.bounding_box_coords[1] < other_particle.bounding_box_coords[0]:
                    break

                if bounding_box_overlap(particle.bounding_box_coords, other_particle.bounding_box_coords):
                    # Check if the particles are close enough to collide
                    yield particle, other_particle

    def handle_interactions(self):
        """Apply torque-based collision response between particles using linked cells."""
        # Get potential pairs of particles that may collide

        for particle in self.particles:
            particle.stress = 0.0

        potential_pairs = self.sweep_and_prune()

        for p1, p2 in potential_pairs:
            overlapping, overlap, contact_point, normal = p1.check_overlap(p2)
            if overlapping:
                self.apply_collision_response(
                    p1, p2, overlap, contact_point, normal)

            penalty = np.exp(overlap) - np.exp(-0.5)
            if overlap > -0.5:
                p1.stress += penalty
                p2.stress += penalty

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

        # move particles apart based on overlap
        # Calculate the distance to move apart
        move_distance = overlap / 2.0
        particle1.position += move_distance * normal
        particle2.position -= move_distance * normal

        # Calculate the force vector based on overlap and normal
        force_magnitude = self.friction * overlap
        force_vector = force_magnitude * normal

        # Apply forces to particles
        particle1.force += force_vector
        particle2.force -= force_vector

        # Calculate the torque based on the contact point and normal
        # Calculate the distance vector from the center of particle2 to the contact point
        r1 = contact_point - particle1.position
        r2 = contact_point - particle2.position

        torque1 = np.cross(r1, force_vector)
        torque2 = np.cross(r2, -force_vector)

        particle1.torque += torque1
        particle2.torque -= torque2

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
                    orientation=particle.orientation - noise,
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

        # remove debug lines
        for line in self.ax.lines:
            line.remove()
        # Handle collisions and apply torques
        self.handle_interactions()

        # Divide particles if necessary
        self.divide_particles()

        # Update all particles
        for particle in self.particles:
            # Grow based on stress
            particle.grow(dt, self.tao_growthrate,
                          self.lambda_sensitivity)

            # Move particle with collision effects and drag
            particle.move(dt)

            # Apply periodic boundary conditions
            self.enforce_periodic_boundaries(particle)

            # Update visual representation
            particle.update_visual_elements(self.ax)

        # Calculate total energy
        self.total_time += dt

    def animate(self, time, base_dt, scaling_factor):
        """Animation function for matplotlib."""

        for i in range(20):
            dt = base_dt / (1 + len(self.particles) * scaling_factor)
            # Update the simulation
            self.update(dt)

        # Return all visual elements that need to be redrawn
        visual_elements = [self.energy_text, self.time_text, self.cell_text]
        self.time_text.set_text(
            f't={time:.2f}, Δt: {dt:.2f}, Particles: {len(self.particles)}')

        # Draw real particles
        for particle in self.particles:
            visual_elements.extend(
                [particle.rod, particle.cap1, particle.cap2, particle.text])

        return visual_elements

    def run_simulation(self, end_time=500, base_dt=0.1, scaling_factor=0.5, interval=50, show_ghosts=False, show_grid=False):
        """Run the simulation animation."""
        self.show_ghosts = show_ghosts
        self.show_grid = show_grid

        # Initial update of visual elements
        for particle in self.particles:
            particle.update_visual_elements(self.ax)

        def frame_generator():
            """Generator for frames."""
            while self.total_time < end_time:
                yield self.total_time

        # Create and run animation
        anim = FuncAnimation(self.fig, self.animate,
                             repeat=False,
                             frames=frame_generator,
                             interval=interval, blit=True, fargs=(base_dt,
                                                                  scaling_factor))
        return anim
