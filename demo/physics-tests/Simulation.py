
from time import monotonic_ns

import matplotlib.pyplot as plt
import nest_asyncio
import numpy as np
from matplotlib.animation import FuncAnimation

nest_asyncio.apply()


class Simulation:
    """
    Simulation environment for a collection of spherocylinder particles
    with torque-based collision response and background drag.
    Uses ghost particles to implement periodic boundary conditions.
    Uses linked cells for efficient collision detection.
    """

    def __init__(self, tao_growthrate=0.1, lambda_sensitivity=0.5, friction=100):
        """
        Initialize the simulation of growing spherocylinder particles.
        """
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
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.ax.set_aspect('equal')

        # Add energy indicator
        self.total_time = 0.0
        self.time_last_frame = monotonic_ns()
        self.ups = 0.0
        self.dt = 0.0

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

        self.potential_pairs = []

        for i in range(len(self.particles)):
            particle = self.particles[i]

            for j in range(i + 1, len(self.particles)):
                other_particle = self.particles[j]

                # Check if the bounding boxes overlap
                if particle.bounding_box_coords[1] < other_particle.bounding_box_coords[0]:
                    break

                if bounding_box_overlap(particle.bounding_box_coords, other_particle.bounding_box_coords):
                    # Check if the particles are close enough to collide
                    self.potential_pairs.append((particle, other_particle))

    def handle_interactions(self):
        """Apply torque-based collision response between particles using linked cells."""
        # Get potential pairs of particles that may collide

        for particle in self.particles:
            particle.stress = 0.0

        for p1, p2 in self.potential_pairs:
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

    def update(self, dt):
        """Update the simulation for one time step."""

        # Handle collisions and apply torques
        self.handle_interactions()

        # Divide particles if necessary

        # Update all particles
        for particle in self.particles:
            particle.move(dt)

            # Grow based on stress
            particle.grow(dt, self.tao_growthrate,
                          self.lambda_sensitivity)

            new_particle = particle.divide()
            if new_particle is not None:
                self.particles.append(new_particle)

        # Calculate total energy
        self.total_time += dt

    def animate(self, frame, base_dt, scaling_factor):
        """Animation function for matplotlib."""

        self.sweep_and_prune()

        UPDATES_PER_FRAME = 20
        for i in range(UPDATES_PER_FRAME):
            self.dt = base_dt / (1 + len(self.particles) * scaling_factor)
            # Update the simulation
            self.update(self.dt)

        for particle in self.particles:
            particle.update_visual_elements(self.ax)

        t_current = monotonic_ns() / 1e9
        delta_time = max((t_current - self.time_last_frame), 1e-9)
        self.ups = UPDATES_PER_FRAME / delta_time
        self.time_last_frame = t_current

    def run_simulation(self, end_time=500, base_dt=0.1, scaling_factor=0.5, interval=50, show_ghosts=False, show_grid=False):
        """Run the simulation animation."""
        self.show_ghosts = show_ghosts
        self.show_grid = show_grid

        def frame_generator():
            """Generator for frames."""
            frame = 0
            while self.total_time < end_time:
                yield frame
                frame += 1

        # Create and run animation
        anim = FuncAnimation(self.fig, self.animate,
                             repeat=False,
                             frames=frame_generator,
                             interval=interval,
                             cache_frame_data=False,
                             fargs=(base_dt,
                                    scaling_factor))
        return anim
