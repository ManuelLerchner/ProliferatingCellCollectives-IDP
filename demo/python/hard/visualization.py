import os

import numpy as np
from quaternion import getDirectionVector


def drawSpherocyllinder(end1, end2, orientation, length, diameter, ax, color, n_phi=12, n_theta=4, n_height=4):
    """Update the visual representation of the 3D spherocylinder.

    This method needs significant changes for proper 3D visualization.
    """

    # Particle properties
    radius = diameter / 2

    # Create orthogonal vectors to the axis
    axis = orientation
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
    height = np.linspace(0, length - diameter, n_height)

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
            pointL = end1 + height[i] * orientation + circle_point
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
                                  np.cos(theta[i]) * orientation)
            pointL = end1 + cap_point
            x_cap1[i, j] = pointL[0]
            y_cap1[i, j] = pointL[1]
            z_cap1[i, j] = pointL[2]

            pointR = end2 + cap_point * np.array([-1, -1, 1])
            x_cap2[i, j] = pointR[0]
            y_cap2[i, j] = pointR[1]
            z_cap2[i, j] = pointR[2]

    # Plot the surfaces

    # Cylindrical body
    ax.plot_surface(x_cyl, y_cyl, z_cyl,
                    color=color,   shade=True)

    # Hemispherical caps
    ax.plot_surface(x_cap1, y_cap1, z_cap1,                    color=color)
    ax.plot_surface(x_cap2, y_cap2, z_cap2, color=color)


def render_particles(ax, C, L):
    """Render the particles in 3D space"""
    # Clear the previous frame

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    zlims = ax.get_zlim()
    ax.cla()

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)

    # aspect ratio

    for i in range(len(L)):
        # Extract position (x, y, z)
        pos = C[i*7:i*7 + 3]

        # Extract quaternion components
        s = C[i*7 + 3]  # quaternion scalar part
        v = C[i*7 + 4:i*7 + 7]  # quaternion vector part

        # Get the particle length
        length = L[i]

        # Convert quaternion to direction vector
        direction = getDirectionVector([s, v[0], v[1], v[2]])

        diameter = 0.5

        # Calculate endpoints of the rod
        start_point = pos - direction * (length / 2 - diameter / 2)
        end_point = pos + direction * (length / 2 - diameter / 2)

        # base color on stress. no stress = green, stress = red interpolated
        c = "green"

        # Draw the particle (spherocylinder)
        drawSpherocyllinder(start_point, end_point, direction,
                            length, diameter, ax, c)
