
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# nbagg


def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v


class SpherocylinderVisualizer:
    def __init__(self):
        # Initialize figure
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initial parameters for two spherocylinders
        self.sc1_pos = np.array([0.0, 0.0, 0.0])
        self.sc1_dir = np.array([1.0, 0.0, 0.0])
        self.sc1_length = 2.0
        self.sc1_diameter = 0.5

        self.sc2_pos = np.array([2.0, 0.5, 0.0])
        self.sc2_dir = np.array([0.0, 1.0, 0.0])
        self.sc2_length = 2.0
        self.sc2_diameter = 0.5

        # Setup the UI sliders
        self.setup_ui()

        # Draw the initial state
        self.update(None)

        plt.show()

    def setup_ui(self):
        # Create sliders for adjusting spherocylinder properties
        slider_color = 'lightgoldenrodyellow'
        slider_ax_x1 = plt.axes(
            [0.1, 0.01, 0.15, 0.02], facecolor=slider_color)
        slider_ax_y1 = plt.axes(
            [0.1, 0.04, 0.15, 0.02], facecolor=slider_color)
        slider_ax_z1 = plt.axes(
            [0.1, 0.07, 0.15, 0.02], facecolor=slider_color)

        slider_ax_dx1 = plt.axes(
            [0.35, 0.01, 0.15, 0.02], facecolor=slider_color)
        slider_ax_dy1 = plt.axes(
            [0.35, 0.04, 0.15, 0.02], facecolor=slider_color)
        slider_ax_dz1 = plt.axes(
            [0.35, 0.07, 0.15, 0.02], facecolor=slider_color)

        slider_ax_x2 = plt.axes(
            [0.6, 0.01, 0.15, 0.02], facecolor=slider_color)
        slider_ax_y2 = plt.axes(
            [0.6, 0.04, 0.15, 0.02], facecolor=slider_color)
        slider_ax_z2 = plt.axes(
            [0.6, 0.07, 0.15, 0.02], facecolor=slider_color)

        slider_ax_dx2 = plt.axes(
            [0.85, 0.01, 0.15, 0.02], facecolor=slider_color)
        slider_ax_dy2 = plt.axes(
            [0.85, 0.04, 0.15, 0.02], facecolor=slider_color)
        slider_ax_dz2 = plt.axes(
            [0.85, 0.07, 0.15, 0.02], facecolor=slider_color)

        slider_ax_len1 = plt.axes(
            [0.35, 0.10, 0.15, 0.02], facecolor=slider_color)
        slider_ax_diam1 = plt.axes(
            [0.35, 0.13, 0.15, 0.02], facecolor=slider_color)
        slider_ax_len2 = plt.axes(
            [0.85, 0.10, 0.15, 0.02], facecolor=slider_color)
        slider_ax_diam2 = plt.axes(
            [0.85, 0.13, 0.15, 0.02], facecolor=slider_color)

        # Create the sliders
        self.slider_x1 = Slider(
            slider_ax_x1, 'SC1 X', -5.0, 5.0, valinit=self.sc1_pos[0])
        self.slider_y1 = Slider(
            slider_ax_y1, 'SC1 Y', -5.0, 5.0, valinit=self.sc1_pos[1])
        self.slider_z1 = Slider(
            slider_ax_z1, 'SC1 Z', -5.0, 5.0, valinit=self.sc1_pos[2])

        self.slider_dx1 = Slider(
            slider_ax_dx1, 'SC1 DirX', -1.0, 1.0, valinit=self.sc1_dir[0])
        self.slider_dy1 = Slider(
            slider_ax_dy1, 'SC1 DirY', -1.0, 1.0, valinit=self.sc1_dir[1])
        self.slider_dz1 = Slider(
            slider_ax_dz1, 'SC1 DirZ', -1.0, 1.0, valinit=self.sc1_dir[2])

        self.slider_x2 = Slider(
            slider_ax_x2, 'SC2 X', -5.0, 5.0, valinit=self.sc2_pos[0])
        self.slider_y2 = Slider(
            slider_ax_y2, 'SC2 Y', -5.0, 5.0, valinit=self.sc2_pos[1])
        self.slider_z2 = Slider(
            slider_ax_z2, 'SC2 Z', -5.0, 5.0, valinit=self.sc2_pos[2])

        self.slider_dx2 = Slider(
            slider_ax_dx2, 'SC2 DirX', -1.0, 1.0, valinit=self.sc2_dir[0])
        self.slider_dy2 = Slider(
            slider_ax_dy2, 'SC2 DirY', -1.0, 1.0, valinit=self.sc2_dir[1])
        self.slider_dz2 = Slider(
            slider_ax_dz2, 'SC2 DirZ', -1.0, 1.0, valinit=self.sc2_dir[2])

        self.slider_len1 = Slider(
            slider_ax_len1, 'SC1 Length', 0.1, 4.0, valinit=self.sc1_length)
        self.slider_diam1 = Slider(
            slider_ax_diam1, 'SC1 Diameter', 0.1, 2.0, valinit=self.sc1_diameter)
        self.slider_len2 = Slider(
            slider_ax_len2, 'SC2 Length', 0.1, 4.0, valinit=self.sc2_length)
        self.slider_diam2 = Slider(
            slider_ax_diam2, 'SC2 Diameter', 0.1, 2.0, valinit=self.sc2_diameter)

        # Connect the sliders to the update function
        self.slider_x1.on_changed(self.update)
        self.slider_y1.on_changed(self.update)
        self.slider_z1.on_changed(self.update)
        self.slider_dx1.on_changed(self.update)
        self.slider_dy1.on_changed(self.update)
        self.slider_dz1.on_changed(self.update)
        self.slider_x2.on_changed(self.update)
        self.slider_y2.on_changed(self.update)
        self.slider_z2.on_changed(self.update)
        self.slider_dx2.on_changed(self.update)
        self.slider_dy2.on_changed(self.update)
        self.slider_dz2.on_changed(self.update)
        self.slider_len1.on_changed(self.update)
        self.slider_diam1.on_changed(self.update)
        self.slider_len2.on_changed(self.update)
        self.slider_diam2.on_changed(self.update)

        # Add reset button
        reset_ax = plt.axes([0.1, 0.13, 0.15, 0.02])
        self.reset_button = Button(
            reset_ax, 'Reset', color=slider_color, hovercolor='0.975')
        self.reset_button.on_clicked(self.reset)

    def l2_norm(self, v):
        """Calculate L2 norm of a vector."""
        return np.sqrt(np.sum(v**2))

    def normalize(self, v):
        """Normalize a vector."""
        norm = self.l2_norm(v)
        if norm > 0:
            return v / norm
        return v

    def cross(self, a, b):
        """Cross product of two vectors."""
        return np.cross(a, b)

    def dot(self, a, b):
        """Dot product of two vectors."""
        return np.dot(a, b)

    def minimum_distance_between_line_segments(self, a0, a1, b0, b1, clamp_all=True):
        """Implementation of the minimum distance between line segments algorithm from the C++ code."""
        clamp_a0 = clamp_a1 = clamp_b0 = clamp_b1 = clamp_all

        A = a1 - a0
        B = b1 - b0
        mag_a = self.l2_norm(A)
        mag_b = self.l2_norm(B)

        _A = A / mag_a if mag_a > 0 else A
        _B = B / mag_b if mag_b > 0 else B

        cross_product = self.cross(_A, _B)
        denom = np.sum(cross_product ** 2)

        # If lines are nearly parallel
        if denom < 1e-10:
            d0 = self.dot(_A, b0 - a0)

            if clamp_a0 or clamp_a1 or clamp_b0 or clamp_b1:
                d1 = self.dot(_A, b1 - a0)

                if d0 <= 0 and d1 <= 0:
                    if clamp_a0 and clamp_b1:
                        if abs(d0) < abs(d1):
                            return a0, b0, self.l2_norm(a0 - b0)
                        return a0, b1, self.l2_norm(a0 - b1)
                elif d0 >= mag_a and d1 >= mag_a:
                    if clamp_a1 and clamp_b0:
                        if abs(d0) < abs(d1):
                            return a1, b0, self.l2_norm(a1 - b0)
                        return a1, b1, self.l2_norm(a1 - b1)

            diff = (_A * d0) + a0 - b0
            return a0, b0, self.l2_norm(diff)

        t = b0 - a0
        c1 = self.dot(_A, _A)
        c2 = self.dot(_A, _B)
        c3 = self.dot(_B, _B)
        c4 = self.dot(_A, t)
        c5 = self.dot(_B, t)

        den = c1 * c3 - c2 * c2
        if abs(den) < 1e-10:
            t0 = 0
            t1 = self.dot(_B, a0 - b0) / c3 if c3 > 1e-10 else 0
        else:
            t0 = (c2 * c5 - c3 * c4) / den
            t1 = (c1 * c5 - c2 * c4) / den

        p_a = a0 + (_A * t0)
        p_b = b0 + (_B * t1)

        # Skip clamping if no constraints
        if not (clamp_a0 or clamp_a1 or clamp_b0 or clamp_b1):
            return p_a, p_b, self.l2_norm(p_a - p_b)

        # First pass: Clamp points directly to endpoints
        if clamp_a0 and t0 < 0:
            p_a = a0
        if clamp_a1 and t0 > mag_a:
            p_a = a1
        if clamp_b0 and t1 < 0:
            p_b = b0
        if clamp_b1 and t1 > mag_b:
            p_b = b1

        # Second pass: Project point B onto segment B after clamping A
        if (clamp_a0 and t0 < 0) or (clamp_a1 and t0 > mag_a):
            dot_b = self.dot(_B, p_a - b0)
            dot_b = max(0.0 if clamp_b0 else float('-inf'),
                        min(mag_b if clamp_b1 else float('inf'), dot_b))
            p_b = b0 + (_B * dot_b)

        # Third pass: Project point A onto segment A after clamping B
        if (clamp_b0 and t1 < 0) or (clamp_b1 and t1 > mag_b):
            dot_a = self.dot(_A, p_b - a0)
            dot_a = max(0.0 if clamp_a0 else float('-inf'),
                        min(mag_a if clamp_a1 else float('inf'), dot_a))
            p_a = a0 + (_A * dot_a)

        return p_a, p_b, self.l2_norm(p_a - p_b)

    def get_collision_info(self, p1, p2, q1, q2, diameter1, diameter2):
        """Get collision information between two spherocylinders."""
        import DCPQuery  # Assuming DCPQuery is a module that provides the required functionality
        dist_min, closest_p1, closest_p2, s, t = DCPQuery.segment_segment_distance(
            p1, p2, q1, q2)

        sep = dist_min-(diameter1 + diameter2) / 2.0

        BUFFER = 0.01
        return {
            "collision": sep < BUFFER,
            "sep": sep,
            "closest_p1": closest_p1,
            "closest_p2": closest_p2,
            "normal": self.normalize(closest_p1 - closest_p2),
        }

    def get_endpoints(self, pos, dir_vec, length):
        """Get the endpoints of a spherocylinder."""
        dir_norm = self.normalize(dir_vec)
        half_length = length / 2.0
        p1 = pos - dir_norm * half_length
        p2 = pos + dir_norm * half_length
        return p1, p2

    def draw_spherocylinder(self, pos, dir_vec, length, diameter, color):
        """Draw a spherocylinder in 3D."""
        dir_norm = self.normalize(dir_vec)
        half_length = length / 2.0

        # Calculate endpoints
        p1 = pos - dir_norm * half_length
        p2 = pos + dir_norm * half_length

        # Draw the cylindrical part
        radius = diameter / 2.0

        # Create a perpendicular vector to the direction
        if np.allclose(dir_norm, [0, 0, 1]) or np.allclose(dir_norm, [0, 0, -1]):
            perp1 = np.array([1, 0, 0])
        else:
            perp1 = self.normalize(self.cross(dir_norm, [0, 0, 1]))

        perp2 = self.cross(dir_norm, perp1)

        # Create cylinder using wireframe
        theta = np.linspace(0, 2 * np.pi, 20)
        segments = 10  # Number of segments along the cylinder

        for i in range(segments + 1):
            fraction = i / segments
            center = p1 * (1 - fraction) + p2 * fraction

            # Create a circle at this position
            x_circle = []
            y_circle = []
            z_circle = []

            for t in theta:
                x = center[0] + radius * \
                    (perp1[0] * np.cos(t) + perp2[0] * np.sin(t))
                y = center[1] + radius * \
                    (perp1[1] * np.cos(t) + perp2[1] * np.sin(t))
                z = center[2] + radius * \
                    (perp1[2] * np.cos(t) + perp2[2] * np.sin(t))

                x_circle.append(x)
                y_circle.append(y)
                z_circle.append(z)

            # Close the circle
            x_circle.append(x_circle[0])
            y_circle.append(y_circle[0])
            z_circle.append(z_circle[0])

            # Plot this circle
            self.ax.plot(x_circle, y_circle, z_circle,
                         color=color, alpha=0.5, linewidth=1)

        # Add some lines along the cylinder length
        for t in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x_line = []
            y_line = []
            z_line = []

            for i in range(segments + 1):
                fraction = i / segments
                center = p1 * (1 - fraction) + p2 * fraction

                x = center[0] + radius * \
                    (perp1[0] * np.cos(t) + perp2[0] * np.sin(t))
                y = center[1] + radius * \
                    (perp1[1] * np.cos(t) + perp2[1] * np.sin(t))
                z = center[2] + radius * \
                    (perp1[2] * np.cos(t) + perp2[2] * np.sin(t))

                x_line.append(x)
                y_line.append(y)
                z_line.append(z)

            self.ax.plot(x_line, y_line, z_line,
                         color=color, alpha=0.5, linewidth=1)

        # Draw the spheres at endpoints using wireframe instead of surface
        # This avoids the compatibility issues with plot_surface
        for center in [p1, p2]:
            # Create a wireframe sphere
            for i in range(10):  # Latitude
                theta = i * np.pi / 10
                x_circle = []
                y_circle = []
                z_circle = []

                for j in range(21):  # Longitude
                    phi = j * 2 * np.pi / 20
                    x = center[0] + radius * np.sin(theta) * np.cos(phi)
                    y = center[1] + radius * np.sin(theta) * np.sin(phi)
                    z = center[2] + radius * np.cos(theta)

                    x_circle.append(x)
                    y_circle.append(y)
                    z_circle.append(z)

                self.ax.plot(x_circle, y_circle, z_circle,
                             color=color, alpha=0.5, linewidth=1)

            for j in range(10):  # Longitude
                phi = j * 2 * np.pi / 10
                x_circle = []
                y_circle = []
                z_circle = []

                for i in range(11):  # Latitude
                    theta = i * np.pi / 10
                    x = center[0] + radius * np.sin(theta) * np.cos(phi)
                    y = center[1] + radius * np.sin(theta) * np.sin(phi)
                    z = center[2] + radius * np.cos(theta)

                    x_circle.append(x)
                    y_circle.append(y)
                    z_circle.append(z)

                self.ax.plot(x_circle, y_circle, z_circle,
                             color=color, alpha=0.5, linewidth=1)

        # Draw the axis line
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [
                     p1[2], p2[2]], color='black', linewidth=1)

        return p1, p2

    def update(self, val):
        self.ax.clear()

        # Get values from sliders
        self.sc1_pos = np.array(
            [self.slider_x1.val, self.slider_y1.val, self.slider_z1.val])
        self.sc2_pos = np.array(
            [self.slider_x2.val, self.slider_y2.val, self.slider_z2.val])

        # Get direction vectors and normalize them
        self.sc1_dir = np.array(
            [self.slider_dx1.val, self.slider_dy1.val, self.slider_dz1.val])
        self.sc2_dir = np.array(
            [self.slider_dx2.val, self.slider_dy2.val, self.slider_dz2.val])

        if self.l2_norm(self.sc1_dir) > 0:
            self.sc1_dir = self.sc1_dir / self.l2_norm(self.sc1_dir)
        if self.l2_norm(self.sc2_dir) > 0:
            self.sc2_dir = self.sc2_dir / self.l2_norm(self.sc2_dir)

        self.sc1_length = self.slider_len1.val
        self.sc1_diameter = self.slider_diam1.val
        self.sc2_length = self.slider_len2.val
        self.sc2_diameter = self.slider_diam2.val

        # Draw the spherocylinders
        p1, p2 = self.draw_spherocylinder(
            self.sc1_pos, self.sc1_dir, self.sc1_length, self.sc1_diameter, 'blue')
        q1, q2 = self.draw_spherocylinder(
            self.sc2_pos, self.sc2_dir, self.sc2_length, self.sc2_diameter, 'red')

        # Compute collision
        collision_info = self.get_collision_info(
            p1, p2, q1, q2, self.sc1_diameter, self.sc2_diameter)

        # Set up the plot limits
        max_val = max(
            np.max(np.abs(self.sc1_pos)) + self.sc1_length + self.sc1_diameter,
            np.max(np.abs(self.sc2_pos)) + self.sc2_length + self.sc2_diameter,
            5.0
        )
        self.ax.set_xlim(-max_val, max_val)
        self.ax.set_ylim(-max_val, max_val)
        self.ax.set_zlim(-max_val, max_val)

        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Spherocylinder Collision Visualization')

        # Set equal aspect ratio for all axes
        self.ax.set_box_aspect([1, 1, 1])

        info_text = ""

        # Highlight collision points
        self.ax.scatter([collision_info["closest_p1"][0]], [collision_info["closest_p1"][1]],
                        [collision_info["closest_p1"][2]], color='green', s=100, label='Point on SC1')
        self.ax.scatter([collision_info["closest_p2"][0]], [collision_info["closest_p2"][1]],
                        [collision_info["closest_p2"][2]], color='yellow', s=100, label='Point on SC2')

        # Draw a line connecting collision points
        self.ax.plot([collision_info["closest_p1"][0], collision_info["closest_p2"][0]],
                     [collision_info["closest_p1"][1],
                      collision_info["closest_p2"][1]],
                     [collision_info["closest_p1"][2],
                      collision_info["closest_p2"][2]],
                     'k--', linewidth=1)

        # Show collision info as text
        info_text = f"Collision Detected!\n\n" if collision_info['collision'] else f"No Collision Detected!\n\n"   \
            f"sep: {collision_info['sep']:.4f}\n" \
            f"Distance: {self.l2_norm(collision_info['closest_p1'] - collision_info['closest_p2']):.4f}\n" \
            f"Required distance: {(self.sc1_diameter + self.sc2_diameter) / 2:.4f}\n\n"

        # Add a legend
        self.ax.legend()

        # Draw the info box
        bbox_props = dict(boxstyle="round,pad=0.5",
                          facecolor="wheat", alpha=0.7)
        plt.figtext(0.02, 0.5, info_text, va="center",
                    ha="left", bbox=bbox_props, fontsize=9)

        # Use draw() instead of draw_idle() for more reliability
        self.fig.canvas.draw()

    def reset(self, event):
        # Reset all sliders to initial values
        self.slider_x1.reset()
        self.slider_y1.reset()
        self.slider_z1.reset()
        self.slider_dx1.reset()
        self.slider_dy1.reset()
        self.slider_dz1.reset()
        self.slider_x2.reset()
        self.slider_y2.reset()
        self.slider_z2.reset()
        self.slider_dx2.reset()
        self.slider_dy2.reset()
        self.slider_dz2.reset()
        self.slider_len1.reset()
        self.slider_diam1.reset()
        self.slider_len2.reset()
        self.slider_diam2.reset()
        self.update(None)


if __name__ == "__main__":
    visualizer = SpherocylinderVisualizer()
