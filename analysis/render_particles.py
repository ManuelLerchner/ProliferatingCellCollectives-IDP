import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

# --- Visualization setup ---
fig, ax = plt.subplots(figsize=(12, 12))

ax.set_aspect('equal')


def draw_particles(data):
    ax.clear()

    ax.set_aspect('equal')

    # color scale from 1 to 2 - dark green to light green
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'custom_cmap', [(0.2, 0.3, 0), (0, 1, 0)])
    norm = mpl.colors.Normalize(vmin=1, vmax=2)

    for i, row in data.iterrows():
        x, y = row['x'], row['y']
        angle = row['orientation_angle'] * 180 / np.pi
        length = row['lengths_x']

        fill_color = cmap(norm(row['lengths_x']))

        # ellipse
        ellipse = mpl.patches.Ellipse(
            (x, y), length, 0.5, angle=angle, fill=True, fc=fill_color, ec=None)
        ax.add_patch(ellipse)

    # reset x and y limits
    ax.set_xlim(data['x'].min()-5, data['x'].max()+5)
    ax.set_ylim(data['y'].min()-5, data['y'].max()+5)

    plt.pause(0.01)
