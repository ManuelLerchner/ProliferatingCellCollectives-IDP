import matplotlib as mpl
# patches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def draw_particles(data, ax=None, show=True):
    """
    Draws particles as ellipses with orientation and length.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: ['x', 'y', 'orientation_angle', 'lengths_x']
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure/axis is created.
    show : bool, optional
        If True, updates the plot interactively. If False, just returns fig, ax.

    Returns
    -------
    fig, ax : tuple
        The matplotlib figure and axis objects.
    """

    # Create new fig/ax if none is given
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
        ax.clear()

    ax.set_aspect('equal')

    # color scale from 1 to 2 - dark green to light green
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'custom_cmap', [(0.2, 0.3, 0), (0, 1, 0)]
    )
    norm = mpl.colors.Normalize(vmin=1, vmax=2)

    for _, row in data.iterrows():
        x, y = row['x'], row['y']
        angle = row['orientation_angle'] * 180 / np.pi
        length = row['length']

        fill_color = cmap(norm(length))

        ellipse = mpl.patches.Ellipse(
            (x, y),
            width=length,
            height=0.5,
            angle=angle,
            fill=True,
            fc=fill_color,
            ec=None
        )
        ax.add_patch(ellipse)

    # reset x and y limits
    ax.set_xlim(data['x'].min() - 5, data['x'].max() + 5)
    ax.set_ylim(data['y'].min() - 5, data['y'].max() + 5)

    if show:
        plt.pause(0.01)

    return fig, ax


def draw_particles_cluster(data, colors):
    """
    Draw particles as ellipses using positions from a DataFrame,
    colored according to a precomputed RGB color array.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns ['x', 'y', 'orientation_angle', 'lengths_x']
    colors : np.ndarray
        Nx3 array of RGB colors (0..1)

    Returns
    -------
    fig, ax : tuple
    """
    data = data.reset_index(drop=True)
    positions = data[['x', 'y']].values
    n_cells = len(data)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    for i in range(n_cells):
        x = data.iloc[i]['x']
        y = data.iloc[i]['y']
        angle = data.iloc[i]['orientation_angle'] * 180 / np.pi
        length = data.iloc[i]['length']

        fill_color = tuple(colors[i])

        ellipse = patches.Ellipse(
            (x, y),
            width=length,
            height=0.5,
            angle=angle,
            fc=fill_color,
            ec=None,
            fill=True
        )
        ax.add_patch(ellipse)

    ax.set_xlim(positions[:, 0].min() - 5, positions[:, 0].max() + 5)
    ax.set_ylim(positions[:, 1].min() - 5, positions[:, 1].max() + 5)

    return fig, ax
