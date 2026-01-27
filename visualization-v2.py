import matplotlib

matplotlib.use("QtAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_data(
    X,
    Y,
    Z,
    z_split_pct=0.6,
    bottom_hue_range=(0.3, 0.6),
    top_hue_range=(0.5, 0.8),
    cmap_x_below=cm.Greens,
    cmap_x_above=cm.Oranges,
    cmap_y_below=cm.Blues,
    cmap_y_above=cm.Reds,
):
    """
    Create a 3D wireframe plot with color zones split at a percentage of the z-axis.

    Hue gradients vary in the z-axis, with a hard color change at the split plane.

    Parameters:
        X, Y, Z: 2D arrays from meshgrid defining the surface
        z_split_pct: percentage (0-1) along z-axis where color changes (default 0.6)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Calculate the z-axis split value
    z_min, z_max = Z.min(), Z.max()
    z_split_val = z_min + z_split_pct * (z_max - z_min)

    print(f"Z-axis range: {z_min:.2f} to {z_max:.2f}")
    print(f"{z_split_pct * 100:.0f}% split point: z = {z_split_val:.2f}")

    # Collect all segments and colors for batch rendering
    all_segments = []
    all_colors = []

    def get_gradient_color(z_val, is_below, cmap_below, cmap_above):
        """
        Get color from gradient based on z position within region.
        """
        if is_below:
            t = (z_val - z_min) / (z_split_val - z_min) if z_split_val > z_min else 0.5
            # Normalize within [z_min, z_split_val] -> bottom_hue_range
            return cmap_below(bottom_hue_range[0] + bottom_hue_range[1] * t)
        else:
            t = (
                (z_val - z_split_val) / (z_max - z_split_val)
                if z_max > z_split_val
                else 0.5
            )
            # Normalize within [z_split_val, z_max] -> top_hue_range
            return cmap_above(top_hue_range[0] + top_hue_range[1] * t)

    def collect_line_segments(x_coords, y_coords, z_coords, cmap_below, cmap_above):
        """
        Collect line segments with colors for batch rendering.
        """
        for i in range(len(x_coords) - 1):
            x0, x1 = x_coords[i], x_coords[i + 1]
            y0, y1 = y_coords[i], y_coords[i + 1]
            z0, z1 = z_coords[i], z_coords[i + 1]

            below_0 = z0 < z_split_val
            below_1 = z1 < z_split_val

            if below_0 == below_1:
                # Entire segment on one side - use midpoint for color
                z_mid = (z0 + z1) / 2
                color = get_gradient_color(z_mid, below_0, cmap_below, cmap_above)
                all_segments.append([(x0, y0, z0), (x1, y1, z1)])
                all_colors.append(color)
            else:
                # Segment crosses threshold - interpolate crossing point
                t = (z_split_val - z0) / (z1 - z0)
                x_cross = x0 + t * (x1 - x0)
                y_cross = y0 + t * (y1 - y0)
                z_cross = z_split_val

                # Collect both segments with gradient colors
                if below_0:
                    color1 = get_gradient_color(
                        (z0 + z_cross) / 2, True, cmap_below, cmap_above
                    )
                    color2 = get_gradient_color(
                        (z_cross + z1) / 2, False, cmap_below, cmap_above
                    )
                else:
                    color1 = get_gradient_color(
                        (z0 + z_cross) / 2, False, cmap_below, cmap_above
                    )
                    color2 = get_gradient_color(
                        (z_cross + z1) / 2, True, cmap_below, cmap_above
                    )
                all_segments.append([(x0, y0, z0), (x_cross, y_cross, z_cross)])
                all_colors.append(color1)
                all_segments.append([(x_cross, y_cross, z_cross), (x1, y1, z1)])
                all_colors.append(color2)

    # Collect all segments
    for i in range(X.shape[0]):
        collect_line_segments(X[i, :], Y[i, :], Z[i, :], cmap_x_below, cmap_x_above)
    for j in range(X.shape[1]):
        collect_line_segments(X[:, j], Y[:, j], Z[:, j], cmap_y_below, cmap_y_above)

    # Create a single Line3DCollection for all segments
    line_collection = Line3DCollection(all_segments, colors=all_colors, linewidths=1.5)
    ax.add_collection3d(line_collection)

    # Set axis limits
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(z_min, z_max)

    # Add labels and title
    ax.set_xlabel("X Axis", fontsize=12)
    ax.set_ylabel("Y Axis", fontsize=12)
    ax.set_zlabel("Z Axis", fontsize=12)
    ax.set_title(
        f"3D Wireframe with Gradient Colors\nColor split at {z_split_pct * 100:.0f}% Z-axis",
        fontsize=14,
        pad=20,
    )

    # Add a horizontal plane indicator at the split point
    verts = [
        [(X.min(), Y.min()), (X.max(), Y.min()), (X.max(), Y.max()), (X.min(), Y.max())]
    ]
    plane = PolyCollection(verts, alpha=0.15, facecolors="gray", edgecolors="none")
    ax.add_collection3d(plane, zs=z_split_val, zdir="z")

    # Add legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=cmap_x_below(0.6),
            lw=2,
            label="X-direction (below)",
        ),
        Line2D(
            [0],
            [0],
            color=cmap_x_above(0.6),
            lw=2,
            label="X-direction (above)",
        ),
        Line2D(
            [0],
            [0],
            color=cmap_y_below(0.6),
            lw=2,
            label="Y-direction (below)",
        ),
        Line2D(
            [0],
            [0],
            color=cmap_y_above(0.6),
            lw=2,
            label="Y-direction (above)",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=2,
            alpha=0.3,
            label=f"{z_split_pct * 100:.0f}% Z split plane",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Set viewing angle
    ax.view_init(elev=25, azim=45)

    # display the plot
    plt.tight_layout()

    # Check for QtAgg backend
    backend = matplotlib.get_backend()
    print(f" Using backend: {backend}")

    plt.show()

    # return figure and axis references for any further manipulation
    return fig, ax


# Generate data for parallel cosine curves with more diverse shapes
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Amplitude scaling factor: starts small on left, grows toward right
# Maps X from [-2, 2] to [0.3, 1.0]
amplitude_scale = 0.3 + 0.7 * (X - x.min()) / (x.max() - x.min())

# Create cosine curves with amplitude that grows along X
Z = amplitude_scale * (
    np.cos(2 * np.pi * X) * (1 + 0.3 * Y)
    + 0.6 * np.cos(1.5 * np.pi * Y + 0.5 * X)
    + 0.3 * np.sin(3 * X * Y)
)

print(f"Data shapes - X: {X.shape}, Y: {Y.shape}, Z: {Z.shape}")
print(f"Z range: {Z.min():.2f} to {Z.max():.2f}")
print(
    f"Amplitude scale range: {amplitude_scale.min():.2f} to {amplitude_scale.max():.2f}"
)

# Call the plotting function
fig, ax = plot_data(X, Y, Z)
