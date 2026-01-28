import matplotlib

# for better graph outputs than the default rendering backend
# (must be called before importing everything else)
matplotlib.use("QtAgg")

from matplotlib.collections import PolyCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def depth_sort_segments(segments, colors, ax):
    """Sort segments by distance from camera (back-to-front)."""
    elev = np.deg2rad(ax.elev)
    azim = np.deg2rad(ax.azim)

    # Camera direction vector (pointing toward the camera)
    cam_dir = np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )

    # Compute centroid of each segment and project onto camera direction
    centroids = segments.mean(axis=1)
    depths = centroids @ cam_dir

    # Sort back-to-front (farthest first, so they get drawn first/behind)
    sort_idx = np.argsort(depths)

    return segments[sort_idx], colors[sort_idx]


def plot_data(
    X,
    Y,
    Z,
    z_split_pct=0.6,
    cmap_x_below=LinearSegmentedColormap.from_list("custom", ["#00BFFF", "#0000CD"]),
    cmap_x_above=LinearSegmentedColormap.from_list("custom", ["#FFD700", "#DC143C"]),
    cmap_y_below=LinearSegmentedColormap.from_list("custom", ["#7FFF00", "#006400"]),
    cmap_y_above=LinearSegmentedColormap.from_list("custom", ["#FFA07A", "#FF4500"]),
    enable_depth_sort=True,
    line_width=1.5,
):
    """
    Create a 3D wireframe plot with color zones split at a percentage of the z-axis.

    Color gradients vary in the z-axis, with a hard split at the split plane.

    Parameters:
        X, Y, Z: 2D arrays from meshgrid defining the surface
        z_split_pct: percentage (0-1) along z-axis where color changes (default 0.6)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Calculate and cache all min/max values
    z_min, z_max = Z.min(), Z.max()
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_split_val = z_min + z_split_pct * (z_max - z_min)

    print(f"Z-axis range: {z_min:.2f} to {z_max:.2f}")
    print(f"{z_split_pct * 100:.0f}% split point: z = {z_split_val:.2f}")

    # Pre-calculate frequently used values (avoid division by zero)
    z_split_range = z_split_val - z_min if z_split_val > z_min else 1.0
    z_upper_range = z_max - z_split_val if z_max > z_split_val else 1.0

    def process_direction_segments(X_arr, Y_arr, Z_arr, axis, cmap_below, cmap_above):
        """
        Process all segments for one direction using fully vectorized operations.
        axis=1 for X-direction (along columns), axis=0 for Y-direction (along rows)
        """
        if axis == 1:
            # X-direction: shape (rows, cols) -> segments along cols
            x0 = X_arr[:, :-1].ravel()
            x1 = X_arr[:, 1:].ravel()
            y0 = Y_arr[:, :-1].ravel()
            y1 = Y_arr[:, 1:].ravel()
            z0 = Z_arr[:, :-1].ravel()
            z1 = Z_arr[:, 1:].ravel()
        else:
            # Y-direction: shape (rows, cols) -> segments along rows
            x0 = X_arr[:-1, :].ravel()
            x1 = X_arr[1:, :].ravel()
            y0 = Y_arr[:-1, :].ravel()
            y1 = Y_arr[1:, :].ravel()
            z0 = Z_arr[:-1, :].ravel()
            z1 = Z_arr[1:, :].ravel()

        n_segments = len(x0)

        # Vectorized threshold checking
        below_0 = z0 < z_split_val
        below_1 = z1 < z_split_val
        same_side = below_0 == below_1
        cross_mask = ~same_side

        # Count segments (crossing segments produce 2 each)
        n_cross = np.sum(cross_mask)
        n_same = n_segments - n_cross
        total_output = n_same + 2 * n_cross

        # Preallocate output arrays
        segments = np.empty((total_output, 2, 3), dtype=np.float64)
        colors = np.empty((total_output, 4), dtype=np.float64)

        # Process same-side segments (vectorized)
        if n_same > 0:
            same_idx = np.where(same_side)[0]
            z_mid = (z0[same_idx] + z1[same_idx]) / 2
            is_below = below_0[same_idx]

            # Compute t values for color mapping
            t_below = np.clip((z_mid - z_min) / z_split_range, 0, 1)
            t_above = np.clip((z_mid - z_split_val) / z_upper_range, 0, 1)
            t_color = np.where(is_below, t_below, t_above)

            # Get colors from appropriate colormaps
            colors_below = cmap_below(t_color)
            colors_above = cmap_above(t_color)
            same_colors = np.where(is_below[:, np.newaxis], colors_below, colors_above)

            # Build segments
            segments[:n_same, 0, 0] = x0[same_idx]
            segments[:n_same, 0, 1] = y0[same_idx]
            segments[:n_same, 0, 2] = z0[same_idx]
            segments[:n_same, 1, 0] = x1[same_idx]
            segments[:n_same, 1, 1] = y1[same_idx]
            segments[:n_same, 1, 2] = z1[same_idx]
            colors[:n_same] = same_colors

        # Process crossing segments (vectorized)
        if n_cross > 0:
            cross_idx = np.where(cross_mask)[0]
            x0_c, x1_c = x0[cross_idx], x1[cross_idx]
            y0_c, y1_c = y0[cross_idx], y1[cross_idx]
            z0_c, z1_c = z0[cross_idx], z1[cross_idx]
            below_0_c = below_0[cross_idx]

            # Calculate crossing points
            t = (z_split_val - z0_c) / (z1_c - z0_c)
            x_cross = x0_c + t * (x1_c - x0_c)
            y_cross = y0_c + t * (y1_c - y0_c)

            # Compute colors for both sub-segments
            z_mid_1 = (z0_c + z_split_val) / 2
            z_mid_2 = (z_split_val + z1_c) / 2

            t_below_1 = np.clip((z_mid_1 - z_min) / z_split_range, 0, 1)
            t_above_1 = np.clip((z_mid_1 - z_split_val) / z_upper_range, 0, 1)
            t_below_2 = np.clip((z_mid_2 - z_min) / z_split_range, 0, 1)
            t_above_2 = np.clip((z_mid_2 - z_split_val) / z_upper_range, 0, 1)

            # First sub-segment colors
            t_color_1 = np.where(below_0_c, t_below_1, t_above_1)
            colors_1_below = cmap_below(t_color_1)
            colors_1_above = cmap_above(t_color_1)
            cross_colors_1 = np.where(
                below_0_c[:, np.newaxis], colors_1_below, colors_1_above
            )

            # Second sub-segment colors (opposite side)
            t_color_2 = np.where(~below_0_c, t_below_2, t_above_2)
            colors_2_below = cmap_below(t_color_2)
            colors_2_above = cmap_above(t_color_2)
            cross_colors_2 = np.where(
                ~below_0_c[:, np.newaxis], colors_2_below, colors_2_above
            )

            # Build crossing segments
            base_idx = n_same
            for i in range(n_cross):
                idx1 = base_idx + 2 * i
                idx2 = base_idx + 2 * i + 1

                segments[idx1, 0] = [x0_c[i], y0_c[i], z0_c[i]]
                segments[idx1, 1] = [x_cross[i], y_cross[i], z_split_val]
                colors[idx1] = cross_colors_1[i]

                segments[idx2, 0] = [x_cross[i], y_cross[i], z_split_val]
                segments[idx2, 1] = [x1_c[i], y1_c[i], z1_c[i]]
                colors[idx2] = cross_colors_2[i]

        return segments, colors

    # Process X-direction and Y-direction segments
    x_segments, x_colors = process_direction_segments(
        X, Y, Z, axis=1, cmap_below=cmap_x_below, cmap_above=cmap_x_above
    )
    y_segments, y_colors = process_direction_segments(
        X, Y, Z, axis=0, cmap_below=cmap_y_below, cmap_above=cmap_y_above
    )

    # Combine all segments
    all_segments = np.concatenate([x_segments, y_segments], axis=0)
    all_colors = np.concatenate([x_colors, y_colors], axis=0)

    # Apply depth sorting to the first/static renders
    sorted_segments, sorted_colors = depth_sort_segments(all_segments, all_colors, ax)

    # Create a single Line3DCollection for all segments
    line_collection = Line3DCollection(
        sorted_segments, colors=sorted_colors, linewidths=line_width
    )
    ax.add_collection3d(line_collection)

    # Set up dynamic depth sorting on view changes
    if enable_depth_sort:

        def on_draw(_event):
            nonlocal all_segments, all_colors
            sorted_segs, sorted_cols = depth_sort_segments(all_segments, all_colors, ax)
            line_collection.set_segments(sorted_segs)
            line_collection.set_colors(sorted_cols)

        fig.canvas.mpl_connect("draw_event", on_draw)

    # Set axis limits (using cached values)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
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

    # Add a horizontal plane indicator at the split point (using cached values)
    verts = [[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]]
    plane = PolyCollection(verts, alpha=0.15, facecolors="gray", edgecolors="none")
    ax.add_collection3d(plane, zs=z_split_val, zdir="z")

    # Add legend showing gradient endpoints
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=cmap_x_below(0.6),
            lw=2,
            marker="o",
            markersize=4,
            markerfacecolor=cmap_x_below(0.6),
            label="X below",
        ),
        Line2D(
            [0],
            [0],
            color=cmap_x_above(0.6),
            lw=2,
            marker="o",
            markersize=4,
            markerfacecolor=cmap_x_above(0.6),
            label="X above",
        ),
        Line2D(
            [0],
            [0],
            color=cmap_y_below(0.6),
            lw=2,
            marker="o",
            markersize=4,
            markerfacecolor=cmap_y_below(0.6),
            label="Y below",
        ),
        Line2D(
            [0],
            [0],
            color=cmap_y_above(0.6),
            lw=2,
            marker="o",
            markersize=4,
            markerfacecolor=cmap_y_above(0.6),
            label="Y above",
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
