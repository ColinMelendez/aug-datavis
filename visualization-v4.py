import numpy as np
import plotly.graph_objects as go


def create_solid_color_segments(X_arr, Y_arr, Z_arr, axis, z_split_val):
    """
    Create simplified line segments that are split at the boundary.
    Returns 4 lists of coordinates for different color combinations:
    - x_below, y_below, z_below: lines below the split plane
    - x_above, y_above, z_above: lines above the split plane
    """
    # axis=1 for X-direction (along columns), axis=0 for Y-direction (along rows)
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

    # Initialize coordinate lists for below and above segments
    below_coords = {"x": [], "y": [], "z": []}
    above_coords = {"x": [], "y": [], "z": []}

    # Check which segments cross the boundary
    below_0 = z0 < z_split_val
    below_1 = z1 < z_split_val
    same_side = below_0 == below_1
    cross_mask = ~same_side

    # Process segments that don't cross the boundary
    same_idx = np.where(same_side)[0]
    for idx in same_idx:
        if below_0[idx]:  # Both points are below
            below_coords["x"].extend([x0[idx], x1[idx], None])
            below_coords["y"].extend([y0[idx], y1[idx], None])
            below_coords["z"].extend([z0[idx], z1[idx], None])
        else:  # Both points are above
            above_coords["x"].extend([x0[idx], x1[idx], None])
            above_coords["y"].extend([y0[idx], y1[idx], None])
            above_coords["z"].extend([z0[idx], z1[idx], None])

    # Process segments that cross the boundary
    cross_idx = np.where(cross_mask)[0]
    if len(cross_idx) > 0:
        x0_c, x1_c = x0[cross_idx], x1[cross_idx]
        y0_c, y1_c = y0[cross_idx], y1[cross_idx]
        z0_c, z1_c = z0[cross_idx], z1[cross_idx]

        # Calculate crossing points using linear interpolation
        t = (z_split_val - z0_c) / (z1_c - z0_c)
        x_cross = x0_c + t * (x1_c - x0_c)
        y_cross = y0_c + t * (y1_c - y0_c)

        # Split each crossing segment into two non-crossing segments
        for i in range(len(cross_idx)):
            # First segment (from starting point to boundary)
            if below_0[cross_idx[i]]:  # Starts below, goes above
                below_coords["x"].extend([x0_c[i], x_cross[i], None])
                below_coords["y"].extend([y0_c[i], y_cross[i], None])
                below_coords["z"].extend([z0_c[i], z_split_val, None])
                above_coords["x"].extend([x_cross[i], x1_c[i], None])
                above_coords["y"].extend([y_cross[i], y1_c[i], None])
                above_coords["z"].extend([z_split_val, z1_c[i], None])
            else:  # Starts above, goes below
                above_coords["x"].extend([x0_c[i], x_cross[i], None])
                above_coords["y"].extend([y0_c[i], y_cross[i], None])
                above_coords["z"].extend([z0_c[i], z_split_val, None])
                below_coords["x"].extend([x_cross[i], x1_c[i], None])
                below_coords["y"].extend([y_cross[i], y1_c[i], None])
                below_coords["z"].extend([z_split_val, z1_c[i], None])

    return below_coords, above_coords


def plot_data(
    X,
    Y,
    Z,
    z_split_pct=0.6,
    line_width=3,
):
    """
    Create a 3D wireframe plot with solid colors split at a percentage of z-axis using Plotly.

    Uses 4 solid colors: X-direction (green/orange), Y-direction (blue/red) below/above boundary.

    Parameters:
        X, Y, Z: 2D arrays from meshgrid defining the surface
        z_split_pct: percentage (0-1) along z-axis where color changes (default 0.6)
        line_width: width of the lines (default 3)
    """
    # Calculate split values
    z_min, z_max = Z.min(), Z.max()
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_split_val = z_min + z_split_pct * (z_max - z_min)

    print(f"Z-axis range: {z_min:.2f} to {z_max:.2f}")
    print(f"{z_split_pct * 100:.0f}% split point: z = {z_split_val:.2f}")

    # Process X-direction and Y-direction segments
    x_below, x_above = create_solid_color_segments(
        X, Y, Z, axis=1, z_split_val=z_split_val
    )
    y_below, y_above = create_solid_color_segments(
        X, Y, Z, axis=0, z_split_val=z_split_val
    )

    # Create Plotly figure
    fig = go.Figure()

    # Add 4 solid color traces for different direction/boundary combinations
    if len(x_below["x"]) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=x_below["x"],
                y=x_below["y"],
                z=x_below["z"],
                mode="lines",
                line=dict(color="green", width=line_width),
                name="X-direction (below)",
                showlegend=True,
                hoverinfo="none",
            )
        )

    if len(x_above["x"]) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=x_above["x"],
                y=x_above["y"],
                z=x_above["z"],
                mode="lines",
                line=dict(color="orange", width=line_width),
                name="X-direction (above)",
                showlegend=True,
                hoverinfo="none",
            )
        )

    if len(y_below["x"]) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=y_below["x"],
                y=y_below["y"],
                z=y_below["z"],
                mode="lines",
                line=dict(color="blue", width=line_width),
                name="Y-direction (below)",
                showlegend=True,
                hoverinfo="none",
            )
        )

    if len(y_above["x"]) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=y_above["x"],
                y=y_above["y"],
                z=y_above["z"],
                mode="lines",
                line=dict(color="red", width=line_width),
                name="Y-direction (above)",
                showlegend=True,
                hoverinfo="none",
            )
        )

    # Add split plane using a mesh
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    zz = np.ones_like(xx) * z_split_val

    fig.add_trace(
        go.Mesh3d(
            x=xx,
            y=yy,
            z=zz,
            opacity=0.15,
            color="lightgray",
            name="Split plane",
            showlegend=False,
        )
    )

    # Add split plane trace for legend
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=10, color="gray", opacity=0.3),
            name=f"{z_split_pct * 100:.0f}% Z split plane",
            showlegend=True,
        )
    )

    # Update layout
    fig.update_layout(
        title={
            "text": f"3D Wireframe with Solid Colors<br>Color split at {z_split_pct * 100:.0f}% Z-axis",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 14},
        },
        scene=dict(
            xaxis=dict(title="X Axis", tickfont={"size": 12}, range=[x_min, x_max]),
            yaxis=dict(title="Y Axis", tickfont={"size": 12}, range=[y_min, y_max]),
            zaxis=dict(title="Z Axis", tickfont={"size": 12}, range=[z_min, z_max]),
            camera=dict(
                eye=dict(
                    x=1.2, y=1.2, z=0.8
                )  # Similar to matplotlib's elev=25, azim=45
            ),
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Show the plot
    print(" Using Plotly backend with WebGL rendering")
    fig.show()

    # Return figure for any further manipulation
    return fig


# Sample data --------------------------------------------------------------------

# Generate sample data
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

amplitude_scale = 0.3 + 0.7 * (X - x.min()) / (x.max() - x.min())
Z = amplitude_scale * (
    np.cos(2 * np.pi * X) * (1 + 0.3 * Y)
    + 0.6 * np.cos(1.5 * np.pi * Y + 0.5 * X)
    + 0.3 * np.sin(3 * X * Y)
)

fig = plot_data(X, Y, Z)
