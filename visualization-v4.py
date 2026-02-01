import numpy as np
import plotly.graph_objects as go


def create_split_segments(X_arr, Y_arr, Z_arr, axis, z_split_val):
    """
    Create line segments split at the boundary into below/above groups.
    Returns two tuples: (below_x, below_y, below_z), (above_x, above_y, above_z)
    Each with None separators for Plotly.
    """
    if axis == 1:
        x0, x1 = X_arr[:, :-1].ravel(), X_arr[:, 1:].ravel()
        y0, y1 = Y_arr[:, :-1].ravel(), Y_arr[:, 1:].ravel()
        z0, z1 = Z_arr[:, :-1].ravel(), Z_arr[:, 1:].ravel()
    else:
        x0, x1 = X_arr[:-1, :].ravel(), X_arr[1:, :].ravel()
        y0, y1 = Y_arr[:-1, :].ravel(), Y_arr[1:, :].ravel()
        z0, z1 = Z_arr[:-1, :].ravel(), Z_arr[1:, :].ravel()

    below_0 = z0 < z_split_val
    below_1 = z1 < z_split_val
    crosses = below_0 != below_1

    t = np.where(crosses, (z_split_val - z0) / (z1 - z0), 0)
    x_cross, y_cross = x0 + t * (x1 - x0), y0 + t * (y1 - y0)

    below = {"x": [], "y": [], "z": []}
    above = {"x": [], "y": [], "z": []}

    for i in range(len(x0)):
        if crosses[i]:
            # Split at intersection - one segment below, one above
            if below_0[i]:  # starts below, ends above
                below["x"].extend([x0[i], x_cross[i], None])
                below["y"].extend([y0[i], y_cross[i], None])
                below["z"].extend([z0[i], z_split_val, None])
                above["x"].extend([x_cross[i], x1[i], None])
                above["y"].extend([y_cross[i], y1[i], None])
                above["z"].extend([z_split_val, z1[i], None])
            else:  # starts above, ends below
                above["x"].extend([x0[i], x_cross[i], None])
                above["y"].extend([y0[i], y_cross[i], None])
                above["z"].extend([z0[i], z_split_val, None])
                below["x"].extend([x_cross[i], x1[i], None])
                below["y"].extend([y_cross[i], y1[i], None])
                below["z"].extend([z_split_val, z1[i], None])
        elif below_0[i]:  # both below
            below["x"].extend([x0[i], x1[i], None])
            below["y"].extend([y0[i], y1[i], None])
            below["z"].extend([z0[i], z1[i], None])
        else:  # both above
            above["x"].extend([x0[i], x1[i], None])
            above["y"].extend([y0[i], y1[i], None])
            above["z"].extend([z0[i], z1[i], None])

    return below, above


def plot_data(
    X,
    Y,
    Z,
    z_split_pct=0.6,
    line_width=2,
    x_below_colorscale=[[0, "lightgreen"], [1, "darkblue"]],
    x_above_colorscale=[[0, "orange"], [1, "red"]],
    y_below_colorscale=[[0, "darkblue"], [1, "cyan"]],
    y_above_colorscale=[[0, "gold"], [1, "magenta"]],
):
    """
    Create a 3D wireframe plot with gradient colors split at a percentage of z-axis using Plotly.

    Segments are physically split at the boundary plane, then each region gets its own gradient.

    Parameters:
        X, Y, Z: 2D arrays from meshgrid defining the surface
        z_split_pct: percentage (0-1) along z-axis where color changes (default 0.6)
        line_width: width of the lines (default 2)
    """
    # Calculate split values
    z_min, z_max = Z.min(), Z.max()
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_split_val = z_min + z_split_pct * (z_max - z_min)

    print(f"Z-axis range: {z_min:.2f} to {z_max:.2f}")
    print(f"{z_split_pct * 100:.0f}% split point: z = {z_split_val:.2f}")

    # Process X-direction and Y-direction segments, split at boundary
    x_below, x_above = create_split_segments(X, Y, Z, axis=1, z_split_val=z_split_val)
    y_below, y_above = create_split_segments(X, Y, Z, axis=0, z_split_val=z_split_val)

    def filter_unused(z_list):
        """Replace None with z_min for color array."""
        return [z if z is not None else z_min for z in z_list]

    # Create Plotly figure
    fig = go.Figure()

    # Add 4 traces: X-below, X-above, Y-below, Y-above
    # Each config: (coords, colorscale, cmin, cmax, name)
    trace_configs = [
        (
            x_below,
            x_below_colorscale,
            z_min,
            z_split_val,
            "X-direction (below)",
        ),
        (
            x_above,
            x_above_colorscale,
            z_split_val,
            z_max,
            "X-direction (above)",
        ),
        (
            y_below,
            y_below_colorscale,
            z_min,
            z_split_val,
            "Y-direction (below)",
        ),
        (
            y_above,
            y_above_colorscale,
            z_split_val,
            z_max,
            "Y-direction (above)",
        ),
    ]

    for coords, colorscale, cmin, cmax, name in trace_configs:
        if not coords["x"]:
            continue
        # Add the line trace (hidden from legend)
        fig.add_trace(
            go.Scatter3d(
                x=coords["x"],
                y=coords["y"],
                z=coords["z"],
                mode="lines",
                line=dict(
                    color=filter_unused(coords["z"]),
                    colorscale=colorscale,
                    width=line_width,
                    cmin=cmin,
                    cmax=cmax,
                ),
                name=name,
                legendgroup=name,
                showlegend=True,
                hoverinfo="none",
            )
        )
        # Add legend-only trace with midpoint color marker so we can control the color separately
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=[None],
        #         y=[None],
        #         z=[None],
        #         mode="markers",
        #         marker=dict(size=10, colorscale=colorscale, symbol="square"),
        #         name=name,
        #         legendgroup=name,
        #         showlegend=True,
        #     )
        # )

    # Update layout
    fig.update_layout(
        title={
            "text": f"3D Wireframe with Split Gradients<br>Split at {z_split_pct * 100:.0f}% Z-axis",
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
