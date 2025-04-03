import numpy as np
import plotly.graph_objects as go
from fastdtw import fastdtw

if __name__ == "__main__":

    # Generate angles from 0 to 2π
    point_count_1 = 20
    point_count_2 = 8
    angles1 = np.linspace(0, 2 * np.pi, point_count_1)
    angles2 = np.linspace(0, 2 * np.pi, point_count_2)

    # Compute sine and cosine values
    trace1_y = np.sin(angles1)
    # trace2_values = np.cos(angles2)
    # trace2_values = np.sin(angles2)
    trace2_y = np.sin(angles2 - np.pi / 8)  # Shift right by π/8
    trace2_y = trace2_y + 1

    trace1_z = [i for i in range(len(trace1_y))]
    trace2_z = [i for i in range(len(trace2_y))]

    padding = True
    if padding:
        trace2_y_pad = [0 for i in range(point_count_1 - point_count_2)]
        trace2_y_pad.extend(trace2_y)
        trace2_y = trace2_y_pad
        angles2 = angles1
        trace2_z = [i for i in range(len(trace2_z))]

    if len(trace1_y) == len(trace2_y):
        mappings = [(i, i) for i in range(point_count_1)]
    else:
        dist, mappings = fastdtw(trace1_y, trace2_y, 1)
    print(mappings)
    # mode = "markers"
    mode = "lines+markers"

    # Create the plot
    fig = go.Figure()

    # Add sine trace
    fig.add_trace(
        go.Scatter(
            x=angles1,
            y=trace1_y,
            # z=[i for i in range(len(trace1_y))],
            #  mode="lines",
            mode=mode,
            name=f"Sequence 1: {point_count_1} points",
        )
    )

    # Add cosine trace
    fig.add_trace(
        go.Scatter(
            x=angles2,
            y=trace2_y,
            # z=[i for i in range(len(trace2_y))],
            #  mode="lines",
            mode=mode,
            name=f"Sequence 2 {point_count_2} points",
        )
    )
    # Add green lines for mappings
    for i, j in mappings:
        fig.add_trace(
            go.Scatter(
                x=[angles1[i], angles2[j]],
                y=[trace1_y[i], trace2_y[j]],
                # z=[trace1_z[i], trace2_z[j]],
                mode="lines",
                line=dict(color="green", width=2),
                showlegend=False,  # Hide individual mappings from legend
            )
        )

    # Customize layout
    if len(trace1_y) == len(trace2_y):
        if padding:
            title = "Mapping between Unequal Sequences with Padding"
        else:
            title = "Mapping between Equal-Length Sequences"

    else:
        title = f"DTW Mapping between Unequal-Length Sequences"
    fig.update_layout(
        title=title,
        xaxis_title="Angle (radians)",
        yaxis_title="Value",
        legend_title="Functions",
    )

    # Show the plot
    fig.show()
