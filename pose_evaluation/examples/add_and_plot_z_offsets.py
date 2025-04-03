from collections import defaultdict
import json
import time
from pathlib import Path
from typing import Union

import random
import numpy as np
import numpy.ma as ma
import plotly.graph_objects as go

import numpy as np
import numpy.ma as ma
import plotly.graph_objects as go
import pandas as pd
from pose_format import Pose
from pyzstd import decompress
import plotly.graph_objects as go
from tqdm import tqdm

from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import (
    DTWAggregatedPowerDistanceMeasure,
    DTWAggregatedScipyDistanceMeasure,
    DTWDTAIImplementationDistanceMeasure,
    DTWOptimizedDistanceMeasure,
)
from pose_evaluation.metrics.pose_processors import (
    NormalizePosesProcessor,
    GetHandsOnlyHolisticPoseProcessor,
    InterpolateAllToSetFPSPoseProcessor,
    ReduceHolisticPoseProcessor,
    ZeroPadShorterPosesProcessor,
    AddTOffsetsToZPoseProcessor,
    get_standard_pose_processors,
    AddFramebasedOffsetToZPoseProcessor,
    SetZToTPoseProcessor,
    SetZToZeroProcessor,
)


def get_pose_data(file_path: Union[Path, str]) -> Pose:
    """Loads a .pose or .pose.zst, returns a Pose object"""
    file_path = Path(file_path)
    if file_path.name.endswith(".pose.zst"):
        return Pose.read(decompress(file_path.read_bytes()))
    else:
        return Pose.read(file_path.read_bytes())


def plot_xyz_trajectories(
    trajectories,
    labels,
    xaxis_title="X",
    yaxis_title="Y",
    zaxis_title="Z",
    title="Keypoint Trajectories",
    use_gradient=False,
    color_scale="Viridis",
    mode="lines+markers",
):
    """Plots multiple x, y, z coordinate trajectories using Plotly, grouping left and right hand trajectories.
       Ensures only points where all (x, y, z) are unmasked are included.
       Optionally applies a color gradient over the frame index.

    Args:
      trajectories: A list of NumPy masked arrays, each of shape (num_frames, 3), representing
        the x, y, z coordinates of a keypoint over time.
      labels: A list of labels corresponding to each trajectory.
      use_gradient: Whether to apply a color gradient over frame indices.
      color_scale: The color scale to use if gradient is enabled.
      mode: Plotly mode for traces (e.g., "markers", "lines", "lines+markers").
    """
    if len(trajectories) != len(labels):
        raise ValueError("Number of trajectories and labels must match")

    fig = go.Figure()
    grouped_traces = {"LEFT_HAND": [], "RIGHT_HAND": [], "OTHER": []}

    # Group trajectories
    for trajectory, label in zip(trajectories, labels):
        trajectory = ma.array(trajectory)  # Ensure it's a masked array
        valid_mask = ~ma.getmaskarray(trajectory)  # Boolean mask for valid values
        valid_rows = valid_mask.all(axis=1)  # Keep only rows where ALL (x, y, z) are valid
        valid_data = trajectory[valid_rows]  # Extract valid (fully unmasked) points
        frame_indices = np.arange(trajectory.shape[0])[valid_rows]  # Frame indices for valid points

        if valid_data.shape[0] == 0:  # Skip if all points are invalid
            continue

        if "LEFT_HAND" in label:
            grouped_traces["LEFT_HAND"].append((valid_data, frame_indices))
        elif "RIGHT_HAND" in label:
            grouped_traces["RIGHT_HAND"].append((valid_data, frame_indices))
        else:
            grouped_traces["OTHER"].append((valid_data, frame_indices, label))

    # Function to combine and add grouped traces with optional gradient
    def add_group_trace(group_name, color):
        if grouped_traces[group_name]:  # Only add if there's data
            combined_data = np.vstack([data for data, _ in grouped_traces[group_name]])
            combined_frames = np.concatenate([frames for _, frames in grouped_traces[group_name]])

            marker_args = (
                {"size": 4, "color": color}
                if not use_gradient
                else {
                    "size": 4,
                    "color": combined_frames,
                    "colorscale": color_scale,
                    "colorbar": {"title": "Frame Index"},
                }
            )

            fig.add_trace(
                go.Scatter3d(
                    x=combined_data[:, 0],
                    y=combined_data[:, 1],
                    z=combined_data[:, 2],
                    mode=mode,
                    marker=marker_args,
                    name=group_name,
                )
            )

    # Add grouped traces
    add_group_trace("LEFT_HAND", "blue")
    add_group_trace("RIGHT_HAND", "red")

    # Add other traces individually
    for trajectory, frame_indices, label in grouped_traces["OTHER"]:
        marker_args = (
            {"size": 4}
            if not use_gradient
            else {"size": 4, "color": frame_indices, "colorscale": color_scale, "colorbar": {"title": "Frame Index"}}
        )

        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2], mode=mode, marker=marker_args, name=label
            )
        )

    # Set camera view so Y is vertical and Z is horizontal
    camera = dict(
        up=dict(x=0, y=-1, z=0),  # Set Y as the vertical axis
        eye=dict(x=1.5, y=-0.5, z=1.5),  # View from the front, aligning Z horizontally
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
            camera=camera,  # Apply camera settings
        ),
        legend=dict(x=-0.2),
        title=title,
    )

    fig.show()


# def plot_xyz_trajectory(
#     keypoint_trajectory, xaxis_title="X", yaxis_title="Y", zaxis_title="Z", title="Keypoint Trajectory"
# ):
#     """Plots the x, y, z coordinates of a keypoint trajectory using Plotly, ignoring masked points.

#     Args:
#       keypoint_trajectory: A NumPy masked array of shape (num_frames, 3) representing
#         the x, y, z coordinates of a keypoint over time.
#     """
#     # Ensure keypoint_trajectory is a masked array
#     keypoint_trajectory = ma.array(keypoint_trajectory)

#     # Mask invalid (masked) points
#     valid_mask = ~ma.getmaskarray(keypoint_trajectory).any(axis=1)

#     fig = go.Figure(
#         data=[
#             go.Scatter3d(
#                 x=keypoint_trajectory[valid_mask, 0],
#                 y=keypoint_trajectory[valid_mask, 1],
#                 z=keypoint_trajectory[valid_mask, 2],
#                 mode="lines+markers",
#                 marker=dict(size=4),
#             )
#         ]
#     )

#     fig.update_layout(
#         scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title, zaxis_title=zaxis_title), title=title
#     )

#     fig.show()


def get_name_mapping_for_pose_keypoint_indices(pose: Pose):
    mapping_dict = {}
    index = 0
    for c in pose.header.components:
        for p_name in c.points:
            mapping_dict[index] = f"{c.name}:{p_name}"
            index += 1
    return mapping_dict


def print_z_extremes(arr):
    # Extract Z values (last dimension index 2)
    z_values = arr[..., 2]

    # Find min and max of Z
    z_min = np.min(z_values)
    z_max = np.max(z_values)

    print("Min Z:", z_min)
    print("Max Z:", z_max)


if __name__ == "__main__":
    poses_folder = Path(r"C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose")
    pose_files = list(poses_folder.glob("*.pose.zst"))
    desired_gloss = "HOUSE"
    plot = True
    print_pos = True
    print_z_vals = True

    pose_files = [file for file in pose_files if desired_gloss in file.name]
    print(f"Found {len(pose_files)} poses")
    pose_file = random.choice(pose_files)
    print(pose_file.name)
    pose = get_pose_data(pose_file)
    pose = GetHandsOnlyHolisticPoseProcessor().process_pose(pose)
    print(f"random pose: {pose}")
    name_mappings = get_name_mapping_for_pose_keypoint_indices(pose)
    if print_pos:

        for i, pos in enumerate(pose.body.points_perspective()[0].squeeze()):
            print(i, pos)

    labels = []
    trajectories = []
    for i, trajectory in enumerate(pose.body.points_perspective().squeeze()):
        name = name_mappings[i]
        # print(i, name, trajectory.shape)
        labels.append(name)
        trajectories.append(trajectory)

    # exit()
    if plot:
        plot_xyz_trajectories(
            labels=labels,
            trajectories=trajectories,
            title=f"{pose_file.name} Hands with Z Unmodified",
            mode="markers",
            use_gradient=True,
        )
    if print_z_vals:
        print_z_extremes(pose.body.points_perspective())

    ########################################################
    # Run the preprocessor
    # pose = SetZToTPoseProcessor().process_pose(pose)
    # pose = AddTOffsetsToZPoseProcessor().process_pose(pose)
    pose = SetZToZeroProcessor().process_pose(pose)
    ########################################################

    name_mappings = get_name_mapping_for_pose_keypoint_indices(pose)
    labels = []
    trajectories = []
    for i, trajectory in enumerate(pose.body.points_perspective().squeeze()):
        name = name_mappings[i]
        # print(i, name, trajectory.shape)
        labels.append(name)
        trajectories.append(trajectory)

    if plot:
        plot_xyz_trajectories(
            labels=labels,
            trajectories=trajectories,
            zaxis_title="T",
            title=f"{pose_file.name} Hands with Z = 0",
            mode="markers",
            use_gradient=True,
        )

    if print_pos:
        print(name_mappings[0])
        for i, pos in enumerate(pose.body.points_perspective()[0].squeeze()):
            # print(pos.shape)
            print(i, pos)

    if print_z_vals:
        print_z_extremes(pose.body.points_perspective())
    print(pose_file.name)
