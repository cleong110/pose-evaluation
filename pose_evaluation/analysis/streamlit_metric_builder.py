import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

from pose_evaluation.evaluation.create_metrics import (
    DEFAULT_METRIC_PARAMETERS,
    construct_metric,
)
from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.utils.pose_utils import pose_slice_frames

# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/4
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


@st.cache_resource
def load_pose_from_bytes(pose_bytes: bytes) -> Pose:
    return Pose.read(pose_bytes)


def get_filename(path_str: str) -> str:
    return Path(path_str).name


def select_file(key_prefix: str) -> tuple[Pose, bytes]:
    method = st.selectbox(f"How to select files for {key_prefix}", options=["Dataset", "Path"])
    if method == "Dataset":
        dataset_df = pd.read_csv("/opt/home/cleong/projects/pose-evaluation/dataset_dfs/asl-citizen.csv")
        selected_gloss = st.selectbox(
            f"Which gloss to use for {key_prefix}",
            options=dataset_df["GLOSS"].unique().tolist(),
            key=f"{key_prefix} gloss select",
        )
        mask = dataset_df["GLOSS"] == selected_gloss
        dataset_df.loc[mask, "FILENAME"] = dataset_df.loc[mask, "POSE_FILE_PATH"].apply(get_filename)
        gloss_df = dataset_df[dataset_df["GLOSS"] == selected_gloss]
        selected_filename = st.selectbox(
            f"Which filename to use for {key_prefix}?",
            options=gloss_df["FILENAME"].tolist(),
            key=f"{key_prefix} file select",
        )
        pose_path = gloss_df.loc[gloss_df["FILENAME"] == selected_filename, "POSE_FILE_PATH"].iloc[0]
    elif method == "Path":
        pose_path = st.text_input(
            f"Enter path to target file for {key_prefix}",
            value="/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/downloads/ase/Chronological Bible Translation in American Sign Language (119 Introductions and Passages expanded with More Information)/CBT-001-ase-3-Passage _ God Creates the World.pose",
            key=f"{key_prefix} pose path input",
        )

    pose_bytes = Path(pose_path).read_bytes()
    pose = load_pose_from_bytes(pose_bytes)
    return pose, pose_bytes


def get_pose_gif(
    pose: Pose,
    step: int = 1,
    start_frame: int | None = None,
    end_frame: int | None = None,
    fps: int | None = None,
):
    if fps is not None:
        pose.body.fps = fps
    v = PoseVisualizer(pose)
    frames = list(v.draw())
    frames = frames[start_frame:end_frame:step]
    return v.save_gif(None, frames=frames)


# def pose_sliding_window(
#     pose: Pose,
#     window_length: int,
#     stride: int = 1,
#     start: int = 0,
#     end: int | None = None,
# ) -> Iterator[tuple[int, Pose]]:
#     """
#     Yield sliding window slices of a pose.

#     Parameters
#     ----------
#     - pose (Pose): The original pose object.
#     - window_length (int): Number of frames in each window.
#     - stride (int): Step size between windows.
#     - start (int): Starting frame index.
#     - end (int | None): Optional end frame index (exclusive). Defaults to end of pose.

#     Yields
#     ------
#     - Pose: A windowed slice of the pose.

#     """
#     total_frames = len(pose.body.data)
#     end = end if end is not None else total_frames

#     if end > total_frames:
#         raise ValueError(f"End index {end} exceeds pose length {total_frames}")
#     if window_length <= 0:
#         raise ValueError("window_length must be > 0")
#     if stride <= 0:
#         raise ValueError("stride must be > 0")

#     for i in range(start, end - window_length + 1, stride):
#         # st.write(i)
#         yield i, pose_slice_frames(pose, i, i + window_length)


def get_sliding_window_frame_indexes(
    total_frames: int,
    window_length: int,
    stride: int,
    start: int = 0,
    end: int | None = None,
) -> list[int]:
    end = end if end is not None else total_frames

    if end > total_frames:
        raise ValueError(f"End index {end} exceeds pose length {total_frames}")
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be > 0")

    return list(range(start, end - window_length + 1, stride))


@st.cache_data(show_spinner="Scoring windows...")
def score_windows(
    query_pose_bytes: bytes,
    target_pose_bytes: bytes,
    window_length: int,
    stride: int,
    start: int,
    end: int,
    metric_name: str,
) -> pd.DataFrame:
    query_pose = load_pose_from_bytes(query_pose_bytes)
    target_pose = load_pose_from_bytes(target_pose_bytes)

    frame_indexes = get_sliding_window_frame_indexes(
        total_frames=len(target_pose.body.data),
        window_length=window_length,
        stride=stride,
        start=start,
        end=end,
    )

    scores = []
    for i in frame_indexes:
        window = pose_slice_frames(target_pose, i, i + window_length)
        scores.append(metric.score(query_pose, window))

    return pd.DataFrame(
        {
            "Window Starting Frame Index": frame_indexes,
            "Score": scores,
        }
    )


st.title("Pose Evaluation Metric Builder")

# --- Sidebar for configuration ---
st.sidebar.header("Metric Parameters")

# Distance Measure selection
distance_measure_option = st.sidebar.selectbox(
    "Distance Measure",
    [
        "dtaiDTWAggregatedDistanceMeasureFast",
        "AggregatedPowerDistance",
    ],
)

# Map string name to object
distance_measure = {
    "dtaiDTWAggregatedDistanceMeasureFast": DTWDTAIImplementationDistanceMeasure(
        name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True
    ),
    "AggregatedPowerDistance": AggregatedPowerDistance(),
}[distance_measure_option]

# Dynamic selection widgets in sidebar
z_speed = st.sidebar.selectbox("Z Speed", DEFAULT_METRIC_PARAMETERS["z_speeds"])
default_distance = st.sidebar.selectbox("Default Distance", DEFAULT_METRIC_PARAMETERS["default_distances"])
masked_fill_value = st.sidebar.selectbox("Masked Fill Value", [None] + DEFAULT_METRIC_PARAMETERS["masked_fill_values"])
trim = st.sidebar.selectbox("Trim Meaningless Frames", DEFAULT_METRIC_PARAMETERS["trim_values"])
normalize = st.sidebar.selectbox("Normalize", DEFAULT_METRIC_PARAMETERS["normalize_values"])
keypoint_selection = st.sidebar.selectbox(
    "Keypoint Selection",
    DEFAULT_METRIC_PARAMETERS["keypoint_selection_strategies"],
    index=DEFAULT_METRIC_PARAMETERS["keypoint_selection_strategies"].index("hands"),
)
fps = st.sidebar.selectbox("FPS", DEFAULT_METRIC_PARAMETERS["fps_values"])
sequence_alignment = st.sidebar.selectbox(
    "Sequence Alignment",
    DEFAULT_METRIC_PARAMETERS["sequence_alignment_strategies"],
    index=DEFAULT_METRIC_PARAMETERS["sequence_alignment_strategies"].index("dtw"),
)
reduce_common_components = st.sidebar.checkbox("Reduce to Common Components", value=True)

# --- Main panel ---
st.write("Use the sidebar to configure the metric parameters.")
metric = None


metric = construct_metric(
    distance_measure=distance_measure,
    z_speed=z_speed,
    default_distance=default_distance,
    trim_meaningless_frames=trim,
    normalize=normalize,
    sequence_alignment=sequence_alignment,
    keypoint_selection=keypoint_selection,
    fps=fps,
    masked_fill_value=masked_fill_value,
    reduce_poses_to_common_components=reduce_common_components,
)

st.success("Metric constructed!")
st.write("### Metric Name")
st.code(metric.name)

st.write("### Metric Signature")
st.code(metric.get_signature().format())

st.write("---")


# query_file = st.text_input("Enter path to query file", value="pose_evaluation/utils/test/test_data/mediapipe")


# target_pose = Pose.read(Path(target_pose_path).read_bytes())

col1, col2 = st.columns(2)

with col1:
    query_pose, query_pose_bytes = select_file("query")
    st.write("Query Pose:")
    st.write(query_pose)

with col2:
    target_pose, target_pose_bytes = select_file("target")
    st.write("Pose to Search inside")
    st.write(target_pose)


if st.button("Calculate Score!"):
    score = metric.score(query_pose, target_pose)
    st.success(score)


st.write("---")


# Pose object must already be loaded, e.g.
# pose = load_pose_file(Path("example.pose"))

st.sidebar.header("Sliding Window Parameters")

# Total number of frames in the pose
total_frames = len(target_pose.body.data)

window_length = st.sidebar.number_input(
    "Window length (frames)",
    min_value=1,
    max_value=total_frames,
    value=min(int(len(query_pose.body.data) * 1.1), len(target_pose.body.data)),
    step=1,
)

stride = st.sidebar.number_input(
    "Stride (frames)", min_value=1, max_value=total_frames, value=len(query_pose.body.data) // 3, step=1
)

start = st.sidebar.number_input("Start frame index", min_value=0, max_value=total_frames - 1, value=0, step=1)

end = st.sidebar.number_input(
    "End frame index (exclusive)", min_value=1, max_value=total_frames, value=total_frames, step=1
)

top_n = st.number_input("How many best matches to show?", min_value=1, max_value=100, value=5, step=1)

# Call the sliding window function
if st.sidebar.button("Generate Windows"):
    expected_windows = max(0, (end - start - window_length) // stride + 1)

    if expected_windows == 0:
        st.warning("No windows can be generated with current parameters.")
    else:
        scores_df = score_windows(
            query_pose_bytes=query_pose_bytes,
            target_pose_bytes=target_pose_bytes,
            window_length=window_length,
            stride=stride,
            start=start,
            end=end,
            metric_name=metric.name,
        )

        fig = px.line(scores_df, x="Window Starting Frame Index", y="Score", title="Distances")
        st.plotly_chart(fig)

        fps = target_pose.body.fps
        top_matches_df = scores_df.sort_values("Score", ascending=True).head(top_n)

        st.markdown(f"### 🥇 Top {top_n} Matches")
        for idx, row in top_matches_df.iterrows():
            frame_index = int(row["Window Starting Frame Index"])
            seconds = frame_index / fps
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)

            st.write(f"## **Match {idx + 1}**")
            st.write(f"Score: `{row['Score']:.4f}`")
            st.write(f"Starts at frame `{frame_index}`, approx `{minutes}m {remaining_seconds}s` into the video")
            st.markdown("---")

            # if st.button(f"Visualize match {idx + 1}"):
            window_pose = pose_slice_frames(target_pose, frame_index, frame_index + window_length)

            st.write("### Window visualization")
            pose_bytes = get_pose_gif(
                pose=window_pose,
            )
            # st.write(pose_bytes)
            if pose_bytes is not None:
                st.image(pose_bytes)

            st.write("### Processed visualization")
            processed_window_pose = metric.process_poses([window_pose, query_pose])[0]
            st.write(processed_window_pose)
            pose_bytes = get_pose_gif(
                pose=processed_window_pose,
            )
            # st.write(pose_bytes)
            if pose_bytes is not None:
                st.image(pose_bytes)
