import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

from pose_evaluation.analysis.plotly_pose_visualizer import PlotlyVisualizer
from pose_evaluation.evaluation.create_metrics import (
    DEFAULT_METRIC_PARAMETERS,
    construct_metric,
)
from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.utils.pose_utils import pose_slice_frames

# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/4
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


def construct_metric_from_args(metric_args: dict):
    """Reconstruct the metric from a dictionary of arguments."""
    return construct_metric(
        distance_measure=metric_args["distance_measure"],
        z_speed=metric_args["z_speed"],
        default_distance=metric_args["default_distance"],
        trim_meaningless_frames=metric_args["trim"],
        normalize=metric_args["normalize"],
        sequence_alignment=metric_args["sequence_alignment"],
        keypoint_selection=metric_args["keypoint_selection"],
        fps=metric_args["fps"],
        masked_fill_value=metric_args["masked_fill_value"],
        reduce_poses_to_common_components=metric_args["reduce_common_components"],
    )


@st.cache_resource
def load_pose_from_bytes(pose_bytes: bytes) -> Pose:
    return Pose.read(pose_bytes)


def get_filename(path_str: str) -> str:
    return Path(path_str).name


def select_files(key_prefix: str) -> list[tuple[Pose, bytes]]:
    method = st.selectbox(f"How to select files for {key_prefix}", options=["Dataset", "Path"])
    pose_results = []

    if method == "Dataset":
        dataset_df = pd.read_csv("/opt/home/cleong/projects/pose-evaluation/dataset_dfs/asl-citizen.csv")
        st.dataframe(dataset_df)

        selected_glosses = st.multiselect(
            f"Select gloss(es) for {key_prefix}",
            options=dataset_df["GLOSS"].unique().tolist(),
            key=f"{key_prefix} gloss multiselect",
        )

        if selected_glosses:
            gloss_df = dataset_df[dataset_df["GLOSS"].isin(selected_glosses)].copy()
            gloss_df["FILENAME"] = gloss_df["POSE_FILE_PATH"].apply(get_filename)

            use_all = st.checkbox("Use all pose files for selected gloss(es)?", key=f"{key_prefix} use all checkbox")

            if use_all:
                selected_files_df = gloss_df
            else:
                selected_filenames = st.multiselect(
                    f"Select file(s) to use for {key_prefix}",
                    options=gloss_df["FILENAME"].tolist(),
                    key=f"{key_prefix} file multiselect",
                )
                selected_files_df = gloss_df[gloss_df["FILENAME"].isin(selected_filenames)]

            for _, row in selected_files_df.iterrows():
                pose_path = row["POSE_FILE_PATH"]
                pose_bytes = Path(pose_path).read_bytes()
                pose = load_pose_from_bytes(pose_bytes)
                gloss = row.get("GLOSS", f"Query {len(pose_results) + 1}")
                filename = get_filename(row["POSE_FILE_PATH"])
                label = f"{gloss} – {filename}"
                pose_results.append((pose, pose_bytes, label))

    elif method == "Path":
        input_paths = st.text_area(
            f"Enter one or more pose file paths for {key_prefix} (one per line)",
            value="/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/downloads/ase/Chronological Bible Translation in American Sign Language (119 Introductions and Passages expanded with More Information)/CBT-001-ase-3-Passage _ God Creates the World.pose",
            key=f"{key_prefix} pose path input",
        )
        for pose_path in input_paths.strip().splitlines():
            if pose_path.strip():
                pose_bytes = Path(pose_path.strip()).read_bytes()
                pose = load_pose_from_bytes(pose_bytes)
                pose_results.append((pose, pose_bytes, f"Query {len(pose_results) + 1}"))

    return pose_results


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
            f"Which file(s) to use for {key_prefix}?",
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
    st.write(f"Visualizing pose:{pose.body.data.shape}, sum {pose.body.data.sum()}")

    # st.write(f"Visualizing pose:{pose.body.data.shape}, sum {pose.body.data.sum()}")
    if fps is not None:
        pose.body.fps = fps
    v = PoseVisualizer(pose)
    frames = list(v.draw())
    frames = frames[start_frame:end_frame:step]

    # st.write(f"last frame:{frames[-1].shape}, sum {frames[-1].sum()}")
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


def score_windows(
    query_pose_bytes: bytes,
    target_pose_bytes: bytes,
    window_length: int,
    stride: int,
    start: int,
    end: int,
    metric_name: str,
    metric: DistanceMetric,
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
    time_seconds = []
    time_labels = []
    for i in frame_indexes:
        window = pose_slice_frames(target_pose, i, i + window_length)
        scores.append(metric.score(query_pose, window))
        seconds = i / target_pose.body.fps
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        time_seconds.append(seconds)
        time_labels.append(f"{minutes}:{remaining_seconds:02d}")

    return pd.DataFrame(
        {
            "Window Starting Frame Index": frame_indexes,
            "Time (s)": time_seconds,
            "Time Label": time_labels,
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
fps = st.sidebar.selectbox("FPS Interpolation", DEFAULT_METRIC_PARAMETERS["fps_values"])
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
    query_items = select_files("query")  # List of (Pose, bytes)
    st.write(f"Selected {len(query_items)} Query Poses:")
    for i, (query_pose, query_pose_bytes, gloss_label) in enumerate(query_items):
        # pose, _ = item
        st.write(f"Pose {i} ({gloss_label}): shape {query_pose.body.data.shape}")

with col2:
    target_pose, target_pose_bytes = select_file("target")  # Still just one
    st.write("Pose to Search inside")
    st.write(target_pose)

st.write("---")

# Sidebar params
st.sidebar.header("Sliding Window Parameters")
total_frames = len(target_pose.body.data)
if not query_items:
    st.warning("No query items selected. Please select one or more before generating windows.")
    st.stop()

max_query_length = max(len(pose.body.data) for pose, _, _ in query_items)
# st.write(f"Max Query video Length: {max_query_length}")
# default_window_length = min(int(max_query_length * 1.1), total_frames)
default_window_length_multiplier = 1.1

window_length_multiplier = st.sidebar.number_input(
    "Window length (multiplier)",
    min_value=0.1,
    max_value=2.0,
    value=default_window_length_multiplier,
    step=0.1,
)

# stride = st.sidebar.number_input("Stride (frames)", 1, total_frames, value=int(default_window_length / 3), step=1)
stride_proportion = st.sidebar.number_input(
    "Stride (proportion of window)", min_value=0.1, max_value=1.0, value=0.3, step=0.1
)
start = st.sidebar.number_input("Start frame index", 0, total_frames - 1, value=0, step=1)
end = st.sidebar.number_input("End frame index (exclusive)", 1, total_frames, value=total_frames, step=1)
top_n = st.sidebar.number_input("How many best matches to show?", 1, 100, value=5, step=1)

# Compute scores
if st.sidebar.button("Generate Windows"):
    # expected_windows = max(0, (end - start - window_length) // stride + 1)

    # if expected_windows == 0:
    #     st.warning("No windows can be generated with current parameters.")
    # else:
    all_scores = []

    progress_text = f"Running sliding windows for {len(query_items)} query videos"
    my_bar = st.progress(0, text=progress_text)

    for i, (query_pose, query_pose_bytes, gloss_label) in enumerate(query_items):
        my_bar.progress((i + 1) / len(query_items), text=progress_text)
        query_window_length = int(query_pose.body.data.shape[0] * window_length_multiplier)
        query_stride = int(query_window_length * stride_proportion)
        st.write(
            f"Query window length: Query pose length {query_pose.body.data.shape[0]} * multiplier {window_length_multiplier} = {query_window_length}"
        )
        st.write(f"query_stride: window length {query_window_length} * {stride_proportion} = {query_stride}")
        scores_df = score_windows(
            query_pose_bytes=query_pose_bytes,
            target_pose_bytes=target_pose_bytes,
            # window_length=window_length,
            window_length=query_window_length,
            # stride=stride,
            stride=query_stride,
            start=start,
            end=end,
            metric_name=metric.name,
            metric=metric,
        )
        scores_df["Query"] = gloss_label
        scores_df["Gloss"] = gloss_label.split("–")[0]
        all_scores.append(scores_df)

    combined_scores = pd.concat(all_scores, ignore_index=True)
    st.dataframe(combined_scores)

    fig = px.line(
        combined_scores,
        x="Time (s)",
        # x="Time Label",
        y="Score",
        color="Query",
        title="Sliding Window Distance Scores by Query",
        hover_data={
            "Time Label": True,
            "Time (s)": False,  # Optional: hide raw seconds
            "Window Starting Frame Index": True,
            "Score": ":.4f",
        },
    )
    fig.update_layout(
        xaxis_title="Time (m:s)",
        xaxis_tickformat=".1f",  # Show ~1 decimal minute
    )
    st.plotly_chart(fig)

    # Show top matches for each query
    fps = target_pose.body.fps
    for query_label, df in combined_scores.groupby("Query"):
        query_pose = next(pose for pose, _, gloss in query_items if gloss == query_label)
        query_window_length = int(query_pose.body.data.shape[0] * window_length_multiplier)
        if st.checkbox(f"Show matches for {query_label}?"):
            st.markdown(f"### 🥇 Top {top_n} Matches for {query_label}")
            top_matches_df = df.sort_values("Score", ascending=True).head(top_n)

            for idx, row in top_matches_df.iterrows():
                frame_index = int(row["Window Starting Frame Index"])
                seconds = frame_index / fps
                minutes = int(seconds // 60)
                remaining_seconds = int(seconds % 60)

                st.write(f"## **Window {idx + 1}**")
                st.write(f"Score: `{row['Score']:.4f}`")
                st.write(f"Starts at frame `{frame_index}`, approx `{minutes}m {remaining_seconds}s` into the video")
                st.markdown("---")

                window_pose = pose_slice_frames(target_pose, frame_index, frame_index + query_window_length)
                if st.checkbox(f"Visualize Match {idx + 1}"):
                    st.write("### Processed visualization")
                    processed_window_pose = metric.process_poses([window_pose, query_pose])[0]
                    st.write(processed_window_pose)

                    viz = PlotlyVisualizer(processed_window_pose)
                    fig = viz.get_3d_animation()
                    st.plotly_chart(fig, key=f"{idx + 1} Plotly Visualization")
