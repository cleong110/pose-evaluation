"""Given two poses, one longer, slide the smaller across the larger"""

import argparse
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from pose_format import Pose
from tqdm import tqdm

from pose_evaluation.evaluation.create_metrics import (
    construct_metric,
)
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import DatasetDFCol
from pose_evaluation.evaluation.dataset_parsing.gloss_utils import text_to_glosses
from pose_evaluation.evaluation.load_splits_and_run_metrics import combine_dataset_dfs
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.utils.pose_utils import pose_slice_frames

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@contextmanager
def timed_section(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    log.info(f"[TIMER] {name} took {end - start:.2f} seconds.")


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
    query_pose: Pose,
    target_pose: Pose,
    stride: int,
    start: int,
    end: int,
    window_length: int,
    metric: DistanceMetric,
) -> pd.DataFrame:
    # TODO: use Pose.read start_frame, end_frame?
    # https://pose-format.readthedocs.io/en/latest/_modules/pose_format/pose_body.html#PoseBody.read

    frame_indexes = get_sliding_window_frame_indexes(
        total_frames=len(target_pose.body.data),
        window_length=window_length,
        stride=stride,
        start=start,
        end=end,
    )

    scores = []
    time_seconds = []
    end_frame_indexes = []
    for i in tqdm(frame_indexes, desc="Scoring windows"):
        window_pose = pose_slice_frames(target_pose, i, i + window_length)
        scores.append(metric.score(query_pose, window_pose))
        seconds = i / target_pose.body.fps
        time_seconds.append(seconds)
        end_frame_indexes.append(i + len(window_pose.body.data))

    return pd.DataFrame(
        {
            "start_frame": frame_indexes,
            "end_frame": end_frame_indexes,
            # TODO: add in end_frame
            "Time (s)": time_seconds,
            "score": scores,
        }
    )


def get_query_poses(
    dataset_df,
    query_glosses,
    sample_count_per_gloss: int = 5,
):
    gloss_df = dataset_df[dataset_df[DatasetDFCol.GLOSS].isin(query_glosses)].copy()

    # Group by gloss and sample
    sampled_df = gloss_df.groupby(DatasetDFCol.GLOSS, group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_count_per_gloss), random_state=42)
    )

    for row in sampled_df[[DatasetDFCol.GLOSS, DatasetDFCol.POSE_FILE_PATH]].itertuples(index=False, name=None):
        gloss, pose_path = row
        log.info(f"{gloss}, {pose_path}")
        yield gloss, Pose.read(Path(pose_path).read_bytes()), Path(pose_path)


def search_with_gloss_list(
    ref_pose_path: Path,
    dataset_df: Path,
    query_glosses: list[str],
    query_sample_count: int,
    metric: DistanceMetric,
    stride: int,
    start_frame: int,
    end_frame: int | None,
) -> pd.DataFrame:
    log.info(f"Reference Pose: {ref_pose_path}")
    log.info(f"Dataset df: {dataset_df}")
    log.info(f"Query Glosses: {query_glosses}")
    log.info(f"Query Count per Gloss: {query_sample_count}")
    log.info(f"Metric: {metric.name}")
    log.info(f"Stride: {stride}")
    log.info(f"Start Frame: {start_frame}")
    log.info(f"End Frame: {end_frame}")

    log.info(dataset_df.head())

    with timed_section("Query Pose Loading"):
        glosses_and_poses = list(get_query_poses(dataset_df, query_glosses, query_sample_count))
    log.info(f"Loaded {len(glosses_and_poses)} query poses.")

    for i, (gloss, pose, pose_path) in enumerate(glosses_and_poses):
        log.info(f"{gloss} pose {i}, {pose.body.data.shape}, {pose_path.name}")

    with timed_section("Reference Pose Loading"):
        ref_pose = Pose.read(ref_pose_path.read_bytes())
    log.info(f"Loaded ref pose: {ref_pose.body.data.shape}")

    log.info("---- Windowing Strategy: Fixed Window, Mean Query Length * 1.2")
    query_window_length = np.mean([q.body.data.shape[0] for g, q, p in glosses_and_poses])
    query_window_length = int(query_window_length * 1.2)
    log.info(f"Calculated Window Length: {query_window_length}")
    query_stride = query_window_length // 3
    log.info(f"Stride (auto-calculated): {query_stride}")

    all_scores = []

    with timed_section("Sliding Window Scoring"):
        for query_gloss, query_pose, query_pose_path in tqdm(glosses_and_poses, desc="Processing Queries"):
            scores_df = score_windows(
                query_pose=query_pose,
                target_pose=ref_pose,
                window_length=query_window_length,
                stride=query_stride,
                start=start_frame,
                end=end_frame or ref_pose.body.data.shape[0],
                metric=metric,
            )
            scores_df["Query"] = query_pose_path.name
            scores_df["GLOSS"] = query_gloss
            all_scores.append(scores_df)

    combined_scores = pd.concat(all_scores, ignore_index=True)
    log.info(f"Combined dataframe shape: {combined_scores.shape}")

    return combined_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slide a query pose across a reference pose and compute similarity scores."
    )

    parser.add_argument("ref_pose", type=Path, help="Path to the reference pose file to search within.")

    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("/opt/home/cleong/projects/pose-evaluation/dataset_dfs/asl-citizen.csv"),
        help="Path to the dataset CSV file (default: %(default)s).",
    )

    parser.add_argument(
        "--query-glosses",
        type=str,
        # required=True,
        help="Comma-separated list of glosses to spot for, e.g., --query-glosses 'APPLE,BOOK'. Either this or --query-text is required.",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        # required=True,
        help="Text to search for. From this a set of query glosses will be generated. Either this or --query-glosses is required.",
    )

    parser.add_argument(
        "--query-sample-count",
        type=int,
        default=5,
        help="How many poses from the dataset to use as queries. Default: %(default)s.",
    )

    parser.add_argument(
        "--metric-name",
        type=str,
        default="untrimmed_normalizedbyshoulders_hands_defaultdist10.0_nointerp_dtw_fillmasked0.0_dtaiDTWAggregatedDistanceMetricFast",
        help="Name of the distance metric to use (default: %(default)s).",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Stride for sliding window (default: %(default)s frames).",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index in the reference pose (default: %(default)s).",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End frame index in the reference pose (default: till end of pose).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".") / "sliding_window_outputs",
        help="Where to save the output figures.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    with timed_section("Dataset Loading"):
        dataset_df = combine_dataset_dfs(dataset_df_files=[args.dataset_csv], splits=["test"])

    query_glosses = []
    if args.query_glosses is not None:
        for g in args.query_glosses.split(","):
            query_glosses.append(g.strip())

    if args.query_text is not None:
        vocab = dataset_df["GLOSS"].unique().tolist()
        query_glosses = text_to_glosses(args.query_text, vocab)
        print(f"Parsed Glosses {query_glosses} from query text")

    keypoint_selections = [
        "removelegsandworld",
        "reduceholistic",
        "hands",
        "youtubeaslkeypoints",
        "fingertips",
    ]

    with timed_section("Metric Construction"):
        metric = construct_metric(
            distance_measure=DTWDTAIImplementationDistanceMeasure(
                name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True
            ),
            z_speed=None,
            default_distance=10.0,
            trim_meaningless_frames=False,
            normalize=True,
            sequence_alignment="dtw",
            keypoint_selection="fingertips",
            fps=None,
            masked_fill_value=0.0,
        )
    log.info(f"Constructed metric: {metric.name}")

    scores_df = search_with_gloss_list(
        ref_pose_path=args.ref_pose,
        dataset_df=dataset_df,
        query_glosses=query_glosses,
        query_sample_count=args.query_sample_count,
        metric=metric,
        stride=args.stride,
        start_frame=args.start,
        end_frame=args.end,
    )
    log.info(scores_df)

    fig = px.line(
        scores_df,
        x="Time (s)",
        y="score",
        color="GLOSS",  # Color by GLOSS
        line_group="Query",  # Individual line per Query exemplar
        title="Gloss Similarities",
        hover_data={
            "Time (s)": False,
            "start_frame": True,
            "end_frame": True,
            "score": ":.4f",
            "Query": True,
            "GLOSS": True,
        },
    )
    fig.update_layout(
        xaxis_title="Time (m:s)",
        xaxis_tickformat=".1f",  # Show ~1 decimal minute
    )
    fig.update_layout(
        legend={
            "orientation": "h",  # Optional: Make the legend horizontal
            "yanchor": "top",  # Anchor the legend's top to the specified y-coordinate
            "y": -0.3,  # Adjust this value to move the legend further down
            "xanchor": "center",  # Anchor the legend's center to the specified x-coordinate
            "x": 0.5,  # Center the legend horizontally
        }
    )
    # fig.show()
    # Save plot to HTML for interactive viewing later
    out_dir = args.output_dir / f"{metric.name}" / "_".join(query_glosses)
    out_dir.mkdir(exist_ok=True, parents=True)
    html_out = out_dir / "distance_scores_plot.html"
    fig.write_html(html_out)
    logging.info(html_out.resolve())

    # Optionally also save as static image (requires kaleido)
    img_out = out_dir / "distance_scores_plot.png"
    fig.write_image(img_out)
    logging.info(img_out.resolve())
