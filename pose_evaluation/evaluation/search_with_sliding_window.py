import argparse
import itertools
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from pose_format import Pose
from tqdm import tqdm

from pose_evaluation.evaluation.create_metric import construct_metric
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import DatasetDFCol
from pose_evaluation.evaluation.load_splits_and_run_metrics import combine_dataset_dfs
from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol
from pose_evaluation.evaluation.sliding_window import (
    FixedWindowsSetByQueryMeanLengthsStrategy,
    FixedWindowStrategy,
    score_windows_from_ranges,
)
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.pose_processors import TrimMeaninglessFramesPoseProcessor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@contextmanager
def timed_section(name: str):
    start = time.perf_counter()
    result = {}
    yield result
    end = time.perf_counter()
    duration = end - start
    result["duration"] = duration
    log.info(f"[TIMER] {name} took {duration:.2f} seconds.")


def get_query_poses(
    dataset_df,
    query_glosses,
    sample_count_per_gloss: int = 5,
    trim=True,
):
    gloss_df = dataset_df[dataset_df[DatasetDFCol.GLOSS].isin(query_glosses)].copy()

    # Group by gloss and sample
    sampled_df = gloss_df.groupby(DatasetDFCol.GLOSS, group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_count_per_gloss), random_state=42)
    )

    processors = []
    if trim:
        # isolated signs often have dead time around the actual signing
        processors.append(TrimMeaninglessFramesPoseProcessor())

    for row in sampled_df[[DatasetDFCol.GLOSS, DatasetDFCol.POSE_FILE_PATH]].itertuples(index=False, name=None):
        gloss, pose_path = row
        log.info(f"{gloss}, {pose_path}")
        pose = Pose.read(Path(pose_path).read_bytes())
        for processor in processors:
            pose = processor.process_pose(pose)

        yield gloss, pose, Path(pose_path)


def search_with_gloss_list(
    ref_pose_path: Path,
    dataset_df: Path,
    query_glosses: list[str],
    query_sample_count: int,
    keypoint_selection: str,
    masked_fill_value: float,
    stride: int,
    start_frame: int,
    end_frame: int | None,
    out_dir: Path,
    trim_queries: bool = True,
    interpolate_ref: float | None = None,
) -> pd.DataFrame:
    with timed_section("Metric Construction"):
        metric = construct_metric(
            distance_measure=DTWDTAIImplementationDistanceMeasure(
                name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True
            ),
            z_speed=None,
            default_distance=10.0,
            trim_meaningless_frames=trim_queries,
            normalize=True,
            sequence_alignment="dtw",
            keypoint_selection=keypoint_selection,
            fps=None,
            masked_fill_value=masked_fill_value,
        )
        log.info(f"Constructed metric: {metric.name}")

    log.info(f"Reference Pose: {ref_pose_path}")
    log.info(f"Dataset df: {dataset_df}")
    log.info(f"Query Glosses: {query_glosses}")
    log.info(f"Query Count per Gloss: {query_sample_count}")
    log.info(f"Metric: {metric.name}")
    log.info(f"Stride: {stride}")
    log.info(f"Start Frame in long video: {start_frame}")
    log.info(f"End Frame in long video: {end_frame}")
    if end_frame is not None:
        log.info(f"Segment in long video: {end_frame - start_frame}")

    log.info(dataset_df.head())

    with timed_section("Query Pose Loading"):
        glosses_and_poses = list(get_query_poses(dataset_df, query_glosses, query_sample_count, trim=trim_queries))
    log.info(f"Loaded {len(glosses_and_poses)} query poses.")

    for i, (gloss, pose, pose_path) in enumerate(glosses_and_poses):
        log.info(f"{gloss} pose {i}, {pose.body.data.shape}, {pose_path.name}")

    with timed_section("Reference Pose Loading"):
        ref_pose = Pose.read(ref_pose_path.read_bytes())
    log.info(f"Loaded ref pose: {ref_pose.body.data.shape}.")

    window_strats = []

    fixed_windows = [6, 24, 48]
    # fixed_windows = [6, 8, 10, 12, 16, 20, 24]
    # fixed_strides = [1, 2, 4, 6]
    fixed_strides = [1, 6]

    for fixed_window_length, fixed_stride in itertools.product(fixed_windows, fixed_strides):
        window_strategy = FixedWindowStrategy(window_length=fixed_window_length, stride=fixed_stride)
        window_strats.append(window_strategy)

    # FixedWindowsSetByQueryMeanLengthsStrategy
    mean_query_length = np.mean([q.body.data.shape[0] for g, q, p in glosses_and_poses])
    multipliers = [0.1, 0.2, 0.5, 0.7]
    stride_divisors = [None, 2]
    for stride_divisor, mult in itertools.product(stride_divisors, multipliers):
        window_strategy = FixedWindowsSetByQueryMeanLengthsStrategy(
            mean_query_pose_length=mean_query_length, window_length_multiplier=mult, stride_divisor=stride_divisor
        )
        window_strats.append(window_strategy)

    for window_strategy in window_strats:
        log.info(
            f"---- Windowing Strategy: {window_strategy.name}",
        )
        log.info("")

        all_scores = []

        with timed_section("Sliding Window Scoring"):
            for query_gloss, query_pose, query_pose_path in tqdm(glosses_and_poses, desc="Processing Queries"):
                windows = window_strategy.get_windows(ref_pose, start_frame, end_frame)
                log.info(f"{window_strategy.get_name()} generated {len(windows)}")
                log.debug(f"Windows Generated: {windows}")
                scores_df = score_windows_from_ranges(
                    query_pose=query_pose,
                    target_pose=ref_pose,
                    windows=windows,
                    metric=metric,
                )
                scores_df["Query"] = query_pose_path.name
                scores_df[ScoreDFCol.GLOSS_A] = query_gloss
                all_scores.append(scores_df)

        combined_scores = pd.concat(all_scores, ignore_index=True)
        log.info(f"Combined dataframe shape: {combined_scores.shape}")

        scores_df = combined_scores
        log.info(scores_df)

        fig = px.line(
            scores_df,
            x="Time (s)",
            y=ScoreDFCol.SCORE,
            color=ScoreDFCol.GLOSS_A,  # Color by GLOSS
            line_group="Query",  # Individual line per Query exemplar
            title=f"Gloss Distances \n(DTW+Norm+Masked Fill {masked_fill_value}), \nWindow Strategy {window_strategy}",
            hover_data={
                "Time (s)": False,
                "start_frame": True,
                "end_frame": True,
                ScoreDFCol.SCORE: ":.4f",
                "Query": True,
                ScoreDFCol.GLOSS_A: True,
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

        out_dir = (
            args.output_dir
            / "_".join(query_glosses)
            / f"searchingfrom_frame{start_frame}_to_{end_frame}"
            / f"query_sample_count{query_sample_count}"
            / "dtw"
            / keypoint_selection
            / f"trim_queries_{trim_queries}"
            / f"maskedfill{masked_fill_value}"
            / f"{window_strategy.get_name()}"
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        # Save plot to HTML for interactive viewing later
        html_out = out_dir / "distance_scores_plot.html"
        fig.write_html(html_out)
        logging.info(html_out.resolve())

        # Optionally also save as static image (requires kaleido)
        img_out = out_dir / "distance_scores_plot.png"
        fig.write_image(img_out)
        logging.info(img_out.resolve())

        parquet_out = out_dir / "window_scores.parquet"
        scores_df.to_parquet(parquet_out, index=False)
        logging.info(parquet_out.resolve())

    return combined_scores


if __name__ == "__main__":
    # Full default lists
    default_keypoint_selections = [
        "removelegsandworld",
        "reduceholistic",
        "hands",
        "youtubeaslkeypoints",
        "fingertips",
    ]
    default_fillmask_vals = [0.0, 10.0, 100]

    parser = argparse.ArgumentParser(
        description="Slide query pose(s) across a reference pose and compute similarity scores."
    )

    parser.add_argument("ref_pose", type=Path, help="Path to the reference pose file to search within.")

    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("/opt/home/cleong/projects/pose-evaluation/dataset_dfs/asl-citizen.csv"),
        help="Path to the dataset CSV file (default: %(default)s), from which to take query glosses.",
    )

    parser.add_argument(
        "--query-glosses",
        type=str,
        help="Comma-separated list of glosses to spot for, e.g., --query-glosses 'APPLE,BOOK'. Either this or --query-text is required.",
    )
    parser.add_argument(
        "--query-sample-count",
        type=int,
        default=5,
        help="How many poses from the dataset to use as queries. Default: %(default)s.",
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

    parser.add_argument(
        "--keypoint-selections",
        type=str,
        default="all",
        help=f"Comma-separated list of keypoint selections (default: all = {default_keypoint_selections}).",
    )

    parser.add_argument(
        "--masked-fill-values",
        type=str,
        default="all",
        help=f"Comma-separated list of masked fill values (default: all = {default_fillmask_vals}).",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Parse keypoint selections
    if args.keypoint_selections == "all":
        keypoint_selections = default_keypoint_selections
    else:
        keypoint_selections = [s.strip() for s in args.keypoint_selections.split(",")]

    # Parse masked fill values
    if args.masked_fill_values == "all":
        fillmask_vals = default_fillmask_vals
    else:
        fillmask_vals = [float(v.strip()) for v in args.masked_fill_values.split(",")]

    with timed_section("Dataset Loading"):
        dataset_df = combine_dataset_dfs(dataset_df_files=[args.dataset_csv], splits=["test"])

    query_glosses = []
    if args.query_glosses is not None:
        query_glosses = [g.strip() for g in args.query_glosses.split(",")]

    for keypoint_selection, masked_fill_value in itertools.product(keypoint_selections, fillmask_vals):
        scores_df = search_with_gloss_list(
            ref_pose_path=args.ref_pose,
            dataset_df=dataset_df,
            query_glosses=query_glosses,
            query_sample_count=args.query_sample_count,
            keypoint_selection=keypoint_selection,
            masked_fill_value=masked_fill_value,
            stride=args.stride,
            start_frame=args.start,
            end_frame=args.end,
            out_dir=args.output_dir,
        )


# python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/search_with_sliding_window.py "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/downloads/ase/Chronological Bible Translation in American Sign Language (119 Introductions and Passages)/CBT-001-ase-3-Passage _ God Creates the World.pose" --query-glosses "GOD,HEAVEN,EARTH,DAY,MORNING,LIGHT,REFRIGERATOR,CELERY" --output-dir ./sliding_window_outputs/god_creates_the_world_short/ --end 2880 --keypoint-selections "hands,youtubeaslkeypoints,reduceholistic"
