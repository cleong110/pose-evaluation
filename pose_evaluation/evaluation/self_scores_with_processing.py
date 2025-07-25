import argparse
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
from pose_format import Pose
from tqdm import tqdm

from pose_evaluation.evaluation.create_metric import DEFAULT_METRIC_PARAMETERS, construct_metric
from pose_evaluation.evaluation.score_dataframe_format import ScoreDFCol
from pose_evaluation.metrics.base import Score
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.pose_processors import PoseProcessor


class SliceStepProcessor(PoseProcessor):
    def __init__(
        self,
        by=1,
    ) -> None:
        super().__init__(name="slice_step")
        self.by = by

    def process_pose(self, pose: Pose) -> Pose:
        pose = pose.copy()
        pose = pose.slice_step(self.by)
        return pose


def compare_with_self_processed(
    pose: Pose, metric: DistanceMetric, processor_pipeline: list[PoseProcessor]
) -> tuple[Score, float]:
    processed_pose = pose.copy()

    start = time.perf_counter()
    for processor in processor_pipeline:
        processed_pose = processor.process_pose(processed_pose)

    print(f"Original Pose: fps={pose.body.fps}, shape = {pose.body.data.shape}")
    print(
        f"Processed ({processor_pipeline}) Pose: fps={processed_pose.body.fps}, shape = {processed_pose.body.data.shape}"
    )
    score = metric.score_with_signature(pose, processed_pose)
    print(f"Score: {score.score}")
    duration = time.perf_counter() - start
    return score, duration


def main(pose_dir: Path, output_dir: Path):
    print(f"Pose directory: {pose_dir}")
    print(f"Output directory: {output_dir}")
    pose_paths = pose_dir.rglob("*.pose")
    output_dir.mkdir(exist_ok=True, parents=True)

    slice_step_by_values = list(range(11))
    results = defaultdict(list)
    for pose_path in tqdm(pose_paths, desc="Scoring"):
        print(pose_path)
        pose = Pose.read(pose_path.read_bytes())
        for keypoint_selection in DEFAULT_METRIC_PARAMETERS["keypoint_selection_strategies"]:
            metric = construct_metric(
                distance_measure=DTWDTAIImplementationDistanceMeasure(
                    name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True
                ),
                z_speed=None,
                default_distance=0.0,
                trim_meaningless_frames=False,
                normalize=True,
                sequence_alignment="dtw",
                keypoint_selection=keypoint_selection,
                fps=None,
                masked_fill_value=0.0,
            )

            for by in slice_step_by_values:
                pipeline = []
                if by > 0:
                    pipeline.append(SliceStepProcessor(by=by))
                score, score_time = compare_with_self_processed(pose, metric, pipeline)
                # class ScoreDFCol:
                # GLOSS_A_PATH = "GLOSS_A_PATH"
                # GLOSS_B_PATH = "GLOSS_B_PATH"
                # GLOSS_A = "GLOSS_A"
                # GLOSS_B = "GLOSS_B"
                # SCORE = "SCORE"
                # METRIC = "METRIC"
                # SIGNATURE = "SIGNATURE"
                # TIME = "TIME"
                results[ScoreDFCol.SCORE].append(score.score)
                results[ScoreDFCol.TIME].append(score_time)
                results["slice_by"].append(by)
                results["keypoint_selection"].append(keypoint_selection)
                results[ScoreDFCol.METRIC].append(metric.name)
                results[ScoreDFCol.GLOSS_A_PATH].append(str(pose_path))
                results[ScoreDFCol.GLOSS_B_PATH].append(str(pose_path))
                results[ScoreDFCol.SIGNATURE].append(metric.get_signature().format())

    scores_df = pd.DataFrame(results)
    df_out = output_dir / "self_scores_with_slice_step_processing.parquet"
    scores_df.to_parquet(df_out, index=False)

    # === Combined Scatter Plots for All Metrics ===
    output_base = output_dir / "self_scores_with_slice_step_processing"

    # Plot 1: Score vs Slice Step (scatter)
    fig_score_vs_slice = px.scatter(
        scores_df,
        x="slice_by",
        y=ScoreDFCol.SCORE,
        color="keypoint_selection",
        title="Score vs Slice Step (All Metrics)",
        labels={"slice_by": "Slice Step", ScoreDFCol.SCORE: "Score", ScoreDFCol.METRIC: "Metric"},
    )
    fig_score_vs_slice.write_image(f"{output_base}_scores_vs_sliceby.png")
    fig_score_vs_slice.write_html(f"{output_base}_scores_vs_sliceby.html")

    # Plot 2: Computation Time vs Slice Step (scatter)
    fig_time_vs_slice = px.scatter(
        scores_df,
        x="slice_by",
        y=ScoreDFCol.TIME,
        color="keypoint_selection",
        title="Computation Time vs Slice Step (All Metrics)",
        labels={"slice_by": "Slice Step", ScoreDFCol.TIME: "Computation Time (s)", ScoreDFCol.METRIC: "Metric"},
    )
    fig_time_vs_slice.write_image(f"{output_base}_scoretimes_vs_sliceby.png")
    fig_time_vs_slice.write_html(f"{output_base}_scoretimes_vs_sliceby.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .pose files and output results.")
    parser.add_argument("pose_dir", type=Path, help="Directory to recursively search for .pose files.")
    parser.add_argument("output_dir", type=Path, help="Directory to save the output results.")

    args = parser.parse_args()
    main(pose_dir=args.pose_dir, output_dir=args.output_dir)
