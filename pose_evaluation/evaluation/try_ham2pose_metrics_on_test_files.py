import warnings
from itertools import product
from pathlib import Path

import pandas as pd
from pose_format import Pose
from tqdm import tqdm

from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.ham2pose import (
    Ham2PoseDTWMetric,
    Ham2PosenAPEMetric,
    Ham2PosenDTWMetric,
    Ham2PosenMSEMetric,
    get_standard_ham2pose_preprocessors,
)

# from pose_evaluation.metrics.ham2pose import (
#     Ham2PoseDTWMetric,
#     Ham2PosenAPEMetric,
#     Ham2PosenDTWMetric,
#     Ham2PosenMSEMetric,
# )

if __name__ == "__main__":
    poses_dir = Path("pose_evaluation/utils/test/test_data")
    # poses_dir = Path("/opt/home/cleong/projects/pose-evaluation/files_to_check_for_ham2pose/ASL_CITIZEN")
    # poses_dir = Path("/opt/home/cleong/projects/pose-evaluation/files_to_check_for_ham2pose/colin/house")
    # poses_dir = Path("/opt/home/cleong/projects/pose-evaluation/files_to_check_for_ham2pose/colin")
    # poses_dir = Path("/opt/home/cleong/projects/pose-evaluation/files_to_check_for_ham2pose/")
    poses = list(poses_dir.rglob("*.pose"))
    entries = []

    # Dictionary of metric names to distance functions
    distance_metrics = {
        "nMSE": Ham2PosenMSEMetric(),
        "nAPE": Ham2PosenAPEMetric(),
        "DTW": Ham2PoseDTWMetric(),
        "nDTW": Ham2PosenDTWMetric(),
        "dtaiDTWfast": DistanceMetric(
            name="dtaiDTWMetricFast",
            distance_measure=DTWDTAIImplementationDistanceMeasure(
                name="dtaiDTWAggregatedDistanceMeasureFast",
                use_fast=True,
                default_distance=0.0,
            ),
            pose_preprocessors=get_standard_ham2pose_preprocessors(),
        ),
        "dtaiDTWslow": DistanceMetric(
            name="dtaiDTWMetricSlow",
            distance_measure=DTWDTAIImplementationDistanceMeasure(
                name="dtaiDTWAggregatedDistanceMeasure",
                use_fast=False,
                default_distance=0.0,
            ),
            pose_preprocessors=get_standard_ham2pose_preprocessors(),
        ),
    }

    pose_pairs = list(product(poses, poses))

    for pose_ref_path, pose_hyp_path in tqdm(pose_pairs, desc="Scoring pose pairs"):
        # for pose_ref_path, pose_hyp_path in [(poses[0], poses[1])]:
        entry = {
            "ref": pose_ref_path,
            "hyp": pose_hyp_path,
        }

        # print(f"pose pair: \n*\t{pose_ref_path},\n*\t{pose_hyp_path}")

        for name, distance_fn in distance_metrics.items():
            # if name not in [
            #     # "nAPE",
            #     # "nMSE",
            #     # "nDTW",
            #     "DTW",
            # ]:
            #     continue
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                # processed_hyp, processed_ref = distance_fn.process_poses([pose_hyp, pose_ref])
                # entry["hyp_shape"] = pose_hyp.body.data.shape
                # entry["ref_shape"] = pose_ref.body.data.shape
                pose_ref = Pose.read(pose_ref_path.read_bytes())
                pose_hyp = Pose.read(pose_hyp_path.read_bytes())

                try:
                    score = distance_fn.score_with_signature(pose_hyp, pose_ref)
                    entry[name] = score.score
                    entry[f"{name}_signature"] = distance_fn.get_signature().format()
                except ValueError as e:
                    entry[f"{name}_ERROR"] = str(e)

                # Now check for any RuntimeWarnings captured
                for w in caught_warnings:
                    if issubclass(w.category, RuntimeWarning):
                        entry[f"{name}_WARN"] = str(w.message)

        # print(entry)
        # exit()
        entries.append(entry)
        # print(pd.DataFrame(entries))
    # Create DataFrame and write to CSV
    df = pd.DataFrame(entries)
    output_path = poses_dir / "ham2pose_pose-eval_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{len(df)} Results saved to: {output_path.resolve()}")

    # Load and compare with previous results if available
    previous_path = poses_dir / "ham2pose_results.csv"
    if previous_path.exists():
        print(f"Loading previous results from: {previous_path.resolve()}")
        df_prev = pd.read_csv(previous_path)
        print(f"Loaded {len(df_prev)}")

        # Normalize paths in both DataFrames for robust merging
        for col in ["ref", "hyp"]:
            df[col] = df[col].apply(lambda x: str(Path(x).resolve()))
            df_prev[col] = df_prev[col].apply(lambda x: str(Path(x).resolve()))

        # Merge on ref + hyp path strings
        merged = pd.merge(df, df_prev, on=["ref", "hyp"], suffixes=("_new", "_old"))
        print(merged)

        # Compute mean % difference for all metric columns
        metric_names = [
            name for name in distance_metrics.keys() if name in df.columns and f"{name}_old" in merged.columns
        ]

        for metric in metric_names:
            old_vals = merged[f"{metric}_old"]
            new_vals = merged[f"{metric}_new"]

            # Masks for nan positions
            old_nan = old_vals.isna()
            new_nan = new_vals.isna()
            both_nan = old_nan & new_nan
            only_old_nan = old_nan & ~new_nan
            only_new_nan = new_nan & ~old_nan

            # Count NaNs
            print(f"\nMetric: {metric}")
            print(f"  NaN in both  : {both_nan.sum()}")
            print(f"  NaN in old   : {only_old_nan.sum()}")
            print(f"  NaN in new   : {only_new_nan.sum()}")

            # Only compute % diff where both values are present
            valid_mask = ~old_nan & ~new_nan
            if valid_mask.sum() > 0:
                percent_diff = (
                    abs(new_vals[valid_mask] - old_vals[valid_mask])
                    / ((new_vals[valid_mask] + old_vals[valid_mask]) / 2)
                ) * 100
                mean_diff = percent_diff.mean()
                print(f"  Mean % difference (non-NaN only): {mean_diff:.2f}%")
                output_path = poses_dir / f"{metric}ham2pose_pose-eval_and_orig_results_compared.csv"
                merged[[f"{metric}_old", f"{metric}_new", "ref", "hyp", f"{metric}_signature"]].to_csv(
                    output_path, index=False
                )
                print(f"Saved to {output_path}")
            else:
                print("  No valid (non-NaN) comparisons for this metric.")

        # Optional: Compare different metrics (e.g., dtaiDTWfast vs nDTW)
        metric_comparisons = [
            ("dtaiDTWfast", "DTW_old"),
            ("dtaiDTWfast", "nDTW_old"),
            ("dtaiDTWslow", "nDTW_old"),
            ("dtaiDTWslow", "DTW_old"),
            # Add more if needed
        ]

        print(merged.columns)
        for metric_a, metric_b in metric_comparisons:
            if metric_a in merged.columns and metric_b in merged.columns:
                a_vals = merged[metric_a]
                b_vals = merged[metric_b]
                valid_mask = ~a_vals.isna() & ~b_vals.isna()

                print(f"\nCross-metric comparison: {metric_a} vs {metric_b}")
                if valid_mask.sum() == 0:
                    print("  No valid (non-NaN) pairs to compare.")
                    continue

                percent_diff = (
                    abs(a_vals[valid_mask] - b_vals[valid_mask]) / ((a_vals[valid_mask] + b_vals[valid_mask]) / 2)
                ) * 100
                mean_diff = percent_diff.mean()

                print(f"  Mean % difference (non-NaN only): {mean_diff:.2f}%")

                comparison_df = merged.loc[valid_mask, ["ref", "hyp", metric_a, metric_b]].copy()
                comparison_df["percent_diff"] = percent_diff.values
                output_path = poses_dir / f"{metric_a}_vs_{metric_b}_comparison.csv"
                comparison_df.to_csv(output_path, index=False)
                print(f"  Saved to {output_path}")
            else:
                print(f"Don't have results for both {metric_a} and {metric_b}")
    else:
        print(f"No previous results found at {previous_path.resolve()}")
