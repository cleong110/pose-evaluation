#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

CHEATING_METRIC = "CheatingMetric_cheat0.0_defaultdist4.0"
CHEATING_SIGNATURE = (
    "CheatingMetric_cheat0.0_defaultdist4.0|"
    "higher_is_better:no|"
    "pose_preprocessors:[]|"
    "distance_measure:{return4|default_distance:4.0}"
)


def process_file(file_path: Path, out_dir: Path | None):
    df = pd.read_parquet(file_path)

    # Zero the score where GLOSS_A == GLOSS_B
    df.loc[df["GLOSS_A"] == df["GLOSS_B"], "SCORE"] = 0

    # Update METRIC and SIGNATURE
    df["METRIC"] = df["METRIC"].replace(
        to_replace=r"Return4Metric_defaultdist4.0",
        value=CHEATING_METRIC,
        regex=True,
    )
    df["SIGNATURE"] = CHEATING_SIGNATURE

    # Output path
    new_filename = file_path.name.replace("Return4Metric_defaultdist4.0", "CheatingMetric_cheat0.0_defaultdist4.0")
    new_path = (out_dir or file_path.parent) / new_filename

    # Ensure output directory exists
    new_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(new_path, index=False)
    print(f"Saved: {new_path}")


def main(root_path: Path, out_dir: Path | None = None):
    parquet_files = root_path.rglob("*Return4*.parquet")
    for file_path in parquet_files:
        process_file(file_path, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cheating scores from Return4 scores.")
    parser.add_argument("root_path", type=Path, help="Root path to search for Return4 parquet files")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional output directory for saving files (defaults to input file locations)",
    )
    args = parser.parse_args()

    main(args.root_path, args.out_dir)
