import argparse
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client
from tqdm import tqdm


def process_metric_partition_dask(metric_path: Path, output_dir: Path, bins: int = 100):
    df = dd.read_parquet(
        str(metric_path),
        columns=["GLOSS_A", "GLOSS_B", "SCORE"],
        engine="pyarrow",
        split_row_groups=True,
        blocksize=None,  # respect parquet row group boundaries
    )

    metric_name = metric_path.name.replace("METRIC=", "")

    # First pass: min/max of scores (lazily computed)
    score_min, score_max = dd.compute(df["SCORE"].min(), df["SCORE"].max())
    if score_min == score_max:
        print(f"All scores equal in {metric_name}, skipping.")
        return

    bin_edges = np.linspace(score_min, score_max, bins + 1)
    in_class_hist = np.zeros(bins, dtype=np.int64)
    out_class_hist = np.zeros(bins, dtype=np.int64)

    # Second pass: stream through the dataframe in partitions
    for partition in df.to_delayed():
        part_df = dd.from_delayed([partition])

        # Filter in-class and out-of-class
        in_class_scores = part_df[part_df["GLOSS_A"] == part_df["GLOSS_B"]]["SCORE"].compute()
        out_class_scores = part_df[part_df["GLOSS_A"] != part_df["GLOSS_B"]]["SCORE"].compute()

        in_class_hist += np.histogram(in_class_scores, bins=bin_edges)[0]
        out_class_hist += np.histogram(out_class_scores, bins=bin_edges)[0]

    # Normalize
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    in_density = in_class_hist / np.sum(in_class_hist) / np.diff(bin_edges)
    out_density = out_class_hist / np.sum(out_class_hist) / np.diff(bin_edges)

    # Plot
    output_path = output_dir / f"{metric_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, out_density, label="Out-of-class", color="tab:blue")
    plt.plot(bin_centers, in_density, label="In-class", color="tab:orange")
    plt.title(f"Score Distribution for {metric_name}")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Dask-based memory-safe histogram generator")
    parser.add_argument("dataset_path", type=str, help="Path to Hive-partitioned dataset")
    parser.add_argument("--output", type=str, default="score_distributions_dask", help="Output directory")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument("--workers", type=int, default=10, help="Number of Dask workers")
    parser.add_argument("--mem-limit", type=str, default="3GB", help="Per-worker memory limit")
    args = parser.parse_args()

    root = Path(args.dataset_path)
    output_dir = Path(args.output)
    client = Client(
        n_workers=args.workers,
        threads_per_worker=1,
        memory_limit=args.mem_limit,
        config={
            "distributed.worker.memory.target": 0.6,
            "distributed.worker.memory.spill": 0.7,
            "distributed.worker.memory.pause": 0.8,
            "distributed.worker.memory.terminate": 0.95,
        },
    )
    print(f"Dask client started: {client.dashboard_link}")

    metric_partitions = sorted(root.glob("METRIC=*"))
    print(f"Found {len(metric_partitions)} metric partitions.")

    for metric_path in tqdm(metric_partitions, desc="Processing metrics"):
        try:
            process_metric_partition_dask(metric_path, output_dir, bins=args.bins)
        except Exception as e:
            print(f"Failed to process {metric_path.name}: {e}")


if __name__ == "__main__":
    main()
