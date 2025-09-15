import argparse
import logging
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dask.distributed import Client
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def plot_and_save(
    bin_centers: np.ndarray,
    in_density: np.ndarray,
    out_density: np.ndarray,
    metric_name: str,
    output_dir: Path,
) -> None:
    """Save both Matplotlib (PNG/PDF) and Plotly (HTML) versions of the plot."""
    output_path = output_dir / metric_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Matplotlib static plots ---
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, out_density, label="Out-of-class", color="tab:blue")
    plt.plot(bin_centers, in_density, label="In-class", color="tab:orange")
    plt.title(f"Score Distribution for {metric_name}")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"))
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()

    # --- Plotly interactive HTML plot ---
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=out_density,
            mode="lines",
            name="Out-of-class",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=in_density,
            mode="lines",
            name="In-class",
            line=dict(color="orange"),
        )
    )
    fig.update_layout(
        title=f"Score Distribution for {metric_name}",
        xaxis_title="Score",
        yaxis_title="Density",
        template="plotly_white",
    )
    fig.write_html(output_path.with_suffix(".html"))


def process_metric_partition_dask(metric_path: Path, output_dir: Path, bins: int = 100) -> None:
    """Process a single metric partition into histograms and plots."""
    df = dd.read_parquet(
        str(metric_path),
        columns=["GLOSS_A", "GLOSS_B", "SCORE"],
        engine="pyarrow",
        split_row_groups=True,
        blocksize=None,  # respect parquet row group boundaries
    )

    metric_name = metric_path.name.replace("METRIC=", "")

    # First pass: min/max of scores (lazy until compute)
    score_min, score_max = dd.compute(df["SCORE"].min(), df["SCORE"].max())
    if score_min == score_max:
        logging.warning("All scores equal in %s, skipping.", metric_name)
        return

    bin_edges = np.linspace(score_min, score_max, bins + 1)
    in_class_hist = np.zeros(bins, dtype=np.int64)
    out_class_hist = np.zeros(bins, dtype=np.int64)

    # Second pass: process partition by partition
    for partition in df.to_delayed():
        part_df = dd.from_delayed([partition])

        in_class_scores = part_df[part_df["GLOSS_A"] == part_df["GLOSS_B"]]["SCORE"].compute()
        out_class_scores = part_df[part_df["GLOSS_A"] != part_df["GLOSS_B"]]["SCORE"].compute()

        in_class_hist += np.histogram(in_class_scores, bins=bin_edges)[0]
        out_class_hist += np.histogram(out_class_scores, bins=bin_edges)[0]

    # Normalize to density
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    in_density = in_class_hist / np.sum(in_class_hist) / np.diff(bin_edges)
    out_density = out_class_hist / np.sum(out_class_hist) / np.diff(bin_edges)

    # Save plots
    plot_and_save(bin_centers, in_density, out_density, metric_name, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dask-based memory-safe histogram generator")
    parser.add_argument("dataset_path", type=str, help="Path to Hive-partitioned dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="score_distributions_dask",
        help="Output directory",
    )
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument("--workers", type=int, default=30, help="Number of Dask workers")
    parser.add_argument(
        "--mem-limit",
        type=str,
        default="3.5GB",
        help="Per-worker memory limit (e.g., 3GB, 500MB)",
    )
    args = parser.parse_args()

    root = Path(args.dataset_path)
    output_dir = Path(args.output)

    client = Client(
        n_workers=args.workers,
        threads_per_worker=1,
        memory_limit=args.mem_limit,
    )
    logging.info("Dask client started: %s or possibly <remote URL>/gui/proxy/8787/status", client.dashboard_link)

    metric_partitions = sorted(root.glob("METRIC=*"))
    logging.info("Found %d metric partitions.", len(metric_partitions))

    for metric_path in tqdm(metric_partitions, desc="Processing metrics"):
        try:
            process_metric_partition_dask(
                metric_path=metric_path,
                output_dir=output_dir,
                bins=args.bins,
            )
        except Exception as e:
            logging.error("Failed to process %s: %s", metric_path.name, e)


if __name__ == "__main__":
    main()
