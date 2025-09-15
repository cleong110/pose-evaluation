import argparse
import logging
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow.dataset as ds
from dask.distributed import Client
from tqdm import tqdm

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def plot_and_save(
    bin_centers: np.ndarray,
    in_density: np.ndarray,
    out_density: np.ndarray,
    in_count: int,
    out_count: int,
    metric_name: str,
    output_dir: Path,
) -> None:
    """Save Plotly plots in HTML, PNG, and PDF formats."""
    output_path = output_dir / metric_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    # Out-of-class line
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=out_density,
            mode="lines",
            name=f"Out-of-class (n={out_count})",
            line=dict(color="blue"),
        )
    )

    # In-class line
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=in_density,
            mode="lines",
            name=f"In-class (n={in_count})",
            line=dict(color="orange"),
        )
    )

    # Layout
    fig.update_layout(
        title=f"Score Distribution for {metric_name}",
        xaxis_title="Score",
        yaxis_title="Density",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Save outputs
    fig.write_html(output_path.with_suffix(".html"))
    fig.write_image(output_path.with_suffix(".png"))
    fig.write_image(output_path.with_suffix(".pdf"))


# def process_metric_partition_dask(metric_path: Path, output_dir: Path, bins: int = 100) -> None:
#     """Process a single metric partition into histograms and plots."""
#     # df = dd.read_parquet(
#     #     str(metric_path),
#     #     columns=["GLOSS_A", "GLOSS_B", "SCORE"],
#     #     engine="pyarrow",
#     #     split_row_groups=True,
#     #     blocksize=None,
#     # )


def process_metric_partition_dask(
    metric_path: Path,
    output_dir: Path,
    bins: int = 100,
    debug_rows: int | None = None,
) -> None:
    """
    Process a single metric partition into histograms and plots.

    If debug_rows is provided, only the first N rows are read eagerly with
    PyArrow. Otherwise the full dataset is loaded lazily with Dask.
    """
    if debug_rows is None:
        # Full dataset, lazily with Dask
        df = dd.read_parquet(
            str(metric_path),
            columns=["GLOSS_A", "GLOSS_B", "SCORE"],
            engine="pyarrow",
            split_row_groups=True,
            blocksize=None,
        )
    else:
        # Quick sample via PyArrow
        dataset = ds.dataset(metric_path, format="parquet")
        scanner = dataset.scanner(columns=["GLOSS_A", "GLOSS_B", "SCORE"])
        table = scanner.head(debug_rows)
        df = dd.from_pandas(table.to_pandas(), npartitions=4)

    metric_name = metric_path.name.replace("METRIC=", "")

    # Global min/max across all partitions
    score_min, score_max = dd.compute(df["SCORE"].min(), df["SCORE"].max())
    if score_min == score_max:
        logging.warning("All scores equal in %s, skipping.", metric_name)
        return

    bin_edges = np.linspace(score_min, score_max, bins + 1)

    # Lazy histogram per partition
    def partition_hist(partition: pd.DataFrame) -> pd.Series:
        in_class = partition.loc[partition["GLOSS_A"] == partition["GLOSS_B"], "SCORE"].to_numpy()
        out_class = partition.loc[partition["GLOSS_A"] != partition["GLOSS_B"], "SCORE"].to_numpy()
        in_hist, _ = np.histogram(in_class, bins=bin_edges)
        out_hist, _ = np.histogram(out_class, bins=bin_edges)
        return pd.Series(
            {
                "in_hist": in_hist,
                "out_hist": out_hist,
                "in_count": len(in_class),
                "out_count": len(out_class),
            }
        )

    meta = pd.Series(
        {
            "in_hist": np.array([], dtype=int),
            "out_hist": np.array([], dtype=int),
            "in_count": 0,
            "out_count": 0,
        }
    )

    hists = df.map_partitions(partition_hist, meta=meta).compute()
    # Ensure hists is always a DataFrame

    # Sum across partitions (stack arrays to avoid object-dtype issues)
    in_class_hist = np.sum(np.stack(hists["in_hist"]), axis=0)
    out_class_hist = np.sum(np.stack(hists["out_hist"]), axis=0)

    in_count = int(hists["in_count"].sum())
    out_count = int(hists["out_count"].sum())

    # Normalize to density
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    in_density = in_class_hist / np.sum(in_class_hist) / np.diff(bin_edges)
    out_density = out_class_hist / np.sum(out_class_hist) / np.diff(bin_edges)

    # Save plots
    plot_and_save(
        bin_centers,
        in_density,
        out_density,
        in_count,
        out_count,
        metric_name,
        output_dir,
    )


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
        threads_per_worker=2,
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
