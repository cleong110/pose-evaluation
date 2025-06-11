import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import pyarrow as pa
import streamlit as st
from tqdm import tqdm

from pose_evaluation.evaluation.interpret_name import descriptive_name

METRIC_COL = "METRIC"
SIGNATURE_COL = "SIGNATURE"
SHORT_COL = "SHORT"
DESCRIPTIVE_NAME_COL = "DESCRIPTIVE_NAME"


def evaluate_top_k_results(
    top_k_results: dict,
    k: int,
    verbose: bool = False,
) -> float:
    """Evaluate top-k results and return accuracy only."""
    correct = 0
    total = 0

    for (query_path, query_label), neighbors in tqdm(
        top_k_results.items(), desc=f"Processing items, k={k}", disable=True
    ):
        neighbor_labels = [label for _, _, label in neighbors[:k]]
        label_counts = Counter(neighbor_labels)
        predicted_label, predicted_label_count = label_counts.most_common(1)[0]
        is_correct = predicted_label == query_label
        correct += is_correct
        total += 1

        if verbose:
            st.write(
                f"{query_path}: {predicted_label} x {predicted_label_count} (true: {query_label}) "
                f"{'✓' if is_correct else '✗'} ({[f'{label} x {count}' for label, count in label_counts.items()]})"
            )
    st.write(f"Evaluated {total} queries with k={k}, of which {correct} were correctly classified")

    return correct / total if total > 0 else 0.0


@st.cache_data
def analyze_neighbors_file(
    file_path: Path,
    k: int,
    query_path_col: str = "GLOSS_A_PATH",
    neighbor_path_col: str = "GLOSS_B_PATH",
    query_label_col: str = "GLOSS_A",
    neighbor_label_col: str = "GLOSS_B",
    score_col: str = "SCORE",
):
    """Analyze previously saved top-k neighbor results."""
    pattern = r"(.+)_\d+_hyps"
    match = re.match(pattern, path.name)

    if match:
        metric_name = match.group(1)  # Group 1 is the content before '_somenumber_hyps'
    else:
        metric_name = "Unknown: Could not parse: Pattern not found in filename."
    df = pd.read_parquet(path)
    # st.success(f"Loaded file with shape: {df.shape}, metric name: {metric_name}")
    # add "METRIC" column
    df["METRIC"] = metric_name

    grouped = df.groupby([query_path_col, query_label_col])

    top_k_results = {
        (query_path, query_label): list(
            zip(group[score_col], group[neighbor_path_col], group[neighbor_label_col], strict=False)
        )
        for (query_path, query_label), group in grouped
    }
    accuracy = evaluate_top_k_results(top_k_results, k=k)
    st.write(f"\n{metric_name} Accuracy from saved neighbors: {accuracy:.4f}")
    return accuracy, metric_name


st.title("🪵 Parquet Viewer")

# Text input for parquet file path
file_path = st.text_input("Enter path to Parquet file or folder:", "")
file_path = Path(file_path)


parquets = []

# Try loading and displaying the parquet file
if file_path.is_dir():
    parquets = list(file_path.glob("*.parquet"))

else:
    parquets.append(file_path)

dfs = []

files_to_load_count = st.number_input("Files to load", value=5, max_value=len(parquets))
min_k_val = st.number_input("Min K value", value=1, max_value=100)
max_k_val = st.number_input("Max K value", value=max(5, min_k_val), min_value=min_k_val, max_value=100)
parquets = parquets[:files_to_load_count]


knn_accuracies = defaultdict(list)
with st.status(f"Calculating accuracies for {len(parquets)} files...", expanded=True) as status:
    for i, file_path in enumerate(parquets):
        status.update(
            label=f"Calculating accuracy for file {i + 1}/{len(parquets)}, max k={max_k_val}",
            state="running",
            expanded=False,
        )
        for k in range(min_k_val, max_k_val + 1):
            path = Path(file_path)

            if not path.exists():
                st.error("File not found. Please check the path.")
            elif not path.suffix == ".parquet":
                st.warning("The file does not have a .parquet extension. Are you sure it's a Parquet file?")
            else:
                try:
                    accuracy, metric_name = analyze_neighbors_file(path, k=k)
                    knn_accuracies["METRIC"].append(metric_name)
                    knn_accuracies["k"].append(k)
                    knn_accuracies["accuracy"].append(accuracy)

                except pa.lib.ArrowIOError as e:
                    st.exception(
                        f"Arrow I/O Error: Problem accessing or reading the Parquet file '{file_path}'. This might be due to permissions, a corrupted file, or an incomplete download. Details: {e}"
                    )
                except pa.lib.ArrowInvalid as e:
                    st.exception(
                        f"Arrow Invalid Data Error: The Parquet file '{file_path}' is structurally invalid or contains uninterpretable data. This could be due to a malformed file or schema issues. Details: {e}"
                    )
                except OSError as e:
                    # This catches more general OS-level errors beyond FileNotFoundError (which you already handle)
                    # such as disk full, I/O interruptions, etc.
                    st.exception(
                        f"Operating System Error: An unexpected OS error occurred while trying to read '{file_path}'. Details: {e}"
                    )
                except (
                    pd.errors.EmptyDataError
                ):  # Though less common with parquet, good to include if you expect empty but valid files
                    st.exception(f"Empty Data Error: The Parquet file '{file_path}' is valid but contains no data.")

if knn_accuracies:
    accuracy_df = pd.DataFrame(knn_accuracies)
    accuracy_df[DESCRIPTIVE_NAME_COL] = accuracy_df[METRIC_COL].apply(descriptive_name)

    # --- Keyword filtering ---
    exclude = st.text_input("Keywords to exclude? (comma-separated)", value="")
    include = st.text_input("Keywords to include? (comma-separated)", value="")

    metric_series = accuracy_df[METRIC_COL].str.lower()

    match_all = st.checkbox("Require all keywords (AND)?", value=False)  # default is "any" (OR)

    if include:
        keywords = [kw.strip().lower() for kw in include.split(",") if kw.strip()]

        # Ensure your `metric_series` is lowercase for consistent matching
        metric_series = accuracy_df["METRIC"].str.lower()

        if match_all:
            for kw in keywords:
                accuracy_df = accuracy_df[metric_series.str.contains(re.escape(kw), na=False)]
        else:
            pattern = "|".join(map(re.escape, keywords))
            accuracy_df = accuracy_df[metric_series.str.contains(pattern, na=False)]

    if exclude:
        keywords = [kw.strip().lower() for kw in exclude.split(",") if kw.strip()]
        pattern = "|".join(map(re.escape, keywords))
        accuracy_df = accuracy_df[~metric_series.str.contains(pattern, na=False)]

    st.dataframe(accuracy_df)

    accuracy_df_csv_data = accuracy_df.to_csv(index=False)
    st.download_button(
        label="Download KNN Accuracies CSV",
        data=accuracy_df_csv_data,
        file_name=f"knn_metric_accuracies_{len(parquets)}metrics_k1_to_k{max_k_val}.csv",
        mime="text/csv",
    )

    # Create the Plotly plot
    fig = px.line(
        accuracy_df, x="k", y="accuracy", color=DESCRIPTIVE_NAME_COL, title="Pose Distance Metrics KNN accuracy"
    )

    # Display the plot as JSON
    st.plotly_chart(fig, use_container_width=True)

# if dfs:
#     df = pd.concat(dfs)
#     st.subheader(f"Loaded {len(df)} rows from {len(parquets)} files")
#     st.dataframe(df)
# conda activate /opt/home/cleong/envs/pose_eval_src && streamlit run pose_evaluation/analysis/st_parquet_viewer.py
