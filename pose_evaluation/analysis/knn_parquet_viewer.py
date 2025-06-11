import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa
import streamlit as st
from tqdm import tqdm

from pose_evaluation.evaluation.interpret_name import descriptive_name

METRIC_COL = "METRIC"
SIGNATURE_COL = "SIGNATURE"
SHORT_COL = "SHORT"
DESCRIPTIVE_NAME_COL = "DESCRIPTIVE_NAME"


def plot_accuracy_by_k_grouped_color(
    df: pd.DataFrame,
    trace_col: str,  # Unique trace per value in this column
    hover_cols: list[str],  # Columns to show in hover
    color_col: str,  # Group and color lines by this column
    plot_title: Optional[str] = None,
    color_palette: Optional[list] = None,
    switch_to_colorscale_if_over: int = 8,  # Threshold for switching
) -> go.Figure:
    """
    Plot one line per `trace_col`, grouped and colored by `color_col`.
    Legend is grouped using dummy traces as headers for each color group.
    """
    # Get all unique color group values
    unique_highlight_values = sorted(df[color_col].unique())
    num_groups = len(unique_highlight_values)
    st.write(f"Num Groups: {num_groups}")

    # Dynamically choose color palette
    if color_palette is None:
        if num_groups <= 8:
            color_palette = px.colors.qualitative.Set2
        elif num_groups <= 10:
            color_palette = px.colors.qualitative.G10
        elif num_groups <= 11:
            color_palette = px.colors.qualitative.Vivid
        elif num_groups <= 26:
            color_palette = px.colors.qualitative.Alphabet  # 26 colors
        else:
            color_palette = pc.sample_colorscale("Rainbow", [i / max(1, num_groups - 1) for i in range(num_groups)])

    # Map each group to a color
    color_map = dict(zip(unique_highlight_values, color_palette, strict=False))

    fig = go.Figure()

    # Add dummy traces for the legend
    for highlight_val in unique_highlight_values:
        color = color_map.get(highlight_val, "gray")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": color},
                name=str(highlight_val),
                legendgroup=highlight_val,
                showlegend=True,
            )
        )

    # Add actual traces
    for trace_value, group in df.groupby(trace_col):
        color_value = group[color_col].iloc[0]
        color = color_map.get(color_value, "gray")

        hover_texts = ["<br>".join(f"{col}: {row[col]}" for col in hover_cols) for _, row in group.iterrows()]

        fig.add_trace(
            go.Scatter(
                x=group["k"],
                y=group["accuracy"],
                mode="lines",
                name=str(trace_value),
                line={"color": color},
                hovertext=hover_texts,
                hoverinfo="text",
                legendgroup=color_value,
                showlegend=False,  # dummy trace already handles legend
            )
        )

    fig.update_layout(
        title=plot_title or "Accuracy by k",
        xaxis_title="k",
        yaxis_title="accuracy",
        legend_title=color_col,
    )

    return fig


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
    match = re.match(pattern, file_path.name)

    if match:
        metric_name = match.group(1)  # Group 1 is the content before '_somenumber_hyps'
    else:
        metric_name = "Unknown: Could not parse: Pattern not found in filename."
    df = pd.read_parquet(file_path)
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


@st.cache_data
def analyze_neighbor_files(
    file_paths: list[Path],
    min_k_val: int,
    max_k_val: int,
    query_path_col: str = "GLOSS_A_PATH",
    neighbor_path_col: str = "GLOSS_B_PATH",
    query_label_col: str = "GLOSS_A",
    neighbor_label_col: str = "GLOSS_B",
    score_col: str = "SCORE",
):
    knn_accuracies = defaultdict(list)
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
                    knn_accuracies[METRIC_COL].append(metric_name)
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

    return pd.DataFrame(knn_accuracies)


def add_metric_keyword_highlights(df):
    # --- Multi-keyword matching ---
    keyword_input = st.text_input("Search / highlight by keyword(s) (comma-separated)", value="dtw")

    multi_color = st.checkbox("Color bars by individual keyword?", value=True)

    df = df.copy()

    if keyword_input.strip():
        keywords = [k.strip().lower() for k in keyword_input.split(",") if k.strip()]

        def match_keywords(text):
            matched = [kw for kw in keywords if kw in text.lower()]
            if matched:
                return " + ".join(sorted(set(matched))) if multi_color else f"Matched: {', '.join(keywords)}"
            return "Other"

        df["highlight"] = df.apply(
            lambda row: match_keywords(row[METRIC_COL]) if pd.notnull(row[METRIC_COL]) else "Other", axis=1
        )

    else:
        # df["highlight"] = "All"
        df["highlight"] = df[DESCRIPTIVE_NAME_COL]
    return df


st.title("🪵 KNN Viewer")

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

accuracy_df = None
with st.status(f"Calculating accuracies for {len(parquets)} files...", expanded=True) as status:
    accuracy_df = analyze_neighbor_files(file_paths=parquets, min_k_val=min_k_val, max_k_val=max_k_val)
    status.update(
        label=f"Accuracies Calculated: {len(accuracy_df['accuracy'])}",
        state="complete",
        expanded=False,
    )

if len(accuracy_df) > 0:
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
    st.write(f"We have {len(accuracy_df)} rows with {len(accuracy_df[METRIC_COL].unique())} metrics")
    st.dataframe(accuracy_df)

    accuracy_df = add_metric_keyword_highlights(accuracy_df)

    accuracy_df_csv_data = accuracy_df.to_csv(index=False)
    st.download_button(
        label="Download KNN Accuracies CSV",
        data=accuracy_df_csv_data,
        file_name=f"knn_metric_accuracies_{len(parquets)}metrics_k1_to_k{max_k_val}.csv",
        mime="text/csv",
    )

    plot_title = st.text_input(
        label="Plot Title",
        value="Pose Distance Metrics KNN Accuracy, ASL Citizen testXtrain, 1100 test poses, 40154 train poses",
    )

    # Create the Plotly plot
    # fig = px.line(
    #     accuracy_df,
    #     x="k",
    #     y="accuracy",
    #     color=DESCRIPTIVE_NAME_COL,
    #     # color="highlight",
    #     hover_data=[DESCRIPTIVE_NAME_COL, METRIC_COL],
    #     title=plot_title,
    # )

    fig = plot_accuracy_by_k_grouped_color(
        df=accuracy_df,
        trace_col="DESCRIPTIVE_NAME",
        hover_cols=["DESCRIPTIVE_NAME", "METRIC"],
        color_col="highlight",
        plot_title=plot_title,
    )

    # Display the plot as JSON
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # Mean accuracy across all k
    agg_accuracy = (
        accuracy_df.groupby(["METRIC", "DESCRIPTIVE_NAME"])
        .agg(
            avg_accuracy=("accuracy", "mean"),
            min_k=("k", "min"),
            max_k=("k", "max"),
            num_k_values=("k", "nunique"),  # Optional: counts how many unique k values
        )
        .reset_index()
    )

    st.subheader("Average Accuracy across all K:")
    st.dataframe(agg_accuracy)
    agg_accuracy_df_csv_data = accuracy_df.to_csv(index=False)
    st.download_button(
        label="Download Mean KNN Accuracies CSV",
        data=agg_accuracy_df_csv_data,
        file_name=f"knn_metric_meanaccuracies_{len(parquets)}metrics_k1_to_k{max_k_val}.csv",
        mime="text/csv",
    )


# if dfs:
#     df = pd.concat(dfs)
#     st.subheader(f"Loaded {len(df)} rows from {len(parquets)} files")
#     st.dataframe(df)
# conda activate /opt/home/cleong/envs/pose_eval_src && streamlit run pose_evaluation/analysis/st_parquet_viewer.py
