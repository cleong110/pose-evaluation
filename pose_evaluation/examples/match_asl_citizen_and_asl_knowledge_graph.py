from itertools import combinations
import math
from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import seaborn as sns
import gc

from pose_evaluation.examples.load_asl_citizen_df_and_compare import load_dataset_stats

sns.set_theme()
import matplotlib.pyplot as plt


def plot_metric_histogram(
    df: pd.DataFrame,
    col: str,
    metric: str,
    bins: int = 10,
    kde: bool = True,
    color: str = "royalblue",
    show=False,
    out_path: Optional[Path] = None,
):
    """
    Plots a histogram of the 'mean' column, filtering the dataframe by the specified 'metric'.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        metric (str): The metric value to filter the dataframe.
        bins (int): Number of bins in the histogram.
        kde (bool): Whether to include the KDE curve.
        color (str): Color of the histogram bars.
    """
    # Filter dataframe by metric
    df_filtered = df[df["metric"] == metric]

    if df_filtered.empty:
        print(f"No data found for metric: {metric}")
        return

    # Set seaborn style
    # sns.set_style("whitegrid")

    # Create the plot
    plt.figure(figsize=(7, 5))
    sns.histplot(
        df_filtered[col],
        bins=bins,
        kde=kde,
        #  color=color, edgecolor="black"
    )

    # Labels and title
    plt.xlabel(f"{col} Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Mean Intergloss ({metric})", fontsize=14)

    if show:
        plt.show()
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.close()


def plot_metric_scatter_interactive(
    df: pd.DataFrame, metric_x: str, metric_y: str, show: bool = False, html_path: Optional[Path] = None
):
    # Filter for the two specified metrics
    df_x = df[df["metric"] == metric_x].rename(columns={"mean": "score_x"})
    df_y = df[df["metric"] == metric_y].rename(columns={"mean": "score_y"})

    # Merge on Gloss A and Gloss B
    merged_df = df_x.merge(df_y, on=["Gloss A", "Gloss B"], suffixes=("", "_y"))

    # Create labels
    merged_df["label"] = merged_df["Gloss A"] + " / " + merged_df["Gloss B"]

    # Create scatter plot without labels
    fig = px.scatter(
        merged_df,
        x="score_x",
        y="score_y",
        color="known_similar",  # Color by 'known_similar' column
        title=f"{metric_x} vs {metric_y}",
        labels={"score_x": metric_x, "score_y": metric_y},
        color_continuous_scale="Viridis",
    )

    # Add text labels as a separate trace with legend entry
    text_trace = go.Scatter(
        x=merged_df["score_x"],
        y=merged_df["score_y"],
        text=merged_df["label"],
        mode="text",
        textposition="top center",
        name="Labels",  # Separate legend entry
        showlegend=True,  # Allow toggling via legend
    )

    fig.add_trace(text_trace)

    # Improve layout
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        xaxis_title=metric_x,
        yaxis_title=metric_y,
        hovermode="closest",
    )

    if show:
        fig.show()
    if html_path:
        fig.write_html(html_path)

    del fig


def plot_metric_scatter(
    df: pd.DataFrame, metric_x: str, metric_y: str, show: bool = False, png_path: Optional[Path] = None
):
    # Filter for the two specified metrics
    df_x = df[df["metric"] == metric_x].rename(columns={"mean": "score_x"})

    df_y = df[df["metric"] == metric_y].rename(columns={"mean": "score_y"})

    # Merge on Gloss A and Gloss B
    merged_df = df_x.merge(df_y, on=["Gloss A", "Gloss B"], suffixes=("", "_y"))

    # Create labels
    merged_df["label"] = merged_df["Gloss A"] + " / " + merged_df["Gloss B"]
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.scatterplot(
        data=merged_df,
        x="score_x",
        y="score_y",
        alpha=0.7,
        hue="known_similar",  # Color by the 'known_similar' column
        palette={True: "red", False: "blue"},  # Assign custom colors for True/False
    )  # Use seaborn's scatterplot
    plt.xlabel(f"{metric_x}")
    plt.ylabel(f"{metric_y}")
    plt.title(f"Mean Intergloss Scores:\n{metric_x} vs\n {metric_y}")
    plt.grid(True)  # Add grid for readability

    # plt.show()
    if show:
        plt.show()
    if png_path:
        plt.tight_layout()
        plt.savefig(png_path.with_suffix(".png"))

    plt.close()


def create_gloss_tuple(row, column_name_a="subject", column_name_b="object"):
    gloss_1 = row[column_name_a].split(":")[-1].upper()  # Extract and capitalize subject gloss
    gloss_2 = row[column_name_b].split(":")[-1].upper()  # Extract and capitalize object gloss
    return tuple(sorted([gloss_1, gloss_2]))  # Sort and create tuple


if __name__ == "__main__":

    # score_analysis_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\embedding_analysis\score_analysis")
    score_analysis_folder = Path(
        r"C:\Users\Colin\data\similar_but_not_the_same\combined_embedding_and_pose_stats\score_analysis"
    )
    data_folder = Path(r"C:\Users\Colin\data\ASL_Citizen\ASL_Citizen")
    json_name = "asl_citizen_dataset_stats.json"

    df = load_dataset_stats(data_folder, json_name)

    print(df.head())
    print(df.info())
    # plots_folder = score_analysis_folder / "plots"
    # plots_folder.mkdir(exist_ok=True)
    # # EmbeddingDistanceMetric_sem-lex_cosine_out_of_class_scores_by_gloss.csv
    # scores_by_gloss_csvs = list(score_analysis_folder.rglob("*out_of_class_scores_by_gloss.csv"))
    # print(f"Found {len(scores_by_gloss_csvs)} csvs containing scores by gloss")

    # score_by_gloss_dfs_list = []

    # for csv_file in tqdm(scores_by_gloss_csvs, desc="Loading CSVs"):
    #     csv_df = pd.read_csv(csv_file)
    #     score_by_gloss_dfs_list.append(csv_df)

    # scores_by_gloss_df = pd.concat(score_by_gloss_dfs_list)
    # scores_by_gloss_df[["Gloss A", "Gloss B"]] = scores_by_gloss_df["gloss_tuple"].str.extract(
    #     r"\('([^']*)', '([^']*)'\)"
    # )
    # print(scores_by_gloss_df.info())
    # print(scores_by_gloss_df.head())

    ################################################################
    # Adding the ASL Knowledge Graph: alas, none of these are in here.
    asl_knowledge_graph_df = pd.read_csv(score_analysis_folder.parent.parent / "edges_v2_noweights.tsv", delimiter="\t")
    # get the "response" relation
    asl_knowledge_graph_df = asl_knowledge_graph_df[asl_knowledge_graph_df["relation"] == "response"]

    # add gloss_tuple
    asl_knowledge_graph_df["gloss_tuple"] = asl_knowledge_graph_df.apply(create_gloss_tuple, axis=1)
    print(asl_knowledge_graph_df.info())
    print(asl_knowledge_graph_df.head())

    # Extract unique glosses from gloss_tuple column
    unique_glosses = set(gloss for tup in asl_knowledge_graph_df["gloss_tuple"] for gloss in tup)

    # Convert to a sorted list (optional)
    unique_glosses_list = sorted(unique_glosses)

    print(unique_glosses_list)

    exit()
