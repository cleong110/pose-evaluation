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


def create_gloss_tuple(row):
    gloss_1 = row["subject"].split(":")[-1].upper()  # Extract and capitalize subject gloss
    gloss_2 = row["object"].split(":")[-1].upper()  # Extract and capitalize object gloss
    return tuple(sorted([gloss_1, gloss_2], reverse=True))  # Sort and create tuple


if __name__ == "__main__":

    # score_analysis_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\embedding_analysis\score_analysis")
    # score_analysis_folder = Path(
    #     r"C:\Users\Colin\data\similar_but_not_the_same\combined_embedding_and_pose_stats\score_analysis"
    # )
    score_analysis_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\nonsense_metrics\score_analysis")
    plots_folder = score_analysis_folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    # EmbeddingDistanceMetric_sem-lex_cosine_out_of_class_scores_by_gloss.csv
    scores_by_gloss_csvs = list(score_analysis_folder.rglob("*out_of_class_scores_by_gloss.csv"))
    print(f"Found {len(scores_by_gloss_csvs)} csvs containing scores by gloss")

    score_by_gloss_dfs_list = []

    for csv_file in tqdm(scores_by_gloss_csvs, desc="Loading CSVs"):
        csv_df = pd.read_csv(csv_file)
        score_by_gloss_dfs_list.append(csv_df)

    scores_by_gloss_df = pd.concat(score_by_gloss_dfs_list)
    scores_by_gloss_df[["Gloss A", "Gloss B"]] = scores_by_gloss_df["gloss_tuple"].str.extract(
        r"\('([^']*)', '([^']*)'\)"
    )
    print(scores_by_gloss_df.info())
    print(scores_by_gloss_df.head())

    ################################################################
    # Adding the ASL Knowledge Graph: alas, none of these are in here.
    # asl_knowledge_graph_df = pd.read_csv(score_analysis_folder.parent.parent / "edges_v2_noweights.tsv", delimiter="\t")
    # # get the "response" relation
    # asl_knowledge_graph_df = asl_knowledge_graph_df[asl_knowledge_graph_df["relation"] == "response"]

    # # add gloss_tuple
    # asl_knowledge_graph_df["gloss_tuple"] = asl_knowledge_graph_df.apply(create_gloss_tuple, axis=1)
    # print(asl_knowledge_graph_df.info())
    # print(asl_knowledge_graph_df.head())

    # # gloss_tuple_set = set(asl_knowledge_graph_df["gloss_tuple"])
    # scores_by_gloss_df["semantically_related"] = scores_by_gloss_df["gloss_tuple"].isin(
    #     set(asl_knowledge_graph_df["gloss_tuple"])
    # )
    # print(set(asl_knowledge_graph_df["gloss_tuple"]).intersection(set(scores_by_gloss_df["gloss_tuple"])))
    # print(scores_by_gloss_df[scores_by_gloss_df["semantically_related"] == True])
    # exit()

    # Example:EmbeddingDistanceMetric_sem-lex_cosine_out_of_class_scores_by_gloss.csv
    # gloss_tuple	count	mean	max	min	std	known_similar	metric	rank
    # ('DEER', 'MOOSE')	870	0.125122927317674	0.364014387130737	0.0213934779167175	0.0508914505643154	True	EmbeddingDistanceMetric_sem-lex_cosine	1
    # ('HUG', 'LOVE')	930	0.129592590242304	0.372491121292114	0.0322074890136718	0.0600298605817267	True	EmbeddingDistanceMetric_sem-lex_cosine	2
    # ('BUT', 'DIFFERENT')	930	0.139856820337234	0.28130042552948	0.023825466632843	0.0414892420832739	True	EmbeddingDistanceMetric_sem-lex_cosine	3
    # ('FAVORITE', 'GOOD')	32	0.140619456768036	0.271014928817749	0.0476789474487304	0.0497585180016914	False	EmbeddingDistanceMetric_sem-lex_cosine	4
    # ('ANIMAL', 'HAVE')	930	0.157980350461057	0.340065956115723	0.0243424773216247	0.051135208995542	True	EmbeddingDistanceMetric_sem-lex_cosine	5
    # ('CHALLENGE', 'GAME')	930	0.163187250334729	0.346950709819794	0.0585821270942688	0.050448722682526	True	EmbeddingDistanceMetric_sem-lex_cosine	6
    # ('SATURDAY', 'TUESDAY')	31	0.163267293284016	0.318226993083954	0.0510987639427185	0.0666946134674303	True	EmbeddingDistanceMetric_sem-lex_cosine	7
    # ('FAVORITE', 'SPICY')	32	0.169413153082132	0.287134885787964	0.104118287563324	0.0507829287998875	False	EmbeddingDistanceMetric_sem-lex_cosine	8
    # ('FAMILY', 'SANTA')	30	0.174653542041779	0.230535089969635	0.130753219127655	0.0242531823125804	False	EmbeddingDistanceMetric_sem-lex_cosine	9
    # ('FAVORITE', 'TASTE')	928	0.181375090739336	0.422960162162781	0.031768798828125	0.0689190830802117	True	EmbeddingDistanceMetric_sem-lex_cosine	10

    metrics = scores_by_gloss_df["metric"].unique().tolist()
    correlation_plots_folder = plots_folder / "metric_correlations"
    histogram_plots_folder = plots_folder / "metric_histograms"
    correlation_plots_folder.mkdir(exist_ok=True)
    histogram_plots_folder.mkdir(exist_ok=True)

    combinations_count = math.comb(len(metrics), 2)
    print(f"We have intergloss scores for {len(metrics)} metrics, so there are {combinations_count} combinations")

    for metric1, metric2 in tqdm(
        combinations(metrics, 2), desc="generating correlation plots", total=combinations_count
    ):
        plot_metric_scatter(
            scores_by_gloss_df,
            metric1,
            metric2,
            show=False,
            png_path=correlation_plots_folder / f"{metric1}_versus_{metric2}.png",
        )
        plot_metric_scatter_interactive(
            scores_by_gloss_df,
            metric1,
            metric2,
            show=False,
            html_path=correlation_plots_folder / f"{metric1}_versus_{metric2}.html",
        )
        gc.collect()

    for metric in tqdm(metrics, desc="Generating histogram plots"):
        plot_metric_histogram(
            scores_by_gloss_df,
            metric=metric,
            col="mean",
            out_path=histogram_plots_folder / f"{metric}_intergloss_hist.png",
        )
        gc.collect()
