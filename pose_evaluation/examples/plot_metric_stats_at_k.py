from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Union


def plot_metric_traces(df: pd.DataFrame, column_name: str, title="") -> go.Figure:
    """
    Creates an interactive line plot using Plotly, where each unique value of "metric"
    is a separate trace, "k" is on the x-axis, and the specified column is on the y-axis.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing 'metric', 'k', and the target column.
    column_name (str): The name of the column to use for the y-axis.

    Returns:
    go.Figure: A Plotly figure object.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    fig = go.Figure()

    for metric in df["metric"].unique():
        metric_df = df[df["metric"] == metric]
        fig.add_trace(go.Scatter(x=metric_df["k"], y=metric_df[column_name], mode="lines+markers", name=str(metric)))

    if not title:
        title = f"{column_name} vs k"
    fig.update_layout(title=title, xaxis_title="k", yaxis_title=column_name, template="plotly_white")

    return fig


if __name__ == "__main__":

    # analysis_folder = Path(
    #     r"C:\Users\Colin\data\similar_but_not_the_same\combined_embedding_and_pose_stats\score_analysis"
    # )
    analysis_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\nonsense_metrics\score_analysis")

    stats_csvs = analysis_folder.glob("stats_by_metric_at_k*.csv")

    for stat_csv in stats_csvs:
        print(f"Graphing for {stat_csv}")
        metric_stats_at_k_df = pd.read_csv(analysis_folder / stat_csv)
        print(metric_stats_at_k_df)

        for stat in ["recall@k", "precision@k", "mean_match_count@k"]:
            title = ""
            if "excluding" in stat_csv.name:
                title = f"{stat} vs k, Excluding Known Similar Glosses"
            elif "full_population" in stat_csv.name:
                title = f"{stat} vs k, Full Population Including Known Similar Glosses"

            fig = plot_metric_traces(metric_stats_at_k_df, stat, title=title)
            fig.show()
            fig.write_html(analysis_folder / "plots" / f"{title}.html")
