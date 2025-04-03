from pathlib import Path

import pandas as pd
import plotly.express as px


if __name__ == "__main__":
    analysis_folder = Path(
        r"C:\Users\Colin\data\similar_but_not_the_same\combined_embedding_and_pose_stats\score_analysis"
    )
    metric_stats_csv = analysis_folder / "deduplicated_scores.csv"
    metric_stats_df = pd.read_csv(metric_stats_csv)
    metric_stats_df.info()
    fig = px.box(
        metric_stats_df,
        x="metric",
        y="time",
        points=None,
        title="Metric Pairwise Scoring Times (s)",
        color="metric",
    )
    # fig.update_layout(xaxis_type="category")
    # fig.update_layout(xaxis={"categoryorder": "category ascending"})
    fig.update_layout(xaxis={"categoryorder": "trace"})

    fig.update_layout(xaxis_tickangle=-45)  # Rotate x labels for readability
    fig.write_html(analysis_folder / "metric_pairwise_scoring_time_distributions_no_points.html")
