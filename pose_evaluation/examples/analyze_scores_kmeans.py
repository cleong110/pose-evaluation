from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import argparse


def compute_retrieval_stats(group, k):
    top_k = group.head(k)  # Get top-k retrieved samples for this query

    # Correct retrievals
    correct_top_k = (top_k["Gloss A"] == top_k["Gloss B"]).sum()  # Count matches
    total_possible_correct = (group["Gloss A"] == group["Gloss B"]).sum()  # All matches in dataset

    # Compute retrieval metrics
    accuracy_at_k = int(correct_top_k > 0)  # At least 1 correct retrieval
    precision_at_k = correct_top_k / k if k > 0 else 0
    recall_at_k = correct_top_k / total_possible_correct if total_possible_correct > 0 else 0

    # Majority vote classifier: which class appears most in top-k?
    gloss_b_counts = Counter(top_k["Gloss B"])  # Count occurrences of each class
    majority_gloss = max(gloss_b_counts, key=gloss_b_counts.get)  # Most common label in top-k
    majority_gloss_count = gloss_b_counts[majority_gloss] if majority_gloss else 0
    classifier_correct = int(majority_gloss == group["Gloss A"].iloc[0])  # Check majority vote match

    return pd.Series(
        {
            "accuracy@k": accuracy_at_k,
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "classifier_correct": classifier_correct,  # Majority vote classifier correctness
            "majority_gloss": majority_gloss,
            "majority_gloss_count": majority_gloss_count,
            # Add gloss_b_counts here?
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Scores from top-k perspective")
    parser.add_argument("k", type=int, nargs="+", help="List of k values for analysis")

    parser.add_argument(
        "--stats_folder",
        type=Path,
        help="Path to the directory containing score CSVs",
        default=Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis_with_times\scores"),
    )

    parser.add_argument("--known_similar_pairs_csv", type=Path)

    parser.add_argument("--analysis_output_folder", type=Path)

    args = parser.parse_args()

    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis\scores")
    stats_folder = args.stats_folder

    if args.known_similar_pairs_csv is None:
        known_similar_pairs_df = pd.read_csv(stats_folder.parent.parent / "similar_sign_pairs.csv")
    else:
        known_similar_pairs_df = pd.read_csv(args.known_similar_pairs_csv)
    known_similar_pairs_df["gloss_tuple"] = known_similar_pairs_df.apply(
        lambda row: tuple(sorted([row["signA"], row["signB"]])), axis=1
    )
    known_similar_gloss_tuples = known_similar_pairs_df["gloss_tuple"].unique().tolist()

    if args.analysis_output_folder is None:
        analysis_folder = stats_folder.parent / "kmeans_analysis"
    else:
        analysis_folder = args.analysis_output_folder

    all_k_data = []
    for k in args.k:

        k_analysis_folder = analysis_folder / f"k_{k}"

        k_analysis_folder.mkdir(exist_ok=True, parents=True)

        # csv_stats_dfs = []
        # csv_kmeans_dfs = []
        metric_stats_dfs = []

        for i, csv_file in enumerate(tqdm(stats_folder.glob("*.csv"), desc="Loading stats csvs")):
            # print(f"Reading {csv_file}")
            csv_stats_df = pd.read_csv(csv_file)

            # Getting a weird error about cannot sort because one of the columns is bool
            # bool_rows = csv_stats_df[
            #     (csv_stats_df["Gloss A"].apply(lambda x: isinstance(x, bool)))
            #     | (csv_stats_df["Gloss B"].apply(lambda x: isinstance(x, bool)))
            # ]

            # if len(bool_rows) > 0:
            #     print(bool_rows)
            #     print(f"csv_file: {csv_file}")  # TRUE_TELL_MJE_score_results.csv
            #     csv_stats_df[]

            # csv_stats_dfs.append(csv_stats_df)
            # print(f"{len(csv_stats_dfs[-1])} rows loaded")

            # Convert "TRUE"
            csv_stats_df["Gloss A"] = csv_stats_df["Gloss A"].astype(str)
            csv_stats_df["Gloss A"] = csv_stats_df["Gloss A"].str.upper()
            csv_stats_df["Gloss B"] = csv_stats_df["Gloss B"].astype(str)
            csv_stats_df["Gloss B"] = csv_stats_df["Gloss B"].str.upper()

            # get first pair
            file_similar_pair = csv_stats_df["Gloss A"].iloc[0], csv_stats_df["Gloss B"].iloc[0]
            file_metric = csv_stats_df["metric"]
            # print(f"{csv_file} similar pair: {file_similar_pair}")
            assert f"{file_similar_pair[0]}_{file_similar_pair[1]}" in str(
                csv_file
            ), f"{csv_file} + {file_similar_pair}"
            # assert file_similar_pair[1] in str(csv_file)

            csv_kmeans_df = csv_stats_df[csv_stats_df["Gloss B"] != file_similar_pair[1]]
            # print(csv_stats_df)
            # print(csv_kmeans_df)
            # print(csv_kmeans_df["Gloss A"].unique())
            # print(csv_kmeans_df["Gloss B"].unique())
            file_self_scores = csv_kmeans_df[csv_kmeans_df["Gloss A"] == csv_kmeans_df["Gloss B"]]
            file_intraclass_scores = csv_kmeans_df[csv_kmeans_df["Gloss A"] != csv_kmeans_df["Gloss B"]]
            # print(len(file_self_scores))
            # print(len(file_intraclass_scores))
            csv_kmeans_df["Experiment ID"] = i
            csv_kmeans_df = csv_kmeans_df.drop(columns=["Unnamed: 0"])
            csv_kmeans_df = csv_kmeans_df.sort_values("score", ascending=True)
            known_similar_pairs_df["gloss_tuple"] = known_similar_pairs_df.apply(
                lambda row: tuple(sorted([row["signA"], row["signB"]])), axis=1
            )
            # csv_kmeans_df['ref_name'] = csv_kmeans_df.apply(lambda row: sorted([row["signA"], row["signB"]], axis=1)
            csv_kmeans_df["Gloss A Filename"] = csv_kmeans_df["Gloss A Path"].apply(lambda x: Path(x).name)
            csv_kmeans_df["Gloss B Filename"] = csv_kmeans_df["Gloss B Path"].apply(lambda x: Path(x).name)
            # print(csv_kmeans_df[["Gloss A", "Gloss B", "score", "Gloss A Filename", "Gloss B Filename"]].head(10))
            # csv_kmeans_dfs.append(csv_kmeans_df)

            # pull out self-scores for samples
            csv_kmeans_df = csv_kmeans_df[csv_kmeans_df["Gloss A Filename"] != csv_kmeans_df["Gloss B Filename"]]

            top_k_df = csv_kmeans_df.groupby("Gloss A Filename", group_keys=False).apply(  # Group first
                lambda g: g.sort_values(by="score", ascending=True).head(k)
            )  # Sort within each group
            # print(top_k_df[["Gloss A", "Gloss B", "score", "Gloss A Filename", "Gloss B Filename"]])
            summary_df = (
                csv_kmeans_df.groupby("Gloss A Filename", group_keys=False)
                .apply(compute_retrieval_stats, k)
                .reset_index()
            )
            # print(summary_df)
            top_k_df.to_csv(k_analysis_folder / f"{csv_file.stem}_grouped_results.csv")
            summary_df["metric"] = csv_stats_df["metric"]
            # summary_df["signature"] = csv_stats_df["signature"]
            summary_df.to_csv(k_analysis_folder / f"{csv_file.stem}_retrieval_kmeans_summary.csv")
            summary_df["k"] = k

            metric_stats_dfs.append(summary_df)

        metric_stats_df = pd.concat(metric_stats_dfs)
        metric_stats_df["k"] = k
        metric_stats_df.to_csv(k_analysis_folder / f"metric_stats_k_{k}.csv")

        pivot_table = metric_stats_df.pivot_table(
            index="metric", values=["accuracy@k", "precision@k", "recall@k", "classifier_correct"], aggfunc="mean"
        ).reset_index()
        pivot_table["k"] = k
        # print(pivot_table)
        pivot_table.to_csv(k_analysis_folder / f"mean_by_metric_k_{k}.csv")
        all_k_data.append(pivot_table)

        # print(metric_stats_df.head())
        # print(metric_stats_df.info())
        metric_summary_stat_names = ["accuracy@k", "precision@k", "recall@k", "classifier_correct"]
        for stat in metric_summary_stat_names:

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=metric_stats_df, x="metric", y=f"{stat}")
            plt.title(f"{stat} Distribution by Metric (k={k})")
            plt.xticks(rotation=45, ha="right")
            # plt.show()
            plt.tight_layout()
            plt.savefig(k_analysis_folder / f"{stat.replace('@','_at_')}_{k}_distribution.png")

        # stats_df["Gloss A"] = stats_df["Gloss A"].astype(str)
        # stats_df["Gloss B"] = stats_df["Gloss B"].astype(str)
        # print(f"{stats_df}")

        metric_summary = (
            metric_stats_df.groupby("metric")
            .agg(
                {
                    "accuracy@k": ["mean", "std"],
                    "precision@k": ["mean", "std"],
                    "recall@k": ["mean", "std"],
                    "classifier_correct": ["mean"],
                }
            )
            .reset_index()
        )
        print(metric_summary)

        # plt.figure(figsize=(10, 5))
        # sns.violinplot(data=metric_stats_df, x="metric", y="accuracy@k", inner="quartile")
        # plt.title("Accuracy@K Distribution by Metric")
        # plt.xticks(rotation=45)
        # plt.show()

        #
        fig = px.box(
            metric_stats_df,
            x="metric",
            y="accuracy@k",
            points="all",
            title=f"Metric Performance, k={k}",
            color="metric",
        )

        # Add dropdown for selecting metrics
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [{"y": [metric_stats_df[col]]}, {"yaxis.title.text": col}],
                            "label": col,
                            "method": "update",
                        }
                        for col in metric_summary_stat_names
                    ],
                    "direction": "down",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 1.15,
                    "yanchor": "top",
                }
            ]
        )

        fig.update_layout(xaxis_tickangle=-45)
        # fig.show()
        fig.write_html(k_analysis_folder / "metric_performance_distributions_interactive.html")

    tidy_df = pd.concat(all_k_data, ignore_index=True)

    metric_stat_types = ["mean_accuracy@k", "mean_classifier_correct", "mean_precision@k", "mean_recall@k"]

    # Rename columns for clarity
    tidy_df = tidy_df.rename(
        columns={
            "accuracy@k": "mean_accuracy@k",
            "precision@k": "mean_precision@k",
            "recall@k": "mean_recall@k",
            "classifier_correct": "mean_classifier_correct",
        }
    )
    print(tidy_df.head())  # Check structure
    tidy_df.to_csv(analysis_folder / "mean_by_metric_all_k.csv")

    # Loop through each metric and generate a separate plot
    for metric_type in metric_stat_types:
        fig = px.line(
            tidy_df,
            x="k",
            y=metric_type,
            color="metric",
            markers=True,
            labels={"k": "K", metric_type: "Score", "metric": "Metric"},
            title=f"{metric_type} vs. K",
        )
        fig.show()
        fig.write_html(analysis_folder / f"{metric_type}_vs_k.html")
