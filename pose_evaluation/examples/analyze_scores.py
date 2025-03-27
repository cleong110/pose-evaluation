from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import plotly.express as px

if __name__ == "__main__":

    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis_with_times\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\embedding_analysis\scores")
    stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\combined_embedding_and_pose_stats\scores")

    known_similar_pairs_df = pd.read_csv(stats_folder.parent.parent / "similar_sign_pairs.csv")
    known_similar_pairs_df["gloss_tuple"] = known_similar_pairs_df.apply(
        lambda row: tuple(sorted([row["signA"], row["signB"]])), axis=1
    )
    known_similar_gloss_tuples = known_similar_pairs_df["gloss_tuple"].unique().tolist()

    analysis_folder = stats_folder.parent / "score_analysis"
    analysis_folder.mkdir(exist_ok=True)

    csv_stats_dfs = []
    for csv_file in tqdm(stats_folder.glob("*.csv"), desc="Loading stats csvs"):
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

        csv_stats_dfs.append(csv_stats_df)
        # print(f"{len(csv_stats_dfs[-1])} rows loaded")
    stats_df = pd.concat(csv_stats_dfs)

    # Convert "TRUE"
    stats_df["Gloss A"] = stats_df["Gloss A"].astype(str)
    stats_df["Gloss B"] = stats_df["Gloss B"].astype(str)
    print(f"{stats_df}")

    # Normalize path pairs by sorting them
    stats_df["path_tuple"] = stats_df.apply(
        lambda row: tuple(sorted([row["Gloss A Path"], row["Gloss B Path"]])), axis=1
    )

    stats_df["gloss_tuple"] = stats_df.apply(lambda row: tuple(sorted([row["Gloss A"], row["Gloss B"]])), axis=1)

    # # Find duplicate rows based on "metric" and sorted paths
    # duplicates = stats_df[stats_df.duplicated(subset=["metric", "path_tuple"], keep=False)]
    # # Drop helper column
    # duplicates = duplicates.drop(columns=["path_tuple"])
    # print("Duplicate rows:")
    # print(duplicates)

    # keep only the deduplicated
    # stats_df = stats_df.drop_duplicates(subset=["metric", "path_tuple"], keep="first").drop(columns=["path_tuple"])
    stats_df = stats_df.drop_duplicates(subset=["metric", "path_tuple"], keep="first")  # actually, path_tuple is useful
    stats_df.drop(columns=["gloss_tuple", "path_tuple", "Unnamed: 0"]).to_csv(
        analysis_folder / "deduplicated_scores.csv", index=False
    )

    # print(f"Deduplicated Rows:")
    # print(stats_df)
    # print(stats_df.info())

    metric_stats = defaultdict(list)

    metrics_to_analyze = stats_df["metric"].unique()
    # metrics_to_analyze = ["n-dtai-DTW-MJE (fast)", "MJE", "nMJE"]
    for metric in metrics_to_analyze:
        print(f"*" * 50)

        metric_df = stats_df[stats_df["metric"] == metric]
        print(metric_df.head())
        print(metric_df.columns)
        metric_df["time"].head()
        print(metric_df["time"].mean())
        print(len(metric_df[metric_df["time"].isna()]))

        signatures = metric_df["signature"].str.split("=").str[0].unique()
        nan_rows = metric_df[metric_df["signature"].isna()]

        # print("nan_rows:")
        # print(nan_rows)

        # empty_rows = metric_df[metric_df["signature"] == ""]
        # print("empty_rows")
        # print(empty_rows)
        # print(signatures)

        path_tuples = metric_df["path_tuple"].unique()
        gloss_tuples = metric_df["gloss_tuple"].unique()
        metric_glosses = metric_df["Gloss A"].unique().tolist()
        metric_glosses.extend(metric_df["Gloss B"].unique().tolist())
        metric_glosses = set(metric_glosses)
        print(
            f"{metric} has {len(metric_df)} results, covering {len(metric_glosses)} glosses, in {len(gloss_tuples)} combinations with {len(path_tuples)} file combinations"
        )

        # Group by 'gloss_tuple' and compute stats for 'score'
        not_self_score_df = metric_df[metric_df["Gloss A"] != metric_df["Gloss B"]]
        print(f"{metric} has {len(not_self_score_df)} out-of-class-scores: ")
        out_of_class_gloss_stats = (
            not_self_score_df.groupby("gloss_tuple")["score"].agg(["count", "mean", "max", "min", "std"]).reset_index()
        )
        out_of_class_gloss_stats = out_of_class_gloss_stats.sort_values("count", ascending=False)
        out_of_class_gloss_stats["known_similar"] = out_of_class_gloss_stats["gloss_tuple"].isin(
            known_similar_gloss_tuples
        )
        out_of_class_gloss_stats["metric"] = metric

        print(f"Gloss Stats for {metric} (not self-score)")
        print(out_of_class_gloss_stats)

        print(f"Most Similar of {len(out_of_class_gloss_stats)} Pairs by mean {metric} score:")
        metric_most_similar_glosses = out_of_class_gloss_stats.nsmallest(100, "mean")
        print(metric_most_similar_glosses.head(10))

        print(f"Least Similar of {len(out_of_class_gloss_stats)} Pairs by mean {metric} score:")
        metric_least_similar_glosses = out_of_class_gloss_stats.nlargest(100, "mean")
        print(metric_least_similar_glosses.head(10))

        ################
        # Self-scores
        self_scores_df = metric_df[metric_df["Gloss A"] == metric_df["Gloss B"]]
        print(f"{metric} has {len(self_scores_df)} self-scores: ")

        print(f"Gloss stats for self-scores for {metric}")
        gloss_stats_self = (
            self_scores_df.groupby("gloss_tuple")["score"].agg(["count", "mean", "max", "min", "std"]).reset_index()
        )
        print(gloss_stats_self)

        print(f"Most consistent of {len(metric_glosses)} glosses by {metric} std deviation")
        metric_most_consistent_glosses = gloss_stats_self.nsmallest(100, "std")
        print(metric_most_consistent_glosses.head(10))

        print(f"Least consistent of {len(metric_glosses)} glosses by {metric} std deviation")
        metric_least_consistent_glosses = gloss_stats_self.nlargest(100, "std")
        print(metric_least_consistent_glosses.head(10))

        ###############################
        # Add to metric stats
        # "metric": [],
        # "metric_signature": [],
        # "unique_gloss_pairs": [],
        # "total_count": [],
        # "self_scores_count": [],
        # "mean": [],
        # "max": [],
        # "std": [],
        # "std_of_gloss_std": [],
        # "std_of_of_gloss_mean": [],
        # "mean_of_gloss_mean": [],
        # metric_stats["metric"].append(metric)

        ####
        # Average Rank of known_similar scores
        # Rank by "mean" in ascending order (use ascending=False for descending)
        out_of_class_gloss_stats = out_of_class_gloss_stats.sort_values("mean", ascending=True)

        # Add a rank column based on the sorted order
        out_of_class_gloss_stats["rank"] = range(1, len(out_of_class_gloss_stats) + 1)

        # Compute the average rank for known_similar == True
        # avg_rank_of_known_similar = out_of_class_gloss_stats.loc[
        #     out_of_class_gloss_stats["known_similar"], "rank"
        # ].mean()

        # print(f"Average Rank of Known Similar: {avg_rank_of_known_similar}")

        mrr_rank_of_known_similar = (
            1 / out_of_class_gloss_stats.loc[out_of_class_gloss_stats["known_similar"], "rank"]
        ).mean()
        print(f"Mean Reciprocal Rank (MRR) of Known Similar: {mrr_rank_of_known_similar:.4f}")

        assert len(signatures) == 1, signatures
        metric_stats["metric"].append(metric)
        metric_stats["metric_signature"].append(signatures[0])
        metric_stats["unique_gloss_pairs"].append(len(gloss_tuples))
        metric_stats["unique_glosses"].append(len(metric_glosses))
        metric_stats["total_count"].append(len(metric_df))

        metric_stats["self_scores_count"].append(len(self_scores_df))
        metric_stats["mean_self_score"].append(self_scores_df["score"].mean())
        metric_stats["std_self_score"].append(self_scores_df["score"].std())
        metric_stats["mean_of_gloss_self_score_means"].append(gloss_stats_self["mean"].mean())
        metric_stats["std_of_gloss_self_score_means"].append(gloss_stats_self["mean"].std())
        # metric_stats["mean_of_gloss_self_score_stds"].append(gloss_stats_self["std"].mean())
        # metric_stats["std_of_gloss_self_score_stds"].append(gloss_stats_self["std"].std())

        metric_stats["out_of_class_scores_count"].append(len(not_self_score_df))
        metric_stats["mean_out_of_class_score"].append(not_self_score_df["score"].mean())
        metric_stats["std_out_of_class_score"].append(not_self_score_df["score"].std())
        metric_stats["mean_of_out_of_class_score_means"].append(out_of_class_gloss_stats["mean"].mean())
        metric_stats["std_of_out_of_class_score_means"].append(out_of_class_gloss_stats["mean"].std())

        metric_stats["mean_score_time"].append(metric_df["time"].mean())
        metric_stats["std_dev_of_score_time"].append(metric_df["time"].std())

        # metric_stats["average_rank_of_known_similar_glosses"].append(avg_rank_of_known_similar)
        metric_stats["mrr_of_known_similar_glosses"].append(mrr_rank_of_known_similar)

        metric_stats["mean_out_mean_in_ratio"].append(
            not_self_score_df["score"].mean() / self_scores_df["score"].mean()
        )

        print(f"Saving most/least similar, least/most consistent to {analysis_folder}")
        out_of_class_gloss_stats.to_csv(analysis_folder / f"{metric}_out_of_class_scores_by_gloss.csv", index=False)
        metric_most_similar_glosses.to_csv(analysis_folder / f"{metric}_most_similar_glosses.csv", index=False)
        metric_least_similar_glosses.to_csv(analysis_folder / f"{metric}_least_similar_glosses.csv", index=False)
        metric_most_consistent_glosses.to_csv(analysis_folder / f"{metric}_most_consistent_glosses.csv", index=False)
        metric_least_consistent_glosses.to_csv(analysis_folder / f"{metric}_least_consistent_glosses.csv", index=False)

    metric_stats = pd.DataFrame(metric_stats)
    print(metric_stats)
    metric_stats_out = analysis_folder / "stats_by_metric.csv"
    metric_stats.to_csv(metric_stats_out, index=False)

    fig = px.box(
        stats_df,
        x="metric",
        y="time",
        # points=None,
        title="Metric Pairwise Scoring Times (s)",
        color="metric",
    )
    # fig.update_layout(xaxis_type="category")
    # fig.update_layout(xaxis={"categoryorder": "category ascending"})
    fig.update_layout(xaxis={"categoryorder": "trace"})

    fig.update_layout(xaxis_tickangle=-45)  # Rotate x labels for readability
    fig.write_html(analysis_folder / "metric_pairwise_scoring_time_distributions.html")
