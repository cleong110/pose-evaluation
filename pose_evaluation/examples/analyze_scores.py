from collections import defaultdict
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px

tqdm.pandas()


# def mean_correct_and_total_population_sizes_per_gloss(df: pd.DataFrame) -> Tuple[float, float]:
#     df = df.copy()
#     pop_sizes = []
#     correct_sizes = []

#     unique_queries = df["Gloss A Path"].unique()  # Only consider actual queries

#     for query_path in tqdm(unique_queries, "calculating population sizes"):
#         # Find all rows where this query appears in EITHER column
#         relevant_rows = df[(df["Gloss A Path"] == query_path) | (df["Gloss B Path"] == query_path)].copy()

#         # Swap so that `query_path` is always in `Gloss A Path` # done already in main function
#         mask = relevant_rows["Gloss B Path"] == query_path
#         relevant_rows.loc[mask, ["Gloss A Path", "Gloss B Path"]] = relevant_rows.loc[
#             mask, ["Gloss B Path", "Gloss A Path"]
#         ].values
#         relevant_rows.loc[mask, ["Gloss A", "Gloss B"]] = relevant_rows.loc[mask, ["Gloss B", "Gloss A"]].values

#         # Exclude self-comparisons
#         filtered_group = relevant_rows[relevant_rows["Gloss B Path"] != query_path]

#         pop_size = len(filtered_group)
#         correct_size = (filtered_group["Gloss A"] == filtered_group["Gloss B"]).sum()

#         pop_sizes.append(pop_size)
#         correct_sizes.append(correct_size)

#         # print(f"Query: {query_path} | Filtered Group Size: {pop_size} | Correct Size: {correct_size}")

#     # Compute per-query means
#     mean_pop_size = np.mean(pop_sizes) if pop_sizes else 0.0
#     mean_correct_size = np.mean(correct_sizes) if correct_sizes else 0.0

#     return float(mean_correct_size), float(mean_pop_size)


def mean_correct_and_total_population_sizes_per_gloss(df: pd.DataFrame) -> Tuple[float, float]:
    df = df.copy()

    # Get all unique "Gloss A Path" values (queries)
    unique_queries = df["Gloss A Path"].unique()

    # Create lists for population sizes and correct sizes
    pop_sizes = []
    correct_sizes = []

    for query_path in tqdm(unique_queries, "calculating population sizes"):
        # Find all rows where this query appears in EITHER column
        relevant_rows = df[(df["Gloss A Path"] == query_path) | (df["Gloss B Path"] == query_path)].copy()

        # Swap rows where query_path is in Gloss B Path
        mask = relevant_rows["Gloss B Path"] == query_path
        relevant_rows.loc[mask, ["Gloss A Path", "Gloss B Path"]] = relevant_rows.loc[
            mask, ["Gloss B Path", "Gloss A Path"]
        ].values
        relevant_rows.loc[mask, ["Gloss A", "Gloss B"]] = relevant_rows.loc[mask, ["Gloss B", "Gloss A"]].values

        # Exclude self-comparisons
        filtered_group = relevant_rows[relevant_rows["Gloss B Path"] != query_path]

        # Compute population size and correct size
        pop_size = len(filtered_group)
        correct_size = (filtered_group["Gloss A"] == filtered_group["Gloss B"]).sum()

        pop_sizes.append(pop_size)
        correct_sizes.append(correct_size)

    # Compute per-query means
    mean_pop_size = np.mean(pop_sizes) if pop_sizes else 0.0
    mean_correct_size = np.mean(correct_sizes) if correct_sizes else 0.0

    return float(mean_correct_size), float(mean_pop_size)


def precision_at_k(df: pd.DataFrame, k: int) -> float:
    precisions = []

    for gloss_a_path, group in df.groupby("Gloss A Path"):
        # Exclude rows where 'Gloss B Path' matches 'Gloss A Path'
        filtered_group = group[group["Gloss B Path"] != gloss_a_path]

        # Sort by score (lower is better)
        sorted_group = filtered_group.sort_values(by="score", ascending=True)

        # Select top-k results
        top_k = sorted_group.head(k)

        # Compute precision@k as (# of correct matches in top-k) / k
        correct_matches = (top_k["Gloss A"] == top_k["Gloss B"]).sum()
        precisions.append(correct_matches / k)

    # Compute mean precision@k over all groups
    # print("average_precisions:", average_precisions)
    # print("Types:", [type(ap) for ap in average_precisions])
    # print("Shapes:", [np.shape(ap) for ap in average_precisions])
    return sum(precisions) / len(precisions) if precisions else 0.0


def recall_at_k(df: pd.DataFrame, k: int) -> float:
    recalls = []

    for gloss_a_path, group in df.groupby("Gloss A Path"):
        # Exclude trivial cases
        filtered_group = group[group["Gloss B Path"] != gloss_a_path]

        # Sort by score (ascending)
        sorted_group = filtered_group.sort_values(by="score", ascending=True)

        # Count total correct matches in the full set
        total_correct = (sorted_group["Gloss A"] == sorted_group["Gloss B"]).sum()
        if total_correct == 0:
            continue  # Skip if there are no true matches

        # Select top-k results
        top_k = sorted_group.head(k)

        # Compute recall@k
        correct_in_top_k = (top_k["Gloss A"] == top_k["Gloss B"]).sum()
        recalls.append(correct_in_top_k / total_correct)

    # Compute mean recall@k
    return sum(recalls) / len(recalls) if recalls else 0.0


def match_count_at_k(df: pd.DataFrame, k: int) -> int:
    match_count = 0

    for gloss_a_path, group in df.groupby("Gloss A Path"):
        # Exclude trivial cases
        filtered_group = group[group["Gloss B Path"] != gloss_a_path]

        # Sort by score (ascending)
        sorted_group = filtered_group.sort_values(by="score", ascending=True)

        # Select top-k results
        top_k = sorted_group.head(k)

        # Count matches where 'Gloss A' == 'Gloss B'
        match_count += (top_k["Gloss A"] == top_k["Gloss B"]).sum()

    return match_count


def mean_average_precision(df: pd.DataFrame) -> float:
    average_precisions = []

    for gloss_a_path, group in df.groupby("Gloss A Path"):
        # Exclude trivial cases
        filtered_group = group[group["Gloss B Path"] != gloss_a_path]

        # Sort by score (ascending)
        sorted_group = filtered_group.sort_values(by="score", ascending=True)

        # Identify relevant positions
        relevant_mask = sorted_group["Gloss A"].values == sorted_group["Gloss B"].values
        relevant_indices = np.where(relevant_mask)[0]  # Convert to NumPy array of indices

        if relevant_indices.size == 0:
            continue  # Skip if there are no correct matches

        # Compute precision at each relevant index
        ranks = np.arange(1, relevant_indices.size + 1)
        precision_at_ranks = (relevant_indices + 1) / ranks  # Using NumPy broadcasting

        # Average Precision for this query
        ap = np.mean(precision_at_ranks)
        average_precisions.append(ap)

    # Mean Average Precision
    return float(np.mean(average_precisions)) if average_precisions else 0.0


def mean_reciprocal_rank(df: pd.DataFrame) -> float:
    reciprocal_ranks = []

    for gloss_a_path, group in df.groupby("Gloss A Path"):
        # Exclude rows where 'Gloss B Path' matches 'Gloss A Path'
        filtered_group = group[group["Gloss B Path"] != gloss_a_path]

        # Sort by score in ascending order (lower score is better)
        sorted_group = filtered_group.sort_values(by="score", ascending=True)

        # Find the first correct match
        relevant_mask = sorted_group["Gloss A"].values == sorted_group["Gloss B"].values
        relevant_indices = np.where(relevant_mask)[0]  # NumPy array of relevant indices

        if relevant_indices.size > 0:
            rank = relevant_indices[0] + 1  # Convert zero-based index to one-based rank
            reciprocal_ranks.append(1 / rank)

    # Compute Mean Reciprocal Rank
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def mean_match_count_at_k(df: pd.DataFrame, k: int) -> float:
    match_counts = []

    for gloss_a_path, group in df.groupby("Gloss A Path"):
        # Exclude trivial cases
        filtered_group = group[group["Gloss B Path"] != gloss_a_path]

        # Sort by score (ascending)
        sorted_group = filtered_group.sort_values(by="score", ascending=True)

        # Select top-k results
        top_k = sorted_group.head(k)

        # Count matches where 'Gloss A' == 'Gloss B'
        match_count = (top_k["Gloss A"] == top_k["Gloss B"]).sum()
        match_counts.append(match_count)

    # Compute mean match count across all groups
    return sum(match_counts) / len(match_counts) if match_counts else 0.0


# def standardize_path_order(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure `Gloss A Path` always contains the original query path if it exists in the dataset."""
#     df = df.copy()

#     unique_queries = df["Gloss A Path"].unique()  # Only consider actual queries

#     for query_path in tqdm(unique_queries, desc="Fixing Queries after deduplication"):
#         # Find rows where this query appears in Gloss B Path
#         mask = df["Gloss B Path"] == query_path

#         # Swap so that `query_path` is always in `Gloss A Path`
#         df.loc[mask, ["Gloss A Path", "Gloss B Path"]] = df.loc[mask, ["Gloss B Path", "Gloss A Path"]].values
#         df.loc[mask, ["Gloss A", "Gloss B"]] = df.loc[mask, ["Gloss B", "Gloss A"]].values

#     return df


def standardize_path_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `Gloss A Path` always contains the original query path if it exists in the dataset."""
    df = df.copy()

    # Create a mask for rows where `Gloss B Path` appears in `Gloss A Path`
    mask = df["Gloss B Path"].isin(df["Gloss A Path"])

    # Swap columns where the mask is True
    df.loc[mask, ["Gloss A Path", "Gloss B Path"]] = df.loc[mask, ["Gloss B Path", "Gloss A Path"]].values
    df.loc[mask, ["Gloss A", "Gloss B"]] = df.loc[mask, ["Gloss B", "Gloss A"]].values

    return df


if __name__ == "__main__":

    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_analysis_with_times\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\embedding_analysis\scores")
    stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\combined_embedding_and_pose_stats\scores")
    # stats_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same\what_the_heck_why_pop\scores")

    # TODO: check if the number of CSVs has changed. If not, load deduplicated.

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
    from tqdm import tqdm

    # Enable tqdm for pandas

    print("Creating path and gloss tuple pairs")
    # stats_df["path_tuple"] = stats_df.progress_apply(
    #     lambda row: tuple(sorted([row["Gloss A Path"], row["Gloss B Path"]])), axis=1
    # )
    # stats_df["gloss_tuple"] = stats_df.progress_apply(
    #     lambda row: tuple(sorted([row["Gloss A"], row["Gloss B"]])), axis=1
    # )

    # faster version
    stats_df["path_tuple"] = [tuple(x) for x in np.sort(stats_df[["Gloss A Path", "Gloss B Path"]].values, axis=1)]
    stats_df["gloss_tuple"] = [tuple(x) for x in np.sort(stats_df[["Gloss A", "Gloss B"]].values, axis=1)]

    # # Find duplicate rows based on "metric" and sorted paths
    # duplicates = stats_df[stats_df.duplicated(subset=["metric", "path_tuple"], keep=False)]
    # # Drop helper column
    # duplicates = duplicates.drop(columns=["path_tuple"])
    # print("Duplicate rows:")
    # print(duplicates)

    # keep only the deduplicated
    # stats_df = stats_df.drop_duplicates(subset=["metric", "path_tuple"], keep="first").drop(columns=["path_tuple"])
    print(f"We have {len(stats_df)} scores: Deduplicating")
    stats_df = stats_df.drop_duplicates(subset=["metric", "path_tuple"], keep="first")  # actually, path_tuple is useful

    # OK now we lost some cases from Path A, the Queries.
    # but no worries, we still have the duplicates where gloss A was the target instead.
    stats_df = standardize_path_order(stats_df)
    print(f"We now have {len(stats_df)} scores")

    stats_df.drop(columns=["gloss_tuple", "path_tuple", "Unnamed: 0"]).to_csv(
        analysis_folder / "deduplicated_scores.csv", index=False
    )

    # print(f"Deduplicated Rows:")
    # print(stats_df)
    # print(stats_df.info())

    metric_stats = defaultdict(list)
    metric_stats_at_k = defaultdict(list)

    metric_stats_at_k_excluding_known_similar = defaultdict(list)

    metrics_to_analyze = stats_df["metric"].unique()
    print(f"We have results for {len(metrics_to_analyze)}")
    # metrics_to_analyze = ["n-dtai-DTW-MJE (fast)", "MJE", "nMJE"]
    for metric in metrics_to_analyze:
        print(f"*" * 50)
        print(f"METRIC: {metric}")

        metric_df = stats_df[stats_df["metric"] == metric]
        # print(metric_df.head())
        # print(metric_df.columns)
        metric_df["time"].head()
        # print(metric_df["time"].mean())
        # print(len(metric_df[metric_df["time"].isna()]))

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
        metric_df["known_similar"] = metric_df["gloss_tuple"].isin(known_similar_gloss_tuples)
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

        metric_stats["mrr_of_same_gloss_from_full_population_including_known_similar"].append(
            mean_reciprocal_rank(metric_df)
        )
        metric_stats["map_of_same_gloss_from_full_population_including_known_similar"].append(
            mean_average_precision(metric_df)
        )

        ########################
        # example_path = r"C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\19151595519059494-ALASKA.pose.zst"
        # example_path = r"C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\13390250766975487-ALASKA.pose.zst"
        # example_path = r"C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\pose\4026978969748962-ALASKA.pose.zst"
        # example_group = metric_df[metric_df["Gloss A Path"] == example_path]
        print("#" * 30)
        # print(f"Rows OUTSIDE THE FUNCTION where {example_path} is in Gloss A Path:", len(example_group))
        # print("Value Counts OUTSIDE", metric_df["Gloss A Path"].value_counts())
        # Check how many comparisons exist after removing self-matches
        # filtered_group = example_group[example_group["Gloss B Path"] != example_path]
        # print(f"Filtered comparisons for {example_path} (excluding self):", len(filtered_group))

        # Check how many correct matches remain
        # correct_count = (filtered_group["Gloss A"] == filtered_group["Gloss B"]).sum()
        # print(f"Correct matches for {example_path}:", correct_count)
        ########################

        mean_correct_pop_full, mean_total_pop_full = mean_correct_and_total_population_sizes_per_gloss(metric_df)
        metric_stats["mean correct items in query population including known similar"].append(mean_correct_pop_full)
        metric_stats["mean total items in query population including known similar"].append(mean_total_pop_full)
        print(
            f"Average {metric} query population including known-similar signs: {mean_total_pop_full}, of which {mean_correct_pop_full} are correct"
        )

        metric_df_excluding_known_similar = metric_df[metric_df["known_similar"] == False]
        mean_correct_pop_excluding_known_similar, mean_total_pop_excluding_known_similar = (
            mean_correct_and_total_population_sizes_per_gloss(metric_df_excluding_known_similar)
        )
        metric_stats["mean correct items in query population excluding known similar"].append(
            mean_correct_pop_excluding_known_similar
        )
        metric_stats["mean total items in query population excluding known similar"].append(
            mean_total_pop_excluding_known_similar
        )
        print(
            f"Average {metric} query population excluding known-similar signs: {mean_total_pop_excluding_known_similar}, of which {mean_correct_pop_excluding_known_similar} are correct"
        )

        for k in tqdm(range(1, 11), desc="Calculating recall metrics at k"):

            # including known_similar
            metric_stats_at_k["metric"].append(metric)
            metric_stats_at_k["metric_signature"].append(signatures[0])
            metric_stats_at_k["k"].append(k)
            metric_stats_at_k["Mean Total Pop Size"].append(mean_total_pop_full)
            metric_stats_at_k["Mean Correct Pop Size"].append(mean_correct_pop_full)
            metric_stats_at_k["recall@k"].append(recall_at_k(metric_df, k))
            metric_stats_at_k["precision@k"].append(precision_at_k(metric_df, k))
            metric_stats_at_k["mean_match_count@k"].append(mean_match_count_at_k(metric_df, k))

            # excluding known_similar

            metric_stats_at_k_excluding_known_similar["metric"].append(metric)
            metric_stats_at_k_excluding_known_similar["metric_signature"].append(signatures[0])
            metric_stats_at_k_excluding_known_similar["k"].append(k)
            metric_stats_at_k_excluding_known_similar["Mean Total Pop Size"].append(
                mean_total_pop_excluding_known_similar
            )
            metric_stats_at_k_excluding_known_similar["Mean Correct Pop Size"].append(
                mean_correct_pop_excluding_known_similar
            )
            metric_stats_at_k_excluding_known_similar["recall@k"].append(
                recall_at_k(metric_df_excluding_known_similar, k)
            )
            metric_stats_at_k_excluding_known_similar["precision@k"].append(
                precision_at_k(metric_df_excluding_known_similar, k)
            )
            metric_stats_at_k_excluding_known_similar["mean_match_count@k"].append(
                mean_match_count_at_k(metric_df_excluding_known_similar, k)
            )

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

    metric_stats_at_k = pd.DataFrame(metric_stats_at_k)
    metric_stats_at_k_out = analysis_folder / "stats_by_metric_at_k_full_population_including_known_similar.csv"
    metric_stats_at_k.to_csv(metric_stats_at_k_out, index=False)

    metric_stats_at_k_excluding_known_similar_df = pd.DataFrame(metric_stats_at_k_excluding_known_similar)
    metric_stats_at_k_excluding_known_similar_out = analysis_folder / "stats_by_metric_at_k_excluding_known_similar.csv"
    metric_stats_at_k_excluding_known_similar_df.to_csv(metric_stats_at_k_excluding_known_similar_out, index=False)

    fig = px.box(
        stats_df,
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
    fig.write_html(analysis_folder / "metric_pairwise_scoring_time_distributions.html")
