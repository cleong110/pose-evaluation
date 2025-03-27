from collections import defaultdict
from itertools import combinations
import json
import time
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from pose_format import Pose
from pyzstd import decompress
from tqdm import tqdm
import plotly.express as px


from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import (
    DTWAggregatedPowerDistanceMeasure,
    DTWAggregatedScipyDistanceMeasure,
    DTWDTAIImplementationDistanceMeasure,
    DTWOptimizedDistanceMeasure,
)
from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric
from pose_evaluation.metrics.pose_processors import (
    NormalizePosesProcessor,
    GetHandsOnlyHolisticPoseProcessor,
    InterpolateAllToSetFPSPoseProcessor,
    ReduceHolisticPoseProcessor,
    ZeroPadShorterPosesProcessor,
    AddTOffsetsToZPoseProcessor,
    get_standard_pose_processors,
)


def get_pose_data(file_path: Union[Path, str]) -> Pose:
    """Loads a .pose or .pose.zst, returns a Pose object"""
    file_path = Path(file_path)
    if file_path.name.endswith(".pose.zst"):
        return Pose.read(decompress(file_path.read_bytes()))
    else:
        return Pose.read(file_path.read_bytes())


def load_embedding(file_path: Path) -> np.ndarray:
    """
    Load a SignCLIP embedding from a .npy file, ensuring it has the correct shape.

    Args:
        file_path (Path): Path to the .npy file.

    Returns:
        np.ndarray: The embedding with shape (768,).
    """
    embedding = np.load(file_path)
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]  # Reduce shape from (1, 768) to (768,)
    return embedding


def plot_metric_scatter(df: pd.DataFrame, metric_x: str, metric_y: str, html_path: Optional[Path] = None):
    # Filter for the two specified metrics
    df_x = df[df["metric"] == metric_x].rename(columns={"mean_score": "score_x"})
    df_y = df[df["metric"] == metric_y].rename(columns={"mean_score": "score_y"})

    # Merge on Gloss A and Gloss B
    merged_df = df_x.merge(df_y, on=["Gloss A", "Gloss B"], suffixes=("", "_y"))

    # Create labels
    merged_df["label"] = merged_df["Gloss A"] + " / " + merged_df["Gloss B"]

    # Plot
    fig = px.scatter(
        merged_df,
        x="score_x",
        y="score_y",
        text="label",
        title=f"Visual/Semantic Distance: {metric_x} vs {metric_y}",
        labels={"score_x": metric_x, "score_y": metric_y},
    )

    # Improve layout for readability
    fig.update_traces(textposition="top center", marker=dict(size=8, opacity=0.7))
    fig.update_layout(xaxis_title=metric_x, yaxis_title=metric_y, hovermode="closest")

    fig.show()
    fig.write_html(html_path)


def get_embeddings_df_from_folder_of_embeddings(directory: Path, dataset_df: Optional[pd.DataFrame] = None):
    embedding_files = list(directory.rglob("*.npy"))

    # get the video ID (assuming VIDEOID-GLOSS-using-model-MODEL)
    embedding_files_df = pd.DataFrame(
        {
            "Embedding File Path": [str(embedding_file) for embedding_file in embedding_files],
            "Embedding File Name": [str(embedding_file.name) for embedding_file in embedding_files],
        }
    )

    print(embedding_files_df.info())

    embedding_files_df["Embedding File Name"] = embedding_files_df["Embedding File Name"].astype(str)

    # Split by '-' and extract VIDEOID (first part)
    embedding_files_df["Video ID"] = embedding_files_df["Embedding File Name"].str.split("-").str[0]

    # Split by '-using-model-' and extract MODEL (second part)
    embedding_files_df["Embedding Model"] = (
        embedding_files_df["Embedding File Name"]
        .str.split("-using-model-")
        .str[-1]
        .str.replace(".npy", "", regex=False)
    )

    print(embedding_files_df[["Video ID", "Embedding Model"]])
    return embedding_files_df


def get_metrics():
    dtw_mje_dtai_fast = DistanceMetric(
        "n-dtai-DTW-MJE (fast)",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
        pose_preprocessors=get_standard_pose_processors(
            reduce_holistic_to_face_and_upper_body=True, zero_pad_shorter=False
        ),
    )

    dtw_mje_dtai_slow = DistanceMetric(
        "n-dtai-DTW-MJE (slow)",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=False),
        pose_preprocessors=get_standard_pose_processors(
            reduce_holistic_to_face_and_upper_body=True, zero_pad_shorter=False
        ),
    )

    dtw_mje_optimized = DistanceMetric(
        "nDTW-MJE (Optimized)",
        distance_measure=DTWOptimizedDistanceMeasure(),
        pose_preprocessors=get_standard_pose_processors(
            reduce_holistic_to_face_and_upper_body=True, zero_pad_shorter=False
        ),
    )

    dtw_mje = DistanceMetric(
        "nDTW-MJE",
        distance_measure=DTWAggregatedPowerDistanceMeasure(),
        pose_preprocessors=get_standard_pose_processors(
            reduce_holistic_to_face_and_upper_body=True, zero_pad_shorter=False
        ),
    )
    dtw_mje_scipy = DistanceMetric(
        "nDTW-MJE (Scipy)",
        distance_measure=DTWAggregatedScipyDistanceMeasure(),
        pose_preprocessors=get_standard_pose_processors(
            reduce_holistic_to_face_and_upper_body=True, zero_pad_shorter=False
        ),
    )
    nmje = DistanceMetric(
        "nMJE",
        distance_measure=AggregatedPowerDistance(),
        pose_preprocessors=get_standard_pose_processors(reduce_holistic_to_face_and_upper_body=True),
    )

    nmje_120fps = DistanceMetric(
        "nMJE_120fps",
        distance_measure=AggregatedPowerDistance(),
        pose_preprocessors=[
            NormalizePosesProcessor(),
            ReduceHolisticPoseProcessor(),
            InterpolateAllToSetFPSPoseProcessor(fps=120),  # has to be done before zero-padding
            ZeroPadShorterPosesProcessor(),
        ],
    )

    mje = DistanceMetric(
        "MJE",
        distance_measure=AggregatedPowerDistance(),
        pose_preprocessors=get_standard_pose_processors(
            normalize_poses=False, reduce_holistic_to_face_and_upper_body=True
        ),
    )

    dtw_mje_dtai_fast_hands = DistanceMetric(
        "n-dtai-DTW-MJE_fast_hands_only",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
        pose_preprocessors=get_standard_pose_processors(
            remove_world_landmarks=False,
            reduce_poses_to_common_components=False,
            remove_legs=False,
            reduce_holistic_to_face_and_upper_body=False,
            zero_pad_shorter=False,
        ),
    )
    dtw_mje_dtai_fast_hands.add_preprocessor(GetHandsOnlyHolisticPoseProcessor())

    dtw_mje_dtai_fast_fps15 = DistanceMetric(
        "n-dtai-DTW-MJE_fast_fps15",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
        pose_preprocessors=get_standard_pose_processors(
            remove_world_landmarks=False,
            reduce_poses_to_common_components=False,
            remove_legs=False,
            reduce_holistic_to_face_and_upper_body=False,
            zero_pad_shorter=False,
        ),
    )
    dtw_mje_dtai_fast_fps15.add_preprocessor(InterpolateAllToSetFPSPoseProcessor())

    dtw_mje_dtai_fast_fps120 = DistanceMetric(
        "n-dtai-DTW-MJE_fast_fps120",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
        pose_preprocessors=get_standard_pose_processors(
            remove_world_landmarks=False,
            reduce_poses_to_common_components=False,
            remove_legs=False,
            reduce_holistic_to_face_and_upper_body=False,
            zero_pad_shorter=False,
        ),
    )
    dtw_mje_dtai_fast_fps120.add_preprocessor(InterpolateAllToSetFPSPoseProcessor(fps=120))

    dtw_mje_dtai_fast_hands_fps15 = DistanceMetric(
        "n-dtai-DTW-dtw_mje_dtai_fast_hands_fps15",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
        pose_preprocessors=get_standard_pose_processors(
            remove_world_landmarks=False,
            reduce_poses_to_common_components=False,
            remove_legs=False,
            reduce_holistic_to_face_and_upper_body=False,
            zero_pad_shorter=False,
        ),
    )
    dtw_mje_dtai_fast_hands_fps15.add_preprocessor(GetHandsOnlyHolisticPoseProcessor())
    dtw_mje_dtai_fast_hands_fps15.add_preprocessor(InterpolateAllToSetFPSPoseProcessor(fps=15))

    dtw_mje_dtai_fast_hands_fps120 = DistanceMetric(
        "n-dtai-DTW-dtw_mje_dtai_fast_hands_fps120",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
        pose_preprocessors=get_standard_pose_processors(
            remove_world_landmarks=False,
            reduce_poses_to_common_components=False,
            remove_legs=False,
            reduce_holistic_to_face_and_upper_body=False,
            zero_pad_shorter=False,
        ),
    )
    dtw_mje_dtai_fast_hands_fps120.add_preprocessor(GetHandsOnlyHolisticPoseProcessor())
    dtw_mje_dtai_fast_hands_fps120.add_preprocessor(InterpolateAllToSetFPSPoseProcessor(fps=120))

    dtw_mje_dtai_fast_hands_z_offsets = DistanceMetric(
        "n-dtai-DTW-dtw_mje_dtai_fast_hands_z_offsets",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
        pose_preprocessors=get_standard_pose_processors(
            normalize_poses=False,
            remove_world_landmarks=False,
            reduce_poses_to_common_components=False,
            remove_legs=False,
            reduce_holistic_to_face_and_upper_body=False,
            zero_pad_shorter=False,
        ),
    )
    dtw_mje_dtai_fast_hands_z_offsets.add_preprocessor(AddTOffsetsToZPoseProcessor())
    dtw_mje_dtai_fast_hands_z_offsets.add_preprocessor(NormalizePosesProcessor())
    dtw_mje_dtai_fast_hands_z_offsets.add_preprocessor(GetHandsOnlyHolisticPoseProcessor())

    dtw_mje_dtai_fast_hands_z_offsets_fps_15 = DistanceMetric(
        "dtw_mje_dtai_fast_hands_z_offsets_fps_15",
        distance_measure=DTWDTAIImplementationDistanceMeasure(use_fast=True),
        pose_preprocessors=get_standard_pose_processors(
            normalize_poses=False,
            remove_world_landmarks=False,
            reduce_poses_to_common_components=False,
            remove_legs=False,
            reduce_holistic_to_face_and_upper_body=False,
            zero_pad_shorter=False,
        ),
    )
    dtw_mje_dtai_fast_hands_z_offsets_fps_15.add_preprocessor(AddTOffsetsToZPoseProcessor())
    dtw_mje_dtai_fast_hands_z_offsets_fps_15.add_preprocessor(NormalizePosesProcessor())
    dtw_mje_dtai_fast_hands_z_offsets_fps_15.add_preprocessor(GetHandsOnlyHolisticPoseProcessor())
    dtw_mje_dtai_fast_hands_z_offsets_fps_15.add_preprocessor(InterpolateAllToSetFPSPoseProcessor(fps=15))

    cosine_distance_metric = EmbeddingDistanceMetric(model="signclip_semlex", kind="cosine")

    metrics = [
        # cosine_distance_metric,
        dtw_mje_dtai_fast_hands_z_offsets,
        # dtw_mje_dtai_fast_hands_z_offsets_fps_15,
        # dtw_mje_dtai_fast,
        # dtw_mje_dtai_slow,
        # nmje,
        # nmje_120fps,
        # mje,
        # dtw_mje_dtai_fast_hands,
        # dtw_mje_dtai_fast_fps15,
        # dtw_mje_dtai_fast_fps120,
        # dtw_mje_dtai_fast_hands_fps15,
        # dtw_mje_dtai_fast_hands_fps120,
        # dtw_mje_optimized,
        # dtw_mje,
        # dtw_mje_scipy,
    ]

    return metrics


def load_dataset_stats(
    dataset_folder: Path,
    json_name: str,
):
    """Should give a dataframe with 'File Path' being paths to pose files, 'Gloss' being the gloss"""
    data_json_path = Path(dataset_folder) / json_name

    with data_json_path.open() as dj_file:
        data = json.load(dj_file)
    loaded_df = pd.DataFrame.from_dict(data)
    return loaded_df


if __name__ == "__main__":

    DISABLE_PROGRESS_BAR_FOR_REFERENCES = True
    data_folder = Path(r"C:\Users\Colin\data\similar_but_not_the_same")
    # results_folder_base = data_folder / "similar_sign_analysis_with_times"
    results_folder_base = data_folder / "embedding_analysis"
    results_folder_base.mkdir(exist_ok=True)
    json_name = "similar_signs_metadata.json"
    similar_sign_pairs_df = pd.read_csv(Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_pairs.csv"))

    manually_chosen_gloss_pairs = [
        ("RUSSIA", "BRAG"),
        ("SUMMER", "BRAG"),
        # ("HOUSE", "HOME"),
        ("SUMMER", "BLACK"),
        ("SUMMER", "FRIENDLY"),
        ("SAD", "FRIENDLY"),
        ("SAD", "BRAG"),
        # ("FENCE", "HANUKKAH"),  # had to manually copy these, but they're not in the precomputed stats
        # ("THIEF", "STEAL"),
        # ("HEALTH", "BRAVE"),
        # ("LAUGH", "CRY"),
        # ("KING", "QUEEN"),
    ]
    manually_chosen_gloss_pairs_df = pd.DataFrame(
        {
            "signA": [pair[0] for pair in manually_chosen_gloss_pairs],
            "signB": [pair[1] for pair in manually_chosen_gloss_pairs],
        }
    )

    df = load_dataset_stats(data_folder, json_name)

    print(df.head())
    print(df.info())

    similar_sign_pairs_df = similar_sign_pairs_df.sample(frac=1)  # shuffle the dataframe
    # similar_sign_pairs_df = pd.concat([manually_chosen_gloss_pairs_df, similar_sign_pairs_df])
    unique_glosses_from_similar_signs = pd.unique(
        pd.concat([similar_sign_pairs_df["signA"], similar_sign_pairs_df["signB"]])
    )
    print(similar_sign_pairs_df.head())
    print(f"SIMILAR SIGN PAIRS HAS {len(unique_glosses_from_similar_signs)} unique")
    # print())

    metrics = get_metrics()

    # dataset_df["Video ID Numeric"] = pd.to_numeric(dataset_df["Video ID"], errors="coerce")

    embeddings_df = get_embeddings_df_from_folder_of_embeddings(data_folder / "embeddings")
    models = embeddings_df["Embedding Model"].unique().tolist()
    for model in models:
        metrics.append(EmbeddingDistanceMetric(model=model))

    print(embeddings_df.info())

    print("MERGING EMBEDDINGS")

    df = df.merge(embeddings_df, on="Video ID", how="left")

    print(df.head())
    print(df.info())

    print("DROPPING ENTRIES WITH NO EMBEDDINGS")
    df = df.dropna(subset=["Embedding File Path"]).reset_index(drop=True)
    print(df.info())

    semantic_and_visual_diffs = defaultdict(list)
    # for model in models:

    for index, row in tqdm(similar_sign_pairs_df.head(50).iterrows(), desc="Evaluating metrics on similar-sign pairs"):
        print()
        print("SIMILAR PAIR " * 10)

        gloss_a = row["signA"]
        gloss_b = row["signB"]
        print(f"{gloss_a}/{gloss_b}")

        sign_a_df = df[df["Gloss"] == gloss_a]
        sign_b_df = df[df["Gloss"] == gloss_b]
        not_a_or_b_df = df[~df["Gloss"].isin([gloss_a, gloss_b])]

        # print(f"Found {len(sign_a_df)} with gloss {gloss_a}")
        # print(f"Found {len(sign_b_df)} with gloss {gloss_b}")

        for metric in metrics:
            print("*" * 50)
            print(f"METRIC: {metric}")
            # print(f"Shape of first hyp: {hyp_poses[0].body.data.shape}")

            # LOAD ONLY THE EMBEDDINGS OR THIS METRIC
            if isinstance(metric, EmbeddingDistanceMetric):
                this_model = metric.model
            else:
                this_model = "sem-lex"

            sign_a_this_model_df = sign_a_df[sign_a_df["Embedding Model"] == this_model]
            sign_b_this_model_df = sign_b_df[sign_b_df["Embedding Model"] == this_model]
            not_a_or_b_this_model_df = not_a_or_b_df[not_a_or_b_df["Embedding Model"] == this_model]

            hyp_embedding_paths = sign_a_this_model_df["Embedding File Path"].tolist()
            ref_embedding_paths = sign_b_this_model_df["Embedding File Path"].tolist()

            hyp_embeds = [load_embedding(embed_path) for embed_path in hyp_embedding_paths]
            ref_embeds = [load_embedding(embed_path) for embed_path in ref_embedding_paths]

            hyp_pose_paths = sign_a_this_model_df["File Path"].tolist()
            hyp_poses = [
                get_pose_data(pose_path)
                for pose_path in tqdm(hyp_pose_paths, desc=f"loading files for {gloss_a}", disable=True)
            ]

            ref_pose_paths = sign_b_this_model_df["File Path"].tolist()
            ref_poses = [
                get_pose_data(pose_path)
                for pose_path in tqdm(ref_pose_paths, desc=f"loading files for {gloss_b}", disable=True)
            ]

            other_class_count = len(sign_a_this_model_df) * 4  # as in
            random_sample_df_this_model_df = not_a_or_b_this_model_df.sample(n=other_class_count, random_state=42)
            # random_sample_df_this_model_df = random_sample_df[random_sample_df["Embedding Model"] == this_model]
            not_a_or_b_embed_paths = random_sample_df_this_model_df["Embedding File Path"].tolist()
            not_a_or_b_embeds = [load_embedding(embed_path) for embed_path in not_a_or_b_embed_paths]
            not_a_or_b_pose_paths = random_sample_df_this_model_df["File Path"].tolist()
            not_a_or_b_pose_glosses = random_sample_df_this_model_df["Gloss"].tolist()
            not_a_or_b_poses = [
                get_pose_data(pose_path)
                for pose_path in tqdm(
                    not_a_or_b_pose_paths,
                    desc="loading files for 'not either'",
                )
            ]

            results = defaultdict(list)
            results_path = results_folder_base / "scores" / f"{gloss_a}_{gloss_b}_{metric.name}_score_results.csv"
            if results_path.exists():
                print(f"Results for {results_path} already exist. Skipping!")
                continue

            # scores = metric.score_all_with_signature(hyp_poses, ref_poses, progress_bar=True)
            # print(scores)

            score_values = []
            for hyp_pose, hyp_pose_path, hyp_embed, hyp_embed_path in tqdm(
                zip(hyp_poses, hyp_pose_paths, hyp_embeds, hyp_embedding_paths),
                desc=f"{metric.name} scoring {gloss_a} vs {gloss_b}",
            ):
                for ref_pose, ref_pose_path, ref_embed, ref_embed_path in tqdm(
                    zip(ref_poses, ref_pose_paths, ref_embeds, ref_embedding_paths),
                    disable=DISABLE_PROGRESS_BAR_FOR_REFERENCES,
                    total=len(hyp_poses),
                ):

                    if isinstance(metric, EmbeddingDistanceMetric):
                        # print(f"It's an embedding metric!!!!!!!!!!!!!!!!!")
                        hyp_path = hyp_embed_path
                        ref_path = ref_embed_path
                        hyp = hyp_embed
                        ref = ref_embed

                    else:
                        hyp_path = hyp_pose_path
                        ref_path = ref_pose_path
                        hyp = hyp_pose
                        ref = ref_pose
                    # print(hyp_path)
                    start_time = time.perf_counter()

                    score = metric.score_with_signature(hyp, ref)

                    end_time = time.perf_counter()
                    # print("*", score)
                    score_values.append(score.score)
                    results["metric"].append(metric.name)
                    results["signature"].append(score.format())
                    results["score"].append(score.score)
                    results["Gloss A"].append(gloss_a)
                    results["Gloss B"].append(gloss_b)
                    results["Gloss A Path"].append(hyp_path)
                    results["Gloss B Path"].append(ref_path)
                    results["time"].append(end_time - start_time)

            mean_score = np.mean(score_values)
            semantic_and_visual_diffs["Gloss A"].append(gloss_a)
            semantic_and_visual_diffs["Gloss B"].append(gloss_b)
            semantic_and_visual_diffs["metric"].append(metric.name)
            semantic_and_visual_diffs["mean_score"].append(mean_score)

            self_score_values = []
            for hyp_pose, hyp_pose_path, hyp_embed, hyp_embed_path in tqdm(
                zip(hyp_poses, hyp_pose_paths, hyp_embeds, hyp_embedding_paths),
                desc=f"{metric.name} scoring {gloss_a} vs {gloss_a}",
            ):
                for ref_pose, ref_pose_path, ref_embed, ref_embed_path in tqdm(
                    zip(hyp_poses, hyp_pose_paths, hyp_embeds, hyp_embedding_paths),
                    disable=DISABLE_PROGRESS_BAR_FOR_REFERENCES,
                    total=len(hyp_poses),
                ):
                    if isinstance(metric, EmbeddingDistanceMetric):
                        # print(f"It's an embedding metric!!!!!!!!!!!!!!!!!")
                        hyp_path = hyp_embed_path
                        ref_path = ref_embed_path
                        hyp = hyp_embed
                        ref = ref_embed
                    else:
                        hyp_path = hyp_pose_path
                        ref_path = ref_pose_path
                        hyp = hyp_pose
                        ref = ref_pose
                    start_time = time.perf_counter()
                    score = metric.score_with_signature(hyp, ref)
                    end_time = time.perf_counter()
                    self_score_values.append(score.score)
                    results["metric"].append(metric.name)
                    results["signature"].append(score.format())
                    results["score"].append(score.score)
                    results["Gloss A"].append(gloss_a)
                    results["Gloss B"].append(gloss_a)
                    results["Gloss A Path"].append(hyp_path)
                    results["Gloss B Path"].append(ref_path)
                    results["time"].append(end_time - start_time)
            mean_self_score = np.mean(self_score_values)

            self_time = time.perf_counter()

            not_a_or_b_score_values = []
            for hyp_pose, hyp_pose_path, hyp_embed, hyp_embed_path in tqdm(
                zip(hyp_poses, hyp_pose_paths, hyp_embeds, hyp_embedding_paths),
                desc=f"{metric.name} scoring {gloss_a} vs {len(not_a_or_b_poses)} random glosses",
            ):
                for ref_pose, ref_pose_path, ref_embed, ref_embed_path, ref_gloss in tqdm(
                    zip(
                        not_a_or_b_poses,
                        not_a_or_b_pose_paths,
                        not_a_or_b_embeds,
                        not_a_or_b_embed_paths,
                        not_a_or_b_pose_glosses,
                    ),
                    disable=DISABLE_PROGRESS_BAR_FOR_REFERENCES,
                    total=len(hyp_poses),
                ):
                    if isinstance(metric, EmbeddingDistanceMetric):
                        # print(f"It's an embedding metric!!!!!!!!!!!!!!!!!")
                        hyp_path = hyp_embed_path
                        ref_path = ref_embed_path
                        hyp = hyp_embed
                        ref = ref_embed
                    else:
                        hyp_path = hyp_pose_path
                        ref_path = ref_pose_path
                        hyp = hyp_pose
                        ref = ref_pose
                    start_time = time.perf_counter()
                    score = metric.score_with_signature(hyp, ref)
                    end_time = time.perf_counter()
                    not_a_or_b_score_values.append(score.score)
                    results["metric"].append(metric.name)
                    results["signature"].append(score.format())
                    results["score"].append(score.score)
                    results["Gloss A"].append(gloss_a)
                    results["Gloss B"].append(ref_gloss)
                    results["Gloss A Path"].append(hyp_path)
                    results["Gloss B Path"].append(ref_path)
                    results["time"].append(end_time - start_time)

            mean_not_a_or_b_score = np.mean(not_a_or_b_score_values)

            other_time = time.perf_counter()

            print(f"Mean {metric.name} ({len(score_values)} scores) for {gloss_a}/{gloss_b}: {mean_score}")

            print(
                f"Mean self-score {metric.name} ({len(self_score_values)} scores) for {gloss_a}/{gloss_a}: {mean_self_score}"
            )

            print(
                f"Mean {metric.name} ({len(not_a_or_b_score_values)} scores) for items that are NEITHER: {mean_not_a_or_b_score}"
            )
            # print(f"time: {other_time-self_time}")
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(results_path)
            print(f"Wrote {len(results_df)} scores to {results_path}")
            print("*" * 50)
    semantic_and_visual_diffs_df = pd.DataFrame(semantic_and_visual_diffs)
    print(semantic_and_visual_diffs_df)

    semantic_and_visual_diffs_df.to_csv(results_folder_base / "mean_scores_by_gloss_and_stat.csv")

    for metric1, metric2 in combinations(metrics, 2):
        plot_metric_scatter(
            semantic_and_visual_diffs_df,
            metric1.name,
            metric2.name,
            results_folder_base / f"{metric1.name}_versus_{metric2.name}.html",
        )
