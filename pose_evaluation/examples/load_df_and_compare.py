from collections import defaultdict
import json
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pose_format import Pose
from pyzstd import decompress
from tqdm import tqdm

from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import (
    DTWAggregatedPowerDistanceMeasure,
    DTWAggregatedScipyDistanceMeasure,
    DTWDTAIImplementationDistanceMeasure,
    DTWOptimizedDistanceMeasure,
)
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

    metrics = [
        dtw_mje_dtai_fast_hands_z_offsets,
        dtw_mje_dtai_fast_hands_z_offsets_fps_15,
        dtw_mje_dtai_fast,
        # dtw_mje_dtai_slow, # literally takes hours
        nmje,
        nmje_120fps,
        mje,
        dtw_mje_dtai_fast_hands,
        dtw_mje_dtai_fast_fps15,
        dtw_mje_dtai_fast_fps120,
        dtw_mje_dtai_fast_hands_fps15,
        dtw_mje_dtai_fast_hands_fps120,
        dtw_mje_optimized,
        dtw_mje,
        dtw_mje_scipy,
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
    json_name = "similar_signs_metadata.json"
    similar_sign_pairs_df = pd.read_csv(Path(r"C:\Users\Colin\data\similar_but_not_the_same\similar_sign_pairs.csv"))

    df = load_dataset_stats(data_folder, json_name)

    print(df.head())
    print(df.info())

    similar_sign_pairs_df = similar_sign_pairs_df.sample(frac=1)  # shuffle the dataframe
    unique_glosses = pd.unique(pd.concat([similar_sign_pairs_df["signA"], similar_sign_pairs_df["signB"]]))
    print(similar_sign_pairs_df.head())
    print(len(unique_glosses))

    metrics = get_metrics()

    for index, row in tqdm(similar_sign_pairs_df.iterrows(), desc="Evaluating metrics on similar-sign pairs"):

        gloss_a = row["signA"]
        gloss_b = row["signB"]
        print(f"{gloss_a}/{gloss_b}")

        sign_a_df = df[df["Gloss"] == gloss_a]
        sign_b_df = df[df["Gloss"] == gloss_b]
        not_a_or_b_df = df[~df["Gloss"].isin([gloss_a, gloss_b])]

        # print(f"Found {len(sign_a_df)} with gloss {gloss_a}")
        # print(f"Found {len(sign_b_df)} with gloss {gloss_b}")
        hyp_pose_paths = sign_a_df["File Path"].tolist()
        hyp_poses = [
            get_pose_data(pose_path)
            for pose_path in tqdm(hyp_pose_paths, desc=f"loading files for {gloss_a}", disable=True)
        ]

        ref_pose_paths = sign_b_df["File Path"].tolist()
        ref_poses = [
            get_pose_data(pose_path)
            for pose_path in tqdm(ref_pose_paths, desc=f"loading files for {gloss_b}", disable=True)
        ]

        other_class_count = len(sign_a_df) * 4  # as in
        random_sample_df = not_a_or_b_df.sample(n=other_class_count, random_state=42)
        not_a_or_b_pose_paths = random_sample_df["File Path"].tolist()
        not_a_or_b_pose_glosses = random_sample_df["Gloss"].tolist()
        not_a_or_b_poses = [
            get_pose_data(pose_path)
            for pose_path in tqdm(
                not_a_or_b_pose_paths,
                desc="loading files for 'not either'",
            )
        ]

        for metric in metrics:
            print("*" * 50)
            print(f"METRIC: {metric}")
            print(f"Shape of first hyp: {hyp_poses[0].body.data.shape}")

            results = defaultdict(list)
            results_path = (
                data_folder
                / "similar_sign_analysis_with_times"
                / "scores"
                / f"{gloss_a}_{gloss_b}_{metric.name}_score_results.csv"
            )
            if results_path.exists():
                print(f"Results for {results_path} already exist. Skipping!")
                continue

            # scores = metric.score_all_with_signature(hyp_poses, ref_poses, progress_bar=True)
            # print(scores)

            score_values = []
            for hyp, hyp_pose_path in tqdm(
                zip(hyp_poses, hyp_pose_paths), desc=f"{metric.name} scoring {gloss_a} vs {gloss_b}"
            ):
                for ref, ref_pose_path in tqdm(
                    zip(ref_poses, ref_pose_paths), disable=DISABLE_PROGRESS_BAR_FOR_REFERENCES, total=len(hyp_poses)
                ):
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
                    results["Gloss A Path"].append(hyp_pose_path)
                    results["Gloss B Path"].append(ref_pose_path)
                    results["time"].append(end_time - start_time)

            mean_score = np.mean(score_values)

            self_score_values = []
            for hyp, hyp_pose_path in tqdm(
                zip(hyp_poses, hyp_pose_paths), desc=f"{metric.name} scoring {gloss_a} vs {gloss_a}"
            ):
                for ref, ref_pose_path in tqdm(
                    zip(hyp_poses, hyp_pose_paths), disable=DISABLE_PROGRESS_BAR_FOR_REFERENCES, total=len(hyp_poses)
                ):
                    start_time = time.perf_counter()
                    score = metric.score_with_signature(hyp, ref)
                    end_time = time.perf_counter()
                    self_score_values.append(score.score)
                    results["metric"].append(metric.name)
                    results["signature"].append(score.format())
                    results["score"].append(score.score)
                    results["Gloss A"].append(gloss_a)
                    results["Gloss B"].append(gloss_a)
                    results["Gloss A Path"].append(hyp_pose_path)
                    results["Gloss B Path"].append(ref_pose_path)
                    results["time"].append(end_time - start_time)
            mean_self_score = np.mean(self_score_values)

            self_time = time.perf_counter()

            # TODO: self-score with interpolation!

            not_a_or_b_score_values = []
            for hyp, hyp_pose_path in tqdm(
                zip(hyp_poses, hyp_pose_paths),
                desc=f"{metric.name} scoring {gloss_a} vs {len(not_a_or_b_poses)} random glosses",
            ):
                for ref, ref_pose_path, ref_gloss in tqdm(
                    zip(not_a_or_b_poses, not_a_or_b_pose_paths, not_a_or_b_pose_glosses),
                    disable=DISABLE_PROGRESS_BAR_FOR_REFERENCES,
                    total=len(hyp_poses),
                ):
                    start_time = time.perf_counter()
                    score = metric.score_with_signature(hyp, ref)
                    end_time = time.perf_counter()
                    not_a_or_b_score_values.append(score.score)
                    results["metric"].append(metric.name)
                    results["signature"].append(score.format())
                    results["score"].append(score.score)
                    results["Gloss A"].append(gloss_a)
                    results["Gloss B"].append(ref_gloss)
                    results["Gloss A Path"].append(hyp_pose_path)
                    results["Gloss B Path"].append(ref_pose_path)
                    results["time"].append(end_time - start_time)

            mean_not_a_or_b_score = np.mean(not_a_or_b_score_values)

            other_time = time.perf_counter()

            print(f"Mean {metric.name} ({len(score_values)} scores) for {gloss_a}/{gloss_b}: {mean_score}")
            # print(f"time: {pair_time-start}")
            print(
                f"Mean self-score {metric.name} ({len(self_score_values)} scores) for {gloss_a}/{gloss_a}: {mean_self_score}"
            )
            # print(f"time: {self_time-pair_time}")
            print(
                f"Mean {metric.name} ({len(not_a_or_b_score_values)} scores) for items that are NEITHER: {mean_not_a_or_b_score}"
            )
            # print(f"time: {other_time-self_time}")
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(results_path)
            print(f"Wrote {len(results_df)} scores to {results_path}")
            print("*" * 50)
