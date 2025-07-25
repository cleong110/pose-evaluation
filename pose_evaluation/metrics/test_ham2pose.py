from itertools import product
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic
from scipy.spatial.distance import euclidean

from pose_evaluation.metrics.ham2pose import get_standard_ham2pose_preprocessors


# This is the code from Ham2Pose normalize_pose
# https://github.com/J22Melody/iict-eval-private/blob/text2pose/metrics/metrics.py#L15
def legacy_normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(
        pose.header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
        )
    )


# This is the code for Ham2Pose get_pose. The result is 178 key points, normalized
def legacy_get_pose(pose_path: str):
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    if "WORLD_LANDMARKS" in [c.name for c in pose.header.components]:
        pose = pose.get_components(
            [
                "POSE_LANDMARKS",
                "FACE_LANDMARKS",
                "LEFT_HAND_LANDMARKS",
                "RIGHT_HAND_LANDMARKS",
            ]
        )
    if "FACE_LANDMARKS" in [c.name for c in pose.header.components]:
        pose = reduce_holistic(pose)

    pose = legacy_normalize_pose(pose)

    return pose


def legacy_pose_hide_low_conf(pose):
    mask = pose.body.confidence <= 0.2
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask, mask], axis=3)
    masked_data = np.ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data


def legacy_get_idx2weight_mediapipe(pose):
    # TODO: weights
    idx2weight = dict.fromkeys(range(pose.body.data.shape[2]), 1)
    return idx2weight


def legacy_mse(trajectory1, trajectory2):
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    pose1_mask = np.ma.getmask(trajectory1)
    pose2_mask = np.ma.getmask(trajectory2)
    trajectory1[pose1_mask] = 0
    trajectory1[pose2_mask] = 0
    trajectory2[pose1_mask] = 0
    trajectory2[pose2_mask] = 0
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return sq_error.mean()


def legacy_APE(trajectory1, trajectory2):  # noqa: N802
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    pose1_mask = np.ma.getmask(trajectory1)
    pose2_mask = np.ma.getmask(trajectory2)
    trajectory1[pose1_mask] = 0
    trajectory1[pose2_mask] = 0
    trajectory2[pose1_mask] = 0
    trajectory2[pose2_mask] = 0
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return np.sqrt(sq_error).mean()


def legacy_masked_mse(trajectory1, trajectory2, confidence):
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
        confidence = np.concatenate((confidence, np.zeros(diff)))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return (sq_error * confidence).mean()


def legacy_masked_APE(trajectory1, trajectory2, confidence):  # noqa: N802
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
        confidence = np.concatenate((confidence, np.zeros(diff)))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return np.sqrt(sq_error * confidence).mean()


def legacy_masked_euclidean(point1, point2):
    if np.ma.is_masked(point2):  # reference label keypoint is missing
        return 0
    elif np.ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
        return euclidean((0, 0, 0), point2) / 2
    d = euclidean(point1, point2)
    return d


def legacy_unmasked_euclidean(point1, point2):
    if np.ma.is_masked(point2):  # reference label keypoint is missing
        return euclidean((0, 0, 0), point1)
    elif np.ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
        return euclidean((0, 0, 0), point2)
    d = euclidean(point1, point2)
    return d


def legacy_compare_poses_preprocessing(pose1, pose2):
    # reduce pose2 the set of keypoints of pose1 (hypothesis)
    pose_components = [c.name for c in pose1.header.components]
    pose_points = {c.name: c.points for c in pose1.header.components}
    pose2 = pose2.get_components(pose_components, pose_points)

    legacy_pose_hide_low_conf(pose1)
    legacy_pose_hide_low_conf(pose2)

    # poses_data = get_pose_data([pose1, pose2])
    poses_data = [pose1.body.data, pose2.body.data]
    return poses_data


def legacy_compare_poses(pose1, pose2, distance_function="nfastdtw"):
    """Copied code from compare_poses"""
    poses_data = legacy_compare_poses_preprocessing(pose1, pose2)

    total_distance = 0
    idx2weight = legacy_get_idx2weight_mediapipe(pose1)

    for keypoint_idx, weight in idx2weight.items():
        pose1_keypoint_trajectory = poses_data[0][:, :, keypoint_idx, :].squeeze(1)
        pose2_keypoint_trajectory = poses_data[1][:, :, keypoint_idx, :].squeeze(1)

        if distance_function in [legacy_mse, legacy_APE]:
            dist = distance_function(pose1_keypoint_trajectory, pose2_keypoint_trajectory)
        elif distance_function in [legacy_masked_mse, legacy_masked_APE]:
            dist = distance_function(
                pose1_keypoint_trajectory,
                pose2_keypoint_trajectory,
                pose1.body.confidence[:, :, keypoint_idx].squeeze(1),
            )
        elif distance_function == fastdtw:
            dist = distance_function(
                pose1_keypoint_trajectory, pose2_keypoint_trajectory, dist=legacy_unmasked_euclidean
            )[0]
        elif distance_function == "nfastdtw":
            dist = fastdtw(pose1_keypoint_trajectory, pose2_keypoint_trajectory, dist=legacy_masked_euclidean)[0]
        total_distance += dist * weight
    return total_distance / len(idx2weight)


def test_ham2pose_get_pose_matches_legacy(real_pose_file_paths: list[Path]):
    processors = get_standard_ham2pose_preprocessors(include_compare_pose_processors=False)

    for pose_path in real_pose_file_paths:
        raw_pose = Pose.read(pose_path.read_bytes())

        # Legacy pipeline
        expected_pose = legacy_get_pose(str(pose_path))
        # should give us 178 keypoints
        assert expected_pose.body.data.shape[2] == 178

        # New pipeline
        new_pose = raw_pose
        for processor in processors:
            new_pose = processor.process_pose(new_pose)

        # Shape check
        assert expected_pose.body.data.shape == new_pose.body.data.shape, (
            f"Shape mismatch for {pose_path.name}: {expected_pose.body.data.shape} vs {new_pose.body.data.shape}"
        )

        # Data check
        np.testing.assert_allclose(
            new_pose.body.data,
            expected_pose.body.data,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Data mismatch for {pose_path.name}",
        )


def test_ham2pose_compare_pose_processing_matches_legacy(real_pose_file_paths: list[Path]):
    processors = get_standard_ham2pose_preprocessors(include_compare_pose_processors=True)

    for hyp_path, ref_path in product(real_pose_file_paths, repeat=2):
        # Legacy pipeline
        expected_hyp = legacy_get_pose(str(hyp_path))
        expected_ref = legacy_get_pose(str(ref_path))
        expected_hyp_data, expected_ref_data = legacy_compare_poses_preprocessing(expected_hyp, expected_ref)

        # New pipeline
        new_hyp = Pose.read(hyp_path.read_bytes())
        new_ref = Pose.read(ref_path.read_bytes())
        for processor in processors:
            new_hyp, new_ref = processor.process_poses([new_hyp, new_ref])

        # Shape checks
        assert expected_hyp_data.shape == new_hyp.body.data.shape, (
            f"Shape mismatch (HYP) for {hyp_path.name}, {ref_path.name}: "
            f"{expected_hyp_data.shape} vs {new_hyp.body.data.shape}"
        )
        assert expected_ref_data.shape == new_ref.body.data.shape, (
            f"Shape mismatch (REF) for {hyp_path.name}, {ref_path.name}: "
            f"{expected_ref_data.shape} vs {new_ref.body.data.shape}"
        )

        # Content checks
        np.testing.assert_allclose(
            new_hyp.body.data,
            expected_hyp_data,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Data mismatch (HYP) for {hyp_path.name}, {ref_path.name}",
        )
        np.testing.assert_allclose(
            new_ref.body.data,
            expected_ref_data,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Data mismatch (REF) for {hyp_path.name}, {ref_path.name}",
        )
