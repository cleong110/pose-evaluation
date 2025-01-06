from pathlib import Path
from typing import List
import numpy as np
from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, pose_hide_legs


def remove_world_landmarks(pose: Pose):
    remove_specified_landmarks(pose, "POSE_WORLD_LANDMARKS")


def remove_specified_landmarks(pose: Pose, landmark_names: List[str]):
    if isinstance(landmark_names, str):
        landmark_names = [landmark_names]
    components_without_specified_names = [
        c.name for c in pose.header.components if c.name not in landmark_names
    ]
    new_pose = pose.get_components(components_without_specified_names)
    pose.body = new_pose.body
    pose.header = new_pose.header


def get_chosen_components_from_pose(
    pose: Pose, chosen_component_names: List[str]
) -> Pose:
    return pose.get_components(chosen_component_names)


def get_face_and_hands_from_pose(pose: Pose) -> Pose:
    # based on MediaPipe Holistic format.
    components_to_keep = [
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
    ]
    return pose.get_components(components_to_keep)


def get_preprocessed_pose(pose_path: Path | str) -> Pose:

    pose_path = Path(pose_path).resolve()

    with pose_path.open("rb") as f:
        pose = Pose.read(f.read())

    # normalize
    # TODO: confirm this is correct. Previously used pose_format.utils.generic.pose_normalization_info(),
    # but pose-format README advises using pose.normalize_distribution()
    pose.normalize(pose_normalization_info(pose))

    # Drop legs
    pose_hide_legs(pose)

    # not used, typically.
    remove_world_landmarks(pose)

    # hide low conf
    pose_hide_low_conf(pose)

    pose.focus()

    # TODO: prune leading/trailing frames without useful data (e.g. no hands, only zeroes, almost no face)
    for frame_index in enumerate(pose.body.data):
        # https://github.com/rotem-shalev/Ham2Pose/blob/main/metrics.py#L44-L60
        pass

    return pose


# old version: https://github.com/rotem-shalev/Ham2Pose/blob/25a5cd7221dfb81a24088e4f38bca868d8e896fc/metrics.py#L31
# def get_pose(keypoints_path: str, datum_id: str, fps: int = 25):
#     pose = get_pose(keypoints_path, fps)

#     if datum_id in PJM_LEFT_VIDEOS_LST:
#         pose = flip_pose(pose)

#     normalization_info = pose_normalization_info(pose_header)
#     pose = pose.normalize(normalization_info)
#     pose.focus()

#     pose_hide_legs(pose)
#     pose_hide_low_conf(pose)

#     # Prune all leading frames containing only zeros, almost no face, or no hands
#     for i in range(len(pose.body.data)):
#         if pose.body.confidence[i][:, 25:-42].sum() > 35 and \
#                 pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > 0:
#             if i != 0:
#                 pose.body.data = pose.body.data[i:]
#                 pose.body.confidence = pose.body.confidence[i:]
#             break

#     # Prune all trailing frames containing only zeros, almost no face, or no hands
#     for i in range(len(pose.body.data) - 1, 0, -1):
#         if pose.body.confidence[i][:, 25:-42].sum() > 35 and \
#                 pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > 0:
#             if i != len(pose.body.data) - 1:
#                 pose.body.data = pose.body.data[:i + 1]
#                 pose.body.confidence = pose.body.confidence[:i + 1]
#             break

#     return pose


def pose_hide_low_conf(pose: Pose, confidence_threshold: float = 0.2) -> None:
    mask = pose.body.confidence <= confidence_threshold
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask, mask], axis=3)
    masked_data = np.ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data
