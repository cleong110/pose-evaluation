import copy
import json
from pathlib import Path

import pytest
from pose_format import Pose
from pose_format.utils.generic import fake_pose
from pose_format.utils.openpose_135 import OpenPose_Components as openpose_135_components

from pose_evaluation.utils.pose_utils import load_pose_file

utils_standard_mediapipe_landmarks_test_data_dir = (
    Path(__file__).parent / "test" / "test_data" / "mediapipe" / "standard_landmarks"
)


@pytest.fixture(scope="function")
def mediapipe_poses_test_data_paths() -> list[Path]:
    pose_file_paths = list(utils_standard_mediapipe_landmarks_test_data_dir.glob("*.pose"))
    pose_file_paths.sort(key=lambda p: p.name)
    return pose_file_paths


@pytest.fixture(scope="function")
def mediapipe_poses_test_data(mediapipe_poses_test_data_paths) -> list[Pose]:  # pylint: disable=redefined-outer-name
    original_poses = [load_pose_file(pose_path) for pose_path in mediapipe_poses_test_data_paths]
    # I ran into issues where if one test would modify a Pose, it would affect other tests.
    # specifically, pose.header.components[0].name = unsupported_component_name in test_detect_format
    # this ensures we get a fresh object each time.
    return copy.deepcopy(original_poses)


@pytest.fixture(scope="function")
def mediapipe_poses_test_data_refined() -> list[Pose]:  # pylint: disable=redefined-outer-name
    refined_path = utils_standard_mediapipe_landmarks_test_data_dir.parent / "refined_landmarks"
    pose_paths = refined_path.glob("*.pose")
    original_poses = [load_pose_file(pose_path) for pose_path in pose_paths]
    return copy.deepcopy(original_poses)


@pytest.fixture(scope="function")
def mediapipe_poses_test_data_mixed_shapes() -> list[Pose]:  # pylint: disable=redefined-outer-name
    mixed_pair_path = utils_standard_mediapipe_landmarks_test_data_dir.parent / "mixed"

    # Data: <class 'numpy.ma.core.MaskedArray'> (37, 1, 576, 3), float32
    pose1 = "000017451997373907346-LIBRARY.pose"

    # Data: <class 'numpy.ma.core.MaskedArray'> (44, 1, 553, 3), float32
    pose2 = "SbTU6uS4cc1tZpqCnE3g.pose"

    # Data: <class 'numpy.ma.core.MaskedArray'> (111, 1, 553, 3), float32
    pose3 = "Ui9Nq58yYSgkJslO02za.pose"
    # pose_paths = mixed_pair_path.glob("*.pose")
    pose_paths = [mixed_pair_path / fname for fname in [pose1, pose2, pose3]]
    original_poses = [load_pose_file(pose_path) for pose_path in pose_paths]
    return copy.deepcopy(original_poses)


@pytest.fixture
def standard_mediapipe_components_dict() -> dict[str, list[str]]:
    format_json = utils_standard_mediapipe_landmarks_test_data_dir / "mediapipe_components_and_points.json"
    with open(format_json, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def fake_openpose_poses(count: int = 3) -> list[Pose]:
    return [fake_pose(30) for _ in range(count)]


@pytest.fixture
def fake_openpose_135_poses(count: int = 3) -> list[Pose]:
    return [fake_pose(30, components=openpose_135_components) for _ in range(count)]
