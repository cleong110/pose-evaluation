from pathlib import Path

import pytest
from pose_format import Pose


@pytest.fixture
def real_pose_file_paths() -> list[Path]:
    test_files_folder = Path("pose_evaluation") / "utils" / "test" / "test_data" / "mediapipe" / "standard_landmarks"
    return list(test_files_folder.glob("*.pose"))


@pytest.fixture
def real_pose_files(real_pose_file_paths) -> list[Pose]:
    # pose_evaluation/utils/test/test_data/standard_landmarks
    real_pose_files_list = [Pose.read(test_file.read_bytes()) for test_file in real_pose_file_paths]
    return real_pose_files_list


@pytest.fixture
def real_refined_landmark_pose_file_paths() -> list[Path]:
    test_files_folder = Path("pose_evaluation") / "utils" / "test" / "test_data" / "mediapipe" / "refined_landmarks"
    return list(test_files_folder.glob("*.pose"))


@pytest.fixture
def real_mixed_shape_files() -> list[Pose]:
    # pose_evaluation/utils/test/test_data/standard_landmarks
    test_files_folder = Path("pose_evaluation") / "utils" / "test" / "test_data" / "mediapipe" / "mixed"
    real_pose_files_list = [Pose.read(test_file.read_bytes()) for test_file in test_files_folder.glob("*.pose")]
    return real_pose_files_list
