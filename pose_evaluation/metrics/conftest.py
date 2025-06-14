import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from pose_format import Pose


@pytest.fixture(scope="session", autouse=True)
def clean_test_artifacts():
    """Fixture to clean up test artifacts before each test session."""
    test_artifacts_dir = Path(__file__).parent / "tests"  # Using Path
    if test_artifacts_dir.exists():
        shutil.rmtree(test_artifacts_dir)  # shutil.rmtree still works with Path
    test_artifacts_dir.mkdir(parents=True, exist_ok=True)  # Using Path.mkdir
    yield  # This allows the test session to run
    # (Optional) You can add cleanup logic here to run after the session if needed


@pytest.fixture(name="distance_matrix_shape_checker")
def fixture_distance_matrix_shape_checker() -> Callable[[torch.Tensor, torch.Tensor], None]:
    def _check_shape(hyp_count: int, ref_count: int, distance_matrix: torch.Tensor):
        expected_shape = torch.Size([hyp_count, ref_count])
        # "line too long"
        msg = (
            f"For M={hyp_count} hypotheses, N={ref_count} references, "
            f"Distance Matrix should be MxN={expected_shape}. "
            f"Instead, received {distance_matrix.shape}"
        )

        assert distance_matrix.shape == expected_shape, msg

    return _check_shape  # type: ignore


@pytest.fixture(name="distance_range_checker")
def fixture_distance_range_checker() -> Callable[[torch.Tensor | np.ndarray, float, float], None]:
    def _check_range(
        distances: torch.Tensor | np.ndarray,
        min_val: float = 0,
        max_val: float = 2,
    ) -> None:
        max_distance = distances.max().item()
        min_distance = distances.min().item()

        # Use np.isclose for comparisons with tolerance
        assert np.isclose(min_distance, min_val, atol=1e-6) or min_val <= min_distance <= max_val, (
            f"Minimum distance ({min_distance}) is outside the expected range [{min_val}, {max_val}]"
        )
        assert np.isclose(max_distance, max_val, atol=1e-6) or min_val <= max_distance <= max_val, (
            f"Maximum distance ({max_distance}) is outside the expected range [{min_val}, {max_val}]"
        )

    return _check_range


@pytest.fixture
def real_pose_files() -> list[Pose]:
    # pose_evaluation/utils/test/test_data/standard_landmarks
    test_files_folder = Path("pose_evaluation") / "utils" / "test" / "test_data" / "mediapipe" / "standard_landmarks"
    real_pose_files_list = [Pose.read(test_file.read_bytes()) for test_file in test_files_folder.glob("*.pose")]
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
