from pathlib import Path

import pytest
from pose_format import Pose

from pose_evaluation.evaluation.sliding_window import FixedWindowsSetByQueryMeanLengthsStrategy


@pytest.mark.parametrize("mean_query_len", [8, 9, 12, 13, 20])
def test_fixed_window_generates_valid_windows(real_pose_file_paths: list[Path], mean_query_len: int):
    strategy = FixedWindowsSetByQueryMeanLengthsStrategy(mean_query_pose_length=mean_query_len)

    for pose_path in real_pose_file_paths:
        pose = Pose.read(pose_path.read_bytes())
        total_frames = len(pose.body.data)

        if total_frames < strategy.window_length:
            # Skip poses too short for a full window
            continue

        windows = strategy.get_windows(pose=pose, start_frame=0, end_frame=total_frames)

        assert all(isinstance(w, tuple) and len(w) == 2 for w in windows)
        assert all(0 <= start < end <= total_frames for (start, end) in windows)

        # Make sure all windows are the same length
        expected_length = strategy.window_length
        # First N-1 windows should be full length
        if len(windows) > 1:
            assert all((end - start) == expected_length for (start, end) in windows[:-1])

        # Last window can be shorter, but never longer
        assert (windows[-1][1] - windows[-1][0]) <= expected_length

        # Make sure the stride is consistent (between starts)
        if len(windows) >= 2:
            strides = [windows[i + 1][0] - windows[i][0] for i in range(len(windows) - 1)]
            assert all(s == strategy.stride for s in strides)
