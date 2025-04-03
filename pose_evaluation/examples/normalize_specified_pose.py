import argparse
from pathlib import Path
from typing import Union
from pose_format import Pose
from pyzstd import decompress

from pose_evaluation.metrics.pose_processors import NormalizePosesProcessor


def get_pose_data(file_path: Union[Path, str]) -> Pose:
    """Loads a .pose or .pose.zst, returns a Pose object"""
    file_path = Path(file_path)
    if file_path.name.endswith(".pose.zst"):
        return Pose.read(decompress(file_path.read_bytes()))
    else:
        return Pose.read(file_path.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Scores from top-k perspective")

    parser.add_argument(
        "pose_path",
        type=Path,
        help="Path to the pose file",
    )

    parser.add_argument(
        "--out",
        type=Path,
        help="Path to write the normalized pose file",
    )

    args = parser.parse_args()

    pose_path = Path(args.pose_path)
    if args.out is None:
        out_path = pose_path.with_name(f"{pose_path.stem}_normalized.pose")
    else:
        out_path = args.out

    pose = get_pose_data(args.pose_path)
    pose = NormalizePosesProcessor().process_pose(pose)
    print(pose)
    print(out_path)
    with out_path.open("wb") as f:
        pose.write(f)

    print(pose.body.data)
