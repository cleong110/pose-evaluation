from pathlib import Path
import fnmatch
from typing import Optional, List, Dict
import typer

app = typer.Typer()


def collect_files_once(
    base: Path,
    pattern_map: Dict[str, List[str]],
) -> Dict[str, List[Path]]:
    """Walk the directory once and classify files by matching patterns."""
    result = {key: [] for key in pattern_map}
    for f in base.rglob("*"):
        if not f.is_file():
            continue
        for category, patterns in pattern_map.items():
            if any(fnmatch.fnmatch(f.name, pattern) for pattern in patterns):
                result[category].append(f)
    # Sort all lists for consistency
    for files in result.values():
        files.sort()

    for name, paths in result.items():
        typer.echo(f"🎯 Found {len(paths)} {name.replace('_', ' ')}. Samples:")
        for path in paths[:3]:
            typer.echo(f"* {path}")
    return result


def collect_files_main(
    dataset_path: Path,
    pose_files_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    video_files_path: Optional[Path] = None,
    pose_patterns: Optional[List[str]] = None,
    metadata_patterns: Optional[List[str]] = None,
    video_patterns: Optional[List[str]] = None,
):
    """Efficiently collect all files by walking each root directory only once."""
    if pose_patterns is None:
        pose_patterns = ["*.pose", "*.pose.zst"]

    if metadata_patterns is None:
        metadata_patterns = ["*.csv"]

    if video_patterns is None:
        video_patterns = ["*.mp4", "*.avi", "*.mov"]

    result = {}

    search_roots = {
        "pose": (pose_files_path or dataset_path, pose_patterns),
        "metadata": (metadata_path or dataset_path, metadata_patterns),
        "video": (video_files_path or dataset_path, video_patterns),
    }

    # Group by root to avoid repeated walks
    root_to_keys = {}
    for key, (root, patterns) in search_roots.items():
        if patterns is not None:
            root_to_keys.setdefault(root, []).append((key, patterns))

    for root, keys_and_patterns in root_to_keys.items():
        pattern_map = dict(keys_and_patterns)
        root_results = collect_files_once(root, pattern_map)
        result.update({f"{key.upper()}_FILES": root_results[key] for key in pattern_map})

    return result


@app.command()
def collect_files_cli(
    dataset_path: Path = typer.Argument(..., exists=True, file_okay=False),
    pose_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    metadata_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    video_files_path: Optional[Path] = typer.Option(None, exists=True, file_okay=False),
    pose_patterns: List[str] = typer.Option(["*.pose", "*.pose.zst"]),
    metadata_patterns: List[str] = typer.Option(["*.csv"]),
    video_patterns: List[str] = typer.Option(["*.mp4", "*.avi", "*.mov"]),
):
    """CLI wrapper around collect_files_main"""
    # pylint: disable=duplicate-code
    result = collect_files_main(
        dataset_path=dataset_path,
        pose_files_path=pose_files_path,
        metadata_path=metadata_path,
        video_files_path=video_files_path,
        pose_patterns=pose_patterns,
        metadata_patterns=metadata_patterns,
        video_patterns=video_patterns,
    )
    # pylint: enable=duplicate-code

    for name, paths in result.items():
        typer.echo(f"✅ Found {len(paths)} {name.replace('_', ' ')}")


if __name__ == "__main__":
    app()
