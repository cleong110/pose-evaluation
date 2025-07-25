import copy
import re
from typing import Literal

from pose_evaluation.metrics.distance_measure import (
    DistanceMeasure,
)
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.pose_processors import (
    AddTOffsetsToZPoseProcessor,
    FillMaskedOrInvalidValuesPoseProcessor,
    FirstFramePadShorterPosesProcessor,
    GetFingertipsOnlyHolisticPoseProcessor,
    GetHandsOnlyHolisticPoseProcessor,
    GetYoutubeASLKeypointsPoseProcessor,
    HideLegsPosesProcessor,
    InterpolateAllToSetFPSPoseProcessor,
    MaskInvalidValuesPoseProcessor,
    NormalizePosesProcessor,
    ReduceHolisticPoseProcessor,
    ReducePosesToCommonComponentsProcessor,
    RemoveWorldLandmarksProcessor,
    TrimMeaninglessFramesPoseProcessor,
    ZeroPadShorterPosesProcessor,
)

# --- Constants & Regexes ------------------------------------------------
# Signature: default_distance:<float>
_SIGNATURE_RE = re.compile(r"default_distance:([\d.]+)")
# metric: defaultdist<float>
_DEFAULTDIST_RE = re.compile(r"defaultdist([\d.]+)")


DEFAULT_METRIC_PARAMETERS = {
    "z_speeds": [None, 0.1, 1.0, 4.0, 100.0, 1000.0],
    "default_distances": [0.0, 1.0, 10.0, 100.0, 1000.0],
    "masked_fill_values": [0.0, 1.0, 10.0, 100.0, 1000.0],
    "trim_values": [True, False],
    "normalize_values": [True, False],
    "sequence_alignment_strategies": ["zeropad", "padwithfirstframe", "dtw"],
    "keypoint_selection_strategies": [
        "removelegsandworld",
        "reduceholistic",
        "hands",
        "youtubeaslkeypoints",
        "fingertips",
    ],
    "fps_values": [None, 15, 30, 45, 60, 120],
}


def extract_signature_distance(signature: str) -> str | None:
    """
    From a signature string, extract the float following
    'default_distance:'.

    Returns None if not found.
    """
    m = _SIGNATURE_RE.search(signature)
    return float(m.group(1)) if m else None


def extract_metric_name_dist(metric_name: str) -> float | None:
    """
    From a metric_name, extract the float following 'defaultdist'.

    Returns None if not found.
    """
    m = _DEFAULTDIST_RE.search(metric_name)
    return float(m.group(1)) if m else None


def construct_metric(
    distance_measure: DistanceMeasure,
    default_distance=0.0,
    trim_meaningless_frames: bool = True,
    normalize: bool = True,
    sequence_alignment: Literal["zeropad", "dtw", "padwithfirstframe"] = "padwithfirstframe",
    keypoint_selection: Literal[
        "removelegsandworld", "reduceholistic", "hands", "youtubeaslkeypoints"
    ] = "removelegsandworld",
    masked_fill_value: float | None = None,
    fps: int | None = None,
    name: str | None = None,
    z_speed: float | None = None,
    reduce_poses_to_common_components: bool = True,
):
    distance_measure = copy.deepcopy(distance_measure)
    name_pieces = []
    if name is None:
        name = ""
    pose_preprocessors = []

    if trim_meaningless_frames:
        name_pieces.append("startendtrimmed")
        pose_preprocessors.append(TrimMeaninglessFramesPoseProcessor())
    else:
        name_pieces.append("untrimmed")

    if z_speed is not None:
        name_pieces.append(f"zspeed{z_speed}")
        pose_preprocessors.append(AddTOffsetsToZPoseProcessor())

    if normalize:
        name_pieces.append("normalizedbyshoulders")
        pose_preprocessors.append(NormalizePosesProcessor())
    else:
        name_pieces.append("unnormalized")

    #########################################
    # Keypoint Selection strategy
    if keypoint_selection == "hands":
        pose_preprocessors.append(GetHandsOnlyHolisticPoseProcessor())
    elif keypoint_selection == "reduceholistic":
        pose_preprocessors.append(ReduceHolisticPoseProcessor())
    elif keypoint_selection == "youtubeaslkeypoints":
        pose_preprocessors.append(GetYoutubeASLKeypointsPoseProcessor())
    elif keypoint_selection == "fingertips":
        pose_preprocessors.append(GetFingertipsOnlyHolisticPoseProcessor())
    else:
        pose_preprocessors.append(RemoveWorldLandmarksProcessor())
        pose_preprocessors.append(HideLegsPosesProcessor())

    name_pieces.append(keypoint_selection)

    ######################
    # Default Distances
    name_pieces.append(f"defaultdist{default_distance}")
    distance_measure.set_default_distance(default_distance)
    assert f"default_distance:{default_distance}" in distance_measure.get_signature().format(), (
        f"{distance_measure.default_distance}, {default_distance}"
    )

    ##########################################
    # FPS Strategy
    if fps is not None:
        pose_preprocessors.append(InterpolateAllToSetFPSPoseProcessor(fps=fps))
        name_pieces.append(f"interp{fps}")
    else:
        name_pieces.append("nointerp")

    ################################################
    # Sequence Alignment
    # Only can go AFTER things that change the length like Interpolate
    # if not then it's probably dtw, so do nothing
    if sequence_alignment == "zeropad":
        pose_preprocessors.append(ZeroPadShorterPosesProcessor())
    elif sequence_alignment == "padwithfirstframe":
        pose_preprocessors.append(FirstFramePadShorterPosesProcessor())

    name_pieces.append(sequence_alignment)

    ###########################################################
    # Masked and/or Invalid Values Strategy
    if masked_fill_value is not None:
        pose_preprocessors.append(FillMaskedOrInvalidValuesPoseProcessor(masked_fill_value))
        name_pieces.append(f"fillmasked{masked_fill_value}")
    else:
        name_pieces.append("maskinvalidvals")
        pose_preprocessors.append(MaskInvalidValuesPoseProcessor())

    ###################################################################
    # Components/points Alignment
    if reduce_poses_to_common_components:
        pose_preprocessors.append(ReducePosesToCommonComponentsProcessor())

    if "Measure" in distance_measure.name:
        name = f"{distance_measure.name}".replace("Measure", "Metric")
    else:
        name = f"{distance_measure.name}Metric"

    name = "_".join(name_pieces) + "_" + name

    return DistanceMetric(name=name, distance_measure=distance_measure, pose_preprocessors=pose_preprocessors)
