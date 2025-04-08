from typing import Any
from pose_format import Pose
from pose_evaluation.metrics.base_pose_metric import PoseMetric
from pose_evaluation.metrics.distance_measure import DistanceMeasure
from pose_evaluation.metrics.distance_metric import DistanceMetric


class TimeDifferenceMetric(PoseMetric):
    def __init__(self, name: str, distance_measure: DistanceMeasure, **kwargs: Any) -> None:
        super().__init__("TimeDifference", **kwargs)

    def _time(self, pose: Pose):
        return pose.body.data.shape[0] / pose.body.fps

    def _pose_score(self, processed_hypothesis: Pose, processed_reference: Pose) -> float:
        hyp_time = processed_hypothesis.body.data.shape[0] / processed_hypothesis.body.fps
        ref_time = processed_reference
