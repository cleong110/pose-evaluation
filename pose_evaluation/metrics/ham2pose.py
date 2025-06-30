# https://github.com/sign-language-processing/pose-evaluation/issues/31

from typing import Any

import numpy as np
import numpy.ma as ma
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from pose_evaluation.metrics.distance_measure import AggregatedDistanceMeasure
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWAggregatedDistanceMeasure
from pose_evaluation.metrics.pose_processors import (
    HideLowConfPoseProcessor,
    NormalizePosesProcessor,
    ReduceHolisticPoseProcessor,
    ReducePosesToFirstPoseComponentsProcessor,
    RemoveWorldLandmarksProcessor,
)


def get_standard_ham2pose_preprocessors(include_compare_pose_processors=True):
    """Replicate get_pose and normalize_pose from https://github.com/J22Melody/iict-eval-private/blob/text2pose/metrics/metrics.py#L15C1-L33C16"""
    get_pose_preprocessors = [RemoveWorldLandmarksProcessor(), ReduceHolisticPoseProcessor(), NormalizePosesProcessor()]

    compare_poses_preprocessors = [ReducePosesToFirstPoseComponentsProcessor(), HideLowConfPoseProcessor()]
    if not include_compare_pose_processors:
        return get_pose_preprocessors
    else:
        return get_pose_preprocessors + compare_poses_preprocessors


class Ham2PosenMSEDistanceMeasure(AggregatedDistanceMeasure):
    """Distance Measure that replicates `mse` from Ham2Pose. We use"""

    def __init__(self):
        super().__init__(
            name="Ham2Pose_nMSEDistanceMeasure",
            default_distance=0.0,
            aggregation_strategy="mean",
        )

    def _mse_trajectory_distance(self, trajectory1, trajectory2):
        """
        Copied from
        https://github.com/J22Melody/iict-eval-private/blob/text2pose/metrics/ham2pose.py#L102C1-L115C27
        and annotated.
        """
        # We theoretically don't need this, we have Zero-Padding preprocessor
        if len(trajectory1) < len(trajectory2):
            diff = len(trajectory2) - len(trajectory1)
            trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 3))))
        elif len(trajectory2) < len(trajectory1):
            trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 3))))

        # This masks locations where EITHER pose is masked, not the same thing as
        # FillMaskedOrInvalidValuesPoseProcessor
        pose1_mask = np.ma.getmask(trajectory1)
        pose2_mask = np.ma.getmask(trajectory2)
        trajectory1[pose1_mask] = 0
        trajectory1[pose2_mask] = 0
        trajectory2[pose1_mask] = 0
        trajectory2[pose2_mask] = 0

        # Assumption is we've already run the preprocessors
        assert ma.count_masked(trajectory1) == 0
        assert ma.count_masked(trajectory2) == 0
        assert len(trajectory1) == len(trajectory2)
        trajectory1 = np.asarray(trajectory1)
        trajectory2 = np.asarray(trajectory2)

        sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
        return sq_error.mean()

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        """
        Mimic code from compare_poses at
        https://github.com/J22Melody/iict-eval-private/blob/text2pose/metrics/ham2pose.py#L163-L192
        which sums up all the trajectory distances, then takes the mean
        """
        keypoint_count = hyp_data.shape[2]  # Assuming shape: (frames, person, keypoints, xyz)
        trajectory_distances = ma.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_trajectory, ref_trajectory) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            dist = self._mse_trajectory_distance(hyp_trajectory, ref_trajectory)
            trajectory_distances[i] = dist
        trajectory_distances = ma.array(trajectory_distances)
        dist = self._aggregate(trajectory_distances)
        if np.isnan(dist):
            dist = self.default_distance
        return dist


class Ham2PosenMSEMetric(DistanceMetric):
    """Using the `mse` distance, replicates "nMSE" metric from Ham2Pose"""

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        pose_preprocessors = get_standard_ham2pose_preprocessors()
        pose_preprocessors.extend(
            [
                # ZeroPadShorterPosesProcessor(),
                # FillMaskedOrInvalidValuesPoseProcessor(masked_fill_value=0.0), # handled internally
            ]
        )
        distance_measure = Ham2PosenMSEDistanceMeasure()
        super().__init__(
            name="Ham2Pose_nMSE",
            distance_measure=distance_measure,
            pose_preprocessors=pose_preprocessors,
            **kwargs,
        )


class Ham2PoseAPEDistanceMeasure(AggregatedDistanceMeasure):
    """Replicates the 'APE' distance from Ham2Pose"""

    def __init__(self):
        super().__init__(
            name="Ham2Pose_nAPEDistanceMeasure",
            default_distance=0.0,
            aggregation_strategy="mean",
        )

    def _ape_trajectory_distance(self, trajectory1, trajectory2):
        """Copied from Ham2Pose's 'APE' function."""
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

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        """
        Mimic code from compare_poses at
        https://github.com/J22Melody/iict-eval-private/blob/text2pose/metrics/ham2pose.py#L163-L192
        which sums up all the trajectory distances, then takes the mean
        """
        keypoint_count = hyp_data.shape[2]  # Assuming shape: (frames, person, keypoints, xyz)
        trajectory_distances = ma.empty(keypoint_count)  # Preallocate a NumPy array

        for i, (hyp_trajectory, ref_trajectory) in tqdm(
            enumerate(self._get_keypoint_trajectories(hyp_data, ref_data)),
            desc="getting dtw distances for trajectories",
            total=keypoint_count,
            disable=not progress,
        ):
            dist = self._ape_trajectory_distance(hyp_trajectory, ref_trajectory)
            trajectory_distances[i] = dist
        trajectory_distances = ma.array(trajectory_distances)
        dist = self._aggregate(trajectory_distances)
        if np.isnan(dist):
            dist = self.default_distance
        return dist


class Ham2PosenAPEMetric(DistanceMetric):
    """Using the 'APE' distance, replicates 'nAPE' from Ham2Pose"""

    def __init__(
        self,
        # name: str,
        # distance_measure: DistanceMeasure,
        **kwargs: Any,
    ) -> None:
        # standard...
        pose_preprocessors = get_standard_ham2pose_preprocessors()
        pose_preprocessors.extend(
            [
                # ZeroPadShorterPosesProcessor(),
                # FillMaskedOrInvalidValuesPoseProcessor(masked_fill_value=0.0),
            ]
        )
        distance_measure = Ham2PoseAPEDistanceMeasure()
        super().__init__(
            name="Ham2Pose_nAPE",
            distance_measure=distance_measure,
            pose_preprocessors=pose_preprocessors,
            **kwargs,
        )


def unmasked_euclidean(point1, point2):
    """Copied from Ham2Pose"""
    # if not ma.isMaskedArray(point1):
    #     print("point1 is NOT masked array")
    # if not ma.isMaskedArray(point2):
    #     print("point2 is NOT masked array")
    if ma.is_masked(point2):  # reference label keypoint is missing
        return euclidean((0, 0, 0), point1)
    elif ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
        return euclidean((0, 0, 0), point2)
    d = euclidean(point1, point2)
    return d


class Ham2PoseUnmaskedEuclideanDTWDistanceMeasure(DTWAggregatedDistanceMeasure):
    """
    using unmasked_euclidean function, replicates fastdtw from ham2pose
    Ham2Pose does dist = fastdtw(pose1_keypoint_trajectory, pose2_keypoint_trajectory, dist=masked_euclidean)[0] for each keypoint trajectory
    DTWAggregatedDistanceMeasure's get_distance calls
    distance, _ = fastdtw(hyp_trajectory, ref_trajectory, dist=self._calculate_pointwise_distances)
    for each trajectory
    And then finds the mean with self._aggregate(trajectory_distances), mean by default

    So we only need to override _calculate_pointwise_distances in theory,
    but I'm not getting expected results, so we copy the compare_poses code
    """

    def __init__(self):
        super().__init__(
            name="Ham2Pose_UnmaskedEuclideanDTWDistanceMeasure",
            default_distance=0.0,
            aggregation_strategy="mean",
        )

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        idx2weight = dict.fromkeys(range(hyp_data.shape[2]), 1)
        total_distance = 0
        for keypoint_idx, weight in idx2weight.items():
            pose1_keypoint_trajectory = hyp_data[:, :, keypoint_idx, :].squeeze(1)
            pose2_keypoint_trajectory = ref_data[:, :, keypoint_idx, :].squeeze(1)
            # print(f"Trajectory shape: {pose1_keypoint_trajectory.shape}")
            # print(f"Sample values: {pose1_keypoint_trajectory[:2]} vs {pose2_keypoint_trajectory[:2]}")
            try:
                traj_dist = fastdtw(pose1_keypoint_trajectory, pose2_keypoint_trajectory, dist=unmasked_euclidean)[0]
            except ValueError:
                # one of the trajectories contains nan: give up and return default distance for the pose pair
                return self.default_distance
            total_distance += traj_dist * weight
        dist = total_distance / len(idx2weight)

        if np.isnan(dist):
            dist = self.default_distance
        return dist


class Ham2PoseDTWMetric(DistanceMetric):
    """
    Uses Ham2PoseUnmaskedEuclideanDTWDistanceMeasure to replicate 'DTW' from Ham2Pose.
    Preprocessing is standard Ham2Pose,
    we just need to call unmasked_euclidean on each trajectory and take the mean
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        pose_preprocessors = get_standard_ham2pose_preprocessors()
        distance_measure = Ham2PoseUnmaskedEuclideanDTWDistanceMeasure()
        super().__init__(
            name="Ham2Pose_DTW",
            distance_measure=distance_measure,
            pose_preprocessors=pose_preprocessors,
            **kwargs,
        )


def masked_euclidean(point1, point2):
    """Copied from Ham2Pose"""
    if ma.is_masked(point2):  # reference label keypoint is missing
        return 0
    elif ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
        return euclidean((0, 0, 0), point2) / 2
    d = euclidean(point1, point2)
    return d


class Ham2PoseMaskedEuclideanDTWDistanceMeasure(DTWAggregatedDistanceMeasure):
    """
    Same as the other one but using MASKED euclidean
    That's the only difference.
    """

    def __init__(self):
        super().__init__(
            name="Ham2Pose_MaskedEuclideanDTWDistanceMeasure",
            default_distance=0.0,
            aggregation_strategy="mean",
        )

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray, progress=False) -> float:
        idx2weight = dict.fromkeys(range(hyp_data.shape[2]), 1)
        total_distance = 0
        for keypoint_idx, weight in idx2weight.items():
            pose1_keypoint_trajectory = hyp_data[:, :, keypoint_idx, :].squeeze(1)
            pose2_keypoint_trajectory = ref_data[:, :, keypoint_idx, :].squeeze(1)
            try:
                dist = fastdtw(pose1_keypoint_trajectory, pose2_keypoint_trajectory, dist=masked_euclidean)[0]
            except ValueError:
                # one of the trajectories contains nan
                return self.default_distance
            total_distance += dist * weight
        dist = total_distance / len(idx2weight)
        if np.isnan(dist):
            dist = self.default_distance
        return dist


class Ham2PosenDTWMetric(DistanceMetric):
    """
    Uses Ham2PoseMaskedEuclideanDTWDistanceMeasure to replicate 'nDTW' from Ham2Pose.
    Everything's the same as 'Ham2PoseDTWMetric' except the use of 'masked_euclidean'
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        pose_preprocessors = get_standard_ham2pose_preprocessors()
        distance_measure = Ham2PoseMaskedEuclideanDTWDistanceMeasure()
        super().__init__(
            name="Ham2Pose_nDTW",
            distance_measure=distance_measure,
            pose_preprocessors=pose_preprocessors,
            **kwargs,
        )
