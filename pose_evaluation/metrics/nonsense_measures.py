from numpy.ma.core import MaskedArray as MaskedArray
import numpy as np

np.random.normal
from pose_evaluation.metrics.distance_measure import DistanceMeasure

# flatten to 1D, absolute value distance
# hash the filenames
# phash the frames
#


from typing import Literal
import numpy as np
from numpy.ma import MaskedArray


class RandomDistanceMeasure(DistanceMeasure):
    def __init__(
        self,
        distribution: Literal[
            "uniform", "normal", "exponential", "beta", "lognormal", "triangular", "zipf"
        ] = "uniform",
        seed: int = 42,
    ) -> None:
        """Initializes the RandomDistanceMeasure with a predefined distribution.

        Args:
            name (str): The name of the distance measure.
            distribution (Literal["uniform", "normal", "exponential", "beta", "lognormal", "triangular", "zipf"]):
                The supported NumPy random distribution.
            seed (int): The random seed for reproducibility.
        """
        super().__init__("RandomDistanceMeasure", default_distance=-1)
        self._rng = np.random.default_rng(seed)

        # Distribution mapping with reasonable defaults
        self._distributions = {
            "uniform": self._rng.uniform,  # Default: uniform(0, 1)
            "normal": self._rng.normal,  # Default: normal(0, 1)
            "exponential": self._rng.exponential,  # Default: lambda=1.0
            "beta": lambda: self._rng.beta(a=2.0, b=2.0),  # Symmetric beta
            "lognormal": lambda: self._rng.lognormal(mean=0.0, sigma=1.0),
            "triangular": lambda: self._rng.triangular(left=0.0, mode=0.5, right=1.0),
            "zipf": lambda: self._rng.zipf(a=2.0),  # Zipf requires a parameter
        }

        if distribution not in self._distributions:
            raise ValueError(f"Unsupported distribution: {distribution}")

        self.distribution = self._distributions[distribution]
        self.seed = seed

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        """Returns a random distance value using the chosen distribution.

        Args:
            hyp_data (MaskedArray): Hypothesis data (not used in randomness).
            ref_data (MaskedArray): Reference data (not used in randomness).

        Returns:
            float: A random distance value.
        """
        return float(self.distribution())  # Ensures return type consistency


class Return4Measure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            "return4",
            default_distance=4,  # chosen by fair dice roll.
            # guaranteed random
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return self.default_distance


class PointCountDifferenceMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            "PointCountDifference",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return np.abs(float(np.prod(hyp_data.shape) - np.prod(ref_data.shape)))


class FrameCountDifferenceMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            "FrameCountDifference",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return np.abs(float(hyp_data.shape[0] - ref_data.shape[0]))


class DifferenceOfSumsMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            "DifferenceOfSums",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return abs(hyp_data.sum() - ref_data.sum())


class AbsMeanCoordinateValueMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__("AbsMeanCoordinateValue", default_distance=-1)

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        combined = np.concatenate((hyp_data, ref_data), axis=0)

        # Compute the mean
        average_value = combined.mean()
        return abs(average_value)


class ReturnPiMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="return_pi",
            default_distance=np.pi,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return self.default_distance


class ReturnEMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="return_e",
            default_distance=np.e,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return self.default_distance


class HashBasedMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="hash_based",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        hyp_hash = abs(hash(hyp_data.tobytes()))
        ref_hash = abs(hash(ref_data.tobytes()))
        return abs(hyp_hash - ref_hash) % 10000  # Keep values reasonable


class ManhattanToMoonMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="manhattan_to_moon",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return np.sum(np.abs(hyp_data - ref_data)) * 384400


class LastElementMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="last_element",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return np.abs(hyp_data.flatten()[-1] - ref_data.flatten()[-1])


class CountSevensMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="count_sevens",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return np.sum(hyp_data == 7) + np.sum(ref_data == 7)


class FirstKeypointDistanceMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="first_keypoint_distance",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return float(np.linalg.norm(hyp_data[0, 0, 0] - ref_data[0, 0, 0]))


class AverageKeypointVarianceMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="average_keypoint_variance",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        return float(np.var(hyp_data) + np.var(ref_data))


class MaskedPointCountDifferenceMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="masked_point_count_diff",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        hyp_masked_count = np.sum(hyp_data.mask)
        ref_masked_count = np.sum(ref_data.mask)
        return np.abs(hyp_masked_count - ref_masked_count)


class MaskedPointProportionDifferenceMeasure(DistanceMeasure):
    def __init__(self) -> None:
        super().__init__(
            name="masked_point_proportion_diff",
            default_distance=-1,
        )

    def get_distance(self, hyp_data: MaskedArray, ref_data: MaskedArray) -> float:
        hyp_total = hyp_data.size
        ref_total = ref_data.size

        hyp_masked_ratio = np.sum(hyp_data.mask) / hyp_total
        ref_masked_ratio = np.sum(ref_data.mask) / ref_total

        return np.abs(hyp_masked_ratio - ref_masked_ratio)
