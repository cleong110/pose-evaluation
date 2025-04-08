# https://colab.research.google.com/drive/1ryzXKJHULCeUFvtd5GUQXs3SCBfnUAlY?usp=sharing


import numpy as np
import numpy.ma as ma
from scipy.stats import wasserstein_distance

from pose_evaluation.metrics.distance_measure import DistanceMeasure


class SlicedWassersteinMeasure(DistanceMeasure):
    def __init__(self, num_projections=50) -> None:
        super().__init__("SlicedWasserstein", default_distance=0)
        self.num_projections = num_projections  # Number of random slices

    def get_distance(self, hyp_data: ma.MaskedArray, ref_data: ma.MaskedArray) -> float:
        hyp_flat = hyp_data.reshape(-1, hyp_data.shape[-1])  # Flatten spatially, keep XYZ
        ref_flat = ref_data.reshape(-1, ref_data.shape[-1])

        total_dist = 0.0
        for _ in range(self.num_projections):
            random_projection = np.random.randn(hyp_flat.shape[1])  # Random slice direction
            hyp_proj = hyp_flat @ random_projection
            ref_proj = ref_flat @ random_projection
            total_dist += wasserstein_distance(hyp_proj, ref_proj)

        return total_dist / self.num_projections  # Average across projections
