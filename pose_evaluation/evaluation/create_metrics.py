"""
Creates metrics by composing various settings and preprocessors together
with DistanceMeasures.

They are given names based on the settings. E.g. 'trimmed_dtw' would be
dynamic time warping with trimming.
"""

import itertools
from pathlib import Path

import pandas as pd

from pose_evaluation.evaluation.create_metric import (
    DEFAULT_METRIC_PARAMETERS,
    construct_metric,
    extract_metric_name_dist,
    extract_signature_distance,
)
from pose_evaluation.evaluation.dataset_parsing.dataset_utils import DatasetDFCol
from pose_evaluation.metrics.distance_measure import (
    AggregatedPowerDistance,
    DistanceMeasure,
)
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
from pose_evaluation.metrics.embedding_distance_metric import EmbeddingDistanceMetric
from pose_evaluation.metrics.ham2pose import (
    Ham2PoseDTWMetric,
    Ham2PosenAPEMetric,
    Ham2PosenDTWMetric,
    Ham2PosenMSEMetric,
)
from pose_evaluation.metrics.nonsense_measures import Return4Measure


def get_embedding_metrics(df: pd.DataFrame) -> list:
    print(f"Getting embedding_metrics from df with {df.columns}")
    if DatasetDFCol.EMBEDDING_MODEL in df:
        for model in df[DatasetDFCol.EMBEDDING_MODEL].unique().tolist():
            yield EmbeddingDistanceMetric(model=f"{model}")
    else:
        raise ValueError(f"No {DatasetDFCol.EMBEDDING_MODEL}")


def get_ham2pose_metrics() -> list:
    return [Ham2PosenMSEMetric(), Ham2PosenAPEMetric(), Ham2PoseDTWMetric(), Ham2PosenDTWMetric()]


def get_metrics(
    measures: list[DistanceMeasure] | None = None,
    include_return4=True,
    metrics_out: Path | None = None,
    include_masked: bool | None = False,
    include_ham2pose: bool = False,
):
    metrics = []

    if measures is None:
        measures = [
            DTWDTAIImplementationDistanceMeasure(name="dtaiDTWAggregatedDistanceMeasureFast", use_fast=True),
            # DTWDTAIImplementationDistanceMeasure(
            #     name="dtaiDTWAggregatedDistanceMeasureSlow", use_fast=False
            # # ),  # super slow
            # DTWOptimizedDistanceMeasure(),
            # DTWAggregatedPowerDistanceMeasure(),
            # DTWAggregatedScipyDistanceMeasure(),
            AggregatedPowerDistance(),
        ]
    measure_names = [measure.name for measure in measures]
    assert len(set(measure_names)) == len(measure_names)

    z_speeds = list(DEFAULT_METRIC_PARAMETERS["z_speeds"])
    default_distances = list(DEFAULT_METRIC_PARAMETERS["default_distances"])
    masked_fill_values = list(DEFAULT_METRIC_PARAMETERS["masked_fill_values"])
    if include_masked:
        masked_fill_values.append(None)

    trim_values = list(DEFAULT_METRIC_PARAMETERS["trim_values"])
    normalize_values = list(DEFAULT_METRIC_PARAMETERS["normalize_values"])
    sequence_alignment_strategies = list(DEFAULT_METRIC_PARAMETERS["sequence_alignment_strategies"])
    keypoint_selection_strategies = list(DEFAULT_METRIC_PARAMETERS["keypoint_selection_strategies"])
    fps_values = list(DEFAULT_METRIC_PARAMETERS["fps_values"])

    # Create all combinations
    metric_combinations = itertools.product(
        measures,
        z_speeds,
        default_distances,
        trim_values,
        normalize_values,
        keypoint_selection_strategies,
        fps_values,
        masked_fill_values,
        sequence_alignment_strategies,
    )

    constructed = []

    # Iterate over them
    for (
        measure,
        z_speed,
        default_distance,
        trim,
        normalize,
        strategy,
        fps,
        masked_fill_value,
        sequence_alignment,
    ) in metric_combinations:
        #############
        # DTW vs other sequence alignments
        # DTW metrics don't use a pose preprocessor, they handle it internally in the DistanceMeasure.
        # so we need to catch that.
        if "dtw" in measure.name.lower() and sequence_alignment != "dtw":
            # we don't want double sequence alignment strategies. No need to do zeropad with dtw
            continue

        if sequence_alignment == "dtw" and "dtw" not in measure.name.lower():
            # doesn't work, creates "dtw" metrics that just fail with ValueError:
            # e.g. "operands could not be broadcast together with shapes (620,1,48,3) (440,1,48,3)""
            continue

        metric = construct_metric(
            distance_measure=measure,
            z_speed=z_speed,
            default_distance=default_distance,
            trim_meaningless_frames=trim,
            normalize=normalize,
            sequence_alignment=sequence_alignment,
            keypoint_selection=strategy,
            fps=fps,
            masked_fill_value=masked_fill_value,
        )

        n, s = metric.name, metric.get_signature().format()

        if "defaultdist0.0" in n:
            assert "default_distance:0.0" in s, f"{n}\n{s}\n{measure}"

        metrics.append(metric)
        constructed.append(
            {
                "measure_name": measure.name,
                "default_distance": default_distance,
                "trim": trim,
                "normalize": normalize,
                "keypoint_selection_strategy": strategy,
                "fps": fps or "nointerp",
                "masked_fill_value": masked_fill_value,
                "sequence_alignment": sequence_alignment,
                "metric_name": metric.name,
                "metric_signature": metric.get_signature().format(),
            }
        )

    # baseline/nonsense measures
    if include_return4:
        metrics.append(
            DistanceMetric(
                name="Return4Metric_defaultdist4.0", distance_measure=Return4Measure(), pose_preprocessors=[]
            )
        )
    if include_ham2pose:
        metrics.extend(get_ham2pose_metrics())

    metric_names = [metric.name for metric in metrics]
    metric_sigs = [metric.get_signature().format() for metric in metrics]
    metric_names_set = set(metric_names)
    metric_sigs_set = set(metric_sigs)
    assert len(metric_names_set) == len(metric_names)
    assert len(metric_sigs_set) == len(metric_sigs)

    for m_name, m_sig in zip(metric_names, metric_sigs, strict=False):
        sig_distance = extract_signature_distance(m_sig)
        try:
            name_distance = extract_metric_name_dist(m_name)
        except IndexError as e:
            print(f"{e} on {m_name}, {m_sig}")
            raise e
        assert sig_distance == name_distance, f"defaultdist for {m_name} does not match signature {m_sig}"

    if metrics_out:
        df = pd.DataFrame(constructed)
        # metrics_out.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
        df.to_csv(metrics_out, index=False)
        print(f"Saved metric configurations to {metrics_out}")

        for column in df.columns:
            uniques = df[column].unique()
            if len(uniques) < 100:
                print(f"{len(uniques)} values for {column}: {uniques.tolist()}")

    return metrics


if __name__ == "__main__":
    metrics = get_metrics(metrics_out="constructed.csv", include_return4=True, include_ham2pose=True)
    metric_names = [m.name for m in metrics]
    metric_sigs = [m.get_signature().format() for m in metrics]

    print(f"Current settings result in the construction of {len(metrics)} metrics")
    print(f"{len(set(metric_names))} unique metric names")
    print(f"{len(set(metric_sigs))} unique metric signatures")

    for n, s in zip(metric_names, metric_sigs, strict=False):
        if "defaultdist0.0" in n:
            assert "default_distance:0.0" in s, f"{n}\n{s}"

        if "return4" in n.lower():
            print(n)

    tiny_csv_for_testing = Path("/opt/home/cleong/projects/pose-evaluation/tiny_csv_for_testing/asl-citizen.csv")
    # tiny_csv_for_testing = Path(
    #     "/opt/home/cleong/projects/pose-evaluation/tiny_csv_for_testing/asl-citizen-no-embeddings.csv"
    # )
    df = pd.read_csv(tiny_csv_for_testing)
    try:
        embedding_metrics = list(get_embedding_metrics(df))
        for embedding_metric in embedding_metrics:
            print(embedding_metric)
            print(embedding_metric.name)
    except ValueError as e:
        print(e)
