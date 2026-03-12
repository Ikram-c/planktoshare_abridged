import logging

import numpy as np

from pre_process.resolution_grouper.models import BucketKey
from pre_process.tar_streamer import ImageRecord
from .models import FilterResult, RuleBasedFilterConfig
from .od_features import (
    compute_background,
    compute_local_background_features,
    extract_od_features,
)
from .rules import BubbleRules, classify_dataframe
from .segmentation import segment, to_grayscale

logger = logging.getLogger(__name__)

_FEATURE_EXPORT_COLS = [
    "label",
    "mean_optical_density",
    "integrated_optical_density",
    "solidity",
    "eccentricity",
    "circularity",
    "intensity_ratio",
    "gradient_rms",
    "intensity_std",
    "bubble_score",
    "is_bubble",
]


class RuleBasedBubbleFilter:

    def __init__(self, config: RuleBasedFilterConfig):
        self._config = config
        self._rules = BubbleRules(
            mod_threshold=config.mod_threshold,
            intensity_ratio_min=config.intensity_ratio_min,
            solidity_min=config.solidity_min,
            eccentricity_max=config.eccentricity_max,
            gradient_rms_max=config.gradient_rms_max,
            intensity_std_max=config.intensity_std_max,
            score_weights=config.score_weights,
            score_threshold=config.score_threshold,
        )
        self._n_passed = 0
        self._n_rejected = 0
        self._n_errors = 0

    def __repr__(self) -> str:
        return (
            f"RuleBasedBubbleFilter("
            f"mod={self._config.mod_threshold}, "
            f"solidity={self._config.solidity_min}, "
            f"score_thresh={self._config.score_threshold}, "
            f"passed={self._n_passed}, rejected={self._n_rejected})"
        )

    def classify_image(self, record: ImageRecord) -> FilterResult:
        image = record["image"]
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        gray = to_grayscale(image) if image.ndim == 3 else image

        labels = segment(
            gray,
            spot_sigma=self._config.spot_sigma,
            outline_sigma=self._config.outline_sigma,
            fallback_to_skimage=self._config.fallback_to_skimage,
        )

        n_objects = int(labels.max())
        if n_objects == 0:
            return FilterResult(
                filename=record["filename"],
                is_bubble=False,
                bubble_score=0.0,
                object_count=0,
            )

        bg_mean = compute_background(gray, labels)
        df = extract_od_features(gray, labels, bg_mean=bg_mean)

        if df.empty:
            return FilterResult(
                filename=record["filename"],
                is_bubble=False,
                bubble_score=0.0,
                object_count=n_objects,
            )

        if self._config.local_background:
            local_df = compute_local_background_features(
                gray, labels, bg_mean, self._config.annulus_width
            )
            if not local_df.empty:
                df = df.merge(local_df, on="label", how="left")
                df["mean_optical_density"] = df["local_mean_od"].combine_first(
                    df["mean_optical_density"]
                )
                df["intensity_ratio"] = df["local_intensity_ratio"].combine_first(
                    df["intensity_ratio"]
                )

        df = classify_dataframe(df, self._rules)

        bubble_fraction = int(df["is_bubble"].sum()) / max(len(df), 1)
        mean_score = float(df["bubble_score"].mean())
        image_is_bubble = bubble_fraction > 0.5

        export_cols = [c for c in _FEATURE_EXPORT_COLS if c in df.columns]
        feature_vector = df[export_cols].to_dict(orient="list")

        return FilterResult(
            filename=record["filename"],
            is_bubble=image_is_bubble,
            bubble_score=mean_score,
            object_count=n_objects,
            feature_vector=feature_vector,
        )

    def filter_record(self, record: ImageRecord) -> tuple[bool, FilterResult]:
        try:
            result = self.classify_image(record)
            if result.is_bubble:
                self._n_rejected += 1
            else:
                self._n_passed += 1
            return not result.is_bubble, result
        except Exception as exc:
            logger.warning("Rule filter error for %s: %s", record["filename"], exc)
            self._n_errors += 1
            return True, FilterResult(
                filename=record["filename"],
                is_bubble=False,
                bubble_score=0.0,
                object_count=0,
            )

    def filter_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> tuple[list[ImageRecord], list[FilterResult]]:
        passed = []
        results = []
        for record in records:
            keep, result = self.filter_record(record)
            results.append(result)
            if keep:
                record["bubble_score"] = result.bubble_score
                record["object_count"] = result.object_count
                passed.append(record)
        logger.info(
            "Bucket %s: %d/%d passed (%.1f%% rejected)",
            key,
            len(passed),
            len(records),
            (1 - len(passed) / max(len(records), 1)) * 100,
        )
        return passed, results

    def filter_buckets(
        self,
        buckets: dict[BucketKey, list[ImageRecord]],
    ) -> tuple[
        dict[BucketKey, list[ImageRecord]],
        dict[BucketKey, list[FilterResult]],
    ]:
        filtered = {}
        all_results = {}
        for key, records in buckets.items():
            passed, results = self.filter_bucket(key, records)
            if passed:
                filtered[key] = passed
            all_results[key] = results
        logger.info(
            "Rule filter complete: %d passed, %d rejected, %d errors",
            self._n_passed,
            self._n_rejected,
            self._n_errors,
        )
        return filtered, all_results

    @property
    def stats(self) -> dict[str, int]:
        return {
            "passed": self._n_passed,
            "rejected": self._n_rejected,
            "errors": self._n_errors,
            "total": self._n_passed + self._n_rejected + self._n_errors,
        }