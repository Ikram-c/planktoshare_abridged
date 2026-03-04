import logging
from collections import defaultdict
from typing import Generator

from tar_streamer.models import ImageRecord
from resolution_grouper.models import BucketKey
from .autoencoder import (
    SupervisedAutoencoder,
    load_checkpoint,
    predict,
)
from .features import (
    add_derived_features,
    aggregate_features,
    extract_features,
)
from .models import FilterConfig, FilterResult
from .segmentation import segment, to_grayscale

logger = logging.getLogger(__name__)


class BubbleFilter:

    def __init__(self, config: FilterConfig):
        self._config = config
        self._model, self._scaler, self._feature_names = load_checkpoint(
            config.checkpoint_path, config.device
        )
        self._n_passed = 0
        self._n_rejected = 0
        self._n_errors = 0

    def __repr__(self) -> str:
        return (
            f"BubbleFilter(threshold={self._config.threshold}, "
            f"features={len(self._feature_names)}, "
            f"passed={self._n_passed}, rejected={self._n_rejected})"
        )

    def classify_image(self, record: ImageRecord) -> FilterResult:
        image = record["image"]
        gray = to_grayscale(image) if hasattr(image, "ndim") else image

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

        feature_df = extract_features(gray, labels)
        feature_df = add_derived_features(feature_df)
        agg = aggregate_features(feature_df)

        score = predict(agg, self._model, self._scaler, self._feature_names)

        return FilterResult(
            filename=record["filename"],
            is_bubble=score >= self._config.threshold,
            bubble_score=score,
            object_count=n_objects,
            feature_vector=agg,
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
            logger.warning("Filter error for %s: %s", record["filename"], exc)
            self._n_errors += 1
            result = FilterResult(
                filename=record["filename"],
                is_bubble=False,
                bubble_score=0.0,
                object_count=0,
            )
            return True, result

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
    ) -> tuple[dict[BucketKey, list[ImageRecord]], dict[BucketKey, list[FilterResult]]]:
        filtered = {}
        all_results = {}
        for key, records in buckets.items():
            passed, results = self.filter_bucket(key, records)
            if passed:
                filtered[key] = passed
            all_results[key] = results

        logger.info(
            "Filter complete: %d passed, %d rejected, %d errors",
            self._n_passed, self._n_rejected, self._n_errors,
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