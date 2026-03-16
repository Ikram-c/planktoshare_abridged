# pre_process/bubble_filter/filter.py
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pre_process.tar_streamer import ImageRecord
from pre_process.resolution_grouper.models import BucketKey
from .autoencoder import load_checkpoint, predict
from .features import add_derived_features, aggregate_features, extract_features
from .models import FilterConfig, FilterResult
from .segmentation import segment, to_grayscale

logger = logging.getLogger(__name__)


class BubbleFilter:

    def __init__(self, config: FilterConfig):
        self._config = config
        self._model = None
        self._scaler = None
        self._feature_names = []

        if not config.checkpoint_path:
            logger.warning(
                "No checkpoint_path configured — bubble filter disabled, all images will pass through"
            )
        else:
            try:
                self._model, self._scaler, self._feature_names = load_checkpoint(
                    config.checkpoint_path, config.device
                )
            except FileNotFoundError:
                logger.warning(
                    "Checkpoint not found at %r — bubble filter disabled, all images will pass through",
                    config.checkpoint_path,
                )
        self._n_passed = 0
        self._n_rejected = 0
        self._n_errors = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        status = "disabled" if self._model is None else f"features={len(self._feature_names)}"
        return (
            f"BubbleFilter(threshold={self._config.threshold}, "
            f"{status}, "
            f"passed={self._n_passed}, rejected={self._n_rejected})"
        )

    @property
    def _filter_active(self) -> bool:
        return self._model is not None

    def __len__(self) -> int:
        return self._n_passed + self._n_rejected + self._n_errors

    def _update_counts(self, *, passed: int = 0, rejected: int = 0, errors: int = 0) -> None:
        with self._lock:
            self._n_passed += passed
            self._n_rejected += rejected
            self._n_errors += errors

    def classify_image(self, record: ImageRecord) -> FilterResult:
        if not self._filter_active:
            return FilterResult(
                filename=record["filename"], is_bubble=False, bubble_score=0.0, object_count=0
            )
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
                filename=record["filename"], is_bubble=False, bubble_score=0.0, object_count=0
            )

        feature_df = add_derived_features(extract_features(gray, labels))
        score = predict(aggregate_features(feature_df), self._model, self._scaler, self._feature_names)

        return FilterResult(
            filename=record["filename"],
            is_bubble=score >= self._config.threshold,
            bubble_score=score,
            object_count=n_objects,
        )

    def filter_record(self, record: ImageRecord) -> tuple[bool, FilterResult]:
        try:
            result = self.classify_image(record)
            self._update_counts(
                passed=int(not result.is_bubble),
                rejected=int(result.is_bubble),
            )
            return not result.is_bubble, result
        except Exception as exc:
            logger.warning("Filter error for %s: %s", record["filename"], exc)
            self._update_counts(errors=1)
            return True, FilterResult(
                filename=record["filename"], is_bubble=False, bubble_score=0.0, object_count=0
            )

    def filter_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> tuple[list[ImageRecord], list[FilterResult]]:
        pairs = [self.filter_record(r) for r in records]
        results = [res for _, res in pairs]
        passed = [
            {**rec, "bubble_score": res.bubble_score, "object_count": res.object_count}
            for (keep, res), rec in zip(pairs, records)
            if keep
        ]
        logger.info(
            "Bucket %s: %d/%d passed (%.1f%% rejected)",
            key, len(passed), len(records),
            (1 - len(passed) / max(len(records), 1)) * 100,
        )
        return passed, results

    def filter_buckets(
        self,
        buckets: dict[BucketKey, list[ImageRecord]],
    ) -> tuple[dict[BucketKey, list[ImageRecord]], dict[BucketKey, list[FilterResult]]]:
        filtered: dict[BucketKey, list[ImageRecord]] = {}
        all_results: dict[BucketKey, list[FilterResult]] = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.filter_bucket, key, records): key
                for key, records in buckets.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                passed, results = future.result()
                all_results[key] = results
                if passed:
                    filtered[key] = passed

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
            "total": len(self),
        }