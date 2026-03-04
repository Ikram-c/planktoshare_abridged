import logging
import math
from collections import defaultdict
from typing import Generator, Iterable

import numpy as np

from pre_process.tar_streamer import ImageRecord
from .models import (
    BucketKey,
    BucketStats,
    GrouperConfig,
    SnapMode,
)

logger = logging.getLogger(__name__)


class ResolutionGrouper:

    def __init__(self, config: GrouperConfig):
        self._config = config
        self._buckets: dict[BucketKey, list[ImageRecord]] = defaultdict(list)
        self._stats: dict[BucketKey, BucketStats] = {}

    def __repr__(self) -> str:
        return (
            f"ResolutionGrouper(tile={self._config.tile_size}, "
            f"snap={self._config.effective_snap_grid}, "
            f"buckets={len(self._buckets)})"
        )

    def __len__(self) -> int:
        return len(self._buckets)

    def __getitem__(self, key: BucketKey) -> list[ImageRecord]:
        return self._buckets[key]

    def __iter__(self) -> Generator[tuple[BucketKey, list[ImageRecord]], None, None]:
        yield from self._buckets.items()

    def __contains__(self, key: BucketKey) -> bool:
        return key in self._buckets

    def _snap(self, value: int) -> int:
        grid = self._config.effective_snap_grid
        mode = self._config.snap_mode
        if mode == SnapMode.EXACT:
            return value
        if mode == SnapMode.CEIL:
            return math.ceil(value / grid) * grid
        if mode == SnapMode.FLOOR:
            return max(grid, math.floor(value / grid) * grid)
        return max(grid, round(value / grid) * grid)

    def compute_key(self, h: int, w: int) -> BucketKey:
        return BucketKey(height=self._snap(h), width=self._snap(w))

    def add(self, record: ImageRecord):
        h, w = record["shape"][:2]
        key = self.compute_key(h, w)
        self._buckets[key].append(record)
        if key not in self._stats:
            self._stats[key] = BucketStats(key=key)
        self._stats[key].update(record)

    def ingest(self, records: Iterable[ImageRecord]) -> "ResolutionGrouper":
        for record in records:
            self.add(record)
        self._apply_bucket_constraints()
        logger.info(
            "Ingested %d images into %d buckets",
            sum(len(v) for v in self._buckets.values()),
            len(self._buckets),
        )
        return self

    def _apply_bucket_constraints(self):
        min_size = self._config.min_bucket_size
        max_size = self._config.max_bucket_size
        keys_to_remove = []

        for key, records in self._buckets.items():
            if len(records) < min_size:
                logger.warning(
                    "Bucket %s has %d images (below min %d), dropping",
                    key, len(records), min_size,
                )
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._buckets[key]
            del self._stats[key]

        if max_size is not None:
            for key in list(self._buckets.keys()):
                if len(self._buckets[key]) > max_size:
                    logger.info(
                        "Bucket %s truncated from %d to %d",
                        key, len(self._buckets[key]), max_size,
                    )
                    self._buckets[key] = self._buckets[key][:max_size]
                    self._stats[key].count = max_size

    def pad_image(self, record: ImageRecord, key: BucketKey) -> np.ndarray:
        image = record["image"]
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        h, w = image.shape[:2]
        target_h, target_w = key.height, key.width

        if h == target_h and w == target_w:
            return image

        if not self._config.pad_to_bucket:
            return image

        if image.ndim == 2:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
            padded[:h, :w] = image
        else:
            c = image.shape[2]
            padded = np.zeros((target_h, target_w, c), dtype=image.dtype)
            padded[:h, :w, :] = image
        return padded

    def iter_padded(
        self, key: BucketKey
    ) -> Generator[tuple[ImageRecord, np.ndarray], None, None]:
        for record in self._buckets[key]:
            padded = self.pad_image(record, key)
            yield record, padded

    @property
    def keys_by_count(self) -> list[BucketKey]:
        return sorted(self._buckets.keys(), key=lambda k: len(self._buckets[k]), reverse=True)

    @property
    def keys_by_resolution(self) -> list[BucketKey]:
        return sorted(self._buckets.keys(), key=lambda k: k.pixel_count)

    @property
    def bucket_stats(self) -> dict[BucketKey, BucketStats]:
        return dict(self._stats)

    @property
    def total_images(self) -> int:
        return sum(len(v) for v in self._buckets.values())

    def summary(self) -> list[dict]:
        return [
            self._stats[key].to_dict()
            for key in self.keys_by_count
            if key in self._stats
        ]