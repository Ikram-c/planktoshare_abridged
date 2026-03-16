import logging
from pathlib import Path
from typing import cast

import numpy as np
import zarr
from zarr.storage import LocalStore

from pre_process.resolution_grouper.models import BucketKey
from pre_process.tar_streamer import ImageRecord

from ._base import BucketProcessor
from .models import ArrayManifest, ZarrWriterConfig

logger = logging.getLogger(__name__)


class ZarrWriter(BucketProcessor):
    """
    Write resolution-grouped image buckets to individual Zarr v2 arrays.

    Inherits compressor setup, thread lock, manifest list, all dunder
    methods, and the parallel execution loop from :class:`BucketProcessor`.
    The sole responsibility of this class is the Zarr-specific write logic
    in :meth:`write_bucket`.

    :param config: Writer settings including output directory, tile size,
        compression, and worker count.
    """

    def __init__(self, config: ZarrWriterConfig) -> None:
        super().__init__(config)
        self._config = config

    def __repr__(self) -> str:
        return (
            f"ZarrWriter("
            f"output={self._config.output_dir!r}, "
            f"tile={self._config.tile_size}, "
            f"codec={self._config.compression_codec.value!r})"
        )

    def _compute_shapes(
        self,
        key: BucketKey,
        n_images: int,
        n_channels: int,
    ) -> tuple[tuple, tuple]:
        tile, h, w = self._config.tile_size, key.height, key.width
        if n_channels > 1:
            return (n_images, h, w, n_channels), (1, tile, tile, n_channels)
        return (n_images, h, w), (1, tile, tile)

    def _process_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> ArrayManifest:
        return self.write_bucket(key, records)

    def write_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> ArrayManifest:
        """
        Write one resolution bucket to a Zarr v2 array on disk.

        Safe to call concurrently from multiple threads; each call writes
        to a unique store path derived from *key*.

        :param key: Resolution group identifier.
        :param records: Image records for this bucket.
        :raises ValueError: If *key* sanitises to an empty string.
        :raises PermissionError: If the resolved store path escapes
            ``output_dir``.
        :return: Manifest describing the written array.
        :rtype: ArrayManifest
        """
        n = len(records)
        dtype = self._resolve_dtype(records)
        n_channels = self._resolve_channels(records)
        shape, chunks = self._compute_shapes(key, n, n_channels)

        safe_name = self._sanitize_bucket_name(str(key))
        output_dir = Path(self._config.output_dir)
        store_path = output_dir / f"bucket_{safe_name}.zarr"
        self._assert_within_dir(store_path, output_dir)
        store_path.parent.mkdir(parents=True, exist_ok=True)

        root = zarr.open_group(
            store=LocalStore(root=str(store_path)),
            mode="w",
            zarr_format=self._config.zarr_format,
        )
        arr = root.create_array(
            name="0",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressors=self._compressor,
            fill_value=0,
        )

        for i, rec in enumerate(records):
            image = rec["image"]
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            arr[i] = self._pad_image(
                image, key.height, key.width, n_channels, dtype
            )

        manifest = ArrayManifest(
            bucket=str(key),
            store_path=str(store_path),
            shape=shape,
            chunks=chunks,
            dtype=str(dtype),
            n_images=n,
            compression=self._compression_label,
        )

        with self._lock:
            self._manifests.append(manifest)

        logger.info(
            "Wrote bucket %s: %d images, shape=%s, dtype=%s -> %s",
            key, n, shape, dtype, store_path,
        )
        return manifest

    def write_all(
        self,
        buckets: dict[BucketKey, list[ImageRecord]],
    ) -> list[ArrayManifest]:
        """
        Write all buckets in parallel; delegates to :meth:`process_all`.

        :param buckets: Mapping of resolution key → image records.
        :return: Manifests in completion order.
        :rtype: list[ArrayManifest]
        """
        return cast(
            list[ArrayManifest],
            self.process_all(buckets, self._config.max_workers),
        )