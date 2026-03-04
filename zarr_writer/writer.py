import logging
import os
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from zarr.storage import LocalStore

from tar_streamer.models import ImageRecord
from resolution_grouper.models import BucketKey
from .models import (
    ArrayManifest,
    CompressionCodec,
    ShuffleMode,
    ZarrWriterConfig,
)

logger = logging.getLogger(__name__)

SHUFFLE_MAP = {
    ShuffleMode.NOSHUFFLE: Blosc.NOSHUFFLE,
    ShuffleMode.SHUFFLE: Blosc.SHUFFLE,
    ShuffleMode.BITSHUFFLE: Blosc.BITSHUFFLE,
}

CODEC_MAP = {
    CompressionCodec.BLOSC_ZSTD: "zstd",
    CompressionCodec.BLOSC_LZ4: "lz4",
    CompressionCodec.BLOSC_LZ4HC: "lz4hc",
    CompressionCodec.ZSTD: "zstd",
}


class ZarrWriter:

    def __init__(self, config: ZarrWriterConfig):
        self._config = config
        self._manifests: list[ArrayManifest] = []

    def __repr__(self) -> str:
        return (
            f"ZarrWriter(output={self._config.output_dir}, "
            f"tile={self._config.tile_size}, "
            f"codec={self._config.compression_codec.value})"
        )

    def _build_compressor(self) -> Blosc:
        return Blosc(
            cname=CODEC_MAP[self._config.compression_codec],
            clevel=self._config.compression_level,
            shuffle=SHUFFLE_MAP[self._config.shuffle],
        )

    def _resolve_dtype(self, records: list[ImageRecord]) -> np.dtype:
        dtypes = {rec["dtype"] for rec in records}
        if len(dtypes) == 1:
            return dtypes.pop()
        ordered = sorted(dtypes, key=lambda d: np.dtype(d).itemsize, reverse=True)
        logger.warning(
            "Mixed dtypes %s in bucket, upcasting to %s",
            [str(d) for d in dtypes],
            ordered[0],
        )
        return ordered[0]

    def _resolve_channels(self, records: list[ImageRecord]) -> int:
        shapes = {rec["shape"] for rec in records}
        channels = set()
        for s in shapes:
            channels.add(s[2] if len(s) == 3 else 1)
        if len(channels) == 1:
            return channels.pop()
        result = max(channels)
        logger.warning(
            "Mixed channel counts %s in bucket, using max=%d",
            channels, result,
        )
        return result

    def _compute_shapes(
        self,
        key: BucketKey,
        n_images: int,
        n_channels: int,
    ) -> tuple[tuple, tuple]:
        tile = self._config.tile_size
        h, w = key.height, key.width

        if n_channels > 1:
            shape = (n_images, h, w, n_channels)
            chunks = (1, tile, tile, n_channels)
        else:
            shape = (n_images, h, w)
            chunks = (1, tile, tile)

        return shape, chunks

    def _pad_image(
        self,
        image: np.ndarray,
        target_h: int,
        target_w: int,
        target_c: int,
        target_dtype: np.dtype,
    ) -> np.ndarray:
        h, w = image.shape[:2]

        if image.ndim == 2 and target_c > 1:
            image = np.stack([image] * target_c, axis=-1)
        elif image.ndim == 3 and image.shape[2] < target_c:
            pad_c = target_c - image.shape[2]
            image = np.concatenate(
                [image, np.zeros((h, w, pad_c), dtype=image.dtype)], axis=2
            )

        image = image.astype(target_dtype)

        if target_c > 1:
            padded = np.zeros((target_h, target_w, target_c), dtype=target_dtype)
            padded[:h, :w, :] = image
        else:
            if image.ndim == 3:
                image = image[:, :, 0]
            padded = np.zeros((target_h, target_w), dtype=target_dtype)
            padded[:h, :w] = image

        return padded

    def write_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> ArrayManifest:
        n = len(records)
        dtype = self._resolve_dtype(records)
        n_channels = self._resolve_channels(records)
        shape, chunks = self._compute_shapes(key, n, n_channels)
        compressor = self._build_compressor()

        store_path = os.path.join(
            self._config.output_dir, f"bucket_{key}.zarr"
        )
        os.makedirs(self._config.output_dir, exist_ok=True)

        store = LocalStore(root=store_path)
        root = zarr.open_group(store=store, mode="w", zarr_format=2)

        arr = root.create_array(
            name="0",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressors=compressor,
            fill_value=0,
        )

        for i, rec in enumerate(records):
            image = rec["image"]
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            padded = self._pad_image(
                image, key.height, key.width, n_channels, dtype
            )
            if padded.ndim == 2:
                arr[i, :, :] = padded
            else:
                arr[i, :, :, :] = padded

        manifest = ArrayManifest(
            bucket=str(key),
            store_path=store_path,
            shape=shape,
            chunks=chunks,
            dtype=str(dtype),
            n_images=n,
            compression=f"{self._config.compression_codec.value}_"
                        f"clevel{self._config.compression_level}_"
                        f"{self._config.shuffle.value}",
        )
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
        for key, records in buckets.items():
            self.write_bucket(key, records)
        return list(self._manifests)

    @property
    def manifests(self) -> list[ArrayManifest]:
        return list(self._manifests)