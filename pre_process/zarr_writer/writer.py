import logging
import os
import shutil

import numpy as np
import zarr
from numcodecs import Blosc
from zarr.storage import LocalStore

from pre_process.resolution_grouper.models import BucketKey
from pre_process.tar_streamer import ImageRecord
from ._array_utils import pad_image, resolve_channels, resolve_dtype
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

    def _compute_shapes(
        self, key: BucketKey, n: int, n_channels: int
    ) -> tuple[tuple, tuple]:
        t = self._config.tile_size
        if n_channels == 1:
            return (n, key.height, key.width), (1, t, t)
        return (n, key.height, key.width, n_channels), (1, t, t, n_channels)

    def write_bucket(
        self, key: BucketKey, records: list[ImageRecord]
    ) -> ArrayManifest:
        n = len(records)
        dtype = resolve_dtype(records)
        n_channels = resolve_channels(records)
        shape, chunks = self._compute_shapes(key, n, n_channels)
        compressor = self._build_compressor()

        final_path = os.path.join(self._config.output_dir, f"bucket_{key}.zarr")
        tmp_path = final_path + ".tmp"

        os.makedirs(self._config.output_dir, exist_ok=True)
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

        store = LocalStore(root=tmp_path)
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
            padded = pad_image(image, key.height, key.width, n_channels, dtype)
            if padded.ndim == 2:
                arr[i, :, :] = padded
            else:
                arr[i, :, :, :] = padded

        store.close()
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.rename(tmp_path, final_path)

        manifest = ArrayManifest(
            bucket=str(key),
            store_path=final_path,
            shape=shape,
            chunks=chunks,
            dtype=str(dtype),
            n_images=n,
            compression=(
                f"{self._config.compression_codec.value}_"
                f"clevel{self._config.compression_level}_"
                f"{self._config.shuffle.value}"
            ),
        )
        self._manifests.append(manifest)
        logger.info(
            "Wrote bucket %s: %d images, shape=%s, dtype=%s -> %s",
            key, n, shape, dtype, final_path,
        )
        return manifest

    def write_all(
        self, buckets: dict[BucketKey, list[ImageRecord]]
    ) -> list[ArrayManifest]:
        for key, records in buckets.items():
            self.write_bucket(key, records)
        return list(self._manifests)

    @property
    def manifests(self) -> list[ArrayManifest]:
        return list(self._manifests)