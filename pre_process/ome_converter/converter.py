import logging
import os
import shutil
import zipfile

import numpy as np
import zarr
from numcodecs import Blosc
from skimage.transform import downscale_local_mean
from zarr.storage import LocalStore

from pre_process.resolution_grouper.models import BucketKey
from pre_process.tar_streamer import ImageRecord
from pre_process.zarr_writer import CODEC_MAP, SHUFFLE_MAP, ZarrWriterConfig
from pre_process.zarr_writer._array_utils import pad_image, resolve_channels, resolve_dtype
from .metadata import build_multiscales_attrs, build_pipeline_metadata
from .models import OmeConverterConfig, OmeManifest

logger = logging.getLogger(__name__)


class OmeZarrConverter:

    def __init__(
        self,
        config: OmeConverterConfig,
        writer_config: ZarrWriterConfig,
    ):
        self._config = config
        self._writer_config = writer_config
        self._manifests: list[OmeManifest] = []

    def __repr__(self) -> str:
        return (
            f"OmeZarrConverter(pixel_size={self._config.pixel_size_um}um, "
            f"pyramid_levels={self._config.pyramid_levels}, "
            f"zip={self._config.zip_store})"
        )

    def _build_compressor(self) -> Blosc:
        return Blosc(
            cname=CODEC_MAP[self._writer_config.compression_codec],
            clevel=self._writer_config.compression_level,
            shuffle=SHUFFLE_MAP[self._writer_config.shuffle],
        )

    def _generate_pyramid(
        self, level0_frames: list[np.ndarray], dtype: np.dtype
    ) -> list[np.ndarray]:
        f = self._config.pyramid_downsample_factor
        levels = [np.stack(level0_frames, axis=0)]
        for _ in range(1, self._config.pyramid_levels):
            prev = levels[-1]
            factors = (1, f, f) if prev.ndim == 3 else (1, f, f, 1)
            down = downscale_local_mean(prev, factors).astype(dtype)
            levels.append(down)
        return levels

    def _write_level(
        self,
        root: zarr.Group,
        name: str,
        data: np.ndarray,
        compressor: Blosc,
        tile: int,
    ) -> None:
        n_channels = data.shape[3] if data.ndim == 4 else 1
        h, w = data.shape[1], data.shape[2]
        clamped = (
            (1, min(tile, h), min(tile, w))
            if n_channels == 1
            else (1, min(tile, h), min(tile, w), n_channels)
        )
        arr = root.create_array(
            name=name,
            shape=data.shape,
            chunks=clamped,
            dtype=data.dtype,
            compressors=compressor,
            fill_value=0,
        )
        arr[:] = data

    def convert_bucket(
        self, key: BucketKey, records: list[ImageRecord]
    ) -> OmeManifest:
        dtype = resolve_dtype(records)
        n_channels = resolve_channels(records)
        t = self._config.tile_size
        n = len(records)
        compressor = self._build_compressor()

        final_path = os.path.join(
            self._config.output_dir, f"bucket_{key}.ome.zarr"
        )
        tmp_path = final_path + ".tmp"
        os.makedirs(self._config.output_dir, exist_ok=True)
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

        shape = (
            (n, key.height, key.width)
            if n_channels == 1
            else (n, key.height, key.width, n_channels)
        )
        chunks = (1, t, t) if n_channels == 1 else (1, t, t, n_channels)

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

        level0_frames: list[np.ndarray] = []
        for i, rec in enumerate(records):
            image = rec["image"]
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            padded = pad_image(image, key.height, key.width, n_channels, dtype)
            if padded.ndim == 2:
                arr[i, :, :] = padded
            else:
                arr[i, :, :, :] = padded
            level0_frames.append(padded)

        if self._config.generate_pyramid and self._config.pyramid_levels > 1:
            pyramid = self._generate_pyramid(level0_frames, dtype)
            for lvl_idx, lvl_data in enumerate(pyramid[1:], start=1):
                self._write_level(root, str(lvl_idx), lvl_data, compressor, t)
        else:
            pyramid = [np.stack(level0_frames, axis=0)]

        pyramid_levels = len(pyramid)
        ndim = 4 if n_channels > 1 else 3

        ms_attrs = build_multiscales_attrs(
            name=f"bucket_{key}",
            pixel_size_um=self._config.pixel_size_um,
            n_levels=pyramid_levels,
            downsample_factor=self._config.pyramid_downsample_factor,
            ndim=ndim,
        )
        for k, v in ms_attrs.items():
            root.attrs[k] = v

        compression_str = (
            f"{self._writer_config.compression_codec.value}_"
            f"clevel{self._writer_config.compression_level}_"
            f"{self._writer_config.shuffle.value}"
        )
        pipeline_meta = build_pipeline_metadata(key, records, t, compression_str)
        for k, v in pipeline_meta.items():
            root.attrs[k] = v

        store.close()
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.rename(tmp_path, final_path)

        zip_path: str | None = None
        if self._config.zip_store:
            zip_path = final_path + ".zip"
            tmp_zip = zip_path + ".tmp"
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as zf:
                for dirpath, _, filenames in os.walk(final_path):
                    for fname in filenames:
                        abs_path = os.path.join(dirpath, fname)
                        zf.write(abs_path, os.path.relpath(abs_path, final_path))
            if os.path.exists(zip_path):
                os.remove(zip_path)
            os.rename(tmp_zip, zip_path)

        manifest = OmeManifest(
            bucket=str(key),
            ome_store_path=final_path,
            zip_path=zip_path,
            n_images=n,
            pyramid_levels=pyramid_levels,
            pixel_size_um=self._config.pixel_size_um,
        )
        self._manifests.append(manifest)
        logger.info(
            "OME-Zarr %s: %d images, %d levels, pixel=%.3fum -> %s",
            key, n, pyramid_levels, self._config.pixel_size_um, final_path,
        )
        return manifest

    def convert_all(
        self, buckets: dict[BucketKey, list[ImageRecord]]
    ) -> list[OmeManifest]:
        for key, records in buckets.items():
            self.convert_bucket(key, records)
        return list(self._manifests)

    @property
    def manifests(self) -> list[OmeManifest]:
        return list(self._manifests)