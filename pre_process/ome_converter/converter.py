import logging
import os
import shutil
import zipfile

import numpy as np
import zarr
from numcodecs import Blosc
from skimage.transform import downscale_local_mean
from zarr.storage import LocalStore, ZipStore

from pre_process.tar_streamer import ImageRecord
from pre_process.resolution_grouper.models import BucketKey
from pre_process.zarr_writer import ZarrWriterConfig
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
        from pre_process.zarr_writer import CODEC_MAP, SHUFFLE_MAP
        return Blosc(
            cname=CODEC_MAP[self._writer_config.compression_codec],
            clevel=self._writer_config.compression_level,
            shuffle=SHUFFLE_MAP[self._writer_config.shuffle],
        )

    def _resolve_dtype(self, records: list[ImageRecord]) -> np.dtype:
        dtypes = {rec["dtype"] for rec in records}
        if len(dtypes) == 1:
            return dtypes.pop()
        return sorted(dtypes, key=lambda d: np.dtype(d).itemsize, reverse=True)[0]

    def _resolve_channels(self, records: list[ImageRecord]) -> int:
        channels = set()
        for rec in records:
            channels.add(rec["shape"][2] if len(rec["shape"]) == 3 else 1)
        return max(channels)

    def _pad_image(
        self, image: np.ndarray, h: int, w: int, c: int, dtype: np.dtype
    ) -> np.ndarray:
        ih, iw = image.shape[:2]
        image = image.astype(dtype)

        if image.ndim == 2 and c > 1:
            image = np.stack([image] * c, axis=-1)
        elif image.ndim == 3 and image.shape[2] < c:
            pad_c = c - image.shape[2]
            image = np.concatenate(
                [image, np.zeros((ih, iw, pad_c), dtype=dtype)], axis=2
            )

        if c > 1:
            padded = np.zeros((h, w, c), dtype=dtype)
            padded[:ih, :iw, :] = image
        else:
            if image.ndim == 3:
                image = image[:, :, 0]
            padded = np.zeros((h, w), dtype=dtype)
            padded[:ih, :iw] = image
        return padded

    def _generate_pyramid(
        self, data: np.ndarray, dtype: np.dtype
    ) -> list[np.ndarray]:
        levels = [data]
        current = data
        factor = self._config.pyramid_downsample_factor

        for _ in range(1, self._config.pyramid_levels):
            ndim = current.ndim
            if ndim == 3:
                factors = (1, factor, factor)
            elif ndim == 4:
                factors = (1, factor, factor, 1)
            else:
                factors = (factor, factor)

            downsampled = downscale_local_mean(current, factors)
            downsampled = downsampled.astype(dtype)
            levels.append(downsampled)
            current = downsampled

        return levels

    @staticmethod
    def _zip_directory(source_dir: str, zip_path: str):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for root, _dirs, files in os.walk(source_dir):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    arcname = os.path.relpath(file_path, source_dir)
                    zf.write(file_path, arcname)

    def convert_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> OmeManifest:
        n = len(records)
        dtype = self._resolve_dtype(records)
        n_channels = self._resolve_channels(records)
        tile = self._config.tile_size
        h, w = key.height, key.width
        compressor = self._build_compressor()

        if n_channels > 1:
            full_shape = (n, h, w, n_channels)
            chunk_shape = (1, tile, tile, n_channels)
            ndim = 4
        else:
            full_shape = (n, h, w)
            chunk_shape = (1, tile, tile)
            ndim = 3

        ome_path = os.path.join(
            self._config.output_dir, f"bucket_{key}.ome.zarr"
        )
        tmp_path = ome_path + ".tmp"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(self._config.output_dir, exist_ok=True)

        store = LocalStore(root=tmp_path)
        root = zarr.open_group(store=store, mode="w", zarr_format=2)

        arr = root.create_array(
            name="0",
            shape=full_shape,
            chunks=chunk_shape,
            dtype=dtype,
            compressors=compressor,
            fill_value=0,
        )

        for i, rec in enumerate(records):
            image = rec["image"]
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            padded = self._pad_image(image, h, w, n_channels, dtype)
            if padded.ndim == 2:
                arr[i, :, :] = padded
            else:
                arr[i, :, :, :] = padded

        n_levels = 1
        if self._config.generate_pyramid:
            full_data = arr[:]
            pyramid = self._generate_pyramid(full_data, dtype)
            n_levels = len(pyramid)

            for lvl_idx in range(1, n_levels):
                lvl_data = pyramid[lvl_idx]
                lvl_chunks = tuple(
                    min(c, s) for c, s in zip(chunk_shape, lvl_data.shape)
                )
                lvl_arr = root.create_array(
                    name=str(lvl_idx),
                    shape=lvl_data.shape,
                    chunks=lvl_chunks,
                    dtype=lvl_data.dtype,
                    compressors=compressor,
                    fill_value=0,
                )
                lvl_arr[:] = lvl_data

        ms_attrs = build_multiscales_attrs(
            name=f"bucket_{key}",
            pixel_size_um=self._config.pixel_size_um,
            n_levels=n_levels,
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
        pipeline_meta = build_pipeline_metadata(
            key, records, tile, compression_str
        )
        for k, v in pipeline_meta.items():
            root.attrs[k] = v

        zip_path = None
        if self._config.zip_store:
            zip_path = ome_path + ".zip"
            self._zip_directory(tmp_path, zip_path)
            logger.info("ZipStore written: %s", zip_path)

        if os.path.exists(ome_path):
            shutil.rmtree(ome_path)
        os.rename(tmp_path, ome_path)

        manifest = OmeManifest(
            bucket=str(key),
            ome_store_path=ome_path,
            zip_path=zip_path,
            n_images=n,
            pyramid_levels=n_levels,
            pixel_size_um=self._config.pixel_size_um,
        )
        self._manifests.append(manifest)

        logger.info(
            "OME-Zarr %s: %d images, %d levels, pixel=%.3fum -> %s",
            key, n, n_levels, self._config.pixel_size_um, ome_path,
        )
        return manifest

    def convert_all(
        self,
        buckets: dict[BucketKey, list[ImageRecord]],
    ) -> list[OmeManifest]:
        for key, records in buckets.items():
            self.convert_bucket(key, records)
        return list(self._manifests)

    @property
    def manifests(self) -> list[OmeManifest]:
        return list(self._manifests)