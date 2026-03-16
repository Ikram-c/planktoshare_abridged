import logging
import shutil
import zipfile
from pathlib import Path
from typing import cast

import numpy as np
import zarr
from skimage.transform import downscale_local_mean
from zarr.storage import LocalStore

from pre_process.resolution_grouper.models import BucketKey
from pre_process.tar_streamer import ImageRecord
from pre_process.zarr_writer import ZarrWriterConfig
from pre_process.zarr_writer._base import BucketProcessor

from .metadata import build_multiscales_attrs, build_pipeline_metadata
from .models import OmeConverterConfig, OmeManifest

logger = logging.getLogger(__name__)


class OmeZarrConverter(BucketProcessor):
    """
    Convert resolution-grouped image buckets to OME-Zarr 0.4 stores.

    Inherits compressor setup, thread lock, manifest list, all dunder
    methods, and the parallel execution loop from :class:`BucketProcessor`.
    The sole responsibility of this class is the OME-Zarr-specific
    conversion logic in :meth:`convert_bucket`.

    :param config: OME-Zarr output settings (pixel size, pyramid, zip).
    :param writer_config: Compression and worker settings shared with the
        plain Zarr writer; passed to :class:`BucketProcessor` via
        ``super().__init__``.
    """

    def __init__(
        self,
        config: OmeConverterConfig,
        writer_config: ZarrWriterConfig,
    ) -> None:
        super().__init__(writer_config)
        self._config = config
        self._writer_config = writer_config

    def __repr__(self) -> str:
        return (
            f"OmeZarrConverter("
            f"pixel_size={self._config.pixel_size_um}um, "
            f"pyramid_levels={self._config.pyramid_levels}, "
            f"zip={self._config.zip_store})"
        )

    def _generate_pyramid(
        self,
        data: np.ndarray,
        dtype: np.dtype,
    ) -> list[np.ndarray]:
        factor = self._config.pyramid_downsample_factor
        ndim = data.ndim
        if ndim == 3:
            factors = (1, factor, factor)
        elif ndim == 4:
            factors = (1, factor, factor, 1)
        else:
            factors = (factor, factor)
        levels = [data]
        for _ in range(1, self._config.pyramid_levels):
            levels.append(
                downscale_local_mean(levels[-1], factors).astype(dtype)
            )
        return levels

    @staticmethod
    def _zip_directory(source_dir: Path, zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for file_path in source_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(source_dir))

    def _process_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> OmeManifest:
        return self.convert_bucket(key, records)

    def convert_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> OmeManifest:
        """
        Convert one resolution bucket to an OME-Zarr 0.4 store on disk.

        Safe to call concurrently from multiple threads; each call writes
        to a unique store path derived from *key*.

        :param key: Resolution group identifier.
        :param records: Image records for this bucket.
        :raises ValueError: If *key* sanitises to an empty string.
        :raises PermissionError: If any resolved output path escapes
            ``output_dir``.
        :raises RuntimeError: If a symlink is detected at the temporary
            write path.
        :return: Manifest describing the written OME-Zarr store.
        :rtype: OmeManifest
        """
        n = len(records)
        dtype = self._resolve_dtype(records)
        n_channels = self._resolve_channels(records)
        tile, h, w = self._config.tile_size, key.height, key.width

        if n_channels > 1:
            full_shape = (n, h, w, n_channels)
            chunk_shape = (1, tile, tile, n_channels)
            ndim = 4
        else:
            full_shape = (n, h, w)
            chunk_shape = (1, tile, tile)
            ndim = 3

        safe_name = self._sanitize_bucket_name(str(key))
        output_dir = Path(self._config.output_dir)
        ome_path = output_dir / f"bucket_{safe_name}.ome.zarr"
        tmp_path = ome_path.with_suffix(".zarr.tmp")

        self._assert_within_dir(ome_path, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if tmp_path.is_symlink():
            raise RuntimeError(
                f"Refusing to remove tmp path {tmp_path}: unexpected symlink "
                "present. This may indicate a filesystem attack. "
                "Remove it manually to proceed."
            )
        try:
            shutil.rmtree(tmp_path)
        except FileNotFoundError:
            pass

        root = zarr.open_group(
            store=LocalStore(root=str(tmp_path)),
            mode="w",
            zarr_format=2,
        )
        arr = root.create_array(
            name="0",
            shape=full_shape,
            chunks=chunk_shape,
            dtype=dtype,
            compressors=self._compressor,
            fill_value=0,
        )

        for i, rec in enumerate(records):
            image = rec["image"]
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            arr[i] = self._pad_image(image, h, w, n_channels, dtype)

        n_levels = 1
        if self._config.generate_pyramid:
            pyramid = self._generate_pyramid(arr[:], dtype)
            n_levels = len(pyramid)
            for lvl_idx, lvl_data in enumerate(pyramid[1:], start=1):
                lvl_chunks = tuple(
                    min(c, s) for c, s in zip(chunk_shape, lvl_data.shape)
                )
                lvl_arr = root.create_array(
                    name=str(lvl_idx),
                    shape=lvl_data.shape,
                    chunks=lvl_chunks,
                    dtype=lvl_data.dtype,
                    compressors=self._compressor,
                    fill_value=0,
                )
                lvl_arr[:] = lvl_data

        root.attrs.update(
            {
                **build_multiscales_attrs(
                    name=f"bucket_{safe_name}",
                    pixel_size_um=self._config.pixel_size_um,
                    n_levels=n_levels,
                    downsample_factor=self._config.pyramid_downsample_factor,
                    ndim=ndim,
                ),
                **build_pipeline_metadata(
                    key, records, tile, self._compression_label
                ),
            }
        )

        zip_path = None
        if self._config.zip_store:
            zip_path = ome_path.with_suffix(".zarr.zip")
            self._assert_within_dir(zip_path, output_dir)
            self._zip_directory(tmp_path, zip_path)
            logger.info("ZipStore written: %s", zip_path)

        if ome_path.exists():
            shutil.rmtree(ome_path)
        tmp_path.rename(ome_path)

        manifest = OmeManifest(
            bucket=str(key),
            ome_store_path=str(ome_path),
            zip_path=str(zip_path) if zip_path else None,
            n_images=n,
            pyramid_levels=n_levels,
            pixel_size_um=self._config.pixel_size_um,
        )

        with self._lock:
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
        """
        Convert all buckets in parallel; delegates to :meth:`process_all`.

        :param buckets: Mapping of resolution key → image records.
        :return: Manifests in completion order.
        :rtype: list[OmeManifest]
        """
        return cast(
            list[OmeManifest],
            self.process_all(buckets, self._writer_config.max_workers),
        )