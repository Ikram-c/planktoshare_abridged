"""
Abstract base class and shared utilities for bucket processors.

``CODEC_MAP`` and ``SHUFFLE_MAP`` live here so that both
:class:`~zarr_writer.writer.ZarrWriter` and
:class:`~ome_converter.converter.OmeZarrConverter` can resolve codec names
without duplicating the mapping or creating a cross-module import cycle.
"""
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import numpy as np
from numcodecs import Blosc

from pre_process.resolution_grouper.models import BucketKey
from pre_process.tar_streamer import ImageRecord

from .models import (
    BaseManifest,
    CompressionCodec,
    ShuffleMode,
    ZarrWriterConfig,
)

logger = logging.getLogger(__name__)

_UNSAFE_FILENAME_RE = re.compile(r'[/\\:*?"<>|\x00\s]')

SHUFFLE_MAP: dict[ShuffleMode, int] = {
    ShuffleMode.NOSHUFFLE: Blosc.NOSHUFFLE,
    ShuffleMode.SHUFFLE: Blosc.SHUFFLE,
    ShuffleMode.BITSHUFFLE: Blosc.BITSHUFFLE,
}

CODEC_MAP: dict[CompressionCodec, str] = {
    CompressionCodec.BLOSC_ZSTD: "zstd",
    CompressionCodec.BLOSC_LZ4: "lz4",
    CompressionCodec.BLOSC_LZ4HC: "lz4hc",
    CompressionCodec.ZSTD: "zstd",
}


class BucketProcessor(ABC):
    """
    Abstract base class for :class:`~zarr_writer.writer.ZarrWriter` and
    :class:`~ome_converter.converter.OmeZarrConverter`.

    Provides all shared initialisation, dunder methods, the parallel
    execution loop, and stateless image-processing helpers. Subclasses
    must implement :meth:`_process_bucket` and :meth:`__repr__`.

    Thread safety
    -------------
    Shared mutable state (``_manifests``) is guarded by ``_lock``.
    ``_compressor`` and ``_compression_label`` are set once in
    ``__init__`` from immutable config and never mutated, so they are
    safe to read from any thread without synchronisation.

    :param writer_config: Compression and worker settings shared by both
        the plain Zarr writer and the OME-Zarr converter.
    """

    def __init__(self, writer_config: ZarrWriterConfig) -> None:
        self._compressor = Blosc(
            cname=CODEC_MAP[writer_config.compression_codec],
            clevel=writer_config.compression_level,
            shuffle=SHUFFLE_MAP[writer_config.shuffle],
        )
        self._compression_label = (
            f"{writer_config.compression_codec.value}_"
            f"clevel{writer_config.compression_level}_"
            f"{writer_config.shuffle.value}"
        )
        self._manifests: list[BaseManifest] = []
        self._lock = threading.Lock()

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def _process_bucket(
        self,
        key: BucketKey,
        records: list[ImageRecord],
    ) -> BaseManifest:
        """
        Process a single resolution bucket and return its manifest.

        Implemented by each subclass as :meth:`write_bucket` or
        :meth:`convert_bucket`. Called internally by :meth:`process_all`.

        :param key: Resolution group identifier.
        :param records: Image records belonging to this bucket.
        :return: Manifest record for the written output.
        :rtype: BaseManifest
        """

    def __len__(self) -> int:
        return len(self._manifests)

    def __iter__(self) -> Iterator[BaseManifest]:
        return iter(list(self._manifests))

    def __bool__(self) -> bool:
        return bool(self._manifests)

    @property
    def manifests(self) -> list[BaseManifest]:
        """Return a thread-safe snapshot of all manifests written so far."""
        with self._lock:
            return list(self._manifests)

    def process_all(
        self,
        buckets: dict[BucketKey, list[ImageRecord]],
        max_workers: int,
    ) -> list[BaseManifest]:
        """
        Process all buckets in parallel using a thread pool.

        Delegates each bucket to :meth:`_process_bucket`. Completion order
        is non-deterministic. Any exception raised inside a worker is
        re-raised in the calling thread when its future is resolved.

        :param buckets: Mapping of resolution key to image records.
        :param max_workers: Maximum number of concurrent worker threads.
        :return: Manifests in completion order.
        :rtype: list[BaseManifest]
        """
        if not buckets:
            return []
        n_workers = min(max_workers, len(buckets))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._process_bucket, k, v): k
                for k, v in buckets.items()
            }
            return [f.result() for f in as_completed(futures)]

    @staticmethod
    def _sanitize_bucket_name(name: str) -> str:
        """
        Replace filesystem-unsafe characters in a BucketKey string.

        Substitutes path separators, null bytes, and shell-special
        characters with underscores, then strips any leading dots or
        spaces that would produce a hidden file or a name beginning with
        whitespace.

        :param name: Raw ``str(BucketKey)`` value.
        :raises ValueError: If sanitisation yields an empty string.
        :return: Safe filename component (no extension).
        :rtype: str
        """
        sanitized = _UNSAFE_FILENAME_RE.sub("_", name).strip("._")
        if not sanitized:
            raise ValueError(
                f"BucketKey {name!r} yields an empty filename after "
                "sanitisation."
            )
        return sanitized

    @staticmethod
    def _assert_within_dir(path: Path, root: Path) -> None:
        """
        Raise if *path* resolves to a location outside *root*.

        Uses ``os.path.realpath`` and ``os.path.normcase`` rather than
        ``Path.resolve`` + ``relative_to`` to avoid two Windows-specific
        pitfalls:

        * ``Path.resolve()`` may return a ``\\\\?\\``-prefixed
          extended-length path; pathlib parses that prefix as a UNC
          server name, so ``relative_to`` sees the two sides as having
          different anchors even when they point to the same directory.
        * ``relative_to`` raises ``ValueError`` for any mismatch, making
          a broad ``except ValueError`` necessary — which risks swallowing
          unrelated errors from inside the call chain.

        ``os.path.normcase`` lowercases and normalises separators on
        Windows so that case differences never produce a false positive.

        :param path: Candidate output path to validate.
        :param root: Permitted root directory.
        :raises PermissionError: If *path* escapes *root* after resolution.
        """
        real_path = os.path.normcase(os.path.realpath(path))
        real_root = os.path.normcase(os.path.realpath(root))
        real_root_prefix = real_root.rstrip(os.sep) + os.sep
        within = (
            real_path == real_root.rstrip(os.sep)
            or real_path.startswith(real_root_prefix)
        )
        if not within:
            raise PermissionError(
                f"Resolved path {real_path!r} escapes "
                f"output directory {real_root!r}."
            )

    @staticmethod
    def _resolve_dtype(records: list[ImageRecord]) -> np.dtype:
        """
        Return the common dtype for a bucket, upcasting on mixed inputs.

        :param records: Image records belonging to one resolution bucket.
        :return: Resolved :class:`numpy.dtype`.
        :rtype: numpy.dtype
        """
        dtypes = {rec["dtype"] for rec in records}
        if len(dtypes) == 1:
            return dtypes.pop()
        ordered = sorted(
            dtypes, key=lambda d: np.dtype(d).itemsize, reverse=True
        )
        logger.warning(
            "Mixed dtypes %s in bucket; upcasting to %s",
            [str(d) for d in dtypes],
            ordered[0],
        )
        return ordered[0]

    @staticmethod
    def _resolve_channels(records: list[ImageRecord]) -> int:
        """
        Return the channel count for a bucket, taking the maximum on conflict.

        :param records: Image records belonging to one resolution bucket.
        :return: Number of channels.
        :rtype: int
        """
        channels = {
            rec["shape"][2] if len(rec["shape"]) == 3 else 1
            for rec in records
        }
        if len(channels) == 1:
            return channels.pop()
        result = max(channels)
        logger.warning(
            "Mixed channel counts %s in bucket; using max=%d",
            channels,
            result,
        )
        return result

    @staticmethod
    def _pad_image(
        image: np.ndarray,
        target_h: int,
        target_w: int,
        target_c: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        """
        Cast and zero-pad *image* to the target spatial and channel dimensions.

        :param image: Source array with shape ``(H, W)`` or ``(H, W, C)``.
        :param target_h: Target height in pixels.
        :param target_w: Target width in pixels.
        :param target_c: Target channel count.
        :param dtype: Output dtype; cast is applied with ``copy=False``.
        :return: Padded array with shape ``(target_h, target_w)`` or
            ``(target_h, target_w, target_c)``.
        :rtype: numpy.ndarray
        """
        h, w = image.shape[:2]
        image = image.astype(dtype, copy=False)

        if image.ndim == 2 and target_c > 1:
            image = np.stack([image] * target_c, axis=-1)
        elif image.ndim == 3 and image.shape[2] < target_c:
            pad_c = target_c - image.shape[2]
            image = np.concatenate(
                [image, np.zeros((h, w, pad_c), dtype=dtype)], axis=2
            )

        if target_c > 1:
            padded = np.zeros((target_h, target_w, target_c), dtype=dtype)
            padded[:h, :w] = image
        else:
            if image.ndim == 3:
                image = image[..., 0]
            padded = np.zeros((target_h, target_w), dtype=dtype)
            padded[:h, :w] = image

        return padded