from dataclasses import dataclass
from enum import Enum

from pre_process._pre_process_utils.interfaces import BaseManifest, PipelineConfig

__all__ = [
    "ArrayManifest",
    "BaseManifest",
    "CompressionCodec",
    "PipelineConfig",
    "ShuffleMode",
    "ZarrWriterConfig",
]


class CompressionCodec(Enum):
    """Supported Blosc compression codecs."""

    BLOSC_ZSTD = "blosc_zstd"
    BLOSC_LZ4 = "blosc_lz4"
    BLOSC_LZ4HC = "blosc_lz4hc"
    ZSTD = "zstd"


class ShuffleMode(Enum):
    """Blosc byte-shuffle filter modes."""

    NOSHUFFLE = "noshuffle"
    SHUFFLE = "shuffle"
    BITSHUFFLE = "bitshuffle"


@dataclass(frozen=True)
class ZarrWriterConfig(PipelineConfig):
    """
    Immutable configuration for :class:`~zarr_writer.writer.ZarrWriter`.

    All fields map directly to keys under the ``zarr_writer`` YAML section.
    ``max_workers`` falls back to ``concurrency.max_workers`` when absent
    from that section.
    """

    output_dir: str = "output"
    tile_size: int = 300
    compression_codec: CompressionCodec = CompressionCodec.BLOSC_ZSTD
    compression_level: int = 5
    shuffle: ShuffleMode = ShuffleMode.BITSHUFFLE
    zarr_format: int = 2
    dimension_separator: str = "/"
    max_workers: int = 4

    def __str__(self) -> str:
        return (
            f"ZarrWriterConfig("
            f"output={self.output_dir!r}, "
            f"tile={self.tile_size}, "
            f"codec={self.compression_codec.value!r}, "
            f"level={self.compression_level}, "
            f"workers={self.max_workers})"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ZarrWriterConfig":
        """
        Construct from a validated pipeline config dictionary.

        :param data: Top-level config dict, optionally containing
            ``zarr_writer`` and ``concurrency`` sub-sections.
        :return: Populated :class:`ZarrWriterConfig` instance.
        :rtype: ZarrWriterConfig
        """
        section = data.get("zarr_writer", data)
        concurrency = data.get("concurrency", {})
        return cls(
            output_dir=section.get("output_dir", "output"),
            tile_size=section.get("tile_size", 300),
            compression_codec=CompressionCodec(
                section.get("compression_codec", "blosc_zstd")
            ),
            compression_level=section.get("compression_level", 5),
            shuffle=ShuffleMode(section.get("shuffle", "bitshuffle")),
            zarr_format=section.get("zarr_format", 2),
            dimension_separator=section.get("dimension_separator", "/"),
            max_workers=section.get(
                "max_workers", concurrency.get("max_workers", 4)
            ),
        )


@dataclass
class ArrayManifest(BaseManifest):
    """Record of a single Zarr array written by :class:`ZarrWriter`."""

    bucket: str
    store_path: str
    shape: tuple
    chunks: tuple
    dtype: str
    n_images: int
    compression: str

    def __str__(self) -> str:
        return (
            f"ArrayManifest("
            f"bucket={self.bucket!r}, "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"n_images={self.n_images})"
        )

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "store_path": self.store_path,
            "shape": list(self.shape),
            "chunks": list(self.chunks),
            "dtype": self.dtype,
            "n_images": self.n_images,
            "compression": self.compression,
        }