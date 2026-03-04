from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CompressionCodec(Enum):
    BLOSC_ZSTD = "blosc_zstd"
    BLOSC_LZ4 = "blosc_lz4"
    BLOSC_LZ4HC = "blosc_lz4hc"
    ZSTD = "zstd"


class ShuffleMode(Enum):
    NOSHUFFLE = "noshuffle"
    SHUFFLE = "shuffle"
    BITSHUFFLE = "bitshuffle"


@dataclass(frozen=True)
class ZarrWriterConfig:
    output_dir: str = "output"
    tile_size: int = 300
    compression_codec: CompressionCodec = CompressionCodec.BLOSC_ZSTD
    compression_level: int = 5
    shuffle: ShuffleMode = ShuffleMode.BITSHUFFLE
    zarr_format: int = 2
    dimension_separator: str = "/"
    shard_images: int = 10
    shard_tiles: int = 10

    @classmethod
    def from_dict(cls, data: dict) -> "ZarrWriterConfig":
        section = data.get("zarr_writer", data)
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
            shard_images=section.get("shard_images", 10),
            shard_tiles=section.get("shard_tiles", 10),
        )


@dataclass
class ArrayManifest:
    bucket: str
    store_path: str
    shape: tuple
    chunks: tuple
    dtype: str
    n_images: int
    compression: str
