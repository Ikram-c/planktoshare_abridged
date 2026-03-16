from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
from PIL import Image


class OutputFormat(Enum):
    PIL = "pil"
    NUMPY = "numpy"


class ImageRecord(TypedDict):
    image: np.ndarray | Image.Image
    filename: str
    dtype: np.dtype
    shape: tuple[int, ...]
    tar_path: str


@dataclass(frozen=True)
class StreamConfig:
    tar_paths: list[str]
    output_format: OutputFormat = OutputFormat.NUMPY
    max_images: Optional[int] = None
    min_size: tuple[int, int] = (128, 128)
    max_size: tuple[int, int] = (16384, 16384)
    convert_mode: Optional[str] = None
    extensions: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"
    )

    def __post_init__(self):
        for path in self.tar_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Archive not found: {path}")

    @classmethod
    def from_dict(cls, data: dict) -> "StreamConfig":
        stream = data.get("stream", data)
        extensions = stream.get("extensions", None)
        if extensions is not None:
            extensions = tuple(extensions)
        min_size = stream.get("min_size", (128, 128))
        max_size = stream.get("max_size", (16384, 16384))
        return cls(
            tar_paths=stream["tar_paths"],
            output_format=OutputFormat(stream.get("output_format", "numpy")),
            max_images=stream.get("max_images"),
            min_size=tuple(min_size),
            max_size=tuple(max_size),
            convert_mode=stream.get("convert_mode"),
            extensions=extensions or cls.extensions,
        )


@dataclass(frozen=True)
class ConcurrencyConfig:
    enabled: bool = True
    max_workers: int = 8
    chunk_size: int = 100

    @classmethod
    def from_dict(cls, data: dict) -> "ConcurrencyConfig":
        section = data.get("concurrency", data)
        return cls(
            enabled=section.get("enabled", True),
            max_workers=section.get("max_workers", 8),
            chunk_size=section.get("chunk_size", 100),
        )
