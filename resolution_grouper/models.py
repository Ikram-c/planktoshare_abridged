from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tar_streamer.models import ImageRecord


class SnapMode(Enum):
    CEIL = "ceil"
    FLOOR = "floor"
    ROUND = "round"
    EXACT = "exact"


@dataclass(frozen=True)
class BucketKey:
    height: int
    width: int

    def __str__(self) -> str:
        return f"{self.height}x{self.width}"

    @property
    def pixel_count(self) -> int:
        return self.height * self.width

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


@dataclass(frozen=True)
class GrouperConfig:
    tile_size: int = 300
    snap_grid: Optional[int] = None
    snap_mode: SnapMode = SnapMode.CEIL
    min_bucket_size: int = 1
    max_bucket_size: Optional[int] = None
    pad_to_bucket: bool = True

    @property
    def effective_snap_grid(self) -> int:
        return self.snap_grid if self.snap_grid is not None else self.tile_size

    @classmethod
    def from_dict(cls, data: dict) -> "GrouperConfig":
        section = data.get("grouper", data)
        return cls(
            tile_size=section.get("tile_size", 300),
            snap_grid=section.get("snap_grid"),
            snap_mode=SnapMode(section.get("snap_mode", "ceil")),
            min_bucket_size=section.get("min_bucket_size", 1),
            max_bucket_size=section.get("max_bucket_size"),
            pad_to_bucket=section.get("pad_to_bucket", True),
        )


@dataclass
class BucketStats:
    key: BucketKey
    count: int = 0
    min_original_h: int = 0
    max_original_h: int = 0
    min_original_w: int = 0
    max_original_w: int = 0
    dtypes: set = field(default_factory=set)
    channels: set = field(default_factory=set)

    def update(self, record: ImageRecord):
        h, w = record["shape"][:2]
        c = record["shape"][2] if len(record["shape"]) == 3 else 1
        self.count += 1
        if self.count == 1:
            self.min_original_h = h
            self.max_original_h = h
            self.min_original_w = w
            self.max_original_w = w
        else:
            self.min_original_h = min(self.min_original_h, h)
            self.max_original_h = max(self.max_original_h, h)
            self.min_original_w = min(self.min_original_w, w)
            self.max_original_w = max(self.max_original_w, w)
        self.dtypes.add(str(record["dtype"]))
        self.channels.add(c)

    def to_dict(self) -> dict:
        return {
            "bucket": str(self.key),
            "count": self.count,
            "original_h_range": [self.min_original_h, self.max_original_h],
            "original_w_range": [self.min_original_w, self.max_original_w],
            "dtypes": sorted(self.dtypes),
            "channels": sorted(self.channels),
        }
