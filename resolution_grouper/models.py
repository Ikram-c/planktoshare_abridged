import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pre_process._pre_process_utils.interfaces import PipelineConfig
from pre_process.tar_streamer import ImageRecord

_BUCKET_KEY_RE = re.compile(r"^(\d+)x(\d+)$")


class SnapMode(Enum):
    """Rounding strategy applied when snapping image dimensions to a grid."""

    CEIL = "ceil"
    FLOOR = "floor"
    ROUND = "round"
    EXACT = "exact"


@dataclass(frozen=True)
class BucketKey:
    """
    Immutable resolution-group identifier.

    Encodes the padded height and width of a resolution bucket so it can
    be used as a dict key and round-tripped through JSON via
    :meth:`__str__` and :meth:`from_str`.
    """

    height: int
    width: int

    def __str__(self) -> str:
        return f"{self.height}x{self.width}"

    def __repr__(self) -> str:
        return f"BucketKey(height={self.height}, width={self.width})"

    @classmethod
    def from_str(cls, value: str) -> "BucketKey":
        """
        Reconstruct a :class:`BucketKey` from its ``__str__`` representation.

        Accepts the canonical ``"HxW"`` format produced by :meth:`__str__`,
        e.g. ``"300x600"``. Used to restore typed keys after JSON
        deserialisation.

        :param value: String in ``"HxW"`` format.
        :raises ValueError: If *value* does not match the expected pattern.
        :return: Reconstructed :class:`BucketKey`.
        :rtype: BucketKey
        """
        match = _BUCKET_KEY_RE.fullmatch(value.strip())
        if not match:
            raise ValueError(
                f"Cannot parse {value!r} as a BucketKey. "
                "Expected format: '<height>x<width>', e.g. '300x600'."
            )
        return cls(height=int(match.group(1)), width=int(match.group(2)))

    @property
    def pixel_count(self) -> int:
        """Total pixel area of this bucket."""
        return self.height * self.width

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio; guards against zero-height keys."""
        return self.width / max(self.height, 1)


@dataclass(frozen=True)
class GrouperConfig(PipelineConfig):
    """
    Immutable configuration for the resolution grouper stage.

    All fields map directly to keys under the ``grouper`` YAML section.
    When ``snap_grid`` is absent, :attr:`effective_snap_grid` falls back
    to ``tile_size``.
    """

    tile_size: int = 300
    snap_grid: Optional[int] = None
    snap_mode: SnapMode = SnapMode.CEIL
    min_bucket_size: int = 1
    max_bucket_size: Optional[int] = None
    pad_to_bucket: bool = True

    def __str__(self) -> str:
        return (
            f"GrouperConfig("
            f"tile_size={self.tile_size}, "
            f"snap_grid={self.snap_grid}, "
            f"snap_mode={self.snap_mode.value!r}, "
            f"min_bucket={self.min_bucket_size}, "
            f"pad={self.pad_to_bucket})"
        )

    @property
    def effective_snap_grid(self) -> int:
        """Snap grid in pixels; falls back to ``tile_size`` when unset."""
        return self.snap_grid if self.snap_grid is not None else self.tile_size

    @classmethod
    def from_dict(cls, data: dict) -> "GrouperConfig":
        """
        Construct from a validated pipeline config dictionary.

        :param data: Top-level config dict, optionally containing a
            ``grouper`` sub-section.
        :return: Populated :class:`GrouperConfig` instance.
        :rtype: GrouperConfig
        """
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
    """
    Accumulates per-bucket statistics as images are assigned to a bucket.

    Call :meth:`update` once per :class:`~pre_process.tar_streamer.ImageRecord`
    and retrieve a JSON-serialisable summary via :meth:`to_dict`.
    """

    key: BucketKey
    count: int = 0
    min_original_h: int = 0
    max_original_h: int = 0
    min_original_w: int = 0
    max_original_w: int = 0
    dtypes: set = field(default_factory=set)
    channels: set = field(default_factory=set)

    def __str__(self) -> str:
        return (
            f"BucketStats("
            f"key={self.key}, "
            f"count={self.count}, "
            f"h=[{self.min_original_h},{self.max_original_h}], "
            f"w=[{self.min_original_w},{self.max_original_w}])"
        )

    def update(self, record: ImageRecord) -> None:
        """
        Incorporate a single image record into the running statistics.

        Updates dimension ranges, dtype set, and channel set in-place.

        :param record: Image record to incorporate.
        """
        h, w = record["shape"][:2]
        c = record["shape"][2] if len(record["shape"]) == 3 else 1
        self.count += 1
        if self.count == 1:
            self.min_original_h = self.max_original_h = h
            self.min_original_w = self.max_original_w = w
        else:
            self.min_original_h = min(self.min_original_h, h)
            self.max_original_h = max(self.max_original_h, h)
            self.min_original_w = min(self.min_original_w, w)
            self.max_original_w = max(self.max_original_w, w)
        self.dtypes.add(str(record["dtype"]))
        self.channels.add(c)

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-compatible dictionary.

        :return: Flat mapping of all accumulated statistics.
        :rtype: dict
        """
        return {
            "bucket": str(self.key),
            "count": self.count,
            "original_h_range": [self.min_original_h, self.max_original_h],
            "original_w_range": [self.min_original_w, self.max_original_w],
            "dtypes": sorted(self.dtypes),
            "channels": sorted(self.channels),
        }