# pre_process/bubble_filter/models.py
from dataclasses import dataclass
from typing import Optional


_FILTER_DEFAULTS: dict = {
    "threshold": 0.5,
    "spot_sigma": 3.5,
    "outline_sigma": 1.0,
    "device": "auto",
    "fallback_to_skimage": True,
}


@dataclass(frozen=True)
class FilterConfig:
    checkpoint_path: Optional[str] = None
    threshold: float = 0.5
    spot_sigma: float = 3.5
    outline_sigma: float = 1.0
    device: str = "auto"
    fallback_to_skimage: bool = True

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None and not self.checkpoint_path:
            raise ValueError("checkpoint_path must not be empty if provided")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {self.threshold}"
            )
        if self.spot_sigma <= 0:
            raise ValueError(
                f"spot_sigma must be positive, got {self.spot_sigma}"
            )

    def __str__(self) -> str:
        return (
            f"FilterConfig(threshold={self.threshold}, "
            f"device={self.device!r}, "
            f"checkpoint={self.checkpoint_path!r})"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "FilterConfig":
        sec = data.get("filter", data)
        return cls(
            checkpoint_path=sec.get("checkpoint_path"),
            **{k: sec.get(k, v) for k, v in _FILTER_DEFAULTS.items()},
        )


@dataclass
class FilterResult:
    filename: str
    is_bubble: bool
    bubble_score: float
    object_count: int
    feature_vector: Optional[dict] = None

    def __str__(self) -> str:
        return (
            f"FilterResult({self.filename!r}: {self.label}, "
            f"score={self.bubble_score:.3f}, objects={self.object_count})"
        )

    def __bool__(self) -> bool:
        return not self.is_bubble

    @property
    def label(self) -> str:
        return "bubble" if self.is_bubble else "plankton"