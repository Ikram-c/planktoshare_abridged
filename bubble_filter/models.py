from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class FilterConfig:
    checkpoint_path: str
    threshold: float = 0.5
    spot_sigma: float = 3.5
    outline_sigma: float = 1.0
    device: str = "cpu"
    fallback_to_skimage: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "FilterConfig":
        section = data.get("filter", data)
        return cls(
            checkpoint_path=section["checkpoint_path"],
            threshold=section.get("threshold", 0.5),
            spot_sigma=section.get("spot_sigma", 3.5),
            outline_sigma=section.get("outline_sigma", 1.0),
            device=section.get("device", "cpu"),
            fallback_to_skimage=section.get("fallback_to_skimage", True),
        )


@dataclass
class FilterResult:
    filename: str
    is_bubble: bool
    bubble_score: float
    object_count: int
    feature_vector: Optional[dict] = None
