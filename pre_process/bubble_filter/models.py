from dataclasses import dataclass
from typing import Optional


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


@dataclass(frozen=True)
class RuleBasedFilterConfig:
    spot_sigma: float = 3.5
    outline_sigma: float = 1.0
    fallback_to_skimage: bool = True
    local_background: bool = False
    annulus_width: int = 10
    mod_threshold: float = 0.02
    intensity_ratio_min: float = 0.90
    solidity_min: float = 0.95
    eccentricity_max: float = 0.30
    gradient_rms_max: float | None = None
    intensity_std_max: float | None = None
    score_weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
    score_threshold: float = 0.70

    @classmethod
    def from_dict(cls, data: dict) -> "RuleBasedFilterConfig":
        s = data.get("rule_filter", data)
        sw = s.get("score_weights", (0.4, 0.2, 0.2, 0.2))
        return cls(
            spot_sigma=s.get("spot_sigma", 3.5),
            outline_sigma=s.get("outline_sigma", 1.0),
            fallback_to_skimage=s.get("fallback_to_skimage", True),
            local_background=s.get("local_background", False),
            annulus_width=s.get("annulus_width", 10),
            mod_threshold=s.get("mod_threshold", 0.02),
            intensity_ratio_min=s.get("intensity_ratio_min", 0.90),
            solidity_min=s.get("solidity_min", 0.95),
            eccentricity_max=s.get("eccentricity_max", 0.30),
            gradient_rms_max=s.get("gradient_rms_max"),
            intensity_std_max=s.get("intensity_std_max"),
            score_weights=tuple(sw),
            score_threshold=s.get("score_threshold", 0.70),
        )


@dataclass
class FilterResult:
    filename: str
    is_bubble: bool
    bubble_score: float
    object_count: int
    feature_vector: Optional[dict] = None