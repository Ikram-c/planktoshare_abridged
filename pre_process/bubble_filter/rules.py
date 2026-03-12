import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class BubbleRules:
    mod_threshold: float = 0.02
    intensity_ratio_min: float = 0.90
    solidity_min: float = 0.95
    eccentricity_max: float = 0.30
    gradient_rms_max: float | None = None
    intensity_std_max: float | None = None
    score_weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
    score_threshold: float = 0.70


def _score_row(row: pd.Series, rules: BubbleRules) -> float:
    w = rules.score_weights

    mod_s = float(np.clip(
        1.0 - row["mean_optical_density"] / max(rules.mod_threshold, 1e-9),
        0.0, 1.0,
    ))
    sol_s = float(np.clip(
        (row["solidity"] - rules.solidity_min) / max(1.0 - rules.solidity_min, 1e-9),
        0.0, 1.0,
    ))
    ecc_s = float(np.clip(
        (rules.eccentricity_max - row["eccentricity"]) / max(rules.eccentricity_max, 1e-9),
        0.0, 1.0,
    ))
    ir_s = float(np.clip(
        (row["intensity_ratio"] - rules.intensity_ratio_min)
        / max(1.0 - rules.intensity_ratio_min, 1e-9),
        0.0, 1.0,
    ))
    return w[0] * mod_s + w[1] * sol_s + w[2] * ecc_s + w[3] * ir_s


def classify_dataframe(df: pd.DataFrame, rules: BubbleRules) -> pd.DataFrame:
    df = df.copy()
    df["bubble_score"] = df.apply(lambda row: _score_row(row, rules), axis=1)

    hard = (
        (df["mean_optical_density"] < rules.mod_threshold)
        & (df["intensity_ratio"] > rules.intensity_ratio_min)
        & (df["solidity"] > rules.solidity_min)
        & (df["eccentricity"] < rules.eccentricity_max)
    )
    if rules.gradient_rms_max is not None and "gradient_rms" in df.columns:
        hard &= df["gradient_rms"] < rules.gradient_rms_max
    if rules.intensity_std_max is not None and "intensity_std" in df.columns:
        hard &= df["intensity_std"] < rules.intensity_std_max

    df["is_bubble_hard"] = hard
    df["is_bubble_scored"] = df["bubble_score"] > rules.score_threshold
    df["is_bubble"] = df["is_bubble_hard"] | df["is_bubble_scored"]
    return df