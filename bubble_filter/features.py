# pre_process/bubble_filter/features.py
import logging
from typing import Callable

import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
from skimage.measure import regionprops_table

logger = logging.getLogger(__name__)

_EPS = 1e-6

SKIMAGE_INTENSITY_PROPERTIES = (
    "label",
    "area",
    "perimeter",
    "perimeter_crofton",
    "eccentricity",
    "solidity",
    "extent",
    "axis_major_length",
    "axis_minor_length",
    "equivalent_diameter_area",
    "intensity_mean",
    "intensity_min",
    "intensity_max",
)

_COLUMN_RENAMES: dict[str, str] = {
    "axis_major_length": "major_axis_length",
    "axis_minor_length": "minor_axis_length",
    "intensity_mean": "mean_intensity",
    "intensity_min": "min_intensity",
    "intensity_max": "max_intensity",
}

_DERIVED_RULES: list[tuple[frozenset, str, Callable[[pd.DataFrame], pd.Series]]] = [
    (
        frozenset({"major_axis_length", "minor_axis_length"}),
        "aspect_ratio",
        lambda df: df["minor_axis_length"] / (df["major_axis_length"] + _EPS),
    ),
    (
        frozenset({"area", "perimeter"}),
        "circularity",
        lambda df: (4 * np.pi * df["area"]) / (df["perimeter"] ** 2 + _EPS),
    ),
    (
        frozenset({"area", "perimeter"}),
        "compactness",
        lambda df: df["perimeter"] ** 2 / (df["area"] + _EPS),
    ),
    (
        frozenset({"bbox_width", "bbox_height"}),
        "bbox_aspect_ratio",
        lambda df: df["bbox_width"] / (df["bbox_height"] + _EPS),
    ),
    (
        frozenset({"bbox_width", "bbox_height", "area"}),
        "extent_derived",
        lambda df: df["area"] / (df["bbox_width"] * df["bbox_height"] + _EPS),
    ),
    (
        frozenset({"mean_intensity", "standard_deviation_intensity"}),
        "intensity_cv",
        lambda df: df["standard_deviation_intensity"] / (df["mean_intensity"] + _EPS),
    ),
    (
        frozenset({"mean_distance_to_centroid", "mean_distance_to_mass_center"}),
        "centroid_mass_center_ratio",
        lambda df: df["mean_distance_to_centroid"] / (df["mean_distance_to_mass_center"] + _EPS),
    ),
]


def extract_features_cle(image: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    if not _CLE_AVAILABLE:
        raise ImportError("pyclesperanto_prototype is not installed")
    df_cle = pd.DataFrame(cle.statistics_of_labelled_pixels(image, labels))
    df_sk = pd.DataFrame(
        regionprops_table(
            labels, intensity_image=image,
            properties=["label", "perimeter", "eccentricity", "solidity",
                        "extent", "axis_major_length", "axis_minor_length"],
        )
    )
    return df_cle.merge(df_sk, on="label", how="inner")


def extract_features_skimage(image: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame(
            regionprops_table(
                labels, intensity_image=image,
                properties=SKIMAGE_INTENSITY_PROPERTIES,
            )
        )
        .rename(columns=_COLUMN_RENAMES)
    )


def extract_features(image: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    try:
        return extract_features_cle(image, labels)
    except Exception as exc:
        logger.warning("pyclesperanto feature extraction failed: %s", exc)
    return extract_features_skimage(image, labels)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = frozenset(out.columns)
    for required, new_col, fn in _DERIVED_RULES:
        if required.issubset(cols):
            out[new_col] = fn(out)
    return out


def aggregate_features(df: pd.DataFrame) -> dict[str, float]:
    numeric = df.select_dtypes(include=[np.number])
    return {
        **{
            col: float(vals.mean())
            for col in numeric.columns
            if col != "label" and not (vals := numeric[col].dropna()).empty
        },
        "object_count": float(len(df)),
    }