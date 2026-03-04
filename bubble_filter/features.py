import logging

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

logger = logging.getLogger(__name__)

_CLE_AVAILABLE = False
try:
    import pyclesperanto_prototype as cle

    _CLE_AVAILABLE = True
except ImportError:
    pass

SKIMAGE_PROPERTIES = (
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
)

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


def extract_features_cle(
        image: np.ndarray, labels: np.ndarray
) -> pd.DataFrame:
    if not _CLE_AVAILABLE:
        raise ImportError("pyclesperanto_prototype is not installed")
    stats = cle.statistics_of_labelled_pixels(image, labels)
    df_cle = pd.DataFrame(stats)

    props = regionprops_table(
        labels,
        intensity_image=image,
        properties=[
            "label", "perimeter", "eccentricity", "solidity",
            "extent", "axis_major_length", "axis_minor_length",
        ],
    )
    df_sk = pd.DataFrame(props)
    return df_cle.merge(df_sk, on="label", how="inner")


def extract_features_skimage(
        image: np.ndarray, labels: np.ndarray
) -> pd.DataFrame:
    props = regionprops_table(
        labels,
        intensity_image=image,
        properties=SKIMAGE_INTENSITY_PROPERTIES,
    )
    df = pd.DataFrame(props)
    rename_map = {
        "axis_major_length": "major_axis_length",
        "axis_minor_length": "minor_axis_length",
        "intensity_mean": "mean_intensity",
        "intensity_min": "min_intensity",
        "intensity_max": "max_intensity",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def extract_features(
        image: np.ndarray, labels: np.ndarray
) -> pd.DataFrame:
    if _CLE_AVAILABLE:
        try:
            return extract_features_cle(image, labels)
        except Exception as exc:
            logger.warning("pyclesperanto feature extraction failed: %s", exc)

    return extract_features_skimage(image, labels)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-6
    out = df.copy()

    if {"major_axis_length", "minor_axis_length"}.issubset(out.columns):
        out["aspect_ratio"] = out["minor_axis_length"] / (
                out["major_axis_length"] + epsilon
        )

    if {"area", "perimeter"}.issubset(out.columns):
        out["circularity"] = (4 * np.pi * out["area"]) / (
                out["perimeter"] ** 2 + epsilon
        )
        out["compactness"] = out["perimeter"] ** 2 / (
                out["area"] + epsilon
        )

    if {"bbox_width", "bbox_height"}.issubset(out.columns):
        out["bbox_aspect_ratio"] = out["bbox_width"] / (
                out["bbox_height"] + epsilon
        )
        if "area" in out.columns:
            out["extent_derived"] = out["area"] / (
                    out["bbox_width"] * out["bbox_height"] + epsilon
            )

    if {"mean_intensity", "standard_deviation_intensity"}.issubset(out.columns):
        out["intensity_cv"] = out["standard_deviation_intensity"] / (
                out["mean_intensity"] + epsilon
        )

    if {"mean_distance_to_centroid", "mean_distance_to_mass_center"}.issubset(out.columns):
        out["centroid_mass_center_ratio"] = (
                out["mean_distance_to_centroid"]
                / (out["mean_distance_to_mass_center"] + epsilon)
        )

    return out


def aggregate_features(df: pd.DataFrame) -> dict[str, float]:
    agg = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "label":
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        agg[col] = values.mean()
    agg["object_count"] = float(len(df))
    return agg
