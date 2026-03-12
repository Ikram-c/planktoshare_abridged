import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import dilation, disk


def compute_background(image: np.ndarray, labels: np.ndarray) -> float:
    bg_pixels = image[labels == 0].astype(float)
    if bg_pixels.size == 0:
        return float(np.median(image))
    return float(np.median(bg_pixels))


def _make_mean_od(bg_mean: float):
    def mean_optical_density(regionmask, intensity_image):
        pixels = intensity_image[regionmask].astype(float)
        od = np.clip(-np.log10(pixels / bg_mean + 1e-10), 0.0, None)
        return float(np.mean(od))
    mean_optical_density.__name__ = "mean_optical_density"
    return mean_optical_density


def _make_integrated_od(bg_mean: float):
    def integrated_optical_density(regionmask, intensity_image):
        pixels = intensity_image[regionmask].astype(float)
        od = np.clip(-np.log10(pixels / bg_mean + 1e-10), 0.0, None)
        return float(np.sum(od))
    integrated_optical_density.__name__ = "integrated_optical_density"
    return integrated_optical_density


def gradient_rms(regionmask, intensity_image):
    gy, gx = np.gradient(intensity_image.astype(float))
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    vals = grad_mag[regionmask]
    return float(np.sqrt(np.mean(vals ** 2))) if vals.size > 0 else 0.0


def intensity_std(regionmask, intensity_image):
    vals = intensity_image[regionmask].astype(float)
    return float(np.std(vals)) if vals.size > 0 else 0.0


def extract_od_features(
    image: np.ndarray,
    labels: np.ndarray,
    bg_mean: float | None = None,
) -> pd.DataFrame:
    if bg_mean is None:
        bg_mean = compute_background(image, labels)

    props = regionprops_table(
        labels,
        intensity_image=image,
        properties=[
            "label",
            "area",
            "eccentricity",
            "solidity",
            "extent",
            "perimeter",
            "intensity_mean",
            "intensity_max",
            "intensity_min",
            "euler_number",
        ],
        extra_properties=[
            _make_mean_od(bg_mean),
            _make_integrated_od(bg_mean),
            gradient_rms,
            intensity_std,
        ],
    )
    df = pd.DataFrame(props)
    if df.empty:
        return df

    df["intensity_ratio"] = df["intensity_mean"] / bg_mean
    df["intensity_contrast"] = (bg_mean - df["intensity_mean"]) / bg_mean
    df["circularity"] = (
        4.0 * np.pi * df["area"] / (df["perimeter"] ** 2 + 1e-10)
    )
    df["bg_mean"] = bg_mean
    return df


def compute_local_background_features(
    image: np.ndarray,
    labels: np.ndarray,
    global_bg_mean: float,
    annulus_width: int = 10,
) -> pd.DataFrame:
    results = []
    for region in regionprops(labels, intensity_image=image):
        obj_mask = labels == region.label
        dilated = dilation(obj_mask, disk(annulus_width))
        annulus = dilated & (~obj_mask) & (labels == 0)
        local_bg = (
            float(np.median(image[annulus]))
            if annulus.sum() > 50
            else global_bg_mean
        )
        pixels = image[obj_mask].astype(float)
        od = np.clip(-np.log10(pixels / local_bg + 1e-10), 0.0, None)
        results.append({
            "label": region.label,
            "local_bg": local_bg,
            "local_mean_od": float(np.mean(od)),
            "local_intensity_ratio": float(region.intensity_mean / local_bg),
        })
    if not results:
        return pd.DataFrame(
            columns=["label", "local_bg", "local_mean_od", "local_intensity_ratio"]
        )
    return pd.DataFrame(results)