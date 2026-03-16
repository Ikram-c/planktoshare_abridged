# pre_process/bubble_filter/segmentation.py
import logging
from enum import Enum, auto

import numpy as np
import pyclesperanto_prototype as cle
from csbdeep.utils import normalize
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label as skimage_label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


class SegmentationBackend(Enum):
    CLE = auto()
    SKIMAGE = auto()


def _detect_backend() -> SegmentationBackend:
    try:
        device = cle.get_device()
        logger.info("pyclesperanto using device: %s", device.name)
        return SegmentationBackend.CLE
    except Exception as exc:
        logger.warning(
            "pyclesperanto device unavailable, falling back to scikit-image: %s", exc
        )
        return SegmentationBackend.SKIMAGE


_BACKEND: SegmentationBackend = _detect_backend()


def segment_cle(
    image: np.ndarray,
    spot_sigma: float = 3.5,
    outline_sigma: float = 1.0,
) -> np.ndarray:
    return np.asarray(
        cle.voronoi_otsu_labeling(image, spot_sigma=spot_sigma, outline_sigma=outline_sigma)
    ).astype(np.int32)


def segment_skimage(
    image: np.ndarray,
    spot_sigma: float = 3.5,
    min_object_size: int = 64,
) -> np.ndarray:
    smoothed = gaussian(image, sigma=spot_sigma, preserve_range=True)
    binary = remove_small_objects(
        smoothed > threshold_otsu(smoothed), max_size=min_object_size - 1
    )
    distance = ndi.distance_transform_edt(binary)
    markers = skimage_label(ndi.maximum_filter(distance, size=20) == distance)
    return watershed(-distance, markers, mask=binary).astype(np.int32)


def segment(
    image: np.ndarray,
    spot_sigma: float = 3.5,
    outline_sigma: float = 1.0,
    fallback_to_skimage: bool = True,
) -> np.ndarray:
    if _BACKEND is SegmentationBackend.CLE:
        try:
            return segment_cle(image, spot_sigma, outline_sigma)
        except Exception as exc:
            logger.warning("pyclesperanto segmentation failed: %s", exc)
            if not fallback_to_skimage:
                raise

    logger.info("Using scikit-image fallback segmentation")
    return segment_skimage(image, spot_sigma)


def to_grayscale(
    image: np.ndarray,
    pmin: float = 2.0,
    pmax: float = 99.8,
) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] > 1:
        image = np.mean(image, axis=-1).astype(image.dtype)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return normalize(image, pmin=pmin, pmax=pmax).astype(np.float32)