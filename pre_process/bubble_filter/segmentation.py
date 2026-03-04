import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CLE_AVAILABLE = False
try:
    import pyclesperanto_prototype as cle
    _CLE_AVAILABLE = True
except ImportError:
    pass

_SKIMAGE_AVAILABLE = False
try:
    from skimage.filters import threshold_otsu
    from skimage.measure import label as skimage_label
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi
    _SKIMAGE_AVAILABLE = True
except ImportError:
    pass


def segment_cle(
    image: np.ndarray,
    spot_sigma: float = 3.5,
    outline_sigma: float = 1.0,
) -> np.ndarray:
    if not _CLE_AVAILABLE:
        raise ImportError("pyclesperanto_prototype is not installed")
    labels = cle.voronoi_otsu_labeling(
        image, spot_sigma=spot_sigma, outline_sigma=outline_sigma
    )
    return np.asarray(labels).astype(np.int32)


def segment_skimage(
    image: np.ndarray,
    spot_sigma: float = 3.5,
    min_object_size: int = 64,
) -> np.ndarray:
    if not _SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is not installed")
    from skimage.filters import gaussian

    smoothed = gaussian(image, sigma=spot_sigma, preserve_range=True)
    thresh = threshold_otsu(smoothed)
    binary = smoothed > thresh
    binary = remove_small_objects(binary, max_size=min_object_size - 1)
    distance = ndi.distance_transform_edt(binary)
    coords = ndi.maximum_filter(distance, size=20) == distance
    markers = skimage_label(coords)
    labels = watershed(-distance, markers, mask=binary)
    return labels.astype(np.int32)


def segment(
    image: np.ndarray,
    spot_sigma: float = 3.5,
    outline_sigma: float = 1.0,
    fallback_to_skimage: bool = True,
) -> np.ndarray:
    if _CLE_AVAILABLE:
        try:
            return segment_cle(image, spot_sigma, outline_sigma)
        except Exception as exc:
            logger.warning("pyclesperanto segmentation failed: %s", exc)
            if not fallback_to_skimage:
                raise

    if _SKIMAGE_AVAILABLE and fallback_to_skimage:
        logger.info("Using scikit-image fallback segmentation")
        return segment_skimage(image, spot_sigma)

    raise ImportError(
        "Neither pyclesperanto_prototype nor scikit-image is installed"
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return np.mean(image, axis=-1).astype(image.dtype)