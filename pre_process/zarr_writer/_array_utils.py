import logging

import numpy as np

from pre_process.tar_streamer import ImageRecord

logger = logging.getLogger(__name__)


def resolve_dtype(records: list[ImageRecord]) -> np.dtype:
    dtypes = {rec["dtype"] for rec in records}
    if len(dtypes) == 1:
        return dtypes.pop()
    ordered = sorted(dtypes, key=lambda d: np.dtype(d).itemsize, reverse=True)
    logger.warning(
        "Mixed dtypes %s in bucket, upcasting to %s",
        [str(d) for d in dtypes],
        ordered[0],
    )
    return ordered[0]


def resolve_channels(records: list[ImageRecord]) -> int:
    channels = set()
    for rec in records:
        channels.add(rec["shape"][2] if len(rec["shape"]) == 3 else 1)
    if len(channels) > 1:
        logger.warning("Mixed channel counts %s, using max", channels)
    return max(channels)


def pad_image(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    target_c: int,
    dtype: np.dtype,
) -> np.ndarray:
    image = image.astype(dtype)

    if image.ndim == 2 and target_c > 1:
        image = np.stack([image] * target_c, axis=-1)
    elif image.ndim == 3 and target_c == 1:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] < target_c:
        pad = np.zeros(
            (image.shape[0], image.shape[1], target_c - image.shape[2]),
            dtype=dtype,
        )
        image = np.concatenate([image, pad], axis=-1)

    ih, iw = image.shape[:2]
    if ih < target_h or iw < target_w:
        if image.ndim == 2:
            out = np.zeros((target_h, target_w), dtype=dtype)
        else:
            out = np.zeros((target_h, target_w, image.shape[2]), dtype=dtype)
        out[:ih, :iw] = image
        image = out

    return image