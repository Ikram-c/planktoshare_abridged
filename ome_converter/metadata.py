import logging

from pre_process.resolution_grouper.models import BucketKey
from pre_process.tar_streamer import ImageRecord

logger = logging.getLogger(__name__)

_OME_MODELS_AVAILABLE = False
try:
    from ome_zarr_models.v04.axes import Axis
    from ome_zarr_models.v04.coordinate_transformations import (
        VectorScale,
        VectorTranslation,
    )
    from ome_zarr_models.v04.multiscales import Dataset, Multiscale

    _OME_MODELS_AVAILABLE = True
except ImportError:
    pass

_ZYX_AXES_DICTS: tuple[dict, ...] = (
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
)
_C_AXIS_DICT: dict = {"name": "c", "type": "channel"}


def _build_axes_dict(with_channel: bool) -> tuple[dict, ...]:
    return _ZYX_AXES_DICTS + ((_C_AXIS_DICT,) if with_channel else ())


def _build_axes_models(with_channel: bool) -> tuple:
    axes = (
        Axis(name="z", type="space", unit="micrometer"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    )
    return axes + ((Axis(name="c", type="channel"),) if with_channel else ())


def _build_axes(with_channel: bool) -> tuple:
    return (
        _build_axes_models(with_channel)
        if _OME_MODELS_AVAILABLE
        else _build_axes_dict(with_channel)
    )


def _level_scale_translation(
    pixel_size_um: float,
    downsample_factor: int,
    lvl: int,
    ndim: int,
) -> tuple[list[float], list[float]]:
    factor = downsample_factor ** lvl
    ps = pixel_size_um * factor
    offset = (factor - 1) * pixel_size_um / 2
    if ndim == 3:
        return [1.0, ps, ps], [0.0, offset, offset]
    return [1.0, ps, ps, 1.0], [0.0, offset, offset, 0.0]


def _build_dataset(
    lvl: int,
    scale: list[float],
    translation: list[float],
):
    if _OME_MODELS_AVAILABLE:
        transforms = (VectorScale.build(scale),)
        if lvl > 0:
            transforms += (VectorTranslation.build(translation),)
        return Dataset(path=str(lvl), coordinateTransformations=transforms)
    transforms = [{"type": "scale", "scale": scale}]
    if lvl > 0:
        transforms.append({"type": "translation", "translation": translation})
    return {"path": str(lvl), "coordinateTransformations": transforms}


def build_datasets(
    pixel_size_um: float,
    n_levels: int,
    downsample_factor: int,
    ndim: int,
) -> tuple:
    """
    Build coordinate-transformation datasets for each pyramid level.

    :param pixel_size_um: Physical pixel size in micrometres at level 0.
    :param n_levels: Total number of pyramid levels.
    :param downsample_factor: Spatial downsampling factor between levels.
    :param ndim: Array dimensionality (3 for ZYX, 4 for ZYXC).
    :return: Tuple of dataset objects (model or plain dict, depending on
        whether ``ome_zarr_models`` is installed).
    :rtype: tuple
    """
    return tuple(
        _build_dataset(
            lvl,
            *_level_scale_translation(pixel_size_um, downsample_factor, lvl, ndim),
        )
        for lvl in range(n_levels)
    )


def build_multiscales_attrs(
    name: str,
    pixel_size_um: float,
    n_levels: int,
    downsample_factor: int,
    ndim: int,
) -> dict:
    """
    Build the OME-Zarr 0.4 ``multiscales`` metadata attribute dictionary.

    :param name: Human-readable name for the multiscales group.
    :param pixel_size_um: Physical pixel size in micrometres at level 0.
    :param n_levels: Total number of pyramid levels.
    :param downsample_factor: Spatial downsampling factor between levels.
    :param ndim: Array dimensionality (3 for ZYX, 4 for ZYXC).
    :return: Dictionary ready to merge into a zarr group's ``.attrs``.
    :rtype: dict
    """
    axes = _build_axes(with_channel=ndim == 4)
    datasets = build_datasets(pixel_size_um, n_levels, downsample_factor, ndim)

    if _OME_MODELS_AVAILABLE:
        ms = Multiscale(
            axes=axes,
            datasets=datasets,
            version="0.4",
            name=name,
        )
        return {"multiscales": [ms.model_dump(exclude_none=True)]}

    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": name,
                "axes": [
                    a if isinstance(a, dict) else a.model_dump(exclude_none=True)
                    for a in axes
                ],
                "datasets": list(datasets),
            }
        ]
    }


def build_pipeline_metadata(
    key: BucketKey,
    records: list[ImageRecord],
    tile_size: int,
    compression: str,
) -> dict:
    """
    Build pipeline-specific metadata to store alongside OME-Zarr attributes.

    :param key: Resolution group identifier.
    :param records: Image records belonging to this bucket.
    :param tile_size: Tile size used during writing.
    :param compression: Human-readable compression label string.
    :return: Metadata dict with ``pipeline_metadata`` and
        ``image_metadata`` keys.
    :rtype: dict
    """
    return {
        "pipeline_metadata": {
            "resolution_group": str(key),
            "num_images": len(records),
            "tile_size": tile_size,
            "compression": compression,
            "source_dtype": str(records[0]["dtype"]) if records else "unknown",
        },
        "image_metadata": [
            {
                "index": i,
                "original_filename": rec.get("filename", "unknown"),
                "bubble_detection_score": rec.get("bubble_score"),
                "object_count": rec.get("object_count"),
                "original_shape": list(rec["shape"]),
                "tar_source": rec.get("tar_path", "unknown"),
            }
            for i, rec in enumerate(records)
        ],
    }