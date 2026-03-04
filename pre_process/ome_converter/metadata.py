import logging

from pre_process.tar_streamer import ImageRecord
from pre_process.resolution_grouper.models import BucketKey

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


def build_axes_zyx() -> tuple:
    if not _OME_MODELS_AVAILABLE:
        return _build_axes_zyx_dict()
    return (
        Axis(name="z", type="space", unit="micrometer"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    )


def build_axes_zyxc() -> tuple:
    if not _OME_MODELS_AVAILABLE:
        return _build_axes_zyxc_dict()
    return (
        Axis(name="z", type="space", unit="micrometer"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
        Axis(name="c", type="channel"),
    )


def build_datasets(
        pixel_size_um: float,
        n_levels: int,
        downsample_factor: int,
        ndim: int,
) -> tuple:
    datasets = []
    for lvl in range(n_levels):
        factor = downsample_factor ** lvl
        ps = pixel_size_um * factor
        offset = (factor - 1) * pixel_size_um / 2

        if ndim == 3:
            scale_values = [1.0, ps, ps]
            translation_values = [0.0, offset, offset]
        else:
            scale_values = [1.0, ps, ps, 1.0]
            translation_values = [0.0, offset, offset, 0.0]

        if _OME_MODELS_AVAILABLE:
            transforms = (VectorScale.build(scale_values),)
            if lvl > 0:
                transforms = (
                    VectorScale.build(scale_values),
                    VectorTranslation.build(translation_values),
                )
            ds = Dataset(path=str(lvl), coordinateTransformations=transforms)
        else:
            transforms = [{"type": "scale", "scale": scale_values}]
            if lvl > 0:
                transforms.append(
                    {"type": "translation", "translation": translation_values}
                )
            ds = {"path": str(lvl), "coordinateTransformations": transforms}

        datasets.append(ds)

    return tuple(datasets)


def build_multiscales_attrs(
        name: str,
        pixel_size_um: float,
        n_levels: int,
        downsample_factor: int,
        ndim: int,
) -> dict:
    if ndim == 4:
        axes = build_axes_zyxc()
    else:
        axes = build_axes_zyx()

    datasets = build_datasets(pixel_size_um, n_levels, downsample_factor, ndim)

    if _OME_MODELS_AVAILABLE:
        ms = Multiscale(
            axes=axes,
            datasets=datasets,
            version="0.4",
            name=name,
        )
        return {"multiscales": [ms.model_dump(exclude_none=True)]}

    axes_dicts = [_axis_to_dict(a) for a in axes]
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": name,
                "axes": axes_dicts,
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
    sample_dtype = str(records[0]["dtype"]) if records else "unknown"
    return {
        "pipeline_metadata": {
            "resolution_group": str(key),
            "num_images": len(records),
            "tile_size": tile_size,
            "compression": compression,
            "source_dtype": sample_dtype,
        },
        "image_metadata": [
            {
                "index": i,
                "original_filename": rec.get("filename", "unknown"),
                "bubble_detection_score": rec.get("bubble_score", None),
                "object_count": rec.get("object_count", None),
                "original_shape": list(rec["shape"]),
                "tar_source": rec.get("tar_path", "unknown"),
            }
            for i, rec in enumerate(records)
        ],
    }


def _build_axes_zyx_dict() -> tuple[dict, ...]:
    return (
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    )


def _build_axes_zyxc_dict() -> tuple[dict, ...]:
    return (
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
        {"name": "c", "type": "channel"},
    )


def _axis_to_dict(axis) -> dict:
    if isinstance(axis, dict):
        return axis
    return axis.model_dump(exclude_none=True)
