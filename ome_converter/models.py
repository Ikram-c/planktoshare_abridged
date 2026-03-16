from dataclasses import dataclass

from pre_process._pre_process_utils.interfaces import BaseManifest, PipelineConfig


@dataclass(frozen=True)
class OmeConverterConfig(PipelineConfig):
    """
    Immutable configuration for
    :class:`~ome_converter.converter.OmeZarrConverter`.

    All fields map directly to keys under the ``ome_converter`` YAML section.
    """

    output_dir: str = "output_ome"
    pixel_size_um: float = 0.36
    generate_pyramid: bool = True
    pyramid_levels: int = 3
    pyramid_downsample_factor: int = 2
    zip_store: bool = True
    tile_size: int = 300

    def __str__(self) -> str:
        return (
            f"OmeConverterConfig("
            f"output={self.output_dir!r}, "
            f"pixel_size={self.pixel_size_um}um, "
            f"pyramid_levels={self.pyramid_levels}, "
            f"zip={self.zip_store})"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "OmeConverterConfig":
        """
        Construct from a validated pipeline config dictionary.

        :param data: Top-level config dict, optionally containing an
            ``ome_converter`` sub-section.
        :return: Populated :class:`OmeConverterConfig` instance.
        :rtype: OmeConverterConfig
        """
        section = data.get("ome_converter", data)
        return cls(
            output_dir=section.get("output_dir", "output_ome"),
            pixel_size_um=section.get("pixel_size_um", 0.36),
            generate_pyramid=section.get("generate_pyramid", True),
            pyramid_levels=section.get("pyramid_levels", 3),
            pyramid_downsample_factor=section.get(
                "pyramid_downsample_factor", 2
            ),
            zip_store=section.get("zip_store", True),
            tile_size=section.get("tile_size", 300),
        )


@dataclass
class OmeManifest(BaseManifest):
    """Record of a single OME-Zarr store written by :class:`OmeZarrConverter`."""

    bucket: str
    ome_store_path: str
    zip_path: str | None
    n_images: int
    pyramid_levels: int
    pixel_size_um: float

    def __str__(self) -> str:
        return (
            f"OmeManifest("
            f"bucket={self.bucket!r}, "
            f"n_images={self.n_images}, "
            f"pyramid_levels={self.pyramid_levels}, "
            f"pixel_size={self.pixel_size_um}um)"
        )

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "ome_store_path": self.ome_store_path,
            "zip_path": self.zip_path,
            "n_images": self.n_images,
            "pyramid_levels": self.pyramid_levels,
            "pixel_size_um": self.pixel_size_um,
        }