from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OmeConverterConfig:
    output_dir: str = "output_ome"
    pixel_size_um: float = 0.36
    generate_pyramid: bool = True
    pyramid_levels: int = 3
    pyramid_downsample_factor: int = 2
    zip_store: bool = True
    tile_size: int = 300

    @classmethod
    def from_dict(cls, data: dict) -> "OmeConverterConfig":
        section = data.get("ome_converter", data)
        return cls(
            output_dir=section.get("output_dir", "output_ome"),
            pixel_size_um=section.get("pixel_size_um", 0.36),
            generate_pyramid=section.get("generate_pyramid", True),
            pyramid_levels=section.get("pyramid_levels", 3),
            pyramid_downsample_factor=section.get("pyramid_downsample_factor", 2),
            zip_store=section.get("zip_store", True),
            tile_size=section.get("tile_size", 300),
        )


@dataclass
class OmeManifest:
    bucket: str
    ome_store_path: str
    zip_path: Optional[str]
    n_images: int
    pyramid_levels: int
    pixel_size_um: float
