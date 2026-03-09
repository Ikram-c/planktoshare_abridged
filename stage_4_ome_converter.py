import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import yaml

from pre_process.ome_converter import OmeConverterConfig, OmeZarrConverter
from pre_process.zarr_writer import ZarrWriterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("stage5_ome_converter")


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    :param config_path: The file path to the YAML configuration file.
    :type config_path: str
    :return: A dictionary containing the parsed configuration parameters.
    :rtype: dict
    """
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def safe_write_json(data: dict, file_path: str) -> None:
    """
    Safely write a dictionary to a JSON file while checking for available disk space.

    Serializes the provided data to a JSON string and verifies that the target
    directory has enough free space (including a 100MB buffer) before writing
    the file to disk.

    :param data: The dictionary data to serialize and write.
    :type data: dict
    :param file_path: The destination path for the output JSON file.
    :type file_path: str
    :raises OSError: If there is insufficient disk space to write the file.
    :return: None
    """
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_str = json.dumps(data, indent=2)
    required_bytes = len(json_str.encode("utf-8"))
    free_bytes = shutil.disk_usage(target_path.parent).free
    
    buffer_bytes = 100 * 1024 * 1024
    if required_bytes + buffer_bytes > free_bytes:
        raise OSError(
            f"Insufficient disk space for {file_path}. "
            f"Requires {required_bytes}B, but only {free_bytes}B free."
        )
        
    with open(target_path, "w") as fh:
        fh.write(json_str)


def main():
    """
    Execute Stage 5 of the microscopy image pipeline.

    Loads the filtered buckets and executes the conversion to standard
    OME-Zarr formats. Calculates final pipeline execution timing, constructs
    the comprehensive manifest, and safely writes it to the output directory.
    """
    parser = argparse.ArgumentParser(
        description="Stage 5: OME-Zarr Conversion and Manifest generation"
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--in-data",
        default="intermediate_filtered.json",
        help="Path to load filtered buckets"
    )
    parser.add_argument(
        "--stats-file",
        default="pipeline_stats.json",
        help="Path to pipeline stats file"
    )
    args = parser.parse_args()

    data = load_config(args.config)
    ome_cfg = OmeConverterConfig.from_dict(data)
    writer_cfg = ZarrWriterConfig.from_dict(data)

    with open(args.in_data, "r") as f:
        filtered_buckets = json.load(f)

    with open(args.stats_file, "r") as f:
        pipeline_stats = json.load(f)

    logger.info("Stage 5: Converting to OME-Zarr 0.4")
    t5 = time.perf_counter()

    converter = OmeZarrConverter(ome_cfg, writer_cfg)
    ome_manifests = converter.convert_all(filtered_buckets)

    dt_ome = time.perf_counter() - t5
    logger.info(
        "Stage 5 complete in %.1fs: %d OME-Zarr files",
        dt_ome, len(ome_manifests)
    )

    for m in ome_manifests:
        logger.info(
            "  %s: %d images, %d levels → %s%s",
            m.bucket, m.n_images, m.pyramid_levels, m.ome_store_path,
            f" (zip: {m.zip_path})" if m.zip_path else "",
        )

    t0 = pipeline_stats["timing"].pop("t0", time.perf_counter())
    dt_total = time.perf_counter() - t0
    logger.info("Pipeline complete in %.1fs", dt_total)

    pipeline_stats["timing"]["ome_convert_s"] = round(dt_ome, 2)
    pipeline_stats["timing"]["total_s"] = round(dt_total, 2)
    pipeline_stats["ome_manifests"] = [
        {
            "bucket": m.bucket,
            "ome_store_path": str(m.ome_store_path),
            "zip_path": str(m.zip_path) if m.zip_path else None,
            "n_images": m.n_images,
            "pyramid_levels": m.pyramid_levels,
            "pixel_size_um": m.pixel_size_um,
        }
        for m in ome_manifests
    ]

    manifest_path = str(Path(ome_cfg.output_dir) / "pipeline_manifest.json")
    safe_write_json(pipeline_stats, manifest_path)
    
    logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()