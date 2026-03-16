import argparse
import json
import logging
import time
from pathlib import Path

from pre_process._pre_process_utils.pipeline_utils import (
    deserialise_buckets,
    load_config,
    prompt_for_path,
    safe_write_json,
)
from pre_process.ome_converter import OmeConverterConfig, OmeZarrConverter
from pre_process.zarr_writer import ZarrWriterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("stage_4_ome_converter")

_DEFAULT_CONFIG = "pre_process/config.yaml"
_DEFAULT_IN_DATA = "intermediate_filtered.json"
_DEFAULT_STATS = "pipeline_stats.json"


def main() -> None:
    """
    Execute Stage 4 of the plankton image pipeline: OME-Zarr conversion.

    Reads filtered image buckets from the intermediate JSON produced by
    Stage 3, converts each bucket to OME-Zarr 0.4 format (with optional
    pyramid and zip), then appends timing and manifest data to the shared
    pipeline stats file and writes the final pipeline manifest.
    """
    parser = argparse.ArgumentParser(
        description="Stage 4: OME-Zarr conversion and manifest generation"
    )
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--in-data", help="Path to load filtered buckets")
    parser.add_argument("--stats-file", help="Path to pipeline stats file")
    args = parser.parse_args()

    print("--- Pipeline Configuration ---")
    config_path = args.config or prompt_for_path(
        "Enter path to YAML config", _DEFAULT_CONFIG
    )
    in_data = args.in_data or prompt_for_path(
        "Enter path for input data", _DEFAULT_IN_DATA
    )
    stats_file = args.stats_file or prompt_for_path(
        "Enter path for pipeline stats", _DEFAULT_STATS
    )
    print("------------------------------\n")

    data = load_config(config_path)
    ome_cfg = OmeConverterConfig.from_dict(data)
    writer_cfg = ZarrWriterConfig.from_dict(data)

    filtered_buckets = deserialise_buckets(
        json.loads(Path(in_data).read_text())
    )
    pipeline_stats: dict = json.loads(Path(stats_file).read_text())

    logger.info("Stage 4: Converting to OME-Zarr 0.4")
    t_start = time.perf_counter()

    converter = OmeZarrConverter(ome_cfg, writer_cfg)
    ome_manifests = converter.convert_all(filtered_buckets)

    dt_ome = time.perf_counter() - t_start
    logger.info(
        "Stage 4 complete in %.1fs: %d OME-Zarr files",
        dt_ome,
        len(ome_manifests),
    )

    for m in ome_manifests:
        logger.info(
            "  %s: %d images, %d levels -> %s%s",
            m.bucket,
            m.n_images,
            m.pyramid_levels,
            m.ome_store_path,
            f" (zip: {m.zip_path})" if m.zip_path else "",
        )

    t0 = pipeline_stats["timing"].get("t0")
    dt_total = (time.perf_counter() - t0) if t0 is not None else None
    if dt_total is not None:
        logger.info("Pipeline complete in %.1fs", dt_total)
    else:
        logger.warning(
            "t0 not found in pipeline stats; total elapsed time unavailable."
        )

    pipeline_stats["timing"]["ome_convert_s"] = round(dt_ome, 2)
    if dt_total is not None:
        pipeline_stats["timing"]["total_s"] = round(dt_total, 2)

    pipeline_stats["ome_manifests"] = [m.to_dict() for m in ome_manifests]

    manifest_path = Path(ome_cfg.output_dir) / "pipeline_manifest.json"
    safe_write_json(pipeline_stats, str(manifest_path))
    logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()