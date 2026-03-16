import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pre_process._pre_process_utils.pipeline_utils import (
    deserialise_buckets,
    load_config,
    prompt_for_path,
    safe_write_json,
)
from pre_process.zarr_writer import ZarrWriter, ZarrWriterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("stage_3_zarr_writer")

_DEFAULT_CONFIG = "pre_process/config.yaml"
_DEFAULT_IN_DATA = "intermediate_filtered.json"
_DEFAULT_OUT_MANIFESTS = "intermediate_zarr_manifests.json"
_DEFAULT_STATS = "pipeline_stats.json"


def main() -> None:
    """
    Execute Stage 3 of the plankton image pipeline: Zarr array writing.

    Reads filtered image buckets from the intermediate JSON produced by
    Stage 2, writes each bucket as a compressed Zarr v2 array, then appends
    timing and manifest data to the shared pipeline stats file.
    """
    parser = argparse.ArgumentParser(
        description="Stage 3: Write zarr arrays"
    )
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--in-data", help="Path to load filtered buckets")
    parser.add_argument(
        "--out-manifests", help="Path to save zarr manifests"
    )
    parser.add_argument("--stats-file", help="Path to pipeline stats file")
    args = parser.parse_args()

    print("--- Pipeline Configuration ---")
    config_path = args.config or prompt_for_path(
        "Enter path to YAML config", _DEFAULT_CONFIG
    )
    in_data = args.in_data or prompt_for_path(
        "Enter path for input data", _DEFAULT_IN_DATA
    )
    out_manifests = args.out_manifests or prompt_for_path(
        "Enter path for output manifests", _DEFAULT_OUT_MANIFESTS
    )
    stats_file = args.stats_file or prompt_for_path(
        "Enter path for pipeline stats", _DEFAULT_STATS
    )
    print("------------------------------\n")

    data = load_config(config_path)
    writer_cfg = ZarrWriterConfig.from_dict(data)

    filtered_buckets = deserialise_buckets(
        json.loads(Path(in_data).read_text())
    )
    pipeline_stats: dict = json.loads(Path(stats_file).read_text())

    logger.info("Stage 3: Writing zarr arrays")
    t_start = time.perf_counter()

    writer = ZarrWriter(writer_cfg)
    zarr_manifests = writer.write_all(filtered_buckets)

    dt = time.perf_counter() - t_start
    logger.info(
        "Stage 3 complete in %.1fs: %d arrays", dt, len(zarr_manifests)
    )

    manifest_dicts = [m.to_dict() for m in zarr_manifests]
    pipeline_stats["timing"]["zarr_write_s"] = round(dt, 2)
    pipeline_stats["zarr_manifests"] = manifest_dicts

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(safe_write_json, manifest_dicts, out_manifests)
        f2 = executor.submit(safe_write_json, pipeline_stats, stats_file)
        f1.result()
        f2.result()


if __name__ == "__main__":
    main()