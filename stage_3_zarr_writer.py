import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from pre_process.zarr_writer import ZarrWriter, ZarrWriterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("stage4_zarr_writer")


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
    Execute Stage 4 of the microscopy image pipeline.

    Loads the filtered image buckets and invokes the ZarrWriter to generate
    Zarr array structures on disk. Writes out the generated Zarr manifests and
    updates pipeline statistics utilizing multithreading.
    """
    parser = argparse.ArgumentParser(description="Stage 4: Writing zarr arrays")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--in-data",
        default="intermediate_filtered.json",
        help="Path to load filtered buckets"
    )
    parser.add_argument(
        "--out-manifests",
        default="intermediate_zarr_manifests.json",
        help="Path to save zarr manifests"
    )
    parser.add_argument(
        "--stats-file",
        default="pipeline_stats.json",
        help="Path to pipeline stats file"
    )
    args = parser.parse_args()

    data = load_config(args.config)
    writer_cfg = ZarrWriterConfig.from_dict(data)

    with open(args.in_data, "r") as f:
        filtered_buckets = json.load(f)

    with open(args.stats_file, "r") as f:
        pipeline_stats = json.load(f)

    logger.info("Stage 4: Writing zarr arrays")
    t4 = time.perf_counter()

    writer = ZarrWriter(writer_cfg)
    zarr_manifests = writer.write_all(filtered_buckets)

    dt_zarr = time.perf_counter() - t4
    logger.info(
        "Stage 4 complete in %.1fs: %d arrays",
        dt_zarr, len(zarr_manifests)
    )

    manifest_dicts = [
        {
            "bucket": m.bucket,
            "store_path": str(m.store_path),
            "shape": list(m.shape),
            "chunks": list(m.chunks),
            "dtype": str(m.dtype),
            "n_images": m.n_images,
        }
        for m in zarr_manifests
    ]
    
    pipeline_stats["timing"]["zarr_write_s"] = round(dt_zarr, 2)
    pipeline_stats["zarr_manifests"] = manifest_dicts

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(safe_write_json, manifest_dicts, args.out_manifests)
        f2 = executor.submit(safe_write_json, pipeline_stats, args.stats_file)
        f1.result()
        f2.result()


if __name__ == "__main__":
    main()