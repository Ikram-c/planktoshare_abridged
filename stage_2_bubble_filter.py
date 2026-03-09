import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from pre_process.bubble_filter import BubbleFilter, FilterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("stage3_filter")


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
    Execute Stage 3 of the microscopy image pipeline.

    Parses command-line arguments, loads the intermediate grouped buckets, and
    applies the bubble autoencoder filter to remove unwanted images. Multithreads
    the saving of the updated filtered buckets and pipeline statistics.
    """
    parser = argparse.ArgumentParser(
        description="Stage 3: Bubble Autoencoder Filter"
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--in-data",
        default="intermediate_grouper.json",
        help="Path to load intermediate grouper data"
    )
    parser.add_argument(
        "--out-data",
        default="intermediate_filtered.json",
        help="Path to save filtered buckets"
    )
    parser.add_argument(
        "--stats-file",
        default="pipeline_stats.json",
        help="Path to pipeline stats file"
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip bubble detection filtering"
    )
    args = parser.parse_args()

    data = load_config(args.config)

    with open(args.in_data, "r") as f:
        grouper_dict = json.load(f)

    with open(args.stats_file, "r") as f:
        pipeline_stats = json.load(f)

    if args.skip_filter:
        logger.info("Stage 3: Skipped (--skip-filter)")
        filtered_buckets = grouper_dict
        pipeline_stats["timing"]["filter_s"] = None
        pipeline_stats["filter_stats"] = None
    else:
        logger.info("Stage 3: Filtering with bubble autoencoder")
        t3 = time.perf_counter()

        filter_cfg = FilterConfig.from_dict(data)
        bubble_filter = BubbleFilter(filter_cfg)
        filtered_buckets, _ = bubble_filter.filter_buckets(grouper_dict)

        dt_filter = time.perf_counter() - t3
        stats = bubble_filter.stats
        logger.info(
            "Stage 3 complete in %.1fs: %d passed, %d rejected, %d errors",
            dt_filter, stats["passed"], stats["rejected"], stats["errors"]
        )

        pipeline_stats["timing"]["filter_s"] = round(dt_filter, 2)
        pipeline_stats["filter_stats"] = bubble_filter.stats

    if not filtered_buckets:
        logger.warning("No images remain after filtering. Pipeline complete.")
        return

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(safe_write_json, filtered_buckets, args.out_data)
        f2 = executor.submit(safe_write_json, pipeline_stats, args.stats_file)
        f1.result()
        f2.result()


if __name__ == "__main__":
    main()