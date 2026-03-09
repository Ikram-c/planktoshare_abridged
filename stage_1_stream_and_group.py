import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from pre_process.resolution_grouper import GrouperConfig, ResolutionGrouper
from pre_process.tar_streamer import ConcurrencyConfig, StreamConfig, TarImageStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("stage1_2_stream_group")


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
    Execute Stages 1 and 2 of the microscopy image pipeline.

    Parses command-line arguments, loads the specified configuration, streams
    images from tar archives, groups them into buckets by resolution, and writes
    the intermediate grouping data and timing statistics to local JSON files using
    multithreading.
    """
    parser = argparse.ArgumentParser(
        description="Stage 1 & 2: Tar Streamer and Resolution Grouper"
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--out-data",
        default="intermediate_grouper.json",
        help="Path to save intermediate grouper data"
    )
    parser.add_argument(
        "--out-stats",
        default="pipeline_stats.json",
        help="Path to save pipeline stats"
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    data = load_config(args.config)

    stream_cfg = StreamConfig.from_dict(data)
    conc_cfg = ConcurrencyConfig.from_dict(data)
    grouper_cfg = GrouperConfig.from_dict(data)

    logger.info("Stage 1: Streaming images from %d archives", len(stream_cfg.tar_paths))
    t1 = time.perf_counter()
    streamer = TarImageStream(config=stream_cfg, concurrency=conc_cfg)

    logger.info("Stage 2: Grouping by resolution (tile=%d)", grouper_cfg.tile_size)
    grouper = ResolutionGrouper(grouper_cfg).ingest(streamer)
    dt_stream = time.perf_counter() - t1

    logger.info(
        "Stages 1+2 complete in %.1fs: %d images → %d buckets",
        dt_stream, grouper.total_images, len(grouper)
    )
    
    for entry in grouper.summary():
        logger.info("  %s: %d images", entry["bucket"], entry["count"])

    stats = {
        "timing": {"stream_and_group_s": round(dt_stream, 2), "t0": t0},
        "stream_stats": streamer.stats,
        "grouper_summary": grouper.summary()
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(safe_write_json, dict(grouper), args.out_data)
        f2 = executor.submit(safe_write_json, stats, args.out_stats)
        f1.result()
        f2.result()

    logger.info("Data saved to %s and stats to %s", args.out_data, args.out_stats)


if __name__ == "__main__":
    main()