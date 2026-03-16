import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from pre_process._pre_process_utils.pipeline_utils import (
    load_config,
    prompt_for_path,
    safe_write_json,
)
from pre_process.resolution_grouper import GrouperConfig, ResolutionGrouper
from pre_process.tar_streamer import ConcurrencyConfig, StreamConfig, TarImageStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("stage_1_stream_group")

_DEFAULT_CONFIG = "pre_process/config.yaml"
_DEFAULT_OUT_DATA = "intermediate_grouper.json"
_DEFAULT_OUT_STATS = "pipeline_stats.json"


def main() -> None:
    """
    Execute Stages 1 and 2 of the plankton image pipeline.

    Streams images from tar archives, groups them by resolution into
    buckets, then writes the bucketed records and timing stats to disk.
    Images are serialised via ``_json_default`` (``ndarray.tolist()``)
    so downstream stages can reconstruct them without data loss.
    """
    parser = argparse.ArgumentParser(
        description="Stages 1+2: tar streaming and resolution grouping"
    )
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--out-data", help="Path to save intermediate data")
    parser.add_argument("--out-stats", help="Path to save pipeline stats")
    args = parser.parse_args()

    print("--- Pipeline Configuration ---")
    config_path = args.config or prompt_for_path(
        "Enter path to YAML config", _DEFAULT_CONFIG
    )
    out_data = args.out_data or prompt_for_path(
        "Enter path for intermediate data", _DEFAULT_OUT_DATA
    )
    out_stats = args.out_stats or prompt_for_path(
        "Enter path for pipeline stats", _DEFAULT_OUT_STATS
    )
    print("------------------------------\n")

    t0 = time.perf_counter()
    data = load_config(config_path)

    stream_cfg = StreamConfig.from_dict(data)
    conc_cfg = ConcurrencyConfig.from_dict(data)
    grouper_cfg = GrouperConfig.from_dict(data)

    logger.info(
        "Stage 1: Streaming images from %d archives",
        len(stream_cfg.tar_paths),
    )
    t1 = time.perf_counter()
    streamer = TarImageStream(config=stream_cfg, concurrency=conc_cfg)

    logger.info(
        "Stage 2: Grouping by resolution (tile=%d)",
        grouper_cfg.tile_size,
    )
    grouper = ResolutionGrouper(grouper_cfg).ingest(streamer)
    dt_stream = time.perf_counter() - t1

    logger.info(
        "Stages 1+2 complete in %.1fs: %d images → %d buckets",
        dt_stream,
        grouper.total_images,
        len(grouper),
    )
    for entry in grouper.summary():
        logger.info("  %s: %d images", entry["bucket"], entry["count"])

    stats = {
        "timing": {"stream_and_group_s": round(dt_stream, 2), "t0": t0},
        "stream_stats": streamer.stats,
        "grouper_summary": grouper.summary(),
    }
    grouper_data = {str(k): v for k, v in dict(grouper).items()}

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(safe_write_json, grouper_data, out_data)
        f2 = executor.submit(safe_write_json, stats, out_stats)
        f1.result()
        f2.result()

    logger.info("Data saved to %s and stats to %s", out_data, out_stats)


if __name__ == "__main__":
    main()