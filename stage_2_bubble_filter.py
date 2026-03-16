import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from pre_process._pre_process_utils.pipeline_utils import (
    deserialise_buckets,
    load_config,
    prompt_for_path,
    safe_write_json,
)
from pre_process.bubble_filter import BubbleFilter, FilterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("stage_2_bubble_filter")

_DEFAULT_CONFIG = "pre_process/config.yaml"
_DEFAULT_IN_DATA = "intermediate_grouper.json"
_DEFAULT_OUT_DATA = "intermediate_filtered.json"
_DEFAULT_STATS = "pipeline_stats.json"


@dataclass
class StageArgs:
    """Resolved runtime arguments for Stage 2."""

    config_path: str
    in_data: str
    out_data: str
    stats_file: str
    skip_filter: bool = False


def _resolve_args(args: argparse.Namespace) -> StageArgs:
    print("--- Pipeline Configuration ---")
    resolved = StageArgs(
        config_path=args.config or prompt_for_path(
            "Path to YAML config", _DEFAULT_CONFIG
        ),
        in_data=args.in_data or prompt_for_path(
            "Path for input data", _DEFAULT_IN_DATA
        ),
        out_data=args.out_data or prompt_for_path(
            "Path for output data", _DEFAULT_OUT_DATA
        ),
        stats_file=args.stats_file or prompt_for_path(
            "Path for pipeline stats", _DEFAULT_STATS
        ),
        skip_filter=args.skip_filter,
    )
    print("------------------------------\n")
    return resolved


def _run_filter(
    stage: StageArgs,
    buckets: dict,
    pipeline_stats: dict,
    config: dict,
) -> dict:
    if stage.skip_filter:
        logger.info("Stage 2: Skipped (--skip-filter)")
        pipeline_stats["timing"]["filter_s"] = None
        pipeline_stats["filter_stats"] = None
        return buckets

    logger.info("Stage 2: Filtering with bubble autoencoder")
    t0 = time.perf_counter()

    bubble_filter = BubbleFilter(FilterConfig.from_dict(config))
    filtered_buckets, _ = bubble_filter.filter_buckets(buckets)
    dt = time.perf_counter() - t0

    stats = bubble_filter.stats
    logger.info(
        "Stage 2 complete in %.1fs: %d passed, %d rejected, %d errors",
        dt,
        stats["passed"],
        stats["rejected"],
        stats["errors"],
    )
    pipeline_stats["timing"]["filter_s"] = round(dt, 2)
    pipeline_stats["filter_stats"] = stats
    return filtered_buckets


def main() -> None:
    """
    Execute Stage 2 of the plankton image pipeline: bubble filtering.

    Reads bucketed image records written by Stage 1, reconstructs numpy
    arrays via :func:`deserialise_buckets`, runs the bubble autoencoder
    filter, and writes surviving records to disk using
    :func:`safe_write_json` so images are preserved as nested lists.
    """
    parser = argparse.ArgumentParser(
        description="Stage 2: bubble autoencoder filter"
    )
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--in-data", help="Path to load intermediate data")
    parser.add_argument("--out-data", help="Path to save filtered buckets")
    parser.add_argument("--stats-file", help="Path to pipeline stats file")
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Pass all images through without filtering",
    )
    stage = _resolve_args(parser.parse_args())

    config = load_config(stage.config_path)
    buckets = deserialise_buckets(
        json.loads(Path(stage.in_data).read_text())
    )
    pipeline_stats: dict = json.loads(Path(stage.stats_file).read_text())

    filtered_buckets = _run_filter(stage, buckets, pipeline_stats, config)

    if not filtered_buckets:
        logger.warning("No images remain after filtering. Pipeline complete.")
        return

    serialisable = {str(k): v for k, v in filtered_buckets.items()}
    safe_write_json(serialisable, stage.out_data)
    safe_write_json(pipeline_stats, stage.stats_file)


if __name__ == "__main__":
    main()