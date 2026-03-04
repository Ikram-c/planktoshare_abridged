import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from .models import ConcurrencyConfig, OutputFormat, StreamConfig
from .stream import TarImageStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_config_from_args(args: argparse.Namespace) -> tuple[StreamConfig, ConcurrencyConfig]:
    if args.config:
        config_path = Path(args.config)
        with open(config_path) as fh:
            data = yaml.safe_load(fh)
        stream_cfg = StreamConfig.from_dict(data)
        concurrency_cfg = ConcurrencyConfig.from_dict(data)
        return stream_cfg, concurrency_cfg

    stream_cfg = StreamConfig(
        tar_paths=args.tar_paths,
        output_format=OutputFormat(args.format),
        max_images=args.max_images,
        convert_mode=args.convert_mode,
        min_size=(args.min_width, args.min_height),
        max_size=(args.max_width, args.max_height),
    )
    concurrency_cfg = ConcurrencyConfig(
        enabled=args.concurrent,
        max_workers=args.workers,
    )
    return stream_cfg, concurrency_cfg


def run_stream(stream_cfg: StreamConfig, concurrency_cfg: ConcurrencyConfig):
    streamer = TarImageStream(config=stream_cfg, concurrency=concurrency_cfg)
    logger.info("Manifest: %d images across %d archives", len(streamer), len(stream_cfg.tar_paths))

    for record in streamer:
        shape = record["shape"]
        dtype = record["dtype"]
        logger.info("  %s: shape=%s dtype=%s", record["filename"], shape, dtype)

    stats = streamer.stats
    logger.info(
        "Done: %d yielded, %d skipped, %d total",
        stats["yielded"],
        stats["skipped"],
        stats["total"],
    )


def run_manifest(stream_cfg: StreamConfig, concurrency_cfg: ConcurrencyConfig):
    streamer = TarImageStream(config=stream_cfg, concurrency=concurrency_cfg)
    manifest = streamer.manifest()
    for tar_path, filenames in manifest.items():
        logger.info("%s: %d images", tar_path, len(filenames))
        for name in filenames[:10]:
            logger.info("  %s", name)
        if len(filenames) > 10:
            logger.info("  ... and %d more", len(filenames) - 10)


def main():
    parser = argparse.ArgumentParser(
        prog="tar_streamer",
        description="Stream and inspect images from tar archives",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "tar_paths", nargs="*", default=[],
        help="Paths to tar archives (ignored if --config is provided)",
    )
    parser.add_argument("--format", choices=["pil", "numpy"], default="numpy")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--convert-mode", type=str, default=None)
    parser.add_argument("--min-width", type=int, default=128)
    parser.add_argument("--min-height", type=int, default=128)
    parser.add_argument("--max-width", type=int, default=16384)
    parser.add_argument("--max-height", type=int, default=16384)
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="List image members without decoding",
    )

    args = parser.parse_args()

    if not args.config and not args.tar_paths:
        parser.error("Provide either --config or one or more tar_paths")

    stream_cfg, concurrency_cfg = build_config_from_args(args)

    if args.manifest_only:
        run_manifest(stream_cfg, concurrency_cfg)
    else:
        run_stream(stream_cfg, concurrency_cfg)


if __name__ == "__main__":
    main()