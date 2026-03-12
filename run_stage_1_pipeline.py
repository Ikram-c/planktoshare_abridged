import argparse
import logging
import time

from pre_process.path_utils import (
    dump_profile,
    get_valid_config_path,
    load_config,
    profile_context,
    safe_write_json,
)
from pre_process.resolution_grouper import GrouperConfig, ResolutionGrouper
from pre_process.tar_streamer import ConcurrencyConfig, StreamConfig, TarImageStream
from pre_process.zarr_writer import ZarrWriter, ZarrWriterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("stage1")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: tar → group → zarr")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--out-metadata",
        default="output/stage1_metadata.json",
        help="Path for zarr + bucket metadata JSON",
    )
    parser.add_argument(
        "--out-stats",
        default="output/stage1_stats.json",
        help="Path for stream + grouper stats JSON",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile and dump stage1.prof",
    )
    args = parser.parse_args()

    config_path = get_valid_config_path(args.config)
    data = load_config(config_path)

    stream_cfg = StreamConfig.from_dict(data)
    stream_cfg.validate_paths()
    conc_cfg = ConcurrencyConfig.from_dict(data)
    grouper_cfg = GrouperConfig.from_dict(data)
    writer_cfg = ZarrWriterConfig.from_dict(data)

    with profile_context(args.profile) as profiler:
        t0 = time.perf_counter()

        logger.info("Stage 1: Streaming %d archives", len(stream_cfg.tar_paths))
        streamer = TarImageStream(config=stream_cfg, concurrency=conc_cfg)

        logger.info("Stage 1: Grouping (tile=%d)", grouper_cfg.tile_size)
        grouper = ResolutionGrouper(grouper_cfg).ingest(streamer)
        logger.info(
            "Stage 1: %d images → %d buckets",
            grouper.total_images, len(grouper),
        )
        for entry in grouper.summary():
            logger.info("  %s: %d images", entry["bucket"], entry["count"])

        logger.info("Stage 1: Writing zarr arrays to %s", writer_cfg.output_dir)
        writer = ZarrWriter(writer_cfg)
        zarr_manifests = writer.write_all(dict(grouper))
        logger.info("Stage 1: Wrote %d zarr arrays", len(zarr_manifests))

        dt_total = time.perf_counter() - t0
        logger.info("Stage 1 complete in %.1fs", dt_total)

        metadata = {
            "zarr_dir": writer_cfg.output_dir,
            "total_s": round(dt_total, 2),
            "bucket_meta": {
                str(key): [
                    {
                        "index": i,
                        "filename": rec["filename"],
                        "tar_path": rec["tar_path"],
                        "original_shape": list(rec["shape"]),
                        "dtype": str(rec["dtype"]),
                    }
                    for i, rec in enumerate(records)
                ]
                for key, records in grouper
            },
            "zarr_manifests": [
                {
                    "bucket": m.bucket,
                    "store_path": m.store_path,
                    "shape": list(m.shape),
                    "chunks": list(m.chunks),
                    "dtype": m.dtype,
                    "n_images": m.n_images,
                    "compression": m.compression,
                }
                for m in zarr_manifests
            ],
        }
        stats = {
            "stream_stats": streamer.stats,
            "grouper_summary": grouper.summary(),
        }

        safe_write_json(metadata, args.out_metadata)
        safe_write_json(stats, args.out_stats)
        logger.info("Metadata → %s | Stats → %s", args.out_metadata, args.out_stats)

    dump_profile(profiler, "stage1.prof")


if __name__ == "__main__":
    main()