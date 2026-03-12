import argparse
import logging
import time
from pathlib import Path

from pre_process.bubble_filter import (
    BubbleFilter,
    FilterConfig,
    RuleBasedBubbleFilter,
    RuleBasedFilterConfig,
)
from pre_process.ome_converter import OmeConverterConfig, OmeZarrConverter
from pre_process.path_utils import (
    get_valid_config_path,
    load_config,
    safe_write_json,
)
from pre_process.resolution_grouper import GrouperConfig, ResolutionGrouper
from pre_process.tar_streamer import ConcurrencyConfig, StreamConfig, TarImageStream
from pre_process.zarr_writer import ZarrWriter, ZarrWriterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")

_FILTER_MODES = ("autoencoder", "rules")


def _build_filter(mode: str, data: dict):
    if mode == "rules":
        return RuleBasedBubbleFilter(RuleBasedFilterConfig.from_dict(data))
    return BubbleFilter(FilterConfig.from_dict(data))


def run(
    config_path: str,
    skip_filter: bool = False,
    skip_zarr: bool = False,
    filter_mode: str = "autoencoder",
):
    t0 = time.perf_counter()
    data = load_config(get_valid_config_path(config_path))

    stream_cfg = StreamConfig.from_dict(data)
    stream_cfg.validate_paths()
    conc_cfg = ConcurrencyConfig.from_dict(data)
    grouper_cfg = GrouperConfig.from_dict(data)
    writer_cfg = ZarrWriterConfig.from_dict(data)
    ome_cfg = OmeConverterConfig.from_dict(data)

    logger.info("Stage 1: Streaming images from %d archives", len(stream_cfg.tar_paths))
    t1 = time.perf_counter()
    streamer = TarImageStream(config=stream_cfg, concurrency=conc_cfg)

    logger.info("Stage 2: Grouping by resolution (tile=%d)", grouper_cfg.tile_size)
    grouper = ResolutionGrouper(grouper_cfg).ingest(streamer)

    dt_stream = time.perf_counter() - t1
    logger.info(
        "Stages 1+2 complete in %.1fs: %d images → %d buckets",
        dt_stream, grouper.total_images, len(grouper),
    )
    for entry in grouper.summary():
        logger.info("  %s: %d images", entry["bucket"], entry["count"])

    dt_filter: float | None = None
    filter_stats: dict | None = None

    if skip_filter:
        logger.info("Stage 3: Skipped (--skip-filter)")
        filtered_buckets = dict(grouper)
    else:
        logger.info("Stage 3: Filtering [mode=%s]", filter_mode)
        t3 = time.perf_counter()
        bubble_filter = _build_filter(filter_mode, data)
        filtered_buckets, _ = bubble_filter.filter_buckets(dict(grouper))
        dt_filter = time.perf_counter() - t3
        filter_stats = bubble_filter.stats
        logger.info(
            "Stage 3 complete in %.1fs: %d passed, %d rejected, %d errors",
            dt_filter,
            filter_stats["passed"],
            filter_stats["rejected"],
            filter_stats["errors"],
        )

    if not filtered_buckets:
        logger.warning("No images remain after filtering. Pipeline complete.")
        return

    if skip_zarr:
        logger.info("Stages 4+5: Skipped (--skip-zarr)")
        return

    logger.info("Stage 4: Writing zarr arrays")
    t4 = time.perf_counter()
    writer = ZarrWriter(writer_cfg)
    zarr_manifests = writer.write_all(filtered_buckets)
    dt_zarr = time.perf_counter() - t4
    logger.info("Stage 4 complete in %.1fs: %d arrays", dt_zarr, len(zarr_manifests))

    logger.info("Stage 5: Converting to OME-Zarr 0.4")
    t5 = time.perf_counter()
    converter = OmeZarrConverter(ome_cfg, writer_cfg)
    ome_manifests = converter.convert_all(filtered_buckets)
    dt_ome = time.perf_counter() - t5
    logger.info("Stage 5 complete in %.1fs: %d OME-Zarr files", dt_ome, len(ome_manifests))

    for m in ome_manifests:
        logger.info(
            "  %s: %d images, %d levels → %s%s",
            m.bucket, m.n_images, m.pyramid_levels, m.ome_store_path,
            f" (zip: {m.zip_path})" if m.zip_path else "",
        )

    dt_total = time.perf_counter() - t0
    logger.info("Pipeline complete in %.1fs", dt_total)

    manifest_data = {
        "filter_mode": filter_mode,
        "timing": {
            "stream_and_group_s": round(dt_stream, 2),
            "filter_s": round(dt_filter, 2) if dt_filter is not None else None,
            "zarr_write_s": round(dt_zarr, 2),
            "ome_convert_s": round(dt_ome, 2),
            "total_s": round(dt_total, 2),
        },
        "stream_stats": streamer.stats,
        "grouper_summary": grouper.summary(),
        "filter_stats": filter_stats,
        "zarr_manifests": [
            {
                "bucket": m.bucket,
                "store_path": m.store_path,
                "shape": list(m.shape),
                "chunks": list(m.chunks),
                "dtype": m.dtype,
                "n_images": m.n_images,
            }
            for m in zarr_manifests
        ],
        "ome_manifests": [
            {
                "bucket": m.bucket,
                "ome_store_path": m.ome_store_path,
                "zip_path": m.zip_path,
                "n_images": m.n_images,
                "pyramid_levels": m.pyramid_levels,
                "pixel_size_um": m.pixel_size_um,
            }
            for m in ome_manifests
        ],
    }
    manifest_path = Path(ome_cfg.output_dir) / "pipeline_manifest.json"
    safe_write_json(manifest_data, manifest_path)
    logger.info("Manifest written to %s", manifest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Microscopy image pipeline: tar → filter → OME-Zarr",
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--skip-filter", action="store_true",
        help="Skip bubble detection filtering (Stage 3)",
    )
    parser.add_argument(
        "--skip-zarr", action="store_true",
        help="Skip zarr/OME-Zarr writing (Stages 4+5)",
    )
    parser.add_argument(
        "--filter-mode",
        choices=_FILTER_MODES,
        default="autoencoder",
        help="Filter backend: 'autoencoder' (default) or 'rules' (training-free)",
    )
    args = parser.parse_args()
    run(
        args.config,
        skip_filter=args.skip_filter,
        skip_zarr=args.skip_zarr,
        filter_mode=args.filter_mode,
    )


if __name__ == "__main__":
    main()