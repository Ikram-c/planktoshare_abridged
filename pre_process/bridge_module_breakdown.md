# Bridge Module — Technical Breakdown

## Module Purpose

This module bridges the image processing pipeline (`tar_streamer` → `resolution_grouper` → `bubble_filter`) to the [planktoshare](https://github.com/geoJoost/planktoshare) plankton classification system. The core problem is a **structural format mismatch**: the pipeline produces resolution-grouped `ImageRecord` dicts with bubble filter metadata flattened across buckets, while planktoshare expects a specific two-level directory of `.tar` archives organised as `{SOURCE_BASE_DIR}/{date_dir}/{cruise_timestamp}.tar`, where each tar is extracted to a temp directory and classified by a FastAI ResNet50 model. The bridge reconstructs this hierarchy from pipeline provenance metadata, re-packages filtered images into correctly-named tars, and orchestrates planktoshare's inference → sampling → report pipeline.

---

## File Map

| File | Role | Key Exports |
|------|------|-------------|
| `models.py` | Config, enums, tar mapping, manifest/result dataclasses | `BridgeConfig`, `ExportManifest`, `BridgeResult`, `PlanktoModel`, `TarMapping` |
| `exporter.py` | Reconstructs planktoshare directory structure from pipeline output | `PlanktoshareExporter` |
| `runner.py` | Calls planktoshare's inference/sampling/report functions | `PlanktoshareRunner` |
| `__init__.py` | Public API surface | Re-exports all above |
| `__main__.py` | CLI with `export`, `run`, `infer`, `report`, `validate` subcommands | `main()` |

---

## The planktoshare Contract

From reading `inference.py`, planktoshare expects:

```
SOURCE_BASE_DIR/
├── 2023-06-15/                              # date directory
│   ├── 2023_MONS_Tridens_1230.tar           # 10-min bin tar
│   ├── 2023_MONS_Tridens_1240.tar
│   └── 2023_MONS_Tridens_1250.tar
├── 2023-06-16/
│   └── ...
```

**Critical constraints extracted from inference.py:**

| Constraint | Source line | Implication |
|------------|------------|-------------|
| Two-level hierarchy | `os.listdir(SOURCE_BASE_DIR)` → `os.listdir(date_dir_path)` | Must be `source/{date}/{*.tar}` |
| Timestamp from filename | `tar_file.split('_')[-1].split('.')[0]` | Tar must be named `*_{NNNN}.tar` |
| 10-minute bin filter | `timestamp.endswith('0')` | Timestamps not ending in `0` are **skipped** |
| CSV skip check | `results_dir / f"{CRUISE}_{date}_{ts}.csv"` | Already-processed bins are skipped |
| Temp extraction | `tarfile.open(tar_path, 'r')` → `extractall(temp_dir)` | Tars must be uncompressed or auto-detectable |
| FastAI `get_image_files` | Scans extracted temp dir | Standard image extensions only |
| Image resize to 300px | `Resize(300, ResizeMethod.Pad, pad_mode='zeros')` | Planktoshare handles its own resizing — export raw |
| `Background.tif` special | Separated from classification, used for geodata | Should be preserved if present |
| Model weights path | `learn.load(model_weights)` with cwd = planktoshare root | FastAI resolves `models/{name}.pth` from cwd |

---

## Architecture

```
┌────────────────────── Your Pipeline ──────────────────────┐
│                                                            │
│  .tar archives → TarImageStream → ResolutionGrouper        │
│                       → BubbleFilter                       │
│                                                            │
│  Output: dict[BucketKey, list[ImageRecord]]                │
│          dict[BucketKey, list[FilterResult]]               │
│                                                            │
│  Each ImageRecord carries:                                 │
│    tar_path = "data/2023_MONS_Tridens/2023-06-15/..._1230.tar"
│    filename = "img_00001.tif"                              │
│    bubble_score, object_count (from filter)                │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌──────────── bridge.PlanktoshareExporter ────────────────────┐
│                                                             │
│  1. Apply bridge-level filters (max_bubble_score, etc.)     │
│  2. Parse (date, timestamp) from each record's tar_path     │
│  3. Snap timestamps to 10-min bins (must end in 0)          │
│  4. Group records by (date, timestamp)                      │
│  5. Write one tar per group: {date}/{cruise}_{ts}.tar       │
│  6. Write metadata CSVs + origin mapping                    │
│                                                             │
│  Output:                                                    │
│    bridge_export/source/2023-06-15/CRUISE_1230.tar          │
│    bridge_export/source/2023-06-15/CRUISE_1240.tar          │
│    bridge_export/metadata/*.csv, *.json                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────── bridge.PlanktoshareRunner ──────────────────────┐
│                                                             │
│  1. Validate: weights exist, train data exists              │
│  2. sys.path.insert(0, planktoshare_root)                   │
│  3. os.chdir(planktoshare_root)                             │
│  4. conduct_plankton_inference(source_dir, ...)             │
│     → iterates date_dirs → tars → extract → FastAI predict  │
│  5. Merge bubble metadata into results                      │
│  6. get_random_samples(results_dir, ...)                    │
│  7. create_word_document(results_dir, ...) → .docx          │
│                                                             │
│  Output: BridgeResult                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. `models.py` — Configuration & Types

### `PlanktoModel` (Enum)

| Value | Weights file | Training dataset | Classes |
|-------|-------------|------------------|---------|
| `RESNET50_DETAILED` | `Plankton_imager_v01_stage-2_Best` | `data/DETAILED_merged` | 49 plankton + non-plankton |
| `OSPAR` | `Plankton_imager_v03_stage-2_Best` | `data/OSPAR_merged` | 6 OSPAR classes (faster) |

### `BridgeConfig`

| Field | Default | Purpose |
|-------|---------|---------|
| `planktoshare_root` | *(required)* | Filesystem path to planktoshare repo clone |
| `cruise_name` | *(required)* | Survey identifier — used in tar naming, CSV naming, report |
| `export_dir` | `"bridge_export"` | Root for all bridge outputs |
| `image_format` | `"png"` | Image encoding inside exported tars |
| `model` | `RESNET50_DETAILED` | Which planktoshare model to use |
| `batch_size` | `128` | FastAI inference batch size |
| `density_constant` | `340` | Pi-10 normalization (340L per 10-min window) |
| `classification_subsample` | `100` | Percentage of images to classify |
| `max_bubble_score` | `1.0` | Bridge-level filter ceiling (tighter than bubble_filter) |
| `min_object_count` | `0` | Bridge-level minimum object count |
| `default_date` | `"1970-01-01"` | Fallback when date cannot be parsed from tar_path |
| `bin_minutes` | `10` | Timestamp snapping interval (must match planktoshare's assumption) |
| `preserve_background_tif` | `True` | Whether to carry Background.tif through export |

### `TarMapping`

Tracks the provenance of each exported tar:

| Field | Purpose |
|-------|---------|
| `date_dir` | Parsed date string (e.g. `"2023-06-15"`) |
| `timestamp` | Snapped timestamp (e.g. `"1230"`) |
| `tar_filename` | Export tar name (e.g. `"2023_MONS_Tridens_1230.tar"`) |
| `records` | List of ImageRecords packed into this tar |
| `is_valid_bin` | `timestamp.endswith("0")` — planktoshare skips invalid bins |

---

## 2. `exporter.py` — Structure Reconstruction

### Origin Parsing

The exporter must reconstruct `(date, timestamp)` from each record's `tar_path`. Two regex patterns extract this:

| Pattern | Target | Example input | Extracted |
|---------|--------|---------------|-----------|
| `\d{4}[-_]\d{2}[-_]\d{2}` | Date from path components | `.../2023-06-15/...` | `"2023-06-15"` |
| `_(\d{3,6})\.tar$` | Timestamp from tar filename | `..._1230.tar` | `"1230"` |

### Timestamp Snapping

Raw timestamps are snapped to `bin_minutes` boundaries to satisfy planktoshare's `endswith('0')` check:

```
snapped = ⌊minutes / bin⌋ × bin
```

For `bin_minutes=10`: timestamp `"1234"` → `⌊1234/10⌋ × 10 = 1230` → `"1230"` (ends in `0` ✓)

### Export Pipeline

| Step | Operation | Output |
|------|-----------|--------|
| 1 | Apply bridge-level bubble filters across all buckets | Passing records |
| 2 | Write per-bucket bubble metadata CSVs | `metadata/bucket_*_metadata.csv` |
| 3 | Group all passing records by `(date, timestamp)` | `dict[(date, ts), list[ImageRecord]]` |
| 4 | For each group: create `source/{date}/` directory | Directory hierarchy |
| 5 | Write `{cruise}_{timestamp}.tar` with encoded images | Planktoshare-compatible tars |
| 6 | Warn if any timestamp doesn't end in `0` | Log warning |
| 7 | Write origin mapping CSV | `metadata/origin_mapping.csv` |
| 8 | Write export manifest JSON | `metadata/export_manifest.json` |

### Output Structure

```
bridge_export/
├── source/                                    # = SOURCE_BASE_DIR for planktoshare
│   ├── 2023-06-15/                            # date directory
│   │   ├── 2023_MONS_Tridens_1230.tar         # 10-min bin
│   │   ├── 2023_MONS_Tridens_1240.tar
│   │   └── 2023_MONS_Tridens_1250.tar
│   └── 2023-06-16/
│       └── ...
├── metadata/
│   ├── export_manifest.json
│   ├── filter_results.csv
│   ├── origin_mapping.csv
│   ├── bucket_600x900_metadata.csv
│   └── bucket_300x300_metadata.csv
└── results/                                   # populated by planktoshare after inference
    ├── raw/
    │   └── 2023_MONS_Tridens_2023-06-15_1230.csv
    ├── processed/
    │   └── ...
    └── bridge_metadata.json                   # merged bubble filter + origin data
```

---

## 3. `runner.py` — Planktoshare Orchestration

### Environment Management

| Concern | Solution |
|---------|----------|
| Import resolution | `sys.path.insert(0, planktoshare_root)` |
| Working directory | `os.chdir(planktoshare_root)` in `try/finally` |
| Absolute source_dir | Convert relative paths to absolute before chdir |
| macOS OpenMP | `os.environ["KMP_DUPLICATE_LIB_OK"] = "True"` |
| FastAI weights | Resolved as `models/{name}.pth` from cwd by FastAI internals |

### `run_inference(source_dir)`

Calls `conduct_plankton_inference` with 8 positional args matching inference.py's signature:

```python
conduct_plankton_inference(
    SOURCE_BASE_DIR,           # Absolute path to source/
    MODEL_NAME,                # "ResNet50-Detailed" or "OSPAR"
    model_weights,             # Path("Plankton_imager_v01_stage-2_Best")
    TRAIN_DATASET,             # Path("data/DETAILED_merged")
    CRUISE_NAME,               # "2023_MONS_Tridens"
    BATCH_SIZE,                # 128
    DENSITY_CONSTANT,          # 340
    CLASSIFICATION_SUBSAMPLE,  # 100
)
```

planktoshare then internally:
1. Iterates `os.listdir(SOURCE_BASE_DIR)` for date directories
2. For each date dir, iterates `sorted(os.listdir(date_dir))` for `.tar` files
3. Parses timestamp from `tar_name.split('_')[-1].split('.')[0]`
4. Skips timestamps not ending in `0`
5. Extracts tar to temp dir, finds images with `get_image_files`
6. Runs FastAI inference on batch, writes per-timestamp CSV

### `validate_export(manifest)`

Pre-flight checks before committing to inference:

| Check | Severity |
|-------|----------|
| Source directory exists | Error |
| Each date directory exists | Warning |
| Each date directory contains tars | Warning |
| All timestamps end in `0` | Warning (planktoshare will skip) |
| At least one image exported | Error |

### `_merge_bubble_metadata(results_dir)`

After inference, writes `bridge_metadata.json` into planktoshare's results directory combining:
- Bubble filter scores/decisions from the pipeline
- Origin mapping (which export tar → which original tar/filename)
- Bridge configuration snapshot

---

## 4. CLI Interface

| Command | What it does |
|---------|-------------|
| `python -m bridge export --config cfg.yaml` | Pipeline → export tars (no inference) |
| `python -m bridge run --config cfg.yaml` | Full end-to-end: pipeline → export → planktoshare |
| `python -m bridge infer --config cfg.yaml --source-dir path/` | Planktoshare inference on existing dir |
| `python -m bridge report --config cfg.yaml --results-dir path/` | Generate report from existing results |
| `python -m bridge validate --config cfg.yaml --manifest path/` | Validate export without inference |

---

## 5. Design Decisions

- **Reconstruct hierarchy from tar_path provenance**: rather than inventing synthetic dates/timestamps, the exporter parses them from each ImageRecord's `tar_path` field, which preserves the original Pi-10 session structure through the entire pipeline
- **Timestamp snapping**: raw timestamps are floored to `bin_minutes` boundaries because planktoshare only processes timestamps ending in `0`. A record from a `_1234.tar` gets snapped to `1230` — without this, planktoshare silently skips the entire tar
- **Absolute source_dir before chdir**: since the runner changes cwd to planktoshare_root for FastAI, any relative source_dir would break. The runner resolves to absolute before chdir
- **No image resizing at export**: planktoshare applies its own `Resize(300, pad_mode='zeros')` during DataBlock construction. Resizing at the bridge would double-degrade quality
- **Uncompressed tars**: planktoshare uses `tarfile.open(path, 'r')` which auto-detects compression, but since the images are already encoded (PNG/JPEG), tar-level compression would waste CPU for negligible gain
- **Metadata sidecar approach**: bubble scores, origin mappings, and filter decisions are exported as CSV/JSON alongside the tars. This preserves provenance without modifying the tar contents planktoshare expects
- **Two-stage bubble filtering**: the bubble_filter runs at threshold 0.5 during pipeline processing. The bridge's `max_bubble_score` (e.g. 0.3) tightens the cut at export time without re-running the autoencoder, allowing iteration on "clean enough for classification"
