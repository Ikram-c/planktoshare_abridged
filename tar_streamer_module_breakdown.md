# Tar Image Streamer Module — Technical Breakdown

## Module Purpose

This module is the **ingestion layer** of the pipeline — it streams images out of one or more `.tar` archives, decodes them via PIL, applies size filtering and optional colour-mode conversion, and yields structured `ImageRecord` dicts. It is the sole data source for every downstream module (`resolution_grouper`, `bubble_filter`, `zarr_writer`, `ome_converter`) and defines the `ImageRecord` TypedDict that serves as the pipeline's universal data interchange type.

---

## File Map

| File | Role | Key Exports |
|------|------|-------------|
| `models.py` | TypedDict, enums, stream config, concurrency config | `ImageRecord`, `OutputFormat`, `StreamConfig`, `ConcurrencyConfig` |
| `stream.py` | Core streaming engine (sequential + concurrent) | `TarImageStream` |
| `__main__.py` | CLI entry point for standalone inspection | `main()` |
| `__init__.py` | Public API surface | Re-exports all above |

---

## 1. `models.py` — Domain Types & Configuration

### `ImageRecord` (TypedDict)

The pipeline's **universal interchange type** — every downstream module depends on this schema:

| Key | Type | Purpose |
|-----|------|---------|
| `image` | `np.ndarray \| Image.Image` | Pixel data (format depends on `OutputFormat`) |
| `filename` | `str` | Member path within the tar archive |
| `dtype` | `np.dtype` | NumPy dtype of the decoded array |
| `shape` | `tuple[int, ...]` | `(H, W)` or `(H, W, C)` |
| `tar_path` | `str` | Filesystem path to the source archive |

- Downstream modules (`resolution_grouper`, `bubble_filter`, `zarr_writer`) may **mutate** this dict by injecting additional keys (e.g. `bubble_score`, `object_count`) — the TypedDict declaration is a minimum schema, not a strict constraint

### `OutputFormat` (Enum)

| Value | `image` field type | Use case |
|-------|-------------------|----------|
| `PIL` | `PIL.Image.Image` | When downstream needs PIL operations (resize, colour convert) |
| `NUMPY` | `np.ndarray` | Default — direct array access for zarr writing and feature extraction |

### `StreamConfig`

| Field | Default | Purpose |
|-------|---------|---------|
| `tar_paths` | *(required)* | Ordered list of `.tar` / `.tar.gz` / `.tar.bz2` archive paths |
| `output_format` | `NUMPY` | Controls whether `image` is PIL or ndarray |
| `max_images` | `None` | Global yield cap across all archives (early termination) |
| `min_size` | `(128, 128)` | Minimum `(width, height)` — images below are skipped |
| `max_size` | `(16384, 16384)` | Maximum `(width, height)` — images above are skipped |
| `convert_mode` | `None` | PIL mode string (e.g. `"RGB"`, `"L"`) — applied before size check |
| `extensions` | `(".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp")` | Whitelist of file extensions to consider as images |

- `__post_init__` validates that all `tar_paths` exist on the filesystem at construction time
- Size filtering uses PIL's `(width, height)` convention — note this is `(W, H)`, transposed relative to NumPy's `(H, W)` shape convention
- `Image.MAX_IMAGE_PIXELS = None` is set at module scope to disable PIL's decompression bomb guard — necessary for large scientific imagery but a deliberate security tradeoff

### `ConcurrencyConfig`

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | `True` | Toggle concurrent vs sequential decoding |
| `max_workers` | `8` | ThreadPoolExecutor thread count |
| `chunk_size` | `100` | Number of tar members read per concurrent batch |

---

## 2. `stream.py` — Streaming Engine

### Class: `TarImageStream`

#### Dunder Protocol

| Method | Behaviour |
|--------|-----------|
| `__repr__` | Shows archive count, open/closed state, yield count |
| `__len__` | **Full archive scan** — opens every tar, counts image members. Expensive; traverses all archives |
| `__iter__` | Delegates to `stream()` — enables `for record in streamer:` usage |

#### Filtering Pipeline

Three filters are applied in sequence before an image is yielded:

| Filter | Method | Stage | Rejection |
|--------|--------|-------|-----------|
| Extension whitelist | `_is_image(member)` | Pre-extraction | Skips non-image members without reading bytes |
| Size bounds | `_check_size(image)` | Post-decode | Rejects images outside `[min_size, max_size]` |
| Global cap | `_at_limit()` | Pre/post-decode | Terminates stream after `max_images` yields |

**Size check algebra:**

```
pass = (min_w ≤ w ≤ max_w) ∧ (min_h ≤ h ≤ max_h)
```

where `(w, h) = image.size` in PIL convention.

#### `_decode(member_name, raw_bytes)` — Thread-Safe Decoder

| Step | Operation |
|------|-----------|
| 1 | Wrap `raw_bytes` in `io.BytesIO` |
| 2 | `Image.open(buf)` — lazy decode header |
| 3 | `image.load()` — force full pixel decode into memory |
| 4 | Optional `image.convert(mode)` — colour mode conversion |
| 5 | Size check → return `None` if out of bounds |

- This method is designed to be **thread-safe**: it takes raw bytes (already extracted from the tar), creates its own `BytesIO` buffer, and returns an independent `Image` object — no shared mutable state
- Returns `(name, None)` on any failure, letting the caller handle skip counting

#### `_to_record(name, image, tar_path)` — Record Construction

| OutputFormat | `image` field | Notes |
|--------------|---------------|-------|
| `NUMPY` | `np.asarray(image)` | Zero-copy view if PIL backend supports it |
| `PIL` | `image` (PIL object) | `dtype` and `shape` still derived from numpy conversion |

- `np.asarray(image)` is always called to extract `dtype` and `shape`, even in PIL mode — this ensures downstream consumers always have array metadata available

#### Sequential Path: `_stream_sequential()`

```
tar_file.getmembers()
    │
    ├── filter: _is_image(member)
    │
    ▼
tqdm progress bar over filtered members
    │
    ├── _at_limit()? → break
    ├── extractfile(member) → None? → skip
    ├── Image.open(file_obj).load()
    ├── convert_mode?
    ├── _check_size()? → skip
    └── yield _to_record()
```

- Reads and decodes one image at a time
- Uses `tqdm` for per-archive progress display
- `file_obj.close()` in `finally` block ensures tar member handles are released

#### Concurrent Path: `_stream_concurrent()`

```
tar_file members (lazy iterator)
    │
    ▼
islice(member_iter, chunk_size) ──► batch of members
    │
    ├── extractfile(member).read() → raw bytes (serial, GIL-bound)
    │
    ▼
ThreadPoolExecutor(max_workers)
    │
    ├── submit(_decode, name, raw_bytes) × chunk_size
    │
    ▼
as_completed(futures)
    │
    ├── image is not None? → yield _to_record()
    └── image is None? → skip
```

**Concurrency model:**

| Phase | Serial / Parallel | Bottleneck |
|-------|-------------------|------------|
| Tar member extraction (`extractfile.read()`) | Serial | I/O bound — tar is a sequential format, cannot seek in parallel |
| Image decoding (`_decode`) | Parallel (threads) | CPU bound — JPEG/PNG decompression releases the GIL in PIL/libjpeg |
| Record construction (`_to_record`) | Serial (in yield) | Minimal — just `np.asarray` |

- The **chunk-and-dispatch** pattern reads `chunk_size` members sequentially into memory, then fans out decoding across `max_workers` threads
- `as_completed` means yield order is **non-deterministic** within a chunk — records may not arrive in tar-member order. This is acceptable because downstream `resolution_grouper` bins by resolution, not by order
- Thread-based (not process-based) because PIL's JPEG/PNG decoders release the GIL during the C-level decompression, making threads effective for this workload without the overhead of multiprocessing serialisation

#### `stream()` — Top-Level Entry

| Step | Operation |
|------|-----------|
| 1 | Reset counters (`_n_yielded`, `_n_skipped`) |
| 2 | Iterate `tar_paths` in order |
| 3 | `tarfile.open(path, mode="r:*")` — auto-detects compression (gzip, bz2, xz, or none) |
| 4 | Dispatch to `_stream_concurrent` or `_stream_sequential` based on config |
| 5 | Early exit if `_at_limit()` between archives |
| 6 | Log final stats |

- `mode="r:*"` is critical — the `*` means "auto-detect compression", supporting `.tar`, `.tar.gz`, `.tar.bz2`, `.tar.xz` transparently

#### `manifest()` — Dry-Run Inspection

- Opens each archive read-only and collects image member names without decoding
- Returns `dict[tar_path → list[filename]]`
- Used by `__main__.py --manifest-only` for lightweight archive inspection

---

## 3. `__main__.py` — CLI Entry Point

Provides a `python -m tar_streamer` interface with two modes:

| Mode | Flag | Operation |
|------|------|-----------|
| Stream | *(default)* | Decode and log shape/dtype for each image |
| Manifest | `--manifest-only` | List members without decoding (fast) |

### Config Resolution

| Priority | Source | Trigger |
|----------|--------|---------|
| 1 | YAML file | `--config path.yaml` |
| 2 | CLI arguments | Positional `tar_paths` + flags |

- YAML path takes precedence — if `--config` is provided, positional `tar_paths` are ignored
- Validates that at least one config source is provided via `parser.error()`

### CLI Arguments

| Argument | Type | Default | Maps to |
|----------|------|---------|---------|
| `tar_paths` | positional, `nargs="*"` | `[]` | `StreamConfig.tar_paths` |
| `--config` | `str` | `None` | YAML config file path |
| `--format` | choice | `"numpy"` | `StreamConfig.output_format` |
| `--max-images` | `int` | `None` | `StreamConfig.max_images` |
| `--convert-mode` | `str` | `None` | `StreamConfig.convert_mode` |
| `--min-width` / `--min-height` | `int` | `128` | `StreamConfig.min_size` |
| `--max-width` / `--max-height` | `int` | `16384` | `StreamConfig.max_size` |
| `--concurrent` | flag | `False` | `ConcurrencyConfig.enabled` |
| `--workers` | `int` | `8` | `ConcurrencyConfig.max_workers` |
| `--manifest-only` | flag | — | Selects `run_manifest` mode |

---

## 4. Throughput Model

For an archive with `N` images, each of average compressed size `S_c` bytes and decompressed pixel count `P`:

**Sequential throughput:**

```
T_seq = N × (t_extract + t_decode + t_convert)
```

where `t_extract ≈ S_c / BW_disk`, `t_decode ≈ P / R_decode`, and `t_convert` is the optional PIL mode conversion.

**Concurrent throughput:**

```
T_conc ≈ N × t_extract + (N / W) × t_decode
```

where `W = max_workers`. The extraction phase remains serial (tar is sequential), but decoding is parallelised. The speedup approaches:

```
Speedup ≈ (t_extract + t_decode) / (t_extract + t_decode / W)
```

- For CPU-bound JPEG decoding where `t_decode >> t_extract`, this approaches `W`
- For I/O-bound scenarios (network-mounted tar, slow disk), extraction dominates and parallelism has diminishing returns
- The `chunk_size` parameter controls memory pressure: at any moment, up to `chunk_size` raw byte buffers are held in memory simultaneously

---

## 5. Data Flow

```
.tar / .tar.gz / .tar.bz2 archives
         │
         ▼
    tarfile.open(mode="r:*")
         │
         ├── _is_image(member)? → extension filter
         │
         ▼
    extractfile(member) → raw bytes
         │
         ▼
    ┌─ Sequential ─────────────────┐  ┌─ Concurrent ──────────────────────┐
    │ Image.open(buf).load()       │  │ islice(members, chunk_size)       │
    │ convert_mode?                │  │ read raw bytes (serial)           │
    │ _check_size()?               │  │ ThreadPool → _decode() × W       │
    │ _to_record()                 │  │ as_completed → _to_record()      │
    └──────────────────────────────┘  └────────────────────────────────────┘
         │
         ▼
    ImageRecord { image, filename, dtype, shape, tar_path }
         │
         ▼
    resolution_grouper.add(record)
```

---

## 6. Design Observations

- **`ImageRecord` as a mutable TypedDict** is a pragmatic but leaky choice: downstream modules inject keys (`bubble_score`, `object_count`) that aren't declared in the TypedDict, making static type checkers unable to catch schema drift. A `@dataclass` with `Optional` fields or a properly extended TypedDict would tighten the contract
- **`__len__` is O(N) over all archives**: it opens and scans every tar member list, which for large archives can take significant time. Callers should cache the result or use `manifest()` if they need member counts without streaming
- **Non-deterministic yield order in concurrent mode**: `as_completed` returns futures in completion order, not submission order. This is fine for the current pipeline (resolution grouper bins by shape) but would be a problem if any downstream consumer depended on tar-member ordering
- **`Image.MAX_IMAGE_PIXELS = None`** is a global side-effect at import time — it disables PIL's decompression bomb protection for the entire process, not just this module. Necessary for large scientific imagery but worth documenting as a security consideration if the process also handles untrusted inputs
- **No resume/checkpoint mechanism**: if the stream is interrupted, there's no way to resume from where it left off. For very large multi-tar ingestion jobs, a progress checkpoint (e.g. last yielded `(tar_index, member_index)`) would enable restartability
- **Memory ceiling in concurrent mode**: `chunk_size` raw byte buffers are held simultaneously in memory. For a chunk of 100 large TIFFs at ~50MB each, that's ~5GB peak. The `chunk_size` default of 100 is reasonable for typical JPEG imagery but may need tuning for high-resolution TIFF workflows
