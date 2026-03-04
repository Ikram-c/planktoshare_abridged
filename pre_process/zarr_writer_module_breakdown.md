# Zarr Writer Module — Technical Breakdown

## Module Purpose

This module materialises resolution-grouped image buckets into **chunked, compressed zarr v2 stores** on local disk. It is the raw array-writing layer that sits beneath `ome_converter` — handling dtype resolution, channel normalisation, spatial padding, tiling, and Blosc compression. Where `ome_converter` adds OME-Zarr 0.4 metadata, pyramids, and ZIP archival on top, this module provides the core array creation and write logic that both standalone and OME-wrapped workflows depend on.

---

## File Map

| File | Role | Key Exports |
|------|------|-------------|
| `models.py` | Compression enums, writer config, output manifest | `CompressionCodec`, `ShuffleMode`, `ZarrWriterConfig`, `ArrayManifest` |
| `writer.py` | Core writer — array construction, padding, compression | `ZarrWriter` |
| `__init__.py` | Public API surface | Re-exports all above |

---

## 1. `models.py` — Configuration & Output Types

### `CompressionCodec` (Enum)

| Value | Blosc codec string | Characteristics |
|-------|---------------------|-----------------|
| `BLOSC_ZSTD` | `"zstd"` | High ratio, moderate speed — good default for archival |
| `BLOSC_LZ4` | `"lz4"` | Fast compression/decompression, lower ratio |
| `BLOSC_LZ4HC` | `"lz4hc"` | LZ4 high-compression variant — slower encode, same fast decode |
| `ZSTD` | `"zstd"` | Alias for `BLOSC_ZSTD` (both map to `"zstd"`) |

### `ShuffleMode` (Enum)

Controls byte-level reordering before compression to improve codec efficiency:

| Mode | Blosc constant | Operation |
|------|----------------|-----------|
| `NOSHUFFLE` | `Blosc.NOSHUFFLE` | No reordering — raw bytes to codec |
| `SHUFFLE` | `Blosc.SHUFFLE` | Byte-shuffle: groups the k-th byte of every element together. For `dtype=uint16`, all MSBs are contiguous, then all LSBs — dramatically improves compression of smooth data |
| `BITSHUFFLE` | `Blosc.BITSHUFFLE` | Bit-shuffle: transposes at the bit level across elements. Even more effective than byte-shuffle for data with limited dynamic range |

**Mathematical basis of shuffle:** Given `N` elements each of `B` bytes, byte-shuffle performs the transposition:

```
byte[i][j] → byte[j][i]     where i ∈ [0, N), j ∈ [0, B)
```

This converts a stream like `[a₀a₁, b₀b₁, c₀c₁, ...]` into `[a₀b₀c₀..., a₁b₁c₁...]`, grouping bytes of similar magnitude and creating long runs of near-identical values for the downstream entropy coder.

### `ZarrWriterConfig`

| Field | Default | Purpose |
|-------|---------|---------|
| `output_dir` | `"output"` | Root directory for `.zarr` stores |
| `tile_size` | `300` | Chunk tile edge `T` for spatial dimensions |
| `compression_codec` | `BLOSC_ZSTD` | Blosc codec selection |
| `compression_level` | `5` | Codec effort level (1–9 for zstd; higher = slower + smaller) |
| `shuffle` | `BITSHUFFLE` | Pre-compression byte reordering strategy |
| `zarr_format` | `2` | Zarr spec version (v2 — the current stable format) |
| `dimension_separator` | `"/"` | Chunk key separator in store (path-style) |
| `shard_images` | `10` | Sharding hint for image (Z) dimension — reserved for future use |
| `shard_tiles` | `10` | Sharding hint for tile dimensions — reserved for future use |

- `shard_images` and `shard_tiles` are declared in config but **not yet consumed** by the writer — these are forward-looking fields for zarr v3 sharding support

### `ArrayManifest`

| Field | Type | Purpose |
|-------|------|---------|
| `bucket` | `str` | String representation of `BucketKey` (e.g. `"600x900"`) |
| `store_path` | `str` | Filesystem path to the `.zarr` directory store |
| `shape` | `tuple` | Full array shape `(N, H, W)` or `(N, H, W, C)` |
| `chunks` | `tuple` | Chunk shape `(1, T, T)` or `(1, T, T, C)` |
| `dtype` | `str` | Resolved dtype string |
| `n_images` | `int` | Number of images written |
| `compression` | `str` | Human-readable compression descriptor |

---

## 2. `writer.py` — Core Write Engine

### Module-Level Mappings

Two dictionaries translate enum values into `numcodecs.Blosc` constructor arguments:

| Map | From | To |
|-----|------|----|
| `CODEC_MAP` | `CompressionCodec` → `str` | Blosc `cname` parameter |
| `SHUFFLE_MAP` | `ShuffleMode` → `int` | Blosc shuffle constant |

These maps are also imported by `ome_converter.converter` for compressor construction, making this module the single source of truth for compression configuration.

### Class: `ZarrWriter`

#### `_build_compressor()` → `Blosc`

Constructs the compressor from config:

```
Blosc(cname=CODEC_MAP[codec], clevel=level, shuffle=SHUFFLE_MAP[mode])
```

- A single compressor instance is shared across all arrays in a store
- Blosc is a meta-compressor: it applies shuffle → split into blocks → compress each block with the selected codec → reassemble

#### `_resolve_dtype(records)` — Type Promotion

| Condition | Strategy |
|-----------|----------|
| All records share one dtype | Return that dtype directly |
| Mixed dtypes present | **Upcast to widest** by `itemsize` with a warning |

Formally:

```
dtype* = argmax_{d ∈ {rec["dtype"] : rec ∈ records}}( np.dtype(d).itemsize )
```

- This is a max-precision rather than safe-cast strategy — it preserves the most information but may silently widen narrow types (e.g. `uint8` images in a bucket with one `uint16` outlier all get stored as `uint16`)

#### `_resolve_channels(records)` — Channel Harmonisation

- Extracts channel count per record: `shape[2]` if 3D, else `1`
- If uniform → return directly
- If mixed → `max(channels)` with a warning
- Downstream `_pad_image` will expand narrower images to match

#### `_compute_shapes(key, n_images, n_channels)` — Array Geometry

| n_channels | Full shape | Chunk shape |
|------------|------------|-------------|
| `1` | `(N, H, W)` | `(1, T, T)` |
| `> 1` | `(N, H, W, C)` | `(1, T, T, C)` |

- The Z (image-index) dimension is always chunked as `1` — each image is an independent chunk slice, enabling random access to individual images without decompressing neighbours
- Spatial dimensions are tiled at `T × T`
- Channel dimension is **not** chunked independently — the full channel vector is always read together, which is optimal for per-pixel operations but suboptimal for single-channel extraction

#### `_pad_image(image, target_h, target_w, target_c, target_dtype)` — Spatial & Channel Normalisation

Transforms an arbitrarily-shaped input image into the bucket's canonical geometry:

| Input state | Transformation |
|-------------|----------------|
| 2D + `target_c > 1` | Replicate across channels: `np.stack([image] * C, axis=-1)` |
| 3D + `shape[2] < target_c` | Zero-pad channel axis: concatenate `zeros(H, W, C − c)` |
| `target_c == 1` + 3D input | Squeeze: `image[:, :, 0]` |
| Spatial dims `< (H, W)` | Zero-pad bottom-right |
| dtype mismatch | Cast via `image.astype(target_dtype)` |

- Padding is always **zero-fill, bottom-right** — maintains top-left origin alignment
- dtype casting happens **before** spatial padding to ensure the zero-fill uses the correct type

#### `write_bucket(key, records)` — Main Write Path

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | Resolve dtype | Max-precision promotion |
| 2 | Resolve channels | Max-channel harmonisation |
| 3 | Compute shapes | `(N, H, W[, C])` + chunk geometry |
| 4 | Build compressor | `Blosc(cname, clevel, shuffle)` |
| 5 | Create output directory | `os.makedirs(exist_ok=True)` |
| 6 | Open zarr v2 group | `LocalStore` → `zarr.open_group(mode="w")` |
| 7 | Create array `"0"` | Full shape, chunked, compressed, `fill_value=0` |
| 8 | Iterate records | Pad each → write to `arr[i]` |
| 9 | Build `ArrayManifest` | Capture all metadata |
| 10 | Append to `_manifests` | Accumulate across calls |

- No atomic rename pattern here (unlike `ome_converter`) — the store is written in-place with `mode="w"`, meaning a crash mid-write leaves a partial store on disk
- The compression descriptor string follows the format `"{codec}_clevel{N}_{shuffle}"` for human readability in manifests and logs

#### `write_all(buckets)` / `manifests` Property

- `write_all` iterates all buckets sequentially, accumulating manifests
- `manifests` returns a defensive copy of the internal list

---

## 3. Compression Pipeline Detail

The full data path from NumPy array to on-disk bytes:

```
np.ndarray slice (1, T, T[, C])
         │
         ▼
    Blosc.BITSHUFFLE (default)
    ┌─────────────────────────────┐
    │ Transpose bits across the   │
    │ T×T×[C] element block:      │
    │   bit[elem_i][bit_j] →      │
    │   bit[bit_j][elem_i]        │
    └─────────────────────────────┘
         │
         ▼
    Split into Blosc internal blocks
         │
         ▼
    Zstandard (default, clevel=5)
    ┌─────────────────────────────┐
    │ LZ77-style matching +       │
    │ Finite State Entropy (tANS) │
    │ encoding per block          │
    └─────────────────────────────┘
         │
         ▼
    Reassemble → chunk file on disk
```

**Why bitshuffle + zstd is a strong default for microscopy/camera data:**

- Camera sensor data has limited effective dynamic range relative to dtype width (e.g. 12-bit values in 16-bit containers), meaning upper bits are often zero or slowly varying
- Bitshuffle concentrates the entropy into fewer byte streams, giving zstd's entropy coder long runs of predictable data
- Zstd at `clevel=5` balances ratio vs throughput well for batch writes

---

## 4. Shape & Chunking Algebra

For a bucket with `N` images of resolution `H × W`, channel count `C`, and tile size `T`:

| Dimension | Array extent | Chunk extent | # Chunks along this axis | Rationale |
|-----------|-------------|--------------|--------------------------|-----------|
| Z (images) | `N` | `1` | `N` | Per-image random access |
| Y (rows) | `H` | `T` | `⌈H/T⌉` | Spatial tiling |
| X (cols) | `W` | `T` | `⌈W/T⌉` | Spatial tiling |
| C (channels) | `C` | `C` | `1` | Read full pixel vector |

Total chunks per bucket:

```
N_chunks = N × ⌈H/T⌉ × ⌈W/T⌉
```

Total chunk files on disk (zarr v2 with `/` separator):

```
store/0/{z}/{y}/{x}    for each (z, y, x) chunk index
```

---

## 5. Relationship to `ome_converter`

| Concern | `zarr_writer` | `ome_converter` |
|---------|---------------|-----------------|
| Array creation | ✓ | ✓ (duplicated logic) |
| Padding | ✓ | ✓ (duplicated logic) |
| Compression | Defines `CODEC_MAP`, `SHUFFLE_MAP` | Imports from `zarr_writer.writer` |
| Pyramids | ✗ | ✓ |
| OME metadata | ✗ | ✓ |
| ZIP archival | ✗ | ✓ |
| Atomic write | ✗ | ✓ (`.tmp` + rename) |

- `ome_converter` depends on `zarr_writer.models.ZarrWriterConfig` for compression settings and imports `CODEC_MAP`/`SHUFFLE_MAP` for compressor construction
- The array creation and padding logic is **duplicated** between the two modules — `ome_converter` reimplements `_pad_image`, `_resolve_dtype`, `_resolve_channels`, and the `create_array` call rather than delegating to `ZarrWriter`

---

## 6. Design Observations

- **`shard_images` and `shard_tiles` are forward-looking**: these config fields anticipate zarr v3's sharding extension, which would allow multiple chunks to be packed into a single file — reducing filesystem inode pressure for large stores. They're plumbed through config but have no effect on the current v2 write path
- **No atomic write safety**: unlike `ome_converter`'s `.tmp` + rename pattern, `ZarrWriter` writes directly to the final path. A crash or interrupt mid-write will leave a corrupt partial store — worth adding the same atomic pattern if this module is used standalone
- **Duplicated logic with `ome_converter`**: the padding, dtype resolution, channel resolution, and array creation code is nearly identical between the two modules. Extracting these into shared utilities (or having `ome_converter` compose a `ZarrWriter` for its level-0 array) would reduce maintenance surface
- **Per-image chunking (Z=1)** is ideal for append-style writes and single-image reads, but suboptimal for operations that slice across the Z axis (e.g. computing a mean image across the stack). The `shard_images` config hints at a future where multiple images share a shard for better sequential-read throughput
