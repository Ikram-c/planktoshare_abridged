# OME-Zarr Converter Module — Technical Breakdown

## Module Purpose

This module converts resolution-grouped image buckets (produced upstream by `resolution_grouper` and `tar_streamer`) into **OME-Zarr 0.4**-compliant stores with optional multi-resolution pyramids and ZIP archival. It is the final materialisation stage of the pipeline, producing self-describing, cloud-ready imaging datasets.

---

## File Map

| File | Role | Key Exports |
|------|------|-------------|
| `models.py` | Immutable config + output manifest dataclasses | `OmeConverterConfig`, `OmeManifest` |
| `metadata.py` | OME-Zarr 0.4 multiscale metadata builder | `build_multiscales_attrs`, `build_pipeline_metadata` |
| `converter.py` | Orchestrator — array construction, pyramids, ZIP | `OmeZarrConverter` |
| `__init__.py` | Public API surface | Re-exports all above |

---

## 1. `models.py` — Configuration & Manifest

### `OmeConverterConfig`

- Frozen dataclass holding conversion parameters
- `from_dict(data)` factory reads from a nested YAML structure, falling back to top-level keys if `ome_converter` section is absent

| Field | Default | Purpose |
|-------|---------|---------|
| `output_dir` | `"output_ome"` | Root directory for `.ome.zarr` stores |
| `pixel_size_um` | `0.36` | Physical pixel pitch in µm — used as the base scale factor `s₀` |
| `generate_pyramid` | `True` | Toggle multi-resolution generation |
| `pyramid_levels` | `3` | Total number of resolution levels `L` (including full-res) |
| `pyramid_downsample_factor` | `2` | Integer downsample factor `f` per level |
| `zip_store` | `True` | Whether to produce a `.ome.zarr.zip` archive |
| `tile_size` | `300` | Chunk tile edge length `T` (spatial dimensions only) |

### `OmeManifest`

- Mutable dataclass returned after each bucket conversion
- Captures the output path, ZIP path, image count, pyramid depth, and pixel size for downstream consumers (e.g. Azure Blob upload)

---

## 2. `metadata.py` — OME-Zarr 0.4 Coordinate Metadata

This file constructs the `multiscales` attribute tree required by the [OME-Zarr 0.4 spec](https://ngff.openmicroscopy.org/0.4/). It has a **dual-path** design: if `ome-zarr-models` is installed, it uses Pydantic-backed typed models (`Axis`, `VectorScale`, etc.); otherwise it falls back to plain `dict` construction.

### Axes Definitions

- `build_axes_zyx()` → 3D stack: `(z, y, x)`, all `"space"` in `"micrometer"`
- `build_axes_zyxc()` → 3D + channel: `(z, y, x, c)`, with `c` typed as `"channel"`

| Axis | Type | Unit | Notes |
|------|------|------|-------|
| `z` | `space` | `micrometer` | Image index within bucket (stack dimension) |
| `y` | `space` | `micrometer` | Row dimension |
| `x` | `space` | `micrometer` | Column dimension |
| `c` | `channel` | — | Present only when `n_channels > 1` |

### `build_datasets()` — Scale & Translation Algebra

For each pyramid level `l ∈ [0, L)`:

- **Downsample factor at level `l`:**

  ```
  fₗ = f ^ l
  ```

- **Physical pixel size at level `l`:**

  ```
  sₗ = s₀ × fₗ
  ```

  where `s₀ = pixel_size_um`.

- **Translation offset at level `l`** (half-pixel shift to maintain pixel-centre alignment):

  ```
  tₗ = (fₗ − 1) × s₀ / 2
  ```

  This ensures that a pixel at level `l` is centred on the corresponding region in level 0, not aligned to its top-left corner.

- **Coordinate transform vectors by dimensionality:**

| ndim | Scale vector | Translation vector |
|------|--------------|--------------------|
| 3 (ZYX) | `[1.0, sₗ, sₗ]` | `[0.0, tₗ, tₗ]` |
| 4 (ZYXC) | `[1.0, sₗ, sₗ, 1.0]` | `[0.0, tₗ, tₗ, 0.0]` |

- Level 0 carries only a `scale` transform; levels `l > 0` carry both `scale` and `translation`

### `build_multiscales_attrs()`

- Assembles the top-level `{"multiscales": [...]}` dict
- Selects ZYX or ZYXC axes based on `ndim`
- Wraps datasets from `build_datasets()` into a single `Multiscale` entry with `version: "0.4"`

### `build_pipeline_metadata()`

- Attaches non-OME provenance metadata to the root group attrs:
  - `pipeline_metadata`: resolution group key, image count, tile size, compression string, source dtype
  - `image_metadata`: per-image list with original filename, shape, bubble detection score, object count, tar source

---

## 3. `converter.py` — Core Conversion Engine

### Class: `OmeZarrConverter`

#### Construction

- Receives both `OmeConverterConfig` and `ZarrWriterConfig` (the latter provides compression codec/level/shuffle settings)
- Maintains an internal `_manifests` accumulator across `convert_bucket` calls

#### `_build_compressor()`

- Reads `CODEC_MAP` and `SHUFFLE_MAP` from `zarr_writer.writer` to translate enum values into `numcodecs.Blosc` parameters
- Returns a single `Blosc` compressor instance shared across all arrays in a store

#### `_resolve_dtype(records)`

- Collects the set of unique dtypes across all `ImageRecord`s
- If homogeneous → returns that dtype directly
- If heterogeneous → selects the **widest** dtype by `itemsize` (max-precision promotion):

  ```
  dtype* = argmax_{d ∈ dtypes}( itemsize(d) )
  ```

#### `_resolve_channels(records)`

- Extracts channel count per record:
  - `shape[2]` if 3D (H, W, C)
  - `1` if 2D (H, W)
- Returns `max(channels)` — the bucket's array will accommodate the widest image

#### `_pad_image(image, h, w, c, dtype)`

Normalises an arbitrary input image to the bucket's canonical `(h, w)` or `(h, w, c)` shape:

| Input condition | Transformation |
|----------------|----------------|
| 2D, `c > 1` | Stack into `(H, W, c)` by repeating across channels |
| 3D, `shape[2] < c` | Zero-pad along channel axis |
| `c == 1`, input is 3D | Squeeze to `(H, W)` by taking `[:, :, 0]` |
| Spatial dims `< (h, w)` | Zero-pad bottom and right edges |

- All operations cast to target `dtype` first

#### `_generate_pyramid(data, dtype)`

- Iterative multi-resolution construction over `L − 1` additional levels
- At each step applies `skimage.transform.downscale_local_mean`:

  ```
  Iₗ₊₁ = downscale_local_mean(Iₗ, factors)
  ```

  where:

| ndim | factors tuple |
|------|---------------|
| 2 | `(f, f)` |
| 3 (Z, Y, X) | `(1, f, f)` — preserves Z (image index) |
| 4 (Z, Y, X, C) | `(1, f, f, 1)` — preserves Z and C |

- `downscale_local_mean` computes the arithmetic mean over each non-overlapping `f × f` spatial block:

  ```
  Iₗ₊₁[z, j, k] = (1/f²) Σ_{Δy,Δx ∈ [0,f)} Iₗ[z, j·f+Δy, k·f+Δx]
  ```

- Each level is cast back to `dtype` to prevent float accumulation drift

#### `convert_bucket(key, records)` — Main Orchestration

**Step-by-step data flow:**

| Step | Operation | Output |
|------|-----------|--------|
| 1 | Resolve dtype and channels | `dtype`, `n_channels` |
| 2 | Determine array shape | `(N, H, W)` or `(N, H, W, C)` |
| 3 | Determine chunk shape | `(1, T, T)` or `(1, T, T, C)` |
| 4 | Create temporary `LocalStore` + zarr v2 group | Empty root group at `*.tmp` |
| 5 | Create level-0 array `"0"` with Blosc compressor | Chunked zarr array |
| 6 | Iterate records → pad → write slice `arr[i]` | Populated level 0 |
| 7 | If pyramids enabled: read full level-0 back, generate pyramid | `list[np.ndarray]` of `L` levels |
| 8 | Write levels `1..L−1` as separate arrays `"1"`, `"2"`, … | Multi-resolution store |
| 9 | Attach `multiscales` attrs (OME-Zarr 0.4) | Spec-compliant metadata |
| 10 | Attach `pipeline_metadata` + `image_metadata` attrs | Provenance tracking |
| 11 | Optionally ZIP the directory (`ZIP_STORED`, no compression) | `.ome.zarr.zip` |
| 12 | Atomic rename `.tmp` → final path | Safe-publish |
| 13 | Build and cache `OmeManifest` | Return value |

- **Chunk shape for pyramid sub-levels** is clamped: `min(chunk_dim, array_dim)` per axis, preventing chunks larger than the array itself at coarse levels
- **ZIP uses `ZIP_STORED`** (no additional compression) because Blosc already compresses chunk data — double-compression would waste CPU and potentially increase file size
- **Atomic rename** (`os.rename`) ensures consumers never see a partially-written store

#### `convert_all(buckets)`

- Iterates all `BucketKey → list[ImageRecord]` pairs and calls `convert_bucket` for each
- Returns accumulated manifests

---

## 4. Data Shape Summary

For a bucket with `N` images of resolution `H × W`, channel count `C`, tile size `T`, downsample factor `f`, and `L` pyramid levels:

| Level `l` | Array name | Shape (C=1) | Shape (C>1) | Chunk shape (C=1) | Chunk shape (C>1) |
|-----------|------------|-------------|-------------|--------------------|--------------------|
| 0 | `"0"` | `(N, H, W)` | `(N, H, W, C)` | `(1, T, T)` | `(1, T, T, C)` |
| 1 | `"1"` | `(N, H/f, W/f)` | `(N, H/f, W/f, C)` | clamped | clamped |
| 2 | `"2"` | `(N, H/f², W/f²)` | `(N, H/f², W/f², C)` | clamped | clamped |
| … | … | … | … | … | … |

---

## 5. Dependency Graph

```
tar_streamer.models.ImageRecord ──┐
                                  ├──► converter.py ──► OmeManifest
resolution_grouper.models.BucketKey ┘        │
                                             ├── metadata.py (OME attrs)
zarr_writer.models.ZarrWriterConfig ─────────┘
zarr_writer.writer.CODEC_MAP / SHUFFLE_MAP ──┘

Optional: ome_zarr_models.v04 (typed Pydantic metadata)
Core: zarr, numcodecs, numpy, skimage, zipfile
```
