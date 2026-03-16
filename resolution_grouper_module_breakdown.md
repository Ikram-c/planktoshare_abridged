# Resolution Grouper Module — Technical Breakdown

## Module Purpose

This module partitions a heterogeneous stream of `ImageRecord`s into **resolution-homogeneous buckets** by snapping each image's spatial dimensions to a configurable grid. It acts as the sorting/binning stage between raw tar extraction (`tar_streamer`) and downstream array construction (`ome_converter`), ensuring that every image within a bucket shares identical `(H, W)` dimensions — a prerequisite for stacking into a single zarr array.

---

## File Map

| File | Role | Key Exports |
|------|------|-------------|
| `models.py` | Enums, config, bucket key, running statistics | `SnapMode`, `BucketKey`, `GrouperConfig`, `BucketStats` |
| `grouper.py` | Core binning engine with snap, pad, and constraint logic | `ResolutionGrouper` |
| `__init__.py` | Public API surface | Re-exports all above |

---

## 1. `models.py` — Domain Types

### `SnapMode` (Enum)

Controls the rounding strategy applied when mapping a raw dimension to the snap grid.

| Mode | Mapping `v → v'` | Behaviour |
|------|-------------------|-----------|
| `CEIL` | `⌈v / g⌉ × g` | Round up — guarantees `v' ≥ v`, so no spatial data is lost |
| `FLOOR` | `max(g, ⌊v / g⌋ × g)` | Round down — clamps to at least one grid cell |
| `ROUND` | `max(g, round(v / g) × g)` | Nearest — minimises absolute padding but may truncate |
| `EXACT` | `v` | No snapping — each unique `(H, W)` gets its own bucket |

### `BucketKey`

- Frozen dataclass `(height: int, width: int)` — serves as the hashable dictionary key for bucket lookup
- `__str__` → `"HxW"` format (used in OME-Zarr store naming downstream)
- Derived properties:

| Property | Formula | Use |
|----------|---------|-----|
| `pixel_count` | `H × W` | Sorting buckets by resolution |
| `aspect_ratio` | `W / max(H, 1)` | Diagnostic — guarded against division by zero |

### `GrouperConfig`

| Field | Default | Purpose |
|-------|---------|---------|
| `tile_size` | `300` | Chunk tile edge `T` — also serves as the **fallback snap grid** |
| `snap_grid` | `None` | Explicit grid spacing `g`; if `None`, inherits `tile_size` |
| `snap_mode` | `CEIL` | Rounding strategy (see table above) |
| `min_bucket_size` | `1` | Buckets with fewer images are **dropped** post-ingestion |
| `max_bucket_size` | `None` | If set, buckets are **truncated** (first-N) to this limit |
| `pad_to_bucket` | `True` | Whether `pad_image` zero-pads undersized images to bucket dims |

- `effective_snap_grid` property: resolves the `snap_grid ?? tile_size` fallback, ensuring grid spacing is always tile-aligned by default — this means bucket dimensions are always multiples of the chunk tile, which avoids partial edge chunks in the downstream zarr store

### `BucketStats`

- Mutable running-statistics accumulator, updated on each `add()` call
- Tracks per-bucket:

| Statistic | Update rule |
|-----------|-------------|
| `count` | Increment by 1 |
| `min_original_h` / `max_original_h` | Running min/max of raw image heights |
| `min_original_w` / `max_original_w` | Running min/max of raw image widths |
| `dtypes` | Set union of `str(dtype)` values |
| `channels` | Set union of channel counts (`shape[2]` or `1`) |

- `to_dict()` serialises for summary reporting / YAML dump

---

## 2. `grouper.py` — Binning Engine

### Class: `ResolutionGrouper`

#### Dunder Protocol

The class exposes a dict-like interface over its internal `_buckets: dict[BucketKey, list[ImageRecord]]`:

| Method | Behaviour |
|--------|-----------|
| `__len__` | Number of active buckets |
| `__getitem__(key)` | Direct bucket access by `BucketKey` |
| `__iter__` | Yields `(BucketKey, list[ImageRecord])` pairs |
| `__contains__(key)` | Membership test |
| `__repr__` | Shows tile size, effective snap grid, bucket count |

#### `_snap(value)` — Grid Quantisation

The core mathematical operation. Given a raw dimension `v` and grid spacing `g`:

```
CEIL:   v' = ⌈v/g⌉ · g
FLOOR:  v' = max(g, ⌊v/g⌋ · g)
ROUND:  v' = max(g, ⌊v/g + 0.5⌋ · g)
EXACT:  v' = v
```

- The `max(g, ...)` guard in FLOOR and ROUND prevents dimensions from collapsing to zero for very small images
- CEIL has no guard because `⌈v/g⌉ ≥ 1` for any `v > 0`

**Effect on bucket formation:** snapping creates equivalence classes over `(H, W)` space. Two images with raw dimensions `(h₁, w₁)` and `(h₂, w₂)` land in the same bucket iff:

```
snap(h₁) = snap(h₂)  ∧  snap(w₁) = snap(w₂)
```

#### `compute_key(h, w)` → `BucketKey`

- Applies `_snap` independently to height and width
- The two dimensions are snapped **independently** — no coupling between H and W, meaning non-square images are not forced into square buckets

#### `add(record)`

- Extracts `(h, w)` from `record["shape"][:2]`
- Computes key via `compute_key`
- Appends record to the appropriate bucket (using `defaultdict(list)`)
- Creates or updates `BucketStats` for the key

#### `ingest(records)` — Bulk Pipeline Entry Point

| Phase | Operation |
|-------|-----------|
| 1 | Iterate all records → `add()` each |
| 2 | Apply bucket constraints (`_apply_bucket_constraints`) |
| 3 | Log summary |
| — | Returns `self` for method chaining |

#### `_apply_bucket_constraints()` — Post-Ingestion Filtering

Two-pass constraint enforcement:

| Pass | Condition | Action |
|------|-----------|--------|
| 1 — Minimum | `len(records) < min_bucket_size` | **Delete** bucket and its stats entirely |
| 2 — Maximum | `len(records) > max_bucket_size` | **Truncate** to first `max_bucket_size` records (FIFO order) |

- Minimum-size pruning runs first, so truncation never operates on already-dropped buckets
- Truncation is order-preserving (slice `[:max_size]`), meaning the records retained depend on ingestion order

#### `pad_image(record, key)` — Spatial Normalisation

Zero-pads an image to the bucket's canonical `(H, W)`:

| Condition | Result |
|-----------|--------|
| `(h, w) == (target_h, target_w)` | Return as-is (fast path) |
| `pad_to_bucket == False` | Return as-is (padding disabled) |
| `ndim == 2` | Allocate `zeros(target_h, target_w)`, copy `[:h, :w]` |
| `ndim == 3` | Allocate `zeros(target_h, target_w, c)`, copy `[:h, :w, :]` |

- Padding is always **bottom-right** — the original image is placed at the top-left origin `(0, 0)`
- The padded region is filled with `0` in the image's native dtype

#### `iter_padded(key)` — Lazy Padded Iterator

- Generator yielding `(record, padded_image)` pairs for a given bucket
- Enables downstream consumers to process one image at a time without materialising the entire padded stack

#### Sorted Access Properties

| Property | Sort key | Order | Use case |
|----------|----------|-------|----------|
| `keys_by_count` | `len(bucket)` | Descending | Process largest buckets first |
| `keys_by_resolution` | `H × W` | Ascending | Process smallest-resolution first |

#### `summary()`

- Returns a list of `BucketStats.to_dict()` entries, ordered by descending image count
- Suitable for logging, JSON serialisation, or pipeline reporting

---

## 3. Snap Grid Algebra — Worked Example

Given `tile_size=300`, `snap_grid=None` (so `g=300`), `snap_mode=CEIL`:

| Raw `(H, W)` | Snapped `(H', W')` | Bucket key | Padding added |
|---------------|---------------------|------------|---------------|
| `(480, 640)` | `(600, 900)` | `600x900` | `(120, 260)` |
| `(500, 700)` | `(600, 900)` | `600x900` | `(100, 200)` |
| `(300, 300)` | `(300, 300)` | `300x300` | `(0, 0)` |
| `(1, 1)` | `(300, 300)` | `300x300` | `(299, 299)` |
| `(601, 601)` | `(900, 900)` | `900x900` | `(299, 299)` |

- Images `(480, 640)` and `(500, 700)` are binned together despite different raw sizes
- The `(1, 1)` extreme case still gets a valid bucket because `⌈1/300⌉ = 1`

---

## 4. Data Flow Through the Module

```
ImageRecord stream
       │
       ▼
   add(record)
       │
       ├── shape[:2] → (h, w)
       ├── _snap(h), _snap(w) → BucketKey
       ├── _buckets[key].append(record)
       └── _stats[key].update(record)
       │
       ▼
   _apply_bucket_constraints()
       │
       ├── drop buckets < min_bucket_size
       └── truncate buckets > max_bucket_size
       │
       ▼
   iter_padded(key) / __iter__
       │
       └── pad_image(record, key) → np.ndarray (H', W') or (H', W', C)
              │
              ▼
         ome_converter.convert_bucket(key, records)
```

---

## 5. Design Observations

- **Grid-tile alignment by default**: since `effective_snap_grid` falls back to `tile_size`, bucket dimensions are inherently multiples of the zarr chunk tile — this eliminates partial-tile edge chunks and maximises compression efficiency in the downstream OME-Zarr store
- **Stateless snapping**: `_snap` is a pure function of `(value, grid, mode)` — the same image will always land in the same bucket regardless of ingestion order
- **No merging of nearby buckets**: the grouper does not attempt to merge buckets with similar-but-different snapped dimensions (e.g. `600x900` and `600x1200`). Each unique `BucketKey` is an independent partition. This is a simplicity/correctness tradeoff — merging would reduce bucket count but require more padding
