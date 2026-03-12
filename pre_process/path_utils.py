import contextlib
import cProfile
import json
import logging
import platform
import pstats
import re
from pathlib import Path
from typing import Generator, Optional, Union

import yaml

_logger = logging.getLogger(__name__)

_SYSTEM = platform.system()
_WIN_ABS_RE = re.compile(r"^[A-Za-z]:[/\\]")
_WSL_ABS_RE = re.compile(r"^/mnt/([a-z])(/|$)")
_PATH_KEY_SUFFIXES = ("_path", "_paths", "_dir")


def normalize_path(p: str) -> str:
    if not isinstance(p, str) or not p.strip():
        return p
    if _SYSTEM == "Windows":
        m = _WSL_ABS_RE.match(p)
        if m:
            drive = m.group(1).upper()
            rest = p[m.end():]
            return f"{drive}:\\{rest.replace('/', chr(92))}"
        return p
    if _SYSTEM == "Linux":
        m = _WIN_ABS_RE.match(p)
        if m:
            drive = p[0].lower()
            rest = p[2:].replace("\\", "/").lstrip("/")
            return f"/mnt/{drive}/{rest}"
        return p
    return p


def normalize_paths_in_config(data: dict) -> dict:
    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _normalize_value(k, _walk(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        return obj

    def _normalize_value(key: str, value):
        if not isinstance(key, str):
            return value
        key_lower = key.lower()
        if any(key_lower.endswith(s) for s in _PATH_KEY_SUFFIXES):
            if isinstance(value, str):
                return normalize_path(value)
            if isinstance(value, list):
                return [normalize_path(v) if isinstance(v, str) else v for v in value]
        return value

    return _walk(data)


def load_config(path: Union[str, Path]) -> dict:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return normalize_paths_in_config(data)


def get_valid_config_path(raw_path: str) -> Path:
    p = Path(normalize_path(raw_path))
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Config path is not a file: {p}")
    return p


def safe_write_json(data: dict, path: Union[str, Path]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(dest)
    _logger.debug("Wrote JSON: %s", dest)


@contextlib.contextmanager
def profile_context(
    enabled: bool = False,
) -> Generator[Optional[cProfile.Profile], None, None]:
    if not enabled:
        yield None
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()


def dump_profile(
    profiler: Optional[cProfile.Profile],
    out_path: str,
    top_n: int = 40,
) -> None:
    if profiler is None:
        return
    profiler.dump_stats(out_path)
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    _logger.info("Top %d cumulative profile entries:", top_n)
    stats.print_stats(top_n)