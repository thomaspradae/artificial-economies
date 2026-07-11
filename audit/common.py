from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected for tracers.
    np = None  # type: ignore[assignment]


def canonical_float(value: Any, digits: int = 10) -> Any:
    """Return a stable JSON-friendly float representation."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return value

    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return float(f"{number:.{digits}g}")


def _as_numpy(value: Any) -> Any:
    if np is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None


def arr_hash(value: Any) -> str:
    """Hash raw array bytes.

    Shape and dtype are intentionally not included in the hash because they are
    emitted as separate fields and compared directly.
    """

    arr = _as_numpy(value)
    if arr is not None:
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(to_builtin(value), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def arr_stats(value: Any) -> dict[str, Any]:
    """Return deterministic summary fields for an array-like value."""

    arr = _as_numpy(value)
    if arr is None:
        return {
            "hash": arr_hash(value),
            "shape": None,
            "dtype": type(value).__name__,
        }

    stats: dict[str, Any] = {
        "hash": arr_hash(arr),
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
    }
    if arr.size:
        numeric = arr.astype("float64", copy=False)
        stats.update(
            {
                "min": canonical_float(numeric.min()),
                "max": canonical_float(numeric.max()),
                "mean": canonical_float(numeric.mean()),
                "std": canonical_float(numeric.std()),
            }
        )
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            channel_means = []
            channel_mins = []
            channel_maxs = []
            channel_hashes = []
            for channel_index in range(arr.shape[-1]):
                channel = np.ascontiguousarray(arr[..., channel_index])
                channel_numeric = channel.astype("float64", copy=False)
                channel_means.append(canonical_float(channel_numeric.mean()))
                channel_mins.append(canonical_float(channel_numeric.min()))
                channel_maxs.append(canonical_float(channel_numeric.max()))
                channel_hashes.append(arr_hash(channel))
            stats.update(
                {
                    "channel_means": channel_means,
                    "channel_mins": channel_mins,
                    "channel_maxs": channel_maxs,
                    "channel_hashes": channel_hashes,
                }
            )
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None})
    return stats


def frame_to_uint8_hwc(value: Any) -> Any:
    """Normalize a frame-like array to uint8 HWC for byte-level comparison."""

    arr = _as_numpy(value)
    if arr is None:
        return value

    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    numeric = arr.astype("float64", copy=False)
    if numeric.size and numeric.min() >= 0.0 and numeric.max() <= 1.0:
        numeric = numeric * 255.0
    return np.ascontiguousarray(np.rint(np.clip(numeric, 0, 255)).astype(np.uint8))


def frame_stats(value: Any) -> dict[str, Any]:
    """Return stats for a frame after normalizing to uint8 HWC."""

    return arr_stats(frame_to_uint8_hwc(value))


def ram_stats(value: Any) -> dict[str, Any] | None:
    """Return hash/stats and byte values for an ALE RAM dump if available."""

    arr = _as_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
    stats = arr_stats(arr)
    stats["bytes"] = [int(item) for item in arr.tolist()]
    return stats


def to_builtin(value: Any) -> Any:
    """Convert common scientific Python values to JSON-safe builtins."""

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, float):
        return canonical_float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    to_builtin(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, path))
        else:
            flat[path] = value
    return flat
