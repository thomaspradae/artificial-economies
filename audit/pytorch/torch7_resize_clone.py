#!/usr/bin/env python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
ResizeFn = Callable[[Array], Array]

CERTIFIED_BYTE_EXACT = False
OUTPUT_SHAPE = (84, 84)


@dataclass(frozen=True)
class ResizeCandidate:
    name: str
    fn: ResizeFn
    family: str
    coordinate: str
    range_mode: str
    cast_rule: str
    border_rule: str
    notes: str = ""


def _as_gray(y: Array) -> Array:
    arr = np.asarray(y)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grayscale fixture, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _quantize(value: Array, rule: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if rule == "trunc":
        out = arr
    elif rule == "floor":
        out = np.floor(arr)
    elif rule == "round_half_up":
        out = np.floor(arr + 0.5)
    elif rule == "rint":
        out = np.rint(arr)
    else:
        raise ValueError(f"unknown cast rule: {rule}")
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def _prepare_range(y: Array, range_mode: str) -> tuple[Array, float]:
    arr = _as_gray(y)
    if range_mode == "uint8":
        return np.ascontiguousarray(arr.astype(np.uint8)), 1.0
    if range_mode == "float255_f32":
        return np.ascontiguousarray(arr.astype(np.float32)), 1.0
    if range_mode == "float255_f64":
        return np.ascontiguousarray(arr.astype(np.float64)), 1.0
    if range_mode == "unit_f32":
        return np.ascontiguousarray(arr.astype(np.float32) / np.float32(255.0)), 255.0
    if range_mode == "unit_f64":
        return np.ascontiguousarray(arr.astype(np.float64) / 255.0), 255.0
    raise ValueError(f"unknown range mode: {range_mode}")


def _cv2_resize(y: Array, interpolation_name: str, range_mode: str, cast_rule: str) -> Array:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on remote env.
        raise RuntimeError("cv2 unavailable") from exc

    interpolation = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
    }[interpolation_name]
    source, multiplier = _prepare_range(y, range_mode)
    resized = cv2.resize(source, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), interpolation=interpolation)
    return _quantize(np.asarray(resized) * multiplier, cast_rule)


def _pil_resize(y: Array, resample_name: str) -> Array:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on remote env.
        raise RuntimeError("PIL unavailable") from exc

    resampling = getattr(Image, "Resampling", Image)
    resample = {
        "bilinear": resampling.BILINEAR,
        "box": resampling.BOX,
        "nearest": resampling.NEAREST,
    }[resample_name]
    source = np.ascontiguousarray(_as_gray(y).astype(np.uint8))
    return np.asarray(Image.fromarray(source).resize((OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]), resample), dtype=np.uint8)


def _coordinate_grid(input_size: int, output_size: int, mode: str) -> Array:
    dst = np.arange(output_size, dtype=np.float64)
    if mode == "half_pixel":
        return (dst + 0.5) * (float(input_size) / float(output_size)) - 0.5
    if mode == "align_corners":
        if output_size == 1:
            return np.zeros(output_size, dtype=np.float64)
        return dst * (float(input_size - 1) / float(output_size - 1))
    if mode == "asymmetric":
        return dst * (float(input_size) / float(output_size))
    if mode == "legacy_center":
        return (dst + 0.5) * (float(input_size - 1) / float(output_size))
    raise ValueError(f"unknown coordinate mode: {mode}")


def _manual_bilinear(y: Array, coordinate: str, range_mode: str, cast_rule: str) -> Array:
    source, multiplier = _prepare_range(y, range_mode)
    source = np.asarray(source, dtype=np.float64)
    in_h, in_w = source.shape
    out_h, out_w = OUTPUT_SHAPE
    weights_y = _bilinear_weights(in_h, out_h, coordinate)
    weights_x = _bilinear_weights(in_w, out_w, coordinate)
    out = (weights_y @ source) @ weights_x.T
    return _quantize(out * multiplier, cast_rule)


_AREA_WEIGHT_CACHE: dict[tuple[int, int, str, bool], Array] = {}


def _area_weights(input_size: int, output_size: int, mode: str, normalize: bool) -> Array:
    key = (input_size, output_size, mode, normalize)
    cached = _AREA_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    scale = float(input_size) / float(output_size)
    weights = np.zeros((output_size, input_size), dtype=np.float64)
    for out_index in range(output_size):
        if mode == "edge_zero":
            start = out_index * scale
            end = (out_index + 1) * scale
            cell_shift = 0.0
        elif mode == "edge_centered":
            start = out_index * scale - 0.5
            end = (out_index + 1) * scale - 0.5
            cell_shift = -0.5
        elif mode == "edge_centered_inclusive":
            start = (out_index + 0.5) * scale - 0.5 - scale / 2.0
            end = (out_index + 0.5) * scale - 0.5 + scale / 2.0
            cell_shift = -0.5
        else:
            raise ValueError(f"unknown area mode: {mode}")

        for in_index in range(input_size):
            left = max(start, in_index + cell_shift)
            right = min(end, in_index + 1.0 + cell_shift)
            overlap = max(0.0, right - left)
            if overlap:
                weights[out_index, in_index] = overlap
        denom = weights[out_index].sum() if normalize else scale
        if denom:
            weights[out_index] /= denom

    _AREA_WEIGHT_CACHE[key] = weights
    return weights


def _manual_area(y: Array, area_mode: str, normalize: bool, range_mode: str, cast_rule: str) -> Array:
    source, multiplier = _prepare_range(y, range_mode)
    source = np.asarray(source, dtype=np.float64)
    in_h, in_w = source.shape
    out_h, out_w = OUTPUT_SHAPE
    weights_y = _area_weights(in_h, out_h, area_mode, normalize)
    weights_x = _area_weights(in_w, out_w, area_mode, normalize)
    resized = (weights_y @ source) @ weights_x.T
    return _quantize(resized * multiplier, cast_rule)


def _weight_lists(weights: Array) -> list[list[tuple[int, float]]]:
    rows: list[list[tuple[int, float]]] = []
    for row in weights:
        entries = []
        for index, value in enumerate(row):
            if value:
                entries.append((index, float(value)))
        rows.append(entries)
    return rows


def _bilinear_weights(input_size: int, output_size: int, coordinate: str) -> Array:
    coords = _coordinate_grid(input_size, output_size, coordinate)
    weights = np.zeros((output_size, input_size), dtype=np.float64)
    for out_index, coord in enumerate(coords):
        coord = min(max(float(coord), 0.0), input_size - 1.0)
        left = int(math.floor(coord))
        right = min(left + 1, input_size - 1)
        frac = coord - left
        weights[out_index, left] += 1.0 - frac
        weights[out_index, right] += frac
    return weights


def _dimension_weights(input_size: int, output_size: int, area_mode: str, normalize: bool, up_coordinate: str) -> Array:
    if input_size > output_size:
        return _area_weights(input_size, output_size, area_mode, normalize)
    return _bilinear_weights(input_size, output_size, up_coordinate)


def _separable_loop_resize_f32(
    y: Array,
    area_mode: str,
    normalize: bool,
    up_coordinate: str,
    range_mode: str,
    cast_rule: str,
) -> Array:
    source, multiplier = _prepare_range(y, range_mode)
    source32 = np.ascontiguousarray(source, dtype=np.float32)
    in_h, in_w = source32.shape
    out_h, out_w = OUTPUT_SHAPE
    weights_y = _weight_lists(_dimension_weights(in_h, out_h, area_mode, normalize, up_coordinate).astype(np.float32))
    weights_x = _weight_lists(_dimension_weights(in_w, out_w, area_mode, normalize, up_coordinate).astype(np.float32))
    out = np.empty((out_h, out_w), dtype=np.float32)
    for oy, y_entries in enumerate(weights_y):
        for ox, x_entries in enumerate(weights_x):
            acc = np.float32(0.0)
            for iy, wy in y_entries:
                wy32 = np.float32(wy)
                for ix, wx in x_entries:
                    acc = np.float32(acc + np.float32(source32[iy, ix] * np.float32(wy32 * np.float32(wx))))
            out[oy, ox] = acc
    return _quantize(out.astype(np.float32) * np.float32(multiplier), cast_rule)


def _separable_matmul_resize_f32(
    y: Array,
    area_mode: str,
    normalize: bool,
    up_coordinate: str,
    range_mode: str,
    cast_rule: str,
) -> Array:
    source, multiplier = _prepare_range(y, range_mode)
    source32 = np.ascontiguousarray(source, dtype=np.float32)
    in_h, in_w = source32.shape
    out_h, out_w = OUTPUT_SHAPE
    weights_y = np.ascontiguousarray(_dimension_weights(in_h, out_h, area_mode, normalize, up_coordinate), dtype=np.float32)
    weights_x = np.ascontiguousarray(_dimension_weights(in_w, out_w, area_mode, normalize, up_coordinate), dtype=np.float32)
    resized = np.matmul(np.matmul(weights_y, source32), weights_x.T)
    return _quantize(np.asarray(resized, dtype=np.float32) * np.float32(multiplier), cast_rule)


def source_coordinate_x(dst_x: int, in_width: int, out_width: int) -> float:
    """Current Stage 2d clone hypothesis for Torch7's horizontal source coordinate.

    This helper is intentionally exposed for diagnostics.  It is not certified
    as the Torch7 contract until Stage 2d passes every oracle fixture.
    """

    if in_width <= out_width:
        if out_width == 1:
            return 0.0
        return float(np.float32(np.float32(dst_x) * np.float32(float(in_width - 1) / float(out_width - 1))))
    scale = np.float32(float(in_width) / float(out_width))
    return float(np.float32(np.float32(dst_x + 0.5) * scale - np.float32(0.5)))


def source_coordinate_y(dst_y: int, in_height: int, out_height: int) -> float:
    """Current Stage 2d clone hypothesis for Torch7's vertical source center."""

    if in_height <= out_height:
        if out_height == 1:
            return 0.0
        return float(np.float32(np.float32(dst_y) * np.float32(float(in_height - 1) / float(out_height - 1))))
    scale = np.float32(float(in_height) / float(out_height))
    return float(np.float32(np.float32(dst_y + 0.5) * scale - np.float32(0.5)))


def interpolate_pixel(
    top_left: float,
    top_right: float,
    bottom_left: float,
    bottom_right: float,
    weight_y: float,
    weight_x: float,
) -> np.float32:
    """Bilinear interpolation helper using explicit float32 accumulation."""

    wx = np.float32(weight_x)
    wy = np.float32(weight_y)
    top = np.float32(np.float32(top_left) * np.float32(1.0 - wx) + np.float32(top_right) * wx)
    bottom = np.float32(np.float32(bottom_left) * np.float32(1.0 - wx) + np.float32(bottom_right) * wx)
    return np.float32(top * np.float32(1.0 - wy) + bottom * wy)


def _dimension_weights_for_shape(input_size: int, output_size: int) -> list[list[tuple[int, float]]]:
    if input_size > output_size:
        weights = _area_weights(input_size, output_size, "edge_zero", False)
    else:
        weights = _bilinear_weights(input_size, output_size, "align_corners")
    return _weight_lists(np.asarray(weights, dtype=np.float64))


def resize_torch7_exact(
    image: np.ndarray,
    out_height: int = 84,
    out_width: int = 84,
) -> np.ndarray:
    """Attempted Python clone of Torch7 ``image.scale(..., 'bilinear')``.

    Stage 2d validates this function against the installed Torch7 oracle.  The
    implementation is deliberately loop-based and float32-heavy so failures can
    be inspected at the coordinate/weight/cast boundary.  Do not treat this as
    a certified DeepMind preprocessor while ``CERTIFIED_BYTE_EXACT`` is false.
    """

    source_uint8 = _as_gray(image)
    source = np.ascontiguousarray(source_uint8.astype(np.float32) / np.float32(255.0))
    in_height, in_width = source.shape
    y_weights = _dimension_weights_for_shape(in_height, out_height)
    x_weights = _dimension_weights_for_shape(in_width, out_width)
    output = np.empty((out_height, out_width), dtype=np.float32)

    for out_y, y_entries in enumerate(y_weights):
        for out_x, x_entries in enumerate(x_weights):
            acc = 0.0
            for in_y, weight_y in y_entries:
                for in_x, weight_x in x_entries:
                    acc += float(source[in_y, in_x]) * float(weight_y) * float(weight_x)
            output[out_y, out_x] = np.float32(acc)

    scaled = np.asarray(output * np.float32(255.0), dtype=np.float32)
    return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))


def candidate_registry() -> list[ResizeCandidate]:
    candidates: list[ResizeCandidate] = []

    for interpolation in ("area", "linear", "nearest", "cubic"):
        for range_mode in ("uint8", "float255_f32", "unit_f32"):
            for cast_rule in ("trunc", "round_half_up"):
                candidates.append(
                    ResizeCandidate(
                        name=f"cv2_{interpolation}_{range_mode}_{cast_rule}",
                        fn=lambda y, interpolation=interpolation, range_mode=range_mode, cast_rule=cast_rule: _cv2_resize(
                            y, interpolation, range_mode, cast_rule
                        ),
                        family=f"opencv_{interpolation}",
                        coordinate="opencv_internal",
                        range_mode=range_mode,
                        cast_rule=cast_rule,
                        border_rule="opencv_internal",
                    )
                )

    for resample in ("bilinear", "box", "nearest"):
        candidates.append(
            ResizeCandidate(
                name=f"pil_{resample}_uint8_internal",
                fn=lambda y, resample=resample: _pil_resize(y, resample),
                family=f"pil_{resample}",
                coordinate="pil_internal",
                range_mode="uint8",
                cast_rule="pil_internal",
                border_rule="pil_internal",
            )
        )

    for coordinate in ("half_pixel", "align_corners", "asymmetric", "legacy_center"):
        for range_mode in ("float255_f32", "float255_f64", "unit_f32", "unit_f64"):
            for cast_rule in ("trunc", "floor", "round_half_up", "rint"):
                candidates.append(
                    ResizeCandidate(
                        name=f"manual_bilinear_{coordinate}_{range_mode}_{cast_rule}",
                        fn=lambda y, coordinate=coordinate, range_mode=range_mode, cast_rule=cast_rule: _manual_bilinear(
                            y, coordinate, range_mode, cast_rule
                        ),
                        family="manual_bilinear",
                        coordinate=coordinate,
                        range_mode=range_mode,
                        cast_rule=cast_rule,
                        border_rule="clamp",
                    )
                )

    for area_mode in ("edge_zero", "edge_centered", "edge_centered_inclusive"):
        for normalize in (False, True):
            norm_name = "normsum" if normalize else "normscale"
            for range_mode in ("float255_f32", "float255_f64", "unit_f32", "unit_f64"):
                for cast_rule in ("trunc", "floor", "round_half_up", "rint"):
                    candidates.append(
                        ResizeCandidate(
                            name=f"manual_area_{area_mode}_{norm_name}_{range_mode}_{cast_rule}",
                            fn=lambda y, area_mode=area_mode, normalize=normalize, range_mode=range_mode, cast_rule=cast_rule: _manual_area(
                                y, area_mode, normalize, range_mode, cast_rule
                            ),
                            family="manual_area",
                            coordinate=area_mode,
                            range_mode=range_mode,
                            cast_rule=cast_rule,
                            border_rule="weighted_overlap",
                            notes=f"normalize={'sum of visible weights' if normalize else 'source/target scale'}",
                        )
                    )

    for up_coordinate in ("half_pixel", "align_corners", "asymmetric"):
        for range_mode in ("unit_f32", "float255_f32"):
            for cast_rule in ("trunc", "floor", "round_half_up"):
                candidates.append(
                    ResizeCandidate(
                        name=f"matmul_f32_mixed_area_edge_centered_inclusive_normsum_up_{up_coordinate}_{range_mode}_{cast_rule}",
                        fn=lambda y, up_coordinate=up_coordinate, range_mode=range_mode, cast_rule=cast_rule: _separable_matmul_resize_f32(
                            y, "edge_centered_inclusive", True, up_coordinate, range_mode, cast_rule
                        ),
                        family="matmul_f32_mixed_area_bilinear",
                        coordinate=f"down=edge_centered_inclusive;up={up_coordinate}",
                        range_mode=range_mode,
                        cast_rule=cast_rule,
                        border_rule="weighted_overlap_or_clamp",
                        notes="normalize=sum of visible weights; vectorized float32 separable accumulation; targeted Stage 2c refinement",
                    )
                )

    return candidates


def resize_210x160_to_84x84_deepmind_y(y_uint8: Array) -> Array:
    """Best current non-certified Torch7 resize candidate.

    Stage 2c must set CERTIFIED_BYTE_EXACT to True before this function is used
    as sacred preprocessing. Until then, it is intentionally just the closest
    documented candidate, not a claimed DeepMind clone.
    """

    arr = _as_gray(y_uint8)
    if arr.shape != (210, 160):
        raise ValueError(f"expected 210x160 luminance frame, got {arr.shape}")
    return resize_torch7_exact(arr, 84, 84)
