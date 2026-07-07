from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple
import numpy as np

AAB_RE = re.compile(r"<box>\s*\[\[([^\]]+)\]\]\s*</box>")
RAB_RE = re.compile(r"<rab>\s*\[\[([^\]]+)\]\]\s*</rab>")
SEG_RE = re.compile(r"<seg>\s*([^<]+)\s*</seg>")


def _quantize(value: float, max_value: float, resolution: int) -> int:
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    return int(np.clip(round(value / max_value * (resolution - 1)), 0, resolution - 1))


def _dequantize(value: int, max_value: float, resolution: int) -> float:
    return float(value) / float(resolution - 1) * max_value


def encode_aab(box: Sequence[float], image_size: Tuple[int, int], resolution: int = 1000) -> str:
    """Encode an axis-aligned box as a language-compatible spatial token."""
    w, h = image_size
    x1, y1, x2, y2 = box
    vals = [_quantize(x1, w, resolution), _quantize(y1, h, resolution), _quantize(x2, w, resolution), _quantize(y2, h, resolution)]
    return f"<box>[[{vals[0]},{vals[1]},{vals[2]},{vals[3]}]]</box>"


def decode_aab(token: str, image_size: Tuple[int, int], resolution: int = 1000) -> Tuple[float, float, float, float]:
    match = AAB_RE.search(token)
    if not match:
        raise ValueError(f"Invalid AAB token: {token}")
    vals = [int(v.strip()) for v in match.group(1).split(",")]
    if len(vals) != 4:
        raise ValueError("AAB token must contain four values")
    w, h = image_size
    return (_dequantize(vals[0], w, resolution), _dequantize(vals[1], h, resolution), _dequantize(vals[2], w, resolution), _dequantize(vals[3], h, resolution))


def encode_rab(rbox: Sequence[float], image_size: Tuple[int, int], resolution: int = 1000, angle_bins: int = 360) -> str:
    """Encode a rotation-aware box: (xc, yc, width, height, angle_degrees)."""
    w, h = image_size
    xc, yc, bw, bh, theta = rbox
    vals = [
        _quantize(xc, w, resolution),
        _quantize(yc, h, resolution),
        _quantize(bw, w, resolution),
        _quantize(bh, h, resolution),
        int(round(((theta % 180.0) / 180.0) * (angle_bins - 1))),
    ]
    return f"<rab>[[{vals[0]},{vals[1]},{vals[2]},{vals[3]},{vals[4]}]]</rab>"


def decode_rab(token: str, image_size: Tuple[int, int], resolution: int = 1000, angle_bins: int = 360) -> Tuple[float, float, float, float, float]:
    match = RAB_RE.search(token)
    if not match:
        raise ValueError(f"Invalid RAB token: {token}")
    vals = [int(v.strip()) for v in match.group(1).split(",")]
    if len(vals) != 5:
        raise ValueError("RAB token must contain five values")
    w, h = image_size
    theta = float(vals[4]) / float(angle_bins - 1) * 180.0
    return (_dequantize(vals[0], w, resolution), _dequantize(vals[1], h, resolution), _dequantize(vals[2], w, resolution), _dequantize(vals[3], h, resolution), theta)


def rbox_to_aab(rbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert a rotated rectangle to its enclosing axis-aligned box."""
    xc, yc, w, h, theta = rbox
    rad = math.radians(theta)
    c, s = math.cos(rad), math.sin(rad)
    corners = []
    for dx in (-w / 2.0, w / 2.0):
        for dy in (-h / 2.0, h / 2.0):
            x = xc + dx * c - dy * s
            y = yc + dx * s + dy * c
            corners.append((x, y))
    xs, ys = zip(*corners)
    return min(xs), min(ys), max(xs), max(ys)


def encode_mask_rle(mask: np.ndarray) -> str:
    """Encode a binary mask into compact run-length encoding."""
    flat = mask.astype(np.uint8).flatten(order="C")
    runs: List[str] = []
    last = int(flat[0]) if flat.size else 0
    count = 0
    for val in flat:
        val = int(val)
        if val == last:
            count += 1
        else:
            runs.append(f"{last}:{count}")
            last, count = val, 1
    if flat.size:
        runs.append(f"{last}:{count}")
    return "<seg>" + ",".join(runs) + "</seg>"


def decode_mask_rle(token: str, shape: Tuple[int, int]) -> np.ndarray:
    match = SEG_RE.search(token)
    if not match:
        raise ValueError(f"Invalid segmentation token: {token}")
    values: List[int] = []
    payload = match.group(1).strip()
    if payload:
        for run in payload.split(","):
            val, length = run.split(":")
            values.extend([int(val)] * int(length))
    arr = np.asarray(values, dtype=np.uint8)
    expected = shape[0] * shape[1]
    if arr.size != expected:
        raise ValueError(f"RLE length {arr.size} does not match expected mask size {expected}")
    return arr.reshape(shape)


def mask_to_aab(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
