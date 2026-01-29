#!/usr/bin/env python3
"""
Feather image edges and blend into background.
Removes hard edges via a smooth alpha falloff so the graphic integrates seamlessly.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def feather_edges(
    input_path: str | Path,
    output_path: str | Path | None = None,
    feather_fraction: float = 0.12,
) -> Path:
    """
    Apply feathered edges so the image blends seamlessly into any background.

    Args:
        input_path: Path to source image.
        output_path: Where to save. If None, saves next to input with '-blended' suffix.
        feather_fraction: How much of width/height to fade (0–1). Default 0.12.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}-blended{input_path.suffix}"
    output_path = Path(output_path)

    img = Image.open(input_path).convert("RGBA")
    w, h = img.size
    rgb = np.array(img)
    alpha = rgb[:, :, 3].astype(np.float64) / 255.0

    # Distance from each edge (0 at edge, 1 at center of edge axis)
    fx = np.linspace(0, 1, w)
    fy = np.linspace(0, 1, h)
    # 0 at edges, 1 in center
    left = fx
    right = 1 - fx
    top = fy
    bottom = 1 - fy
    # Take min across width/height => 0 near any edge
    fade_x = np.minimum(left, right)
    fade_y = np.minimum(top, bottom)
    # 2D mask: 0 at corners/edges, 1 in center
    fade = np.outer(fade_y, fade_x)

    # Feather zone: map 0..feather_fraction -> 0..1, else 1
    # Use smooth (s-curve) falloff
    def smoothstep(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)

    # Pixel distance from "safe" center; 0 = center, 1 = at feather boundary
    # feather_fraction of half-dimension = feather distance from each edge
    ramp = fade / feather_fraction
    mask = smoothstep(ramp)
    # Combine with existing alpha
    new_alpha = alpha * mask
    rgb[:, :, 3] = (np.clip(new_alpha, 0, 1) * 255).astype(np.uint8)

    out = Image.fromarray(rgb, "RGBA")
    out.save(output_path, "PNG")
    return output_path


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not input_file:
        print("Usage: python feather_edges.py <image path> [output path]")
        sys.exit(1)
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    path = feather_edges(input_file, out_file)
    print(f"Saved: {path}")
