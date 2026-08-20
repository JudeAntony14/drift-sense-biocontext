"""Clean inference interface for BioContext.

Given a Reference Image and a (larger) Search Image, returns the center
coordinate (x, y) of the best matching region in the Search Image, per the
Drift-Sense challenge specification. When multiple locations are close to
equally plausible (the expected situation under repetitive DRAM/FinFET
structure), the center-most match among the near-tied candidates is
returned.

Python API:
    from biocontext.inference import locate
    result = locate("search.png", "reference.png")
    print(result.x, result.y, result.confidence)

CLI:
    python -m biocontext.inference --search search.png --reference reference.png \\
        --method biocontext --visualize out.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from biocontext.methods import METHOD_REGISTRY
from biocontext.methods.common import MatchResult

ImageLike = Union[str, Path, np.ndarray]


def _load(img: ImageLike) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    path = str(img)
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return arr


def locate(search_image: ImageLike, reference_image: ImageLike, method: str = "biocontext",
           nominal_scale: float = 10.0, **kwargs) -> MatchResult:
    """Locate `reference_image` inside `search_image`.

    Args:
        search_image: path or ndarray of the (larger) search image.
        reference_image: path or ndarray of the (smaller, shrunk) reference image.
        method: one of "baseline", "retina_only", "context_only", "biocontext".
        nominal_scale: expected search/reference size ratio (default 10, per spec).
        **kwargs: forwarded to the selected method's `match()` function.

    Returns:
        MatchResult with .x, .y (center coordinates in search-image pixels),
        .confidence, .candidates (all retained plausible matches), .method,
        .runtime_s.
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHOD_REGISTRY)}")
    search = _load(search_image)
    ref = _load(reference_image)
    fn = METHOD_REGISTRY[method]
    return fn(search, ref, nominal_scale=nominal_scale, **kwargs)


def locate_all(search_image: ImageLike, reference_image: ImageLike, nominal_scale: float = 10.0):
    """Runs every registered method and returns a dict {method_name: MatchResult}."""
    search = _load(search_image)
    ref = _load(reference_image)
    return {name: fn(search, ref, nominal_scale=nominal_scale) for name, fn in METHOD_REGISTRY.items()}


def main():
    ap = argparse.ArgumentParser(description="BioContext inference CLI")
    ap.add_argument("--search", required=True, help="Path to the Search Image")
    ap.add_argument("--reference", required=True, help="Path to the Reference Image")
    ap.add_argument("--method", default="biocontext", choices=list(METHOD_REGISTRY))
    ap.add_argument("--nominal-scale", type=float, default=10.0)
    ap.add_argument("--visualize", type=str, default=None,
                     help="If set, saves an annotated visualization PNG to this path")
    ap.add_argument("--json", action="store_true", help="Print result as JSON")
    args = ap.parse_args()

    result = locate(args.search, args.reference, method=args.method,
                     nominal_scale=args.nominal_scale)

    if args.json:
        print(json.dumps({
            "x": result.x, "y": result.y, "confidence": result.confidence,
            "method": result.method, "runtime_s": result.runtime_s,
            "n_candidates": len(result.candidates),
        }, indent=2))
    else:
        print(f"method       : {result.method}")
        print(f"predicted xy : ({result.x:.2f}, {result.y:.2f})")
        print(f"confidence   : {result.confidence:.3f}")
        print(f"candidates   : {len(result.candidates)}")
        print(f"runtime      : {result.runtime_s * 1000:.1f} ms")

    if args.visualize:
        from biocontext.viz.visualize import draw_candidates, save
        search_img = _load(args.search)
        vis = search_img.copy()
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR) if vis.ndim == 2 else vis
        px, py = int(round(result.x)), int(round(result.y))
        cv2.drawMarker(vis_bgr, (px, py), (0, 140, 255), cv2.MARKER_CROSS, 26, 3)
        for c in sorted(result.candidates, key=lambda c: -c.combined_score)[:10]:
            x0, y0, x1, y1 = c.bbox
            cv2.rectangle(vis_bgr, (x0, y0), (x1, y1), (0, 210, 210), 1)
        save(vis_bgr, args.visualize)
        print(f"saved visualization to {args.visualize}")


if __name__ == "__main__":
    main()
