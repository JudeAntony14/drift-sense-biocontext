"""Qualitative visualizations for demos and reports: draws the search image
with candidate matches, highlights false (decoy) matches vs the true
target, and marks each method's final selected location.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from biocontext.data.synthetic import GeneratedCase
from biocontext.methods.common import MatchResult

COLOR_TRUE = (0, 220, 0)         # green
COLOR_FALSE = (0, 0, 230)        # red
COLOR_PRED = (255, 140, 0)       # orange
COLOR_CAND = (255, 210, 0)       # cyan-ish yellow
COLOR_REF_BOX = (230, 0, 230)    # magenta


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def draw_candidates(case: GeneratedCase, result: MatchResult, success_px: float = 15.0,
                     max_candidates: int = 10) -> np.ndarray:
    """Draws every retained candidate (yellow), marking those far from the
    true center as false matches (red X) and the true target (green
    circle), plus the method's final prediction (orange)."""
    vis = _to_bgr(case.search_image)
    tx, ty = case.true_center

    cv2.circle(vis, (int(tx), int(ty)), 14, COLOR_TRUE, 3)
    cv2.putText(vis, "ground truth", (int(tx) + 18, int(ty) + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, COLOR_TRUE, 2)

    cands = sorted(result.candidates, key=lambda c: -c.combined_score)[:max_candidates]
    for c in cands:
        d = ((c.x - tx) ** 2 + (c.y - ty) ** 2) ** 0.5
        is_false = d > success_px
        color = COLOR_FALSE if is_false else COLOR_TRUE
        x0, y0, x1, y1 = c.bbox
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
        marker = cv2.MARKER_TILTED_CROSS if is_false else cv2.MARKER_DIAMOND
        cv2.drawMarker(vis, (int(c.x), int(c.y)), color, marker, 12, 1)

    px, py = int(round(result.x)), int(round(result.y))
    cv2.drawMarker(vis, (px, py), COLOR_PRED, cv2.MARKER_CROSS, 26, 3)
    cv2.putText(vis, f"{result.method} prediction", (px + 18, py - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, COLOR_PRED, 2)
    return vis


def draw_reference_panel(case: GeneratedCase, scale_to: int = 220) -> np.ndarray:
    ref = _to_bgr(case.reference_image)
    h, w = ref.shape[:2]
    factor = scale_to / max(h, w)
    ref_big = cv2.resize(ref, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_NEAREST)
    cv2.putText(ref_big, "reference", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(ref_big, "reference", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    return ref_big


def side_by_side_panel(case: GeneratedCase, result: MatchResult, success_px: float = 15.0) -> np.ndarray:
    search_vis = draw_candidates(case, result, success_px=success_px)
    ref_panel = draw_reference_panel(case, scale_to=search_vis.shape[0] // 3)

    canvas = search_vis.copy()
    ph, pw = ref_panel.shape[:2]
    pad = 10
    y0, x0 = pad, canvas.shape[1] - pw - pad
    cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x0 + pw + 4, y0 + ph + 4), (255, 255, 255), -1)
    canvas[y0:y0 + ph, x0:x0 + pw] = ref_panel
    return canvas


def method_comparison_grid(case: GeneratedCase, results: dict, success_px: float = 15.0) -> np.ndarray:
    """Builds a 2x2 grid comparing all four methods on the same case, for
    demo-video / README use."""
    panels = []
    for name, result in results.items():
        panel = draw_candidates(case, result, success_px=success_px, max_candidates=8)
        cv2.putText(panel, name, (14, panel.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 3)
        cv2.putText(panel, name, (14, panel.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (20, 20, 20), 1)
        panels.append(panel)

    while len(panels) < 4:
        panels.append(np.zeros_like(panels[0]))

    top = np.hstack(panels[0:2])
    bottom = np.hstack(panels[2:4])
    grid = np.vstack([top, bottom])
    return grid


def save(img: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)
