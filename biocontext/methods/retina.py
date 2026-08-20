"""Ablation: retina-inspired center-surround preprocessing only.

Applies the Difference-of-Gaussians center-surround transform to both
images to suppress the repetitive periodic background and emphasize local
distinctive structure, then performs plain single-peak multi-scale template
matching (same peak-picking as the classical baseline). This isolates the
contribution of the center-surround representation, independent of the
ant-inspired context verification stage used in the full BioContext method.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .common import Candidate, MatchResult, center_surround, normalized_xcorr_map, to_gray


def match(search_img: np.ndarray, reference_img: np.ndarray, nominal_scale: float = 10.0,
          scale_range: float = 0.25, n_scales: int = 9,
          sigma_center: float = 1.0, sigma_surround: float = 4.0) -> MatchResult:
    t0 = time.time()
    search = to_gray(search_img)
    ref = to_gray(reference_img)

    # NOTE: the reference is captured at native (shrunk) resolution, so its
    # spatial frequencies only match the search image's after it has been
    # resized up to the trial scale. We therefore resize the RAW reference
    # to each trial scale first, and only then apply the center-surround
    # (DoG) operator -- applying DoG at the tiny native resolution and then
    # resizing would distort/alias the receptive-field frequencies.
    search_cs = center_surround(search, sigma_center, sigma_surround)

    scales = np.linspace(nominal_scale * (1 - scale_range), nominal_scale * (1 + scale_range), n_scales)
    best: Candidate | None = None

    for s in scales:
        w = int(round(ref.shape[1] * s))
        h = int(round(ref.shape[0] * s))
        if w < 4 or h < 4 or w > search_cs.shape[1] or h > search_cs.shape[0]:
            continue
        ref_resized = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)
        tmpl = center_surround(ref_resized, sigma_center, sigma_surround)
        score_map = normalized_xcorr_map(search_cs, tmpl)
        if score_map is None:
            continue
        _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
        if best is None or max_val > best.raw_score:
            x0, y0 = max_loc
            cx, cy = x0 + w / 2.0, y0 + h / 2.0
            best = Candidate(x=cx, y=cy, bbox=(x0, y0, x0 + w, y0 + h), scale=s,
                              raw_score=float(max_val), combined_score=float(max_val))

    runtime = time.time() - t0
    if best is None:
        h, w = search.shape
        return MatchResult(x=w / 2, y=h / 2, confidence=0.0, candidates=[], method="retina_only",
                            runtime_s=runtime)

    return MatchResult(x=best.x, y=best.y, confidence=best.raw_score, candidates=[best],
                        method="retina_only", runtime_s=runtime)
