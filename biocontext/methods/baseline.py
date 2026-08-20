"""Classical baseline: multi-scale normalized cross-correlation template
matching (OpenCV TM_CCOEFF_NORMED). This is the "off-the-shelf" localization
approach against which BioContext and its ablations are benchmarked.

No biological priors are used: the raw reference is resized across a range
of scales around the known nominal shrink factor and correlated directly
against the raw search image. The single global best-scoring peak is
returned as the answer, which is exactly what makes this baseline fragile
under repetitive periodic structure.
"""
from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np

from .common import Candidate, MatchResult, normalized_xcorr_map, to_gray


def match(search_img: np.ndarray, reference_img: np.ndarray, nominal_scale: float = 10.0,
          scale_range: float = 0.25, n_scales: int = 9) -> MatchResult:
    t0 = time.time()
    search = to_gray(search_img)
    ref = to_gray(reference_img)

    scales = np.linspace(nominal_scale * (1 - scale_range), nominal_scale * (1 + scale_range), n_scales)
    best: Candidate | None = None

    for s in scales:
        w = int(round(ref.shape[1] * s))
        h = int(round(ref.shape[0] * s))
        if w < 4 or h < 4 or w > search.shape[1] or h > search.shape[0]:
            continue
        tmpl = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)
        score_map = normalized_xcorr_map(search, tmpl)
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
        # degenerate fallback: center of the search image
        h, w = search.shape
        return MatchResult(x=w / 2, y=h / 2, confidence=0.0, candidates=[], method="baseline",
                            runtime_s=runtime)

    return MatchResult(x=best.x, y=best.y, confidence=best.raw_score, candidates=[best],
                        method="baseline", runtime_s=runtime)
