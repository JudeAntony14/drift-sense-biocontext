"""Ablation: ant-inspired surrounding-context verification only (no
center-surround preprocessing).

Desert ants recognize a nest location not from a single memorized snapshot
of the ground beneath them, but by comparing the panoramic view around them
against a remembered panorama -- the *surrounding context*, not just the
immediate local patch, resolves the location. We use the same principle:
a small central crop of the reference (the "local motif") is deliberately
ambiguous under repetitive structure, so instead of trusting the single
best local match, we keep several plausible candidate locations and
re-rank them using how well the *entire* reference footprint (which
includes surrounding context beyond the bare motif) explains each
candidate's neighborhood in the search image.
"""
from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np

from .common import (Candidate, MatchResult, gradient_orientation_score,
                      non_max_suppress_peaks, normalized_xcorr_map, pick_center_most,
                      safe_crop, to_gray)


def _local_crop(ref: np.ndarray, frac: float = 0.45) -> np.ndarray:
    h, w = ref.shape
    ch, cw = max(2, int(h * frac)), max(2, int(w * frac))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return ref[y0:y0 + ch, x0:x0 + cw]


def generate_candidates(search: np.ndarray, ref: np.ndarray, nominal_scale: float,
                         scale_range: float, n_scales: int, top_k: int,
                         min_score: float, local_frac: float = 0.45) -> List[Candidate]:
    """Propose candidate locations using the FULL reference (raw pixels,
    same signal the classical baseline uses), so that in the common,
    unambiguous case this stage is at least as discriminative as the
    baseline. Multiple plausible peaks are retained via non-max suppression
    so that genuinely ambiguous, repetitive-structure cases carry more than
    one hypothesis forward to the context-verification stage, rather than
    committing to a single best guess immediately."""
    scales = np.linspace(nominal_scale * (1 - scale_range), nominal_scale * (1 + scale_range), n_scales)

    all_candidates: List[Candidate] = []
    for s in scales:
        w = int(round(ref.shape[1] * s))
        h = int(round(ref.shape[0] * s))
        if w < 4 or h < 4 or w > search.shape[1] or h > search.shape[0]:
            continue
        tmpl = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)
        score_map = normalized_xcorr_map(search, tmpl)
        if score_map is None:
            continue
        peaks = non_max_suppress_peaks(score_map, (h, w), top_k=top_k, min_score=min_score)
        for (x0, y0, score) in peaks:
            cx, cy = x0 + w / 2.0, y0 + h / 2.0
            all_candidates.append(Candidate(
                x=cx, y=cy, bbox=(x0, y0, x0 + w, y0 + h), scale=s,
                raw_score=score, local_score=score,
            ))

    # keep the strongest few across all scales (avoid combinatorial blow-up)
    all_candidates.sort(key=lambda c: c.raw_score, reverse=True)
    return all_candidates[: max(top_k * 2, 10)]


def context_verify(search: np.ndarray, ref: np.ndarray, candidates: List[Candidate]) -> List[Candidate]:
    for c in candidates:
        s = c.scale
        full_w = int(round(ref.shape[1] * s))
        full_h = int(round(ref.shape[0] * s))
        cx_i, cy_i = int(round(c.x)), int(round(c.y))
        ctx_patch = safe_crop(search, cx_i, cy_i, full_w // 2, full_h // 2)
        if ctx_patch is None:
            c.context_score = 0.0
            c.combined_score = 0.5 * c.local_score
            continue
        ctx_resized = cv2.resize(ctx_patch, (ref.shape[1], ref.shape[0]))
        # normalized correlation between full (context-including) reference
        # and the corresponding full-size neighborhood at the candidate
        ref_f = ref.astype(np.float32) - ref.mean()
        ctx_f = ctx_resized.astype(np.float32) - ctx_resized.mean()
        denom = (np.linalg.norm(ref_f) * np.linalg.norm(ctx_f))
        ncc = float(np.sum(ref_f * ctx_f) / denom) if denom > 1e-6 else 0.0
        grad_score = gradient_orientation_score(ref, ctx_resized)
        c.context_score = float(np.clip(0.6 * ncc + 0.4 * grad_score, -1.0, 1.0))
        c.combined_score = 0.65 * c.local_score + 0.35 * max(c.context_score, 0.0)
    return candidates


def match(search_img: np.ndarray, reference_img: np.ndarray, nominal_scale: float = 10.0,
          scale_range: float = 0.25, n_scales: int = 9, top_k: int = 6,
          min_score: float = 0.15, local_frac: float = 0.45) -> MatchResult:
    t0 = time.time()
    search = to_gray(search_img)
    ref = to_gray(reference_img)

    candidates = generate_candidates(search, ref, nominal_scale, scale_range, n_scales,
                                      top_k, min_score, local_frac)
    if not candidates:
        h, w = search.shape
        return MatchResult(x=w / 2, y=h / 2, confidence=0.0, candidates=[], method="context_only",
                            runtime_s=time.time() - t0)

    candidates = context_verify(search, ref, candidates)
    best = pick_center_most(candidates, score_attr="combined_score")

    runtime = time.time() - t0
    return MatchResult(x=best.x, y=best.y, confidence=best.combined_score, candidates=candidates,
                        method="context_only", runtime_s=runtime)
