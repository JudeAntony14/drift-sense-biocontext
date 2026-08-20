"""BioContext: the full combined pipeline.

    Reference + Search
      -> normalized multi-scale representation
      -> candidate generation (retain multiple plausible peaks, raw pixels)
      -> retina-inspired center-surround rescoring   (suppresses periodicity)
      -> local structural matching                    (refines each candidate)
      -> ant-inspired surrounding-context verification (uses the wider view)
      -> candidate ranking (center-most tie-break)
      -> final (x, y)

Design note on why candidate *proposal* uses raw pixels while the
center-surround (retina) representation is used for *rescoring*: template
matching with a very small kernel (the reference is captured at 1/10th
resolution) against a heavily high-pass-filtered (DoG) representation is
numerically unstable -- most of the signal energy in a tiny kernel is edge
information, and small misalignments from rotation/scale error dominate the
correlation, producing an unreliable peak location (see the `retina_only`
ablation results in the benchmark). We therefore let raw-pixel correlation
do what it is good at (coarse, stable candidate proposal) and use the
retina-inspired whitening where it is genuinely most useful: as an
*independent evidence channel* that specifically suppresses the repetitive
periodic background and highlights whether a candidate's local neighborhood
contains the same distinctive (non-periodic) micro-structure as the
reference -- exactly the cue that periodic raw template matching alone
cannot exploit. This mirrors how biological center-surround processing acts
as a salience pre-processing channel rather than as the sole basis for
gross localization.
"""
from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np

from .ant_context import _local_crop, generate_candidates
from .common import (Candidate, MatchResult, center_surround, gradient_orientation_score,
                      pick_center_most, safe_crop, to_gray)


def _retina_score(ref_patch: np.ndarray, cand_patch: np.ndarray, sigma_center: float,
                   sigma_surround: float) -> float:
    """Correlate the center-surround (DoG) representations of two
    same-content patches. Periodic content is suppressed by the DoG
    operator on both sides, so this score rewards candidates that share the
    reference's *local non-periodic* micro-structure rather than merely its
    generic repeating motif."""
    if cand_patch.shape != ref_patch.shape:
        cand_patch = cv2.resize(cand_patch, (ref_patch.shape[1], ref_patch.shape[0]))
    ref_cs = center_surround(ref_patch, sigma_center, sigma_surround).astype(np.float32)
    cand_cs = center_surround(cand_patch, sigma_center, sigma_surround).astype(np.float32)
    ref_cs -= ref_cs.mean()
    cand_cs -= cand_cs.mean()
    denom = np.linalg.norm(ref_cs) * np.linalg.norm(cand_cs)
    if denom < 1e-6:
        return 0.0
    return float(np.clip(np.sum(ref_cs * cand_cs) / denom, -1.0, 1.0))


def _score_candidates(search_raw: np.ndarray, ref_raw: np.ndarray, candidates: List[Candidate],
                       local_frac: float, sigma_center: float, sigma_surround: float) -> List[Candidate]:
    local_ref = _local_crop(ref_raw, local_frac)
    for c in candidates:
        s = c.scale
        full_w = int(round(ref_raw.shape[1] * s))
        full_h = int(round(ref_raw.shape[0] * s))
        cx_i, cy_i = int(round(c.x)), int(round(c.y))

        # --- retina channel: center-surround correlation on the LOCAL patch
        local_w = int(round(local_ref.shape[1] * s))
        local_h = int(round(local_ref.shape[0] * s))
        local_cand = safe_crop(search_raw, cx_i, cy_i, local_w // 2, local_h // 2)
        retina_score = _retina_score(local_ref, local_cand, sigma_center, sigma_surround) if local_cand is not None else 0.0
        struct_score = gradient_orientation_score(local_ref, local_cand) if local_cand is not None else 0.0

        # --- ant-inspired context channel: full reference footprint (incl.
        # surrounding context beyond the bare motif) vs the candidate's
        # equivalent neighborhood in the search image.
        ctx_patch = safe_crop(search_raw, cx_i, cy_i, full_w // 2, full_h // 2)
        if ctx_patch is not None:
            ctx_resized = cv2.resize(ctx_patch, (ref_raw.shape[1], ref_raw.shape[0]))
            ref_f = ref_raw.astype(np.float32) - ref_raw.mean()
            ctx_f = ctx_resized.astype(np.float32) - ctx_resized.mean()
            denom = np.linalg.norm(ref_f) * np.linalg.norm(ctx_f)
            ctx_ncc = float(np.sum(ref_f * ctx_f) / denom) if denom > 1e-6 else 0.0
            ctx_struct = gradient_orientation_score(ref_raw, ctx_resized)
            ctx_retina = _retina_score(ref_raw, ctx_resized, sigma_center, sigma_surround)
            context_score = float(np.clip(0.4 * ctx_ncc + 0.3 * ctx_struct + 0.3 * ctx_retina, -1.0, 1.0))
        else:
            context_score = 0.0

        c.local_score = float(np.clip(0.6 * c.raw_score + 0.2 * max(struct_score, 0.0)
                                       + 0.2 * max(retina_score, 0.0), 0.0, 1.0))
        c.context_score = context_score
        # local evidence (which already includes a full-reference match) is
        # the primary signal; context/retina channels mainly resolve
        # ambiguity between locally-similar candidates
        c.combined_score = float(np.clip(0.7 * c.local_score + 0.3 * max(context_score, 0.0), -1.0, 1.0))
    return candidates


def match(search_img: np.ndarray, reference_img: np.ndarray, nominal_scale: float = 10.0,
          scale_range: float = 0.25, n_scales: int = 9, top_k: int = 8,
          min_score: float = 0.12, local_frac: float = 0.5,
          sigma_center: float = 1.2, sigma_surround: float = 3.0) -> MatchResult:
    t0 = time.time()
    search_raw = to_gray(search_img)
    ref_raw = to_gray(reference_img)

    candidates = generate_candidates(search_raw, ref_raw, nominal_scale, scale_range, n_scales,
                                      top_k, min_score, local_frac)
    if not candidates:
        h, w = search_raw.shape
        return MatchResult(x=w / 2, y=h / 2, confidence=0.0, candidates=[], method="biocontext",
                            runtime_s=time.time() - t0)

    candidates = _score_candidates(search_raw, ref_raw, candidates, local_frac, sigma_center, sigma_surround)
    best = pick_center_most(candidates, score_attr="combined_score")

    runtime = time.time() - t0
    return MatchResult(x=best.x, y=best.y, confidence=best.combined_score, candidates=candidates,
                        method="biocontext", runtime_s=runtime)
