"""Shared utilities and result types used by all matching methods."""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclasses.dataclass
class Candidate:
    """A candidate match location in the search image."""
    x: float             # center x in search-image coordinates
    y: float              # center y
    bbox: Tuple[int, int, int, int]  # x0,y0,x1,y1
    scale: float          # scale factor used (ref_px -> search_px)
    raw_score: float      # score from the initial peak-detection stage
    local_score: float = 0.0
    context_score: float = 0.0
    combined_score: float = 0.0


@dataclasses.dataclass
class MatchResult:
    """Final output of a matching method, matching the challenge interface."""
    x: float
    y: float
    confidence: float
    candidates: List[Candidate]
    method: str
    runtime_s: float = 0.0


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def center_surround(img: np.ndarray, sigma_center: float = 1.0, sigma_surround: float = 4.0) -> np.ndarray:
    """Retina-inspired center-surround (Difference-of-Gaussians) transform.

    Approximates the antagonistic center-surround receptive fields of retinal
    ganglion cells: a narrow excitatory center minus a broader inhibitory
    surround. This whitens/suppresses slowly-varying, spatially repetitive
    structure (like a periodic DRAM/FinFET cell array) while amplifying
    local edges and anomalies that make a location distinctive.
    """
    img_f = img.astype(np.float32)
    center = cv2.GaussianBlur(img_f, (0, 0), sigma_center)
    surround = cv2.GaussianBlur(img_f, (0, 0), sigma_surround)
    dog = center - surround
    # normalize to a stable 0-255 range for downstream correlation
    dog = dog - dog.min()
    if dog.max() > 1e-6:
        dog = dog / dog.max() * 255.0
    return dog.astype(np.uint8)


def normalized_xcorr_map(search: np.ndarray, template: np.ndarray) -> Optional[np.ndarray]:
    if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
        return None
    if template.shape[0] < 2 or template.shape[1] < 2:
        return None
    return cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)


def non_max_suppress_peaks(score_map: np.ndarray, tmpl_shape: Tuple[int, int], top_k: int = 8,
                            min_score: float = 0.15) -> List[Tuple[int, int, float]]:
    """Return up to top_k local maxima (x, y, score) from a correlation map,
    suppressing a neighborhood around each accepted peak to avoid returning
    near-duplicate detections of the same physical location."""
    h, w = tmpl_shape
    sm = score_map.copy()
    peaks = []
    suppress_r = max(h, w) // 2
    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(sm)
        if max_val < min_score:
            break
        x, y = max_loc
        peaks.append((x, y, float(max_val)))
        y0, y1 = max(0, y - suppress_r), min(sm.shape[0], y + suppress_r + 1)
        x0, x1 = max(0, x - suppress_r), min(sm.shape[1], x + suppress_r + 1)
        sm[y0:y1, x0:x1] = -1.0
    return peaks


def gradient_orientation_score(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Structural similarity based on correlation of gradient-magnitude maps.
    Robust-ish to small illumination differences; used to refine/validate
    local candidate matches independently of the raw-intensity score."""
    if patch_a.shape != patch_b.shape:
        patch_b = cv2.resize(patch_b, (patch_a.shape[1], patch_a.shape[0]))
    ga = cv2.Laplacian(patch_a.astype(np.float32), cv2.CV_32F)
    gb = cv2.Laplacian(patch_b.astype(np.float32), cv2.CV_32F)
    ga = (ga - ga.mean())
    gb = (gb - gb.mean())
    denom = (np.linalg.norm(ga) * np.linalg.norm(gb))
    if denom < 1e-6:
        return 0.0
    return float(np.clip(np.sum(ga * gb) / denom, -1.0, 1.0))


def safe_crop(img: np.ndarray, cx: int, cy: int, half_w: int, half_h: int) -> Optional[np.ndarray]:
    x0, y0 = cx - half_w, cy - half_h
    x1, y1 = cx + half_w, cy + half_h
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None
    return img[y0:y1, x0:x1]


def pick_center_most(candidates: List[Candidate], score_attr: str = "combined_score",
                      rel_epsilon: float = 0.03, min_abs_epsilon: float = 0.005) -> Candidate:
    """Among candidates whose score is within a small tolerance of the best
    score, prefer the one closest to the centroid of that near-tied group.

    This implements the challenge's "center-most match preferred when
    multiple matches are found" requirement: when several locations are
    effectively tied in confidence (the expected situation under heavy
    repetition), we resolve the tie by preferring the most representative /
    central member of the tied cluster rather than an arbitrary one.

    The tolerance is relative to the best score's magnitude (`rel_epsilon`
    fraction), with a small absolute floor, so that a handful of weak,
    incidentally-close-in-score decoys spread across the whole image don't
    get treated as "tied" with the true winner -- only genuinely
    close-scoring candidates are grouped and centroid-resolved.
    """
    if not candidates:
        raise ValueError("no candidates to pick from")
    scores = [getattr(c, score_attr) for c in candidates]
    best_idx = int(np.argmax(scores))
    best_c = candidates[best_idx]
    best = scores[best_idx]
    epsilon = max(min_abs_epsilon, abs(best) * rel_epsilon)

    # spatial radius within which a near-tied candidate is considered part
    # of the SAME physical location cluster as the best candidate (rather
    # than a spatially distant, coincidentally similar-scoring decoy)
    x0, y0, x1, y1 = best_c.bbox
    radius = 1.5 * max(x1 - x0, y1 - y0)

    tied = [c for c, s in zip(candidates, scores)
            if (best - s) <= epsilon and ((c.x - best_c.x) ** 2 + (c.y - best_c.y) ** 2) ** 0.5 <= radius]
    if len(tied) == 1:
        return tied[0]
    cx = float(np.mean([c.x for c in tied]))
    cy = float(np.mean([c.y for c in tied]))
    tied.sort(key=lambda c: (c.x - cx) ** 2 + (c.y - cy) ** 2)
    return tied[0]
