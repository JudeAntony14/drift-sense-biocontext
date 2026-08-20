"""
Synthetic DRAM / FinFET style wafer-inspection dataset generator.

The generator builds a large "Search Image" containing a highly repetitive
periodic cell array (mimicking DRAM bit-cell / FinFET fin arrays), plus a
sparse set of larger, spatially-unique "context markers" (mimicking rare
local defects / metal-routing / alignment marks that are not periodic).

A "Reference Image" is produced by cropping a region around a chosen ground
truth location and down-scaling it by a configurable factor (default 10x,
as specified by the challenge). Because the periodic cell motif repeats
identically across the search image, many locations will look identical
to the reference at the *local* motif level -- these are the deliberate
false-match traps the BioContext pipeline is designed to resolve using
surrounding context.

Everything here is deterministic given a seed, so the benchmark is
reproducible.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
from typing import List, Tuple

import cv2
import numpy as np


@dataclasses.dataclass
class CaseConfig:
    """Configuration for a single generated benchmark case."""

    name: str = "case"
    seed: int = 0

    # search image geometry
    search_size: int = 900
    cell_pitch: int = 26          # spacing of the repeating unit cell (px)
    cell_radius: int = 6          # size of the repeating unit motif (px)
    cell_jitter: float = 0.08     # per-cell random intensity jitter (0-1)
    n_context_markers: int = 14   # sparse, spatially-unique large features
    marker_size_range: Tuple[int, int] = (14, 34)
    structure_type: str = "mixed"  # "cells" | "lines" | "vias" | "mixed"
    via_density: float = 0.35      # fraction of cells that get a via/contact dot
    line_density: float = 0.5      # fraction of rows that get a connecting metal line

    # reference geometry
    shrink_factor: int = 10       # reference is 1/shrink_factor of source crop
    ref_crop_size: int = 340      # size (in search-image px) of the region
                                   # cropped and shrunk to build the reference

    # nuisance factors applied to the reference AFTER cropping/shrinking
    rotation_deg: float = 0.0
    gaussian_noise_std: float = 0.0
    blur_ksize: int = 0           # 0 = no blur, else odd kernel size
    scale_jitter: float = 0.0     # fractional error added to the true 1/10 scale
    intensity_gain: float = 1.0   # multiplicative brightness variation
    intensity_bias: float = 0.0   # additive brightness variation

    # global nuisance applied to the *search* image
    search_noise_std: float = 0.0
    search_blur_ksize: int = 0

    # SEM-like rendering effects (applied to the search image only, since
    # that's what represents the actual "captured" inspection frame)
    sem_effects: bool = True
    vignette_strength: float = 0.10     # radial illumination falloff, 0 = off
    edge_glow_strength: float = 0.18    # bright halos around structure edges
    grain_noise_std: float = 3.5        # fine sensor-grain noise (separate from search_noise_std)
    illumination_gradient: float = 0.05 # linear brightness ramp across the frame

    # where to place the true target ("center", "edge", "corner", "random")
    target_position: str = "random"

    # repetition_level in [0,1]: higher -> cell array is more uniform/regular
    # (harder disambiguation), lower -> more per-cell variety (easier)
    repetition_level: float = 0.85


@dataclasses.dataclass
class GeneratedCase:
    config: CaseConfig
    search_image: np.ndarray          # HxW uint8
    reference_image: np.ndarray       # hxw uint8 (already distorted)
    true_center: Tuple[float, float]  # (x, y) in search image coords
    true_bbox: Tuple[int, int, int, int]  # x0,y0,x1,y1 in search image coords
    decoy_centers: List[Tuple[float, float]]  # other visually-identical cells


def _place_target_position(rng: random.Random, size: int, margin: int, mode: str) -> Tuple[int, int]:
    if mode == "center":
        return size // 2, size // 2
    if mode == "corner":
        return rng.choice([margin, size - margin]), rng.choice([margin, size - margin])
    if mode == "edge":
        if rng.random() < 0.5:
            return rng.randint(margin, size - margin), rng.choice([margin, size - margin])
        return rng.choice([margin, size - margin]), rng.randint(margin, size - margin)
    return rng.randint(margin, size - margin), rng.randint(margin, size - margin)


def _draw_unit_cell(canvas: np.ndarray, cx: int, cy: int, radius: int, intensity: int, shape_id: int) -> None:
    """Draw one repeating DRAM/FinFET-like unit cell centered at (cx, cy)."""
    if shape_id == 0:
        cv2.rectangle(canvas, (cx - radius, cy - radius // 2), (cx + radius, cy + radius // 2), intensity, -1)
    elif shape_id == 1:
        cv2.circle(canvas, (cx, cy), radius, intensity, -1)
    else:
        pts = np.array([
            [cx, cy - radius], [cx + radius, cy + radius // 2], [cx - radius, cy + radius // 2]
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], intensity)
    # thin connecting "wordline" strokes, characteristic of memory arrays
    cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), max(intensity - 40, 0), 1)


def _draw_via(canvas: np.ndarray, cx: int, cy: int, radius: int, intensity: int) -> None:
    """Draw a small via/contact dot -- the round, brighter features that
    connect metal layers in real interconnect stacks."""
    r = max(1, radius // 3)
    cv2.circle(canvas, (cx, cy), r, intensity, -1)
    cv2.circle(canvas, (cx, cy), r + 1, max(intensity - 60, 0), 1)


def _apply_sem_effects(canvas: np.ndarray, cfg: "CaseConfig", rng: random.Random) -> np.ndarray:
    """Applies a stack of cheap-but-plausible SEM/optical-inspection-style
    rendering effects: a linear illumination gradient (uneven scan
    brightness), radial vignette falloff, bright edge "glow" (charging /
    edge-brightening commonly seen in SEM micrographs), and fine grain
    noise on top of any structural noise already applied. None of this
    claims physical accuracy -- it exists purely to make the imagery look
    like an inspection frame rather than a flat vector diagram, and to add
    the kind of low-level nuisance variation a real detector has to be
    robust to.
    """
    if not cfg.sem_effects:
        return canvas

    img = canvas.astype(np.float32)
    h, w = img.shape

    if cfg.edge_glow_strength > 0:
        edges = cv2.Canny(canvas, 40, 120)
        glow = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 2.0)
        if glow.max() > 1e-6:
            glow = glow / glow.max() * 255.0
        img = img + glow * cfg.edge_glow_strength

    if cfg.illumination_gradient > 0:
        angle = rng.uniform(0, 2 * math.pi)
        xs = np.linspace(-1, 1, w)
        ys = np.linspace(-1, 1, h)
        gx, gy = np.meshgrid(xs, ys)
        ramp = gx * math.cos(angle) + gy * math.sin(angle)
        img = img + ramp * (cfg.illumination_gradient * 255.0)

    if cfg.vignette_strength > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        cy0, cx0 = h / 2.0, w / 2.0
        dist = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)
        dist = dist / dist.max()
        vignette = 1.0 - cfg.vignette_strength * (dist ** 2)
        img = img * vignette

    if cfg.grain_noise_std > 0:
        npr = np.random.RandomState(cfg.seed * 65599 + 17)
        img = img + npr.normal(0, cfg.grain_noise_std, img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


def build_search_image(cfg: CaseConfig, rng: random.Random) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    S = cfg.search_size
    canvas = np.full((S, S), 60, dtype=np.uint8)

    # base periodic array (highly repetitive DRAM/FinFET-style structure)
    shape_id = rng.randint(0, 2)
    base_intensity = 170
    cell_centers = []
    n = S // cfg.cell_pitch
    offset = (S - (n - 1) * cfg.cell_pitch) // 2
    draw_vias = cfg.structure_type in ("vias", "mixed")
    draw_lines = cfg.structure_type in ("lines", "mixed")
    for i in range(n):
        for j in range(n):
            cx = offset + i * cfg.cell_pitch
            cy = offset + j * cfg.cell_pitch
            jitter_amt = (1.0 - cfg.repetition_level) * cfg.cell_jitter
            jitter = int(rng.uniform(-1, 1) * jitter_amt * 255)
            intensity = int(np.clip(base_intensity + jitter, 40, 255))
            _draw_unit_cell(canvas, cx, cy, cfg.cell_radius, intensity, shape_id)
            if draw_vias and rng.random() < cfg.via_density:
                _draw_via(canvas, cx, cy, cfg.cell_radius, min(intensity + 40, 255))
            cell_centers.append((cx, cy))

    # repeated "metal line" structures spanning full rows -- another very
    # common, highly-periodic real-layout feature (bitlines/wordlines)
    if draw_lines:
        for j in range(n):
            if rng.random() < cfg.line_density:
                cy = offset + j * cfg.cell_pitch + cfg.cell_pitch // 2 - 1
                if 0 <= cy < S:
                    intensity = int(np.clip(base_intensity - 60 + rng.uniform(-10, 10), 20, 200))
                    cv2.line(canvas, (0, cy), (S, cy), intensity, 1)

    # sparse, spatially unique context markers -> these are what give
    # otherwise-identical cell neighborhoods a distinguishable surrounding
    # context (the "ant navigation" cue).
    margin = cfg.ref_crop_size // 2 + 10
    for _ in range(cfg.n_context_markers):
        mx = rng.randint(margin, S - margin)
        my = rng.randint(margin, S - margin)
        msize = rng.randint(*cfg.marker_size_range)
        mshape = rng.randint(0, 3)
        mint = rng.randint(210, 255) if rng.random() < 0.5 else rng.randint(0, 30)
        if mshape == 0:
            cv2.rectangle(canvas, (mx - msize, my - msize // 3), (mx + msize, my + msize // 3), mint, -1)
        elif mshape == 1:
            cv2.circle(canvas, (mx, my), msize // 2, mint, -1)
        elif mshape == 2:
            cv2.line(canvas, (mx - msize, my - msize), (mx + msize, my + msize), mint, 3)
        else:
            axes = (msize, msize // 2)
            cv2.ellipse(canvas, (mx, my), axes, rng.randint(0, 180), 0, 360, mint, -1)

    if cfg.search_blur_ksize and cfg.search_blur_ksize > 0:
        k = cfg.search_blur_ksize | 1
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    canvas = _apply_sem_effects(canvas, cfg, rng)

    if cfg.search_noise_std > 0:
        npr = np.random.RandomState(cfg.seed * 7919 + 3)
        canvas = canvas.astype(np.float32) + npr.normal(0, cfg.search_noise_std, canvas.shape)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    return canvas, cell_centers


def _find_decoys(cell_centers: List[Tuple[int, int]], true_center: Tuple[int, int], pitch: int,
                  min_dist_factor: float = 3.0) -> List[Tuple[float, float]]:
    """Cells far enough from the true target look locally identical and act
    as plausible false matches for template matching on the raw motif."""
    tx, ty = true_center
    decoys = []
    for (cx, cy) in cell_centers:
        d = math.hypot(cx - tx, cy - ty)
        if d > pitch * min_dist_factor:
            decoys.append((float(cx), float(cy)))
    return decoys


def generate_case(cfg: CaseConfig) -> GeneratedCase:
    rng = random.Random(cfg.seed)
    search_img, cell_centers = build_search_image(cfg, rng)
    S = cfg.search_size

    margin = cfg.ref_crop_size // 2 + 5
    tx, ty = _place_target_position(rng, S, margin, cfg.target_position)
    # snap to nearest actual cell center so the target sits on a real motif
    nearest = min(cell_centers, key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2)
    tx, ty = nearest

    half = cfg.ref_crop_size // 2
    x0, y0 = tx - half, ty - half
    x1, y1 = tx + half, ty + half
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(S, x1), min(S, y1)

    crop = search_img[y0:y1, x0:x1].copy()

    eff_scale = 1.0 / cfg.shrink_factor
    if cfg.scale_jitter:
        eff_scale *= (1.0 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter))
    new_w = max(4, int(round(crop.shape[1] * eff_scale)))
    new_h = max(4, int(round(crop.shape[0] * eff_scale)))
    reference = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if cfg.rotation_deg:
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), cfg.rotation_deg, 1.0)
        reference = cv2.warpAffine(reference, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

    if cfg.blur_ksize and cfg.blur_ksize > 0:
        k = cfg.blur_ksize | 1
        reference = cv2.GaussianBlur(reference, (k, k), 0)

    if cfg.gaussian_noise_std > 0:
        npr = np.random.RandomState(cfg.seed * 104729 + 11)
        reference = reference.astype(np.float32) + npr.normal(0, cfg.gaussian_noise_std, reference.shape)
        reference = np.clip(reference, 0, 255).astype(np.uint8)

    if cfg.intensity_gain != 1.0 or cfg.intensity_bias != 0.0:
        reference = reference.astype(np.float32) * cfg.intensity_gain + cfg.intensity_bias
        reference = np.clip(reference, 0, 255).astype(np.uint8)

    true_center = (float(tx), float(ty))
    true_bbox = (x0, y0, x1, y1)
    decoys = _find_decoys(cell_centers, (tx, ty), cfg.cell_pitch)

    return GeneratedCase(
        config=cfg,
        search_image=search_img,
        reference_image=reference,
        true_center=true_center,
        true_bbox=true_bbox,
        decoy_centers=decoys,
    )


# ---------------------------------------------------------------------------
# Reproducibility: save the actual generated images + ground-truth metadata
# ---------------------------------------------------------------------------

def save_case_artifacts(case: GeneratedCase, out_dir: str) -> dict:
    """Saves the generated Search Image, Reference Image, and a JSON
    metadata file (config, ground-truth center/bbox, decoy count) to
    `out_dir`. Returns the dict that was written to JSON. Every case can be
    regenerated bit-for-bit from `config.seed` and the same config values,
    but we save the actual rendered images too so a reviewer can inspect
    exactly what was fed into the benchmark without re-running anything.
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    search_path = out / f"{case.config.name}_search.png"
    ref_path = out / f"{case.config.name}_reference.png"
    meta_path = out / f"{case.config.name}_meta.json"

    cv2.imwrite(str(search_path), case.search_image)
    cv2.imwrite(str(ref_path), case.reference_image)

    meta = {
        "config": dataclasses.asdict(case.config),
        "true_center": case.true_center,
        "true_bbox": case.true_bbox,
        "n_decoy_cells": len(case.decoy_centers),
        "search_image_path": str(search_path.name),
        "reference_image_path": str(ref_path.name),
        "search_image_shape": list(case.search_image.shape),
        "reference_image_shape": list(case.reference_image.shape),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------------------
# Benchmark suite: a curated collection of CaseConfigs sweeping each factor
# ---------------------------------------------------------------------------

def build_benchmark_suite(n_per_factor: int = 6, base_seed: int = 1000) -> List[CaseConfig]:
    """Builds a controlled benchmark sweeping scale, rotation, noise, blur,
    repetition density and target position, each in isolation on top of a
    shared baseline configuration, plus a block of "combined/hard" cases."""
    cases: List[CaseConfig] = []
    seed = base_seed

    def base(**overrides) -> CaseConfig:
        nonlocal seed
        seed += 1
        cfg = CaseConfig(seed=seed)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    # 1) scale variation (error around the nominal 10x shrink)
    for i, sj in enumerate(np.linspace(0.0, 0.20, n_per_factor)):
        cases.append(base(name=f"scale_{i}", scale_jitter=float(sj)))

    # 2) rotation
    for i, r in enumerate(np.linspace(0, 25, n_per_factor)):
        cases.append(base(name=f"rot_{i}", rotation_deg=float(r)))

    # 3) noise
    for i, ns in enumerate(np.linspace(0, 22, n_per_factor)):
        cases.append(base(name=f"noise_{i}", gaussian_noise_std=float(ns)))

    # 4) blur
    for i, b in enumerate([0, 3, 5, 7, 9, 11][:n_per_factor]):
        cases.append(base(name=f"blur_{i}", blur_ksize=int(b)))

    # 5) repetition density (structural ambiguity)
    for i, rep in enumerate(np.linspace(0.5, 0.98, n_per_factor)):
        cases.append(base(name=f"repetition_{i}", repetition_level=float(rep), n_context_markers=10))

    # 6) target position
    for i, pos in enumerate((["center", "edge", "corner", "random"] * 2)[:n_per_factor]):
        cases.append(base(name=f"position_{i}", target_position=pos))

    # 7) combined hard cases (multiple nuisances at once)
    for i in range(n_per_factor):
        cases.append(base(
            name=f"combined_{i}",
            rotation_deg=float(np.random.RandomState(seed).uniform(3, 15)),
            gaussian_noise_std=float(np.random.RandomState(seed + 1).uniform(5, 15)),
            blur_ksize=3,
            scale_jitter=0.08,
            repetition_level=0.9,
            target_position=random.choice(["edge", "corner", "random"]),
        ))

    return cases
