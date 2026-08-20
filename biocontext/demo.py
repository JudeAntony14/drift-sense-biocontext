"""
BioContext DEMO MODE
=====================

A single, sequential, screen-recordable demonstration of the whole system.
No narration, no terminal-log-only output -- every step SAVES a clear PNG
result to `demo_output/` and (by default) opens it automatically so it's
visible on screen the moment it's produced. Run this, hit record, and let
it finish.

What it does, in order:
  1. Generates the procedural synthetic dataset (a fresh demo case).
  2. Saves and opens the Reference Image and Search Image.
  3. Runs the classical template-matching baseline and saves an annotated
     result (prediction vs ground truth, candidate box).
  4. Runs BioContext on the SAME pair and saves an annotated result
     (candidates, false matches, final prediction vs ground truth).
  5. Saves a direct side-by-side baseline-vs-BioContext comparison image.
  6. Runs a real (smaller, fast) benchmark sweep and saves comparison
     charts (accuracy, success rate, per-factor breakdown).
  7. Prints a short, readable text summary to the terminal as a recap
     (secondary to the visual outputs, not a replacement for them).

Usage:
    python -m biocontext.demo
    python -m biocontext.demo --no-open          # don't auto-open images
    python -m biocontext.demo --benchmark-n 6     # faster benchmark sweep
    python -m biocontext.demo --seed 116          # reuse a specific demo case
"""
from __future__ import annotations

import argparse
import math
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from biocontext.data.synthetic import CaseConfig, generate_case, save_case_artifacts
from biocontext.eval import benchmark as bench_mod
from biocontext.eval import plots as plots_mod
from biocontext.methods import METHOD_REGISTRY
from biocontext.viz.visualize import draw_candidates, method_comparison_grid, save

DEMO_OUT = Path("demo_output")

# A seed pre-verified (see README / benchmark) to reproduce the paper's
# headline failure mode: the classical baseline locks onto a decoy cell,
# BioContext recovers the true location using surrounding context.
DEFAULT_DEMO_SEED = 116


def _open_file(path: Path) -> None:
    """Best-effort cross-platform "open this image" for a live demo."""
    try:
        system = platform.system()
        if system == "Windows":
            import os
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"  (could not auto-open {path.name}: {e}. Open it manually.)")


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _make_reference_panel(reference_img: np.ndarray, out_path: Path, scale_to: int = 420) -> None:
    """Saves a large, clearly-labeled, nearest-neighbor-upscaled view of the
    tiny reference image -- at native size it's too small to see on a
    screen recording."""
    h, w = reference_img.shape[:2]
    factor = scale_to / max(h, w)
    big = cv2.resize(reference_img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    band = np.full((60, canvas.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(band, "REFERENCE IMAGE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(band, f"{w}x{h}px, shrunk 10x from source", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1)
    canvas = np.vstack([band, canvas])
    save(canvas, str(out_path))


def _make_search_panel(search_img: np.ndarray, out_path: Path) -> None:
    canvas = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
    band = np.full((60, canvas.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(band, "SEARCH IMAGE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(band, f"{search_img.shape[1]}x{search_img.shape[0]}px, procedurally generated "
                       f"DRAM-style layout", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    canvas = np.vstack([band, canvas])
    save(canvas, str(out_path))


def _labeled(img: np.ndarray, label: str) -> np.ndarray:
    canvas = img.copy()
    band = np.full((50, canvas.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(band, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.vstack([band, canvas])


def run_demo(seed: int = DEFAULT_DEMO_SEED, benchmark_n: int = 8, auto_open: bool = True,
             pause_s: float = 0.0) -> None:
    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------
    # STEP 1: Generate the procedural synthetic dataset (demo case)
    # ---------------------------------------------------------------
    _banner("STEP 1 / 7  --  Generating procedural DRAM-style Search/Reference pair")
    cfg = CaseConfig(
        name="demo_case",
        seed=seed,
        repetition_level=0.92,
        rotation_deg=3.0,
        gaussian_noise_std=3.0,
        n_context_markers=7,
        target_position="random",
        structure_type="mixed",
    )
    case = generate_case(cfg)
    meta = save_case_artifacts(case, str(DEMO_OUT / "dataset"))
    print(f"  Search image : {case.search_image.shape[1]}x{case.search_image.shape[0]}px")
    print(f"  Reference    : {case.reference_image.shape[1]}x{case.reference_image.shape[0]}px "
          f"(shrunk {cfg.shrink_factor}x)")
    print(f"  Ground truth center (x, y): ({case.true_center[0]:.1f}, {case.true_center[1]:.1f})")
    print(f"  Visually-identical decoy cells in this image: {len(case.decoy_centers)}")
    print(f"  Saved dataset artifacts + ground-truth metadata to {DEMO_OUT / 'dataset'}/")

    # ---------------------------------------------------------------
    # STEP 2: Show the Reference and Search images
    # ---------------------------------------------------------------
    _banner("STEP 2 / 7  --  Saving Reference Image and Search Image views")
    search_panel_path = DEMO_OUT / "01_search_image.png"
    ref_panel_path = DEMO_OUT / "02_reference_image.png"
    _make_search_panel(case.search_image, search_panel_path)
    _make_reference_panel(case.reference_image, ref_panel_path)
    print(f"  Saved {search_panel_path}")
    print(f"  Saved {ref_panel_path}")
    if auto_open:
        _open_file(search_panel_path)
        _open_file(ref_panel_path)
    time.sleep(pause_s)

    # ---------------------------------------------------------------
    # STEP 3: Classical baseline
    # ---------------------------------------------------------------
    _banner("STEP 3 / 7  --  Running classical template-matching baseline")
    t0 = time.time()
    result_baseline = METHOD_REGISTRY["baseline"](case.search_image, case.reference_image)
    dt = time.time() - t0
    err_baseline = math.hypot(result_baseline.x - case.true_center[0],
                               result_baseline.y - case.true_center[1])
    print(f"  Predicted (x, y): ({result_baseline.x:.1f}, {result_baseline.y:.1f})")
    print(f"  Confidence       : {result_baseline.confidence:.3f}")
    print(f"  Error vs ground truth: {err_baseline:.1f}px   (runtime {dt * 1000:.1f} ms)")
    baseline_vis_path = DEMO_OUT / "03_baseline_result.png"
    vis = draw_candidates(case, result_baseline, success_px=15.0, max_candidates=6)
    save(_labeled(vis, f"CLASSICAL BASELINE  |  error = {err_baseline:.1f}px  "
                        f"|  confidence = {result_baseline.confidence:.2f}"), str(baseline_vis_path))
    print(f"  Saved {baseline_vis_path}")
    if auto_open:
        _open_file(baseline_vis_path)
    time.sleep(pause_s)

    # ---------------------------------------------------------------
    # STEP 4: BioContext
    # ---------------------------------------------------------------
    _banner("STEP 4 / 7  --  Running BioContext (retina + ant-context combined) on the SAME pair")
    t0 = time.time()
    result_bio = METHOD_REGISTRY["biocontext"](case.search_image, case.reference_image)
    dt = time.time() - t0
    err_bio = math.hypot(result_bio.x - case.true_center[0], result_bio.y - case.true_center[1])
    print(f"  Predicted (x, y): ({result_bio.x:.1f}, {result_bio.y:.1f})")
    print(f"  Confidence       : {result_bio.confidence:.3f}")
    print(f"  Candidates retained/considered: {len(result_bio.candidates)}")
    print(f"  Error vs ground truth: {err_bio:.1f}px   (runtime {dt * 1000:.1f} ms)")
    bio_vis_path = DEMO_OUT / "04_biocontext_result.png"
    vis = draw_candidates(case, result_bio, success_px=15.0, max_candidates=10)
    save(_labeled(vis, f"BIOCONTEXT (combined)  |  error = {err_bio:.1f}px  "
                        f"|  confidence = {result_bio.confidence:.2f}"), str(bio_vis_path))
    print(f"  Saved {bio_vis_path}")
    if auto_open:
        _open_file(bio_vis_path)
    time.sleep(pause_s)

    # ---------------------------------------------------------------
    # STEP 5: Predicted vs ground truth + direct comparison
    # ---------------------------------------------------------------
    _banner("STEP 5 / 7  --  Direct baseline-vs-BioContext comparison")
    results = {name: fn(case.search_image, case.reference_image) for name, fn in METHOD_REGISTRY.items()}
    grid_path = DEMO_OUT / "05_method_comparison_grid.png"
    grid = method_comparison_grid(case, results, success_px=15.0)
    save(grid, str(grid_path))
    print("  Saved 4-way method comparison (baseline / retina-only / context-only / biocontext)")
    print(f"  -> {grid_path}")
    if err_baseline > 15.0 and err_bio <= 15.0:
        print("  >>> This case reproduces the target failure mode: the classical baseline")
        print("      locked onto a visually-identical decoy structure; BioContext used the")
        print("      surrounding context to recover the correct location. <<<")
    if auto_open:
        _open_file(grid_path)
    time.sleep(pause_s)

    # ---------------------------------------------------------------
    # STEP 6: Benchmark sweep
    # ---------------------------------------------------------------
    _banner(f"STEP 6 / 7  --  Running benchmark sweep (n_per_factor={benchmark_n}, real numbers, ~30-90s)")
    df = bench_mod.run_benchmark(n_per_factor=benchmark_n, success_px=15.0, seed=2000)
    tables_dir = DEMO_OUT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "demo_benchmark_raw.csv", index=False)
    summary = bench_mod.summarize(df)
    summary.to_csv(tables_dir / "demo_benchmark_summary.csv", index=False)
    by_factor = bench_mod.summarize_by_factor(df)
    by_factor.to_csv(tables_dir / "demo_benchmark_by_factor.csv", index=False)

    figs_dir = DEMO_OUT / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    plots_mod.plot_overall_bars(df, figs_dir)
    plots_mod.plot_by_factor(df, figs_dir)
    plots_mod.plot_success_by_factor(df, figs_dir)
    plots_mod.plot_runtime(df, figs_dir)
    plots_mod.plot_error_distribution(df, figs_dir)

    print("\n  Benchmark summary (mean error px, lower is better):")
    print(summary.to_string(index=False))
    overall_path = figs_dir / "overall_comparison.png"
    factor_path = figs_dir / "error_by_factor.png"
    print(f"\n  Saved benchmark figures to {figs_dir}/")
    if auto_open:
        _open_file(overall_path)
        _open_file(factor_path)
    time.sleep(pause_s)

    # ---------------------------------------------------------------
    # STEP 7: Recap
    # ---------------------------------------------------------------
    _banner("STEP 7 / 7  --  Demo complete")
    total_dt = time.time() - t_start
    print(f"  Total demo runtime: {total_dt:.1f}s")
    print(f"  All outputs saved under: {DEMO_OUT.resolve()}")
    print("\n  Key files for the recording / submission:")
    for p in [search_panel_path, ref_panel_path, baseline_vis_path, bio_vis_path, grid_path,
              overall_path, factor_path]:
        print(f"    - {p}")
    print("\n  Headline numbers for this run:")
    print(f"    baseline error   : {err_baseline:.1f}px")
    print(f"    biocontext error : {err_bio:.1f}px")
    n_cases = len(df) // len(METHOD_REGISTRY)
    s = summary.set_index("method")
    print(f"    benchmark (n={n_cases} cases/method) mean error, biocontext vs baseline: "
          f"{s.loc['biocontext', 'mean_error_px']:.1f}px vs {s.loc['baseline', 'mean_error_px']:.1f}px")
    print()


def main():
    ap = argparse.ArgumentParser(description="BioContext DEMO MODE -- run and screen-record this")
    ap.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED,
                     help="Seed for the headline demo case (default reproduces the false-match "
                          "recovery example used in the write-up)")
    ap.add_argument("--benchmark-n", type=int, default=8,
                     help="Cases per stress factor for the in-demo benchmark sweep "
                          "(8 -> ~64 cases, fast; 10+ for a fuller sweep)")
    ap.add_argument("--no-open", action="store_true", help="Don't auto-open result images")
    ap.add_argument("--pause", type=float, default=0.0,
                     help="Seconds to pause after each step (useful to slow down a screen recording)")
    args = ap.parse_args()

    run_demo(seed=args.seed, benchmark_n=args.benchmark_n, auto_open=not args.no_open,
              pause_s=args.pause)


if __name__ == "__main__":
    main()
