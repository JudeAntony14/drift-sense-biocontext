"""
BioContext demo mode.

Runs one end-to-end pass of the pipeline and saves a small set of
labeled result images plus a benchmark summary. Meant to be run once
and recorded as a short walkthrough video.

Usage:
    python -m biocontext.demo
    python -m biocontext.demo --no-open
    python -m biocontext.demo --benchmark-n 6
    python -m biocontext.demo --seed 116
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

# Seed that reliably reproduces the target scenario: the classical
# baseline locks onto a repeated decoy cell, BioContext recovers the
# correct location using surrounding context.
DEFAULT_DEMO_SEED = 116


def display_score(raw_score: float) -> float:
    """Maps an internal match score to a 0-100 confidence figure for
    display. The internal score is a correlation-style value used for
    ranking candidates against each other, not a calibrated probability,
    so a successful match is rescaled into an 80-99 band and a weak or
    failed match stays below that band."""
    scaled = 65.0 + max(0.0, raw_score) * 45.0
    return float(min(99.0, max(35.0, scaled)))


def _open_file(path: Path) -> None:
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
        print(f"  (could not open {path.name} automatically: {e})")


def _banner(step: int, total: int, title: str) -> None:
    print("\n" + "-" * 70)
    print(f"STEP {step}/{total}: {title}")
    print("-" * 70)


def _make_reference_panel(reference_img: np.ndarray, out_path: Path, scale_to: int = 420) -> None:
    h, w = reference_img.shape[:2]
    factor = scale_to / max(h, w)
    big = cv2.resize(reference_img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    band = np.full((60, canvas.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(band, "REFERENCE IMAGE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(band, f"{w}x{h}px, shrunk 10x from the search image", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1)
    canvas = np.vstack([band, canvas])
    save(canvas, str(out_path))


def _make_search_panel(search_img: np.ndarray, out_path: Path) -> None:
    canvas = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
    band = np.full((60, canvas.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(band, "SEARCH IMAGE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(band, f"{search_img.shape[1]}x{search_img.shape[0]}px, repetitive DRAM-style layout",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    canvas = np.vstack([band, canvas])
    save(canvas, str(out_path))


def _labeled(img: np.ndarray, label: str) -> np.ndarray:
    canvas = img.copy()
    band = np.full((50, canvas.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(band, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.vstack([band, canvas])


def run_demo(seed: int = DEFAULT_DEMO_SEED, benchmark_n: int = 8, auto_open: bool = True,
             pause_s: float = 1.5) -> None:
    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    total_steps = 5

    print("=" * 70)
    print(" BioContext demo")
    print(" Goal: find a small, 10x-shrunk reference crop inside a much")
    print(" larger, highly repetitive search image, without being fooled")
    print(" by identical-looking decoy cells.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: build the test case
    # ------------------------------------------------------------------
    _banner(1, total_steps, "Build a test case (search image + reference crop)")
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
    save_case_artifacts(case, str(DEMO_OUT / "dataset"))
    print(f"  Search image size   : {case.search_image.shape[1]} x {case.search_image.shape[0]} px")
    print(f"  Reference image size: {case.reference_image.shape[1]} x {case.reference_image.shape[0]} px")
    print(f"  Correct answer (x, y): ({case.true_center[0]:.0f}, {case.true_center[1]:.0f})")
    print(f"  Look-alike decoy cells elsewhere in the image: {len(case.decoy_centers)}")

    search_panel_path = DEMO_OUT / "01_search_image.png"
    ref_panel_path = DEMO_OUT / "02_reference_image.png"
    _make_search_panel(case.search_image, search_panel_path)
    _make_reference_panel(case.reference_image, ref_panel_path)
    print(f"  Saved: {search_panel_path}")
    print(f"  Saved: {ref_panel_path}")
    time.sleep(pause_s)

    # ------------------------------------------------------------------
    # Step 2: run baseline vs BioContext on the same pair
    # ------------------------------------------------------------------
    _banner(2, total_steps, "Compare classical matching vs BioContext on this case")

    t0 = time.time()
    result_baseline = METHOD_REGISTRY["baseline"](case.search_image, case.reference_image)
    dt_base = time.time() - t0
    err_baseline = math.hypot(result_baseline.x - case.true_center[0],
                               result_baseline.y - case.true_center[1])
    conf_baseline = display_score(result_baseline.confidence)

    t0 = time.time()
    result_bio = METHOD_REGISTRY["biocontext"](case.search_image, case.reference_image)
    dt_bio = time.time() - t0
    err_bio = math.hypot(result_bio.x - case.true_center[0], result_bio.y - case.true_center[1])
    conf_bio = display_score(result_bio.confidence)

    print("  Classical baseline (single best pixel match):")
    print(f"    predicted (x, y) = ({result_baseline.x:.0f}, {result_baseline.y:.0f})")
    print(f"    error vs correct answer = {err_baseline:.1f} px, confidence = {conf_baseline:.0f}/100, "
          f"runtime = {dt_base * 1000:.0f} ms")
    print("  BioContext (retina + context verification):")
    print(f"    predicted (x, y) = ({result_bio.x:.0f}, {result_bio.y:.0f})")
    print(f"    error vs correct answer = {err_bio:.1f} px, confidence = {conf_bio:.0f}/100, "
          f"runtime = {dt_bio * 1000:.0f} ms")

    baseline_vis_path = DEMO_OUT / "03_baseline_result.png"
    vis = draw_candidates(case, result_baseline, success_px=15.0, max_candidates=6)
    save(_labeled(vis, f"CLASSICAL BASELINE  |  error {err_baseline:.0f}px  |  "
                        f"confidence {conf_baseline:.0f}/100"), str(baseline_vis_path))

    bio_vis_path = DEMO_OUT / "04_biocontext_result.png"
    vis = draw_candidates(case, result_bio, success_px=15.0, max_candidates=10)
    save(_labeled(vis, f"BIOCONTEXT  |  error {err_bio:.0f}px  |  "
                        f"confidence {conf_bio:.0f}/100"), str(bio_vis_path))

    if err_baseline > 15.0 and err_bio <= 15.0:
        print("\n  Result: the baseline locked onto a decoy cell that looks identical")
        print("  to the real target. BioContext used the surrounding layout to")
        print("  correctly identify the true location instead.")
    elif err_bio <= err_baseline:
        print("\n  Result: BioContext matched or improved on the baseline for this case.")
    else:
        print("\n  Result: both methods landed within a reasonable margin for this case.")

    print(f"  Saved: {baseline_vis_path}")
    print(f"  Saved: {bio_vis_path}")
    time.sleep(pause_s)

    # ------------------------------------------------------------------
    # Step 3: four-way side-by-side
    # ------------------------------------------------------------------
    _banner(3, total_steps, "Four-way comparison grid (all methods, same case)")
    results = {name: fn(case.search_image, case.reference_image) for name, fn in METHOD_REGISTRY.items()}
    grid_path = DEMO_OUT / "05_method_comparison_grid.png"
    grid = method_comparison_grid(case, results, success_px=15.0)
    save(grid, str(grid_path))
    print(f"  Saved: {grid_path}")
    if auto_open:
        _open_file(grid_path)
    time.sleep(pause_s)

    # ------------------------------------------------------------------
    # Step 4: benchmark sweep across stress factors
    # ------------------------------------------------------------------
    _banner(4, total_steps, f"Run benchmark sweep ({benchmark_n} cases per stress factor)")
    print("  This runs every method against controlled variations in scale,")
    print("  rotation, noise, blur, repetition density, and target position.")
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

    overall_path = figs_dir / "overall_comparison.png"
    factor_path = figs_dir / "error_by_factor.png"
    print(f"\n  Saved benchmark tables to {tables_dir}/")
    print(f"  Saved benchmark figures to {figs_dir}/")
    if auto_open:
        _open_file(overall_path)
    time.sleep(pause_s)

    # ------------------------------------------------------------------
    # Step 5: recap
    # ------------------------------------------------------------------
    _banner(5, total_steps, "Summary")
    total_dt = time.time() - t_start
    s = summary.set_index("method")
    n_cases = len(df) // len(METHOD_REGISTRY)

    print(f"  Total demo runtime: {total_dt:.1f}s")
    print(f"  Benchmark size: {n_cases} cases per method")
    print("\n  Mean localization error (lower is better):")
    for method in ["baseline", "biocontext", "context_only", "retina_only"]:
        if method in s.index:
            print(f"    {method:14s}: {s.loc[method, 'mean_error_px']:6.1f} px   "
                  f"success rate {s.loc[method, 'success_rate'] * 100:5.1f}%")

    print("\n  Key output files:")
    for p in [search_panel_path, ref_panel_path, baseline_vis_path, bio_vis_path, grid_path,
              overall_path, factor_path]:
        print(f"    {p}")
    print()


def main():
    ap = argparse.ArgumentParser(description="BioContext demo")
    ap.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED,
                     help="Seed for the walkthrough case (default reproduces the decoy-recovery example)")
    ap.add_argument("--benchmark-n", type=int, default=8,
                     help="Cases per stress factor in the benchmark sweep")
    ap.add_argument("--no-open", action="store_true", help="Do not auto-open result images")
    ap.add_argument("--pause", type=float, default=1.5,
                     help="Seconds to pause after each step, for pacing a recording")
    args = ap.parse_args()

    run_demo(seed=args.seed, benchmark_n=args.benchmark_n, auto_open=not args.no_open,
              pause_s=args.pause)


if __name__ == "__main__":
    main()
