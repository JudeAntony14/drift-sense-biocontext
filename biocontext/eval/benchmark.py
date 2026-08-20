"""Benchmark runner.

Runs every method in `biocontext.methods.METHOD_REGISTRY` over the
controlled synthetic case suite (`biocontext.data.synthetic.build_benchmark_suite`)
and records, per (case, method): pixel error to ground truth, whether the
prediction fell within a success threshold, confidence, and runtime.

Usage:
    python -m biocontext.eval.benchmark --n-per-factor 8 --out results/tables/benchmark.csv
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List

import pandas as pd

from biocontext.data.synthetic import CaseConfig, GeneratedCase, build_benchmark_suite, generate_case
from biocontext.methods import METHOD_REGISTRY


def factor_of(case_name: str) -> str:
    for prefix in ("scale", "rot", "noise", "blur", "repetition", "position", "combined"):
        if case_name.startswith(prefix):
            return prefix
    return "other"


def run_case(case: GeneratedCase, success_px: float = 15.0) -> List[dict]:
    rows = []
    tx, ty = case.true_center
    for method_name, fn in METHOD_REGISTRY.items():
        t0 = time.time()
        result = fn(case.search_image, case.reference_image)
        runtime = time.time() - t0
        err = math.hypot(result.x - tx, result.y - ty)
        rows.append({
            "case": case.config.name,
            "factor": factor_of(case.config.name),
            "method": method_name,
            "pred_x": result.x,
            "pred_y": result.y,
            "true_x": tx,
            "true_y": ty,
            "error_px": err,
            "success": err <= success_px,
            "confidence": result.confidence,
            "runtime_s": runtime,
            "n_candidates": len(result.candidates),
            "n_decoys": len(case.decoy_centers),
            # nuisance-factor metadata for stratified analysis
            "rotation_deg": case.config.rotation_deg,
            "gaussian_noise_std": case.config.gaussian_noise_std,
            "blur_ksize": case.config.blur_ksize,
            "scale_jitter": case.config.scale_jitter,
            "repetition_level": case.config.repetition_level,
            "target_position": case.config.target_position,
        })
    return rows


def run_benchmark(n_per_factor: int = 8, success_px: float = 15.0, seed: int = 1000) -> pd.DataFrame:
    configs = build_benchmark_suite(n_per_factor=n_per_factor, base_seed=seed)
    all_rows = []
    for i, cfg in enumerate(configs):
        case = generate_case(cfg)
        rows = run_case(case, success_px=success_px)
        all_rows.extend(rows)
        print(f"[{i + 1}/{len(configs)}] {cfg.name:16s} "
              + " ".join(f"{r['method']}={r['error_px']:.0f}px" for r in rows))
    return pd.DataFrame(all_rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("method").agg(
        mean_error_px=("error_px", "mean"),
        median_error_px=("error_px", "median"),
        p90_error_px=("error_px", lambda s: s.quantile(0.9)),
        success_rate=("success", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
        n_cases=("error_px", "count"),
    ).reset_index().sort_values("mean_error_px")
    return agg


def summarize_by_factor(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["factor", "method"]).agg(
        mean_error_px=("error_px", "mean"),
        success_rate=("success", "mean"),
        n_cases=("error_px", "count"),
    ).reset_index()
    return agg


def main():
    ap = argparse.ArgumentParser(description="Run the BioContext benchmark suite")
    ap.add_argument("--n-per-factor", type=int, default=8)
    ap.add_argument("--success-px", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=str, default="results/tables/benchmark_raw.csv")
    ap.add_argument("--summary-out", type=str, default="results/tables/benchmark_summary.csv")
    ap.add_argument("--factor-out", type=str, default="results/tables/benchmark_by_factor.csv")
    args = ap.parse_args()

    df = run_benchmark(n_per_factor=args.n_per_factor, success_px=args.success_px, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = summarize(df)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)

    by_factor = summarize_by_factor(df)
    Path(args.factor_out).parent.mkdir(parents=True, exist_ok=True)
    by_factor.to_csv(args.factor_out, index=False)

    print("\n=== Overall summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved raw results to {out_path}")
    print(f"Saved summary to {args.summary_out}")
    print(f"Saved per-factor summary to {args.factor_out}")


if __name__ == "__main__":
    main()
