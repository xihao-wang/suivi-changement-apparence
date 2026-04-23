#!/usr/bin/env python3
"""Tune the manual trend matcher with Optuna.

This script optimizes the manual trend matching parameters for stronger
cost-matrix discrimination on one sequence.

Objective:
    maximize mean_row_margin = average(second_smallest_cost - smallest_cost)

Larger is better because each row then has a clearer best candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class SweepResult:
    trial_number: int
    trend_scale: float
    appearance_cost_weight: float
    short_memory_size: int
    used_frames: int
    used_rows: int
    mean_row_margin: float
    mean_matrix_range: float
    mean_best_cost: float
    mean_diag_margin: float | None


def _parse_float_range(value: str) -> tuple[float, float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected exactly 2 float values, got: {value}")
    lo, hi = vals
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parse_int_range(value: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected exactly 2 int values, got: {value}")
    lo, hi = vals
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _compute_row_stats(matrix: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    row_margins: list[float] = []
    row_ranges: list[float] = []
    best_costs: list[float] = []

    for row in matrix:
        finite = row[np.isfinite(row)]
        if finite.size < 2:
            continue
        sorted_vals = np.sort(finite)
        row_margins.append(float(sorted_vals[1] - sorted_vals[0]))
        row_ranges.append(float(sorted_vals[-1] - sorted_vals[0]))
        best_costs.append(float(sorted_vals[0]))

    return row_margins, row_ranges, best_costs


def _compute_diag_margin(matrix: np.ndarray) -> float | None:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    diag_count = min(matrix.shape[0], matrix.shape[1])
    margins = []
    for i in range(diag_count):
        diag = matrix[i, i]
        if not np.isfinite(diag):
            continue
        off = np.delete(matrix[i], i)
        off = off[np.isfinite(off)]
        if off.size == 0:
            continue
        margins.append(float(np.min(off) - diag))
    if not margins:
        return None
    return float(np.mean(margins))


def _matrix_for_mode(metric, features: np.ndarray, candidate_tracks: list, use_memory: bool) -> np.ndarray:
    if use_memory:
        return metric.distance_with_memory(features, candidate_tracks)
    return metric.distance_with_trend_only(features, candidate_tracks)


def _evaluate_one_setting(
    sequence_dir: str,
    detection_file: str,
    use_memory: bool,
    min_tracks: int,
    min_detections: int,
    assume_diagonal_correct: bool,
    min_confidence: float,
    nms_max_overlap: float,
    min_detection_height: int,
    max_cosine_distance: float,
    nn_budget: int | None,
):
    from application_util import preprocessing
    from deep_sort import nn_matching
    from deep_sort.tracker import Tracker
    from deep_sort_app import create_detections, gather_sequence_info
    from opts import opt

    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)

    row_margins: list[float] = []
    row_ranges: list[float] = []
    best_costs: list[float] = []
    diag_margins: list[float] = []
    used_frames = 0

    min_frame = seq_info["min_frame_idx"]
    max_frame = seq_info["max_frame_idx"]

    for frame_idx in range(min_frame, max_frame + 1):
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height
        )
        detections = [d for d in detections if d.confidence >= min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        if len(detections) > 0:
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

        if opt.ECC:
            tracker.camera_update(Path(sequence_dir).name, frame_idx)

        tracker.predict()

        confirmed_track_indices = [
            i for i, t in enumerate(tracker.tracks) if t.is_confirmed()
        ]
        candidate_tracks = [tracker.tracks[i] for i in confirmed_track_indices]
        features = np.array([d.feature for d in detections])

        if len(candidate_tracks) >= min_tracks and len(detections) >= min_detections:
            raw_cost = _matrix_for_mode(metric, features, candidate_tracks, use_memory=use_memory)
            margins, ranges, bests = _compute_row_stats(raw_cost)
            if margins:
                row_margins.extend(margins)
                row_ranges.extend(ranges)
                best_costs.extend(bests)
                used_frames += 1
                if assume_diagonal_correct:
                    diag_margin = _compute_diag_margin(raw_cost)
                    if diag_margin is not None:
                        diag_margins.append(diag_margin)

        tracker.update(detections)

    return {
        "used_frames": used_frames,
        "used_rows": len(row_margins),
        "mean_row_margin": _safe_mean(row_margins),
        "mean_matrix_range": _safe_mean(row_ranges),
        "mean_best_cost": _safe_mean(best_costs),
        "mean_diag_margin": _safe_mean(diag_margins) if diag_margins else None,
    }


def _write_csv(path: Path, results: Iterable[SweepResult]) -> None:
    rows = [asdict(r) for r in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune the manual trend matcher with Optuna."
    )
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--detection_file", required=True)
    parser.add_argument("--dataset", default="CustomDemo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--BoT", action="store_true", help="Use BoT configuration")
    parser.add_argument("--ECC", action="store_true", help="Enable ECC")
    parser.add_argument(
        "--use_memory",
        action="store_true",
        help="Tune the full memory+trend path instead of trend-only matching",
    )
    parser.add_argument(
        "--trend_scale_range",
        default="0.001,0.2",
        help="Low,high range for trend_scale",
    )
    parser.add_argument(
        "--appearance_weight_range",
        default="0.0,1.0",
        help="Low,high range for appearance_cost_weight",
    )
    parser.add_argument(
        "--short_memory_size_range",
        default="2,10",
        help="Low,high range for short_memory_size",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Optuna sampler",
    )
    parser.add_argument(
        "--log_trend_scale",
        action="store_true",
        help="Sample trend_scale in log space",
    )
    parser.add_argument(
        "--sampler",
        choices=["tpe", "random"],
        default="tpe",
        help="Optuna sampler type",
    )
    parser.add_argument("--min_tracks", type=int, default=2)
    parser.add_argument("--min_detections", type=int, default=2)
    parser.add_argument(
        "--assume_diagonal_correct",
        action="store_true",
        help="Also compute diagonal-vs-off-diagonal margin for square matrices",
    )
    parser.add_argument("--topk", type=int, default=10, help="Show top-k trials")
    parser.add_argument("--save_csv", default=None)
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URI")
    parser.add_argument("--study_name", default="trend_manual_tuning")
    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument("--min_detection_height", type=int, default=None)
    parser.add_argument("--nms_max_overlap", type=float, default=None)
    parser.add_argument("--max_cosine_distance", type=float, default=None)
    parser.add_argument("--nn_budget", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    opt_argv = [sys.argv[0], args.dataset, args.split, "--trend"]
    if args.BoT:
        opt_argv.append("--BoT")
    if args.ECC:
        opt_argv.append("--ECC")
    if args.use_memory:
        opt_argv.extend(["--ltm_stm", "--memory_aware"])
    sys.argv = opt_argv

    from opts import opt

    trend_lo, trend_hi = _parse_float_range(args.trend_scale_range)
    app_lo, app_hi = _parse_float_range(args.appearance_weight_range)
    mem_lo, mem_hi = _parse_int_range(args.short_memory_size_range)

    if args.log_trend_scale and (trend_lo <= 0 or trend_hi <= 0):
        raise ValueError("--log_trend_scale requires a strictly positive trend_scale_range")

    min_confidence = opt.min_confidence if args.min_confidence is None else args.min_confidence
    min_detection_height = (
        opt.min_detection_height if args.min_detection_height is None else args.min_detection_height
    )
    nms_max_overlap = opt.nms_max_overlap if args.nms_max_overlap is None else args.nms_max_overlap
    max_cosine_distance = (
        opt.max_cosine_distance if args.max_cosine_distance is None else args.max_cosine_distance
    )
    nn_budget = opt.nn_budget if args.nn_budget is None else args.nn_budget

    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        study_name=args.study_name if args.storage else None,
        load_if_exists=bool(args.storage),
    )

    collected_results: list[SweepResult] = []

    def objective(trial: optuna.Trial) -> float:
        trend_scale = trial.suggest_float(
            "trend_scale",
            trend_lo,
            trend_hi,
            log=args.log_trend_scale,
        )
        appearance_weight = trial.suggest_float(
            "appearance_cost_weight",
            app_lo,
            app_hi,
        )
        short_memory_size = trial.suggest_int(
            "short_memory_size",
            mem_lo,
            mem_hi,
        )

        opt.trend_scale = trend_scale
        opt.appearance_cost_weight = appearance_weight
        opt.short_memory_size = short_memory_size
        opt.enable_trend = True
        opt.enable_memory_matching = bool(args.use_memory)
        opt.enable_trend_only_matching = not args.use_memory

        metrics = _evaluate_one_setting(
            sequence_dir=args.sequence_dir,
            detection_file=args.detection_file,
            use_memory=args.use_memory,
            min_tracks=args.min_tracks,
            min_detections=args.min_detections,
            assume_diagonal_correct=args.assume_diagonal_correct,
            min_confidence=min_confidence,
            nms_max_overlap=nms_max_overlap,
            min_detection_height=min_detection_height,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

        result = SweepResult(
            trial_number=trial.number,
            trend_scale=trend_scale,
            appearance_cost_weight=appearance_weight,
            short_memory_size=short_memory_size,
            used_frames=metrics["used_frames"],
            used_rows=metrics["used_rows"],
            mean_row_margin=metrics["mean_row_margin"],
            mean_matrix_range=metrics["mean_matrix_range"],
            mean_best_cost=metrics["mean_best_cost"],
            mean_diag_margin=metrics["mean_diag_margin"],
        )
        collected_results.append(result)

        trial.set_user_attr("used_frames", result.used_frames)
        trial.set_user_attr("used_rows", result.used_rows)
        trial.set_user_attr("mean_matrix_range", result.mean_matrix_range)
        trial.set_user_attr("mean_best_cost", result.mean_best_cost)
        if result.mean_diag_margin is not None:
            trial.set_user_attr("mean_diag_margin", result.mean_diag_margin)

        print(
            f"[trial {trial.number}] "
            f"trend_scale={trend_scale:g}, "
            f"appearance_weight={appearance_weight:g}, "
            f"short_memory_size={short_memory_size}, "
            f"row_margin={result.mean_row_margin:.4f}, "
            f"matrix_range={result.mean_matrix_range:.4f}"
        )
        return result.mean_row_margin

    study.optimize(objective, n_trials=args.n_trials)

    collected_results.sort(
        key=lambda r: (
            -np.nan_to_num(r.mean_row_margin, nan=-1e9),
            -np.nan_to_num(r.mean_matrix_range, nan=-1e9),
            np.nan_to_num(r.mean_best_cost, nan=1e9),
        )
    )

    print("\nBest trial:")
    best = study.best_trial
    print(
        f"trial={best.number}, "
        f"value={best.value:.4f}, "
        f"trend_scale={best.params['trend_scale']:.6g}, "
        f"appearance_cost_weight={best.params['appearance_cost_weight']:.6g}, "
        f"short_memory_size={best.params['short_memory_size']}"
    )

    print("\nTop trials by mean row margin:")
    for rank, result in enumerate(collected_results[: args.topk], start=1):
        diag_str = (
            f", diag_margin={result.mean_diag_margin:.4f}"
            if result.mean_diag_margin is not None
            else ""
        )
        print(
            f"{rank:02d}. trial={result.trial_number}, "
            f"trend_scale={result.trend_scale:g}, "
            f"appearance_weight={result.appearance_cost_weight:g}, "
            f"short_memory_size={result.short_memory_size}, "
            f"frames={result.used_frames}, rows={result.used_rows}, "
            f"row_margin={result.mean_row_margin:.4f}, "
            f"matrix_range={result.mean_matrix_range:.4f}, "
            f"best_cost={result.mean_best_cost:.4f}"
            f"{diag_str}"
        )

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(csv_path, collected_results)
        print(f"\nSaved CSV results to {csv_path}")

    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_trial": {
                "number": best.number,
                "value": best.value,
                "params": best.params,
                "user_attrs": best.user_attrs,
            },
            "all_trials": [asdict(r) for r in collected_results],
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON results to {json_path}")


if __name__ == "__main__":
    main()
