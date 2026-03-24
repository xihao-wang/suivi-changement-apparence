#!/usr/bin/env python3
"""Inspect detection-to-track matching distances for selected frames."""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _format_bbox(tlwh: np.ndarray) -> str:
    return f"[{tlwh[0]:.1f}, {tlwh[1]:.1f}, {tlwh[2]:.1f}, {tlwh[3]:.1f}]"


def _matrix_to_rows(matrix: np.ndarray, row_labels, col_labels):
    rows = []
    for i, row_label in enumerate(row_labels):
        row = {"track": row_label}
        for j, col_label in enumerate(col_labels):
            value = matrix[i, j]
            if np.isinf(value):
                row[col_label] = "inf"
            else:
                row[col_label] = round(float(value), 4)
        rows.append(row)
    return rows


def _print_frame_report(frame_idx, detections, candidate_tracks, raw_cost, gated_cost):
    print(f"\n=== Frame {frame_idx} ===")
    print(f"Detections: {len(detections)}")
    for det_idx, det in enumerate(detections):
        print(
            f"  D{det_idx}: conf={det.confidence:.3f}, bbox={_format_bbox(det.tlwh)}"
        )

    print(f"Confirmed candidate tracks: {len(candidate_tracks)}")
    for track in candidate_tracks:
        print(
            "  "
            f"T{track.track_id}: hits={track.hits}, age={track.age}, "
            f"tsu={track.time_since_update}, bbox={_format_bbox(track.to_tlwh())}"
        )

    if len(candidate_tracks) == 0 or len(detections) == 0:
        print("  No appearance distance matrix for this frame.")
        return

    det_labels = [f"D{j}" for j in range(len(detections))]
    track_labels = [f"T{track.track_id}" for track in candidate_tracks]

    print("Raw memory distance matrix:")
    for row in _matrix_to_rows(raw_cost, track_labels, det_labels):
        print(" ", row)

    print("Gated distance matrix:")
    for row in _matrix_to_rows(gated_cost, track_labels, det_labels):
        print(" ", row)


def inspect_frames(
    sequence_dir,
    detection_file,
    frame_start,
    frame_end,
    min_confidence,
    nms_max_overlap,
    min_detection_height,
    max_cosine_distance,
    nn_budget,
    save_json,
):
    # Import here so this script can parse its own CLI before modules that
    # eagerly import opts.py and consume sys.argv.
    from application_util import preprocessing
    from deep_sort import linear_assignment, nn_matching
    from deep_sort.tracker import Tracker
    from deep_sort_app import create_detections, gather_sequence_info
    from opts import opt

    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)

    reports = []
    min_frame = seq_info["min_frame_idx"]
    max_frame = seq_info["max_frame_idx"]
    frame_start = max(frame_start, min_frame)
    frame_end = min(frame_end, max_frame)

    for frame_idx in range(min_frame, max_frame + 1):
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height
        )
        detections = [d for d in detections if d.confidence >= min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        if opt.ECC:
            tracker.camera_update(Path(sequence_dir).name, frame_idx)

        tracker.predict()

        confirmed_track_indices = [
            i for i, t in enumerate(tracker.tracks) if t.is_confirmed()
        ]
        detection_indices = list(range(len(detections)))
        candidate_tracks = [tracker.tracks[i] for i in confirmed_track_indices]
        features = np.array([detections[i].feature for i in detection_indices])

        if len(candidate_tracks) > 0 and len(detections) > 0:
            raw_cost = tracker.metric.distance_with_memory(features, candidate_tracks)
            gated_cost = linear_assignment.gate_cost_matrix(
                raw_cost.copy(),
                tracker.tracks,
                detections,
                confirmed_track_indices,
                detection_indices,
            )
        else:
            raw_cost = np.zeros((len(candidate_tracks), len(detections)))
            gated_cost = raw_cost.copy()

        if frame_start <= frame_idx <= frame_end:
            _print_frame_report(frame_idx, detections, candidate_tracks, raw_cost, gated_cost)
            reports.append(
                {
                    "frame": frame_idx,
                    "detections": [
                        {
                            "index": det_idx,
                            "confidence": float(det.confidence),
                            "bbox_tlwh": [float(x) for x in det.tlwh],
                        }
                        for det_idx, det in enumerate(detections)
                    ],
                    "tracks": [
                        {
                            "track_id": track.track_id,
                            "hits": track.hits,
                            "age": track.age,
                            "time_since_update": track.time_since_update,
                            "bbox_tlwh": [float(x) for x in track.to_tlwh()],
                        }
                        for track in candidate_tracks
                    ],
                    "raw_cost_matrix": raw_cost.tolist(),
                    "gated_cost_matrix": gated_cost.tolist(),
                }
            )

        tracker.update(detections)

    if save_json is not None:
        save_path = Path(save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"\nSaved debug report to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect detection-to-track distances for a frame range."
    )
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--detection_file", required=True)
    parser.add_argument("--frame_start", type=int, required=True)
    parser.add_argument("--frame_end", type=int, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.8)
    parser.add_argument("--min_detection_height", type=int, default=0)
    parser.add_argument("--nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=None)
    parser.add_argument("--save_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    # Prevent eager opts.py global parsing from seeing this script's CLI.
    # opts.py requires positional `dataset` and `mode`, so provide harmless
    # placeholders before importing modules that depend on it.
    sys.argv = [sys.argv[0], "CustomDemo", "test"]
    inspect_frames(
        sequence_dir=args.sequence_dir,
        detection_file=args.detection_file,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        min_confidence=args.min_confidence,
        nms_max_overlap=args.nms_max_overlap,
        min_detection_height=args.min_detection_height,
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=args.nn_budget,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
