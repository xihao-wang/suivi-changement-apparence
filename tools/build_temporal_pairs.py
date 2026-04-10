#!/usr/bin/env python3
"""Build pair-level training samples for the learned temporal scorer.

Each sample contains:
  - det_feat
  - p_t
  - p_t_i
  - p_t_2i
  - label

Pseudo labels are derived from a tracking result txt by matching detections
to result boxes with IoU.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Build temporal pair dataset from a sequence.")
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--detection_file", required=True)
    parser.add_argument("--result_txt", required=True, help="Tracking result txt used as pseudo labels")
    parser.add_argument("--output_npz", required=True)

    parser.add_argument("--dataset", default="CustomDemo")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--BoT", action="store_true")
    parser.add_argument("--ECC", action="store_true")
    parser.add_argument("--NSA", action="store_true")
    parser.add_argument("--EMA", action="store_true")
    parser.add_argument("--MC", action="store_true")
    parser.add_argument("--woC", action="store_true")
    parser.add_argument("--ltm_stm", action="store_true")
    parser.add_argument("--memory_init", action="store_true")
    parser.add_argument("--memory_aware", action="store_true")
    parser.add_argument("--topk", action="store_true")

    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument("--min_detection_height", type=int, default=None)
    parser.add_argument("--nms_max_overlap", type=float, default=None)
    parser.add_argument("--max_cosine_distance", type=float, default=None)
    parser.add_argument("--nn_budget", type=int, default=None)

    parser.add_argument("--temporal_stride", type=int, default=2)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    return parser.parse_args()


def load_result_rows(result_txt: str):
    by_frame: dict[int, list[dict[str, float]]] = defaultdict(list)
    with open(result_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            tlwh = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            by_frame[frame].append({"track_id": track_id, "tlwh": tlwh})
    return by_frame


def iou_tlwh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def infer_detection_target_ids(detections, result_rows, iou_threshold):
    target_ids = []
    for det in detections:
        best_iou = 0.0
        best_id = None
        for row in result_rows:
            iou = iou_tlwh(det.tlwh, row["tlwh"])
            if iou > best_iou:
                best_iou = iou
                best_id = row["track_id"]
        if best_iou >= iou_threshold:
            target_ids.append(best_id)
        else:
            target_ids.append(None)
    return target_ids


def main():
    args = parse_args()
    opt_argv = [sys.argv[0], args.dataset, args.split]
    for flag, enabled in [
        ("--BoT", args.BoT),
        ("--ECC", args.ECC),
        ("--NSA", args.NSA),
        ("--EMA", args.EMA),
        ("--MC", args.MC),
        ("--woC", args.woC),
        ("--ltm_stm", args.ltm_stm),
        ("--memory_init", args.memory_init),
        ("--memory_aware", args.memory_aware),
        ("--topk", args.topk),
    ]:
        if enabled:
            opt_argv.append(flag)
    sys.argv = opt_argv

    from application_util import preprocessing
    from deep_sort import nn_matching
    from deep_sort.tracker import Tracker
    from deep_sort_app import create_detections, gather_sequence_info
    from opts import opt

    min_confidence = opt.min_confidence if args.min_confidence is None else args.min_confidence
    min_detection_height = (
        opt.min_detection_height if args.min_detection_height is None else args.min_detection_height
    )
    nms_max_overlap = opt.nms_max_overlap if args.nms_max_overlap is None else args.nms_max_overlap
    max_cosine_distance = (
        opt.max_cosine_distance if args.max_cosine_distance is None else args.max_cosine_distance
    )
    nn_budget = opt.nn_budget if args.nn_budget is None else args.nn_budget

    seq_info = gather_sequence_info(args.sequence_dir, args.detection_file)
    result_by_frame = load_result_rows(args.result_txt)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    det_feat_list = []
    p_t_list = []
    p_t_i_list = []
    p_t_2i_list = []
    label_list = []
    frame_list = []
    track_id_list = []
    det_index_list = []

    stride = max(1, int(args.temporal_stride))

    min_frame = seq_info["min_frame_idx"]
    max_frame = seq_info["max_frame_idx"]
    for frame_idx in range(min_frame, max_frame + 1):
        detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        if opt.ECC:
            tracker.camera_update(Path(args.sequence_dir).name, frame_idx)

        tracker.predict()

        candidate_tracks = [t for t in tracker.tracks if t.is_confirmed()]
        result_rows = result_by_frame.get(frame_idx, [])
        det_target_ids = infer_detection_target_ids(detections, result_rows, args.iou_threshold)

        for track in candidate_tracks:
            history = getattr(track, "prot_short_history", [])
            if len(history) < 2 * stride + 1:
                continue

            p_t = np.asarray(history[-1], dtype=np.float32)
            p_t_i = np.asarray(history[-1 - stride], dtype=np.float32)
            p_t_2i = np.asarray(history[-1 - 2 * stride], dtype=np.float32)

            positive_det_indices = [
                det_idx for det_idx, target_id in enumerate(det_target_ids)
                if target_id == track.track_id
            ]
            if not positive_det_indices:
                continue

            valid_det_indices = [
                det_idx for det_idx, target_id in enumerate(det_target_ids)
                if target_id is not None
            ]

            for det_idx in valid_det_indices:
                det = detections[det_idx]
                det_feat = np.asarray(det.feature, dtype=np.float32)
                det_norm = np.linalg.norm(det_feat)
                if det_norm > 1e-12:
                    det_feat = det_feat / det_norm

                det_feat_list.append(det_feat)
                p_t_list.append(p_t)
                p_t_i_list.append(p_t_i)
                p_t_2i_list.append(p_t_2i)
                label_list.append(1.0 if det_target_ids[det_idx] == track.track_id else 0.0)
                frame_list.append(frame_idx)
                track_id_list.append(track.track_id)
                det_index_list.append(det_idx)

        matches, unmatched_tracks, unmatched_detections = tracker._match(detections)
        for track_idx, detection_idx in matches:
            tracker.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            tracker.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            tracker._initiate_track(detections[detection_idx])
        tracker.tracks = [t for t in tracker.tracks if not t.is_deleted()]

        active_targets = [t.track_id for t in tracker.tracks if t.is_confirmed()]
        feat_list, target_list = [], []
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue
            feat_list += track.features
            target_list += [track.track_id for _ in track.features]
            if not opt.EMA:
                track.features = []
        tracker.metric.partial_fit(np.asarray(feat_list), np.asarray(target_list), active_targets)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        det_feat=np.asarray(det_feat_list, dtype=np.float32),
        p_t=np.asarray(p_t_list, dtype=np.float32),
        p_t_i=np.asarray(p_t_i_list, dtype=np.float32),
        p_t_2i=np.asarray(p_t_2i_list, dtype=np.float32),
        label=np.asarray(label_list, dtype=np.float32),
        frame=np.asarray(frame_list, dtype=np.int32),
        track_id=np.asarray(track_id_list, dtype=np.int32),
        det_index=np.asarray(det_index_list, dtype=np.int32),
        temporal_stride=np.asarray([stride], dtype=np.int32),
    )

    num_samples = len(label_list)
    num_pos = int(np.sum(label_list))
    num_neg = num_samples - num_pos
    print(f"Saved pair dataset to: {output_path}")
    print(f"Samples: {num_samples}, positives: {num_pos}, negatives: {num_neg}")


if __name__ == "__main__":
    main()
