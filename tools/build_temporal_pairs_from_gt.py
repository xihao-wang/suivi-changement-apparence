#!/usr/bin/env python3
"""Build temporal pair samples from ground-truth identities.

Unlike the pseudo-label builder, this script maintains one appearance memory per
GT identity. Memory persists across absences, so re-entry frames naturally
produce long-gap positive and negative pairs.
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
    parser = argparse.ArgumentParser(description="Build temporal pair dataset from GT identities.")
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--detection_file", required=True)
    parser.add_argument("--gt_txt", required=True, help="MOT-format ground-truth gt.txt")
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
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--temporal_stride", type=int, default=2)
    parser.add_argument("--short_history_len", type=int, default=5)
    parser.add_argument("--long_history_len", type=int, default=30)
    return parser.parse_args()


def load_gt_rows(gt_txt: str):
    by_frame: dict[int, list[dict[str, float]]] = defaultdict(list)
    with open(gt_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            tlwh = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            if gt_id <= 0 or tlwh[2] <= 0.0 or tlwh[3] <= 0.0:
                continue
            by_frame[frame].append({"gt_id": gt_id, "tlwh": tlwh})
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


def infer_detection_gt_ids(detections, gt_rows, iou_threshold):
    gt_ids = []
    gt_ious = []
    for det in detections:
        best_iou = 0.0
        best_id = None
        for row in gt_rows:
            iou = iou_tlwh(det.tlwh, row["tlwh"])
            if iou > best_iou:
                best_iou = iou
                best_id = row["gt_id"]
        if best_iou >= iou_threshold:
            gt_ids.append(best_id)
            gt_ious.append(best_iou)
        else:
            gt_ids.append(None)
            gt_ious.append(0.0)
    return gt_ids, gt_ious


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
    from deep_sort.track import Track, TrackState
    from deep_sort_app import create_detections, gather_sequence_info
    from opts import opt

    min_confidence = opt.min_confidence if args.min_confidence is None else args.min_confidence
    min_detection_height = (
        opt.min_detection_height if args.min_detection_height is None else args.min_detection_height
    )
    nms_max_overlap = opt.nms_max_overlap if args.nms_max_overlap is None else args.nms_max_overlap

    seq_info = gather_sequence_info(args.sequence_dir, args.detection_file)
    gt_by_frame = load_gt_rows(args.gt_txt)
    memories: dict[int, Track] = {}

    det_feat_list = []
    short_hist_feat_list = []
    long_hist_feat_list = []
    short_hist_len_list = []
    long_hist_len_list = []
    label_list = []
    frame_list = []
    track_id_list = []
    det_index_list = []

    stride = max(1, int(args.temporal_stride))
    short_history_len = max(2, int(args.short_history_len))
    long_history_len = max(1, int(args.long_history_len))
    min_history_observations = 2 * stride + 1

    def build_short_history(track):
        short_memory = getattr(track, "short_memory", [])
        if len(short_memory) == 0:
            return None
        valid_len = min(len(short_memory), short_history_len)
        items = [np.asarray(feat, dtype=np.float32) for feat in short_memory[-short_history_len:]][::-1]
        while len(items) < short_history_len:
            items.append(items[-1])
        return np.stack(items, axis=0), valid_len

    def build_long_history(track):
        long_memory = getattr(track, "long_memory", [])
        if len(long_memory) == 0:
            return None
        valid_len = min(len(long_memory), long_history_len)
        items = [np.asarray(feat, dtype=np.float32) for feat in long_memory[-long_history_len:]]
        while len(items) < long_history_len:
            items.append(items[-1])
        return np.stack(items, axis=0), valid_len

    for frame_idx in range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1):
        detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        det_gt_ids, det_gt_ious = infer_detection_gt_ids(
            detections,
            gt_by_frame.get(frame_idx, []),
            args.iou_threshold,
        )
        valid_det_indices = [
            det_idx for det_idx, gt_id in enumerate(det_gt_ids)
            if gt_id is not None
        ]

        for gt_id, memory in sorted(memories.items()):
            if len(getattr(memory, "det_feat_history", [])) < min_history_observations:
                continue
            short_result = build_short_history(memory)
            long_result = build_long_history(memory)
            if short_result is None or long_result is None:
                continue
            short_hist, short_hist_len = short_result
            long_hist, long_hist_len = long_result

            for det_idx in valid_det_indices:
                det_feat = np.asarray(detections[det_idx].feature, dtype=np.float32)
                det_norm = np.linalg.norm(det_feat)
                if det_norm > 1e-12:
                    det_feat = det_feat / det_norm
                det_feat_list.append(det_feat)
                short_hist_feat_list.append(short_hist)
                long_hist_feat_list.append(long_hist)
                short_hist_len_list.append(short_hist_len)
                long_hist_len_list.append(long_hist_len)
                label_list.append(1.0 if det_gt_ids[det_idx] == gt_id else 0.0)
                frame_list.append(frame_idx)
                track_id_list.append(gt_id)
                det_index_list.append(det_idx)

        # Keep one best observation per GT identity for memory updates.
        best_det_by_gt = {}
        for det_idx in valid_det_indices:
            gt_id = det_gt_ids[det_idx]
            if gt_id not in best_det_by_gt or det_gt_ious[det_idx] > best_det_by_gt[gt_id][0]:
                best_det_by_gt[gt_id] = (det_gt_ious[det_idx], det_idx)
        for gt_id, (_iou, det_idx) in best_det_by_gt.items():
            detection = detections[det_idx]
            if gt_id not in memories:
                memories[gt_id] = Track(
                    detection.to_xyah(),
                    gt_id,
                    n_init=1,
                    max_age=10**9,
                    feature=detection.feature,
                    score=detection.confidence,
                )
                memories[gt_id].state = TrackState.Confirmed
            else:
                memories[gt_id].update(detection)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        det_feat=np.asarray(det_feat_list, dtype=np.float32),
        short_hist_feat=np.asarray(short_hist_feat_list, dtype=np.float32),
        long_hist_feat=np.asarray(long_hist_feat_list, dtype=np.float32),
        short_hist_len=np.asarray(short_hist_len_list, dtype=np.int32),
        long_hist_len=np.asarray(long_hist_len_list, dtype=np.int32),
        label=np.asarray(label_list, dtype=np.float32),
        frame=np.asarray(frame_list, dtype=np.int32),
        track_id=np.asarray(track_id_list, dtype=np.int32),
        det_index=np.asarray(det_index_list, dtype=np.int32),
        temporal_stride=np.asarray([stride], dtype=np.int32),
    )
    num_samples = len(label_list)
    num_pos = int(np.sum(label_list))
    print(f"Saved GT pair dataset to: {output_path}")
    print(f"Samples: {num_samples}, positives: {num_pos}, negatives: {num_samples - num_pos}")


if __name__ == "__main__":
    main()
