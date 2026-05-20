#!/usr/bin/env python3
"""Build temporal sequence samples for DMAT sequence-level training.

Unlike the pair builder, this script groups K consecutive frames of the same
GT identity into sequences, so DMAT can evolve over time during TBPTT training.

Output npz shape:
  pos_det   : (N, K, D)        positive detection feature at each step
  hist_short: (N, K, Ls, D)    track short-term memory snapshot at each step
  hist_long : (N, K, Ll, D)    track long-term memory snapshot at each step
  hist_slen : (N, K)           valid length of short history
  hist_llen : (N, K)           valid length of long history
  neg_det   : (N, K, M, D)     negative detection features sampled per step
  track_id  : (N,)
  start_frame: (N,)
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
    parser = argparse.ArgumentParser(
        description="Build temporal sequence dataset from GT identities for DMAT training."
    )
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

    # Sequence-specific
    parser.add_argument("--seq_len", type=int, default=8,
                        help="Number of frames per sequence (K).")
    parser.add_argument("--seq_stride", type=int, default=4,
                        help="Sliding window stride for sequence extraction.")
    parser.add_argument("--num_negatives", type=int, default=4,
                        help="Number of negative detections sampled per step (M).")
    parser.add_argument("--max_gap", type=int, default=1,
                        help="Max allowed frame gap within a consecutive run.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_gt_rows(gt_txt: str):
    by_frame: dict[int, list[dict]] = defaultdict(list)
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
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def infer_detection_gt_ids(detections, gt_rows, iou_threshold):
    gt_ids, gt_ious = [], []
    for det in detections:
        best_iou, best_id = 0.0, None
        for row in gt_rows:
            iou = iou_tlwh(det.tlwh, row["tlwh"])
            if iou > best_iou:
                best_iou, best_id = iou, row["gt_id"]
        if best_iou >= iou_threshold:
            gt_ids.append(best_id)
            gt_ious.append(best_iou)
        else:
            gt_ids.append(None)
            gt_ious.append(0.0)
    return gt_ids, gt_ious


def _split_into_runs(timeline: list[dict], max_gap: int) -> list[list[dict]]:
    """Split a per-identity frame timeline into consecutive runs."""
    if not timeline:
        return []
    runs, current = [], [timeline[0]]
    for entry in timeline[1:]:
        if entry["frame"] - current[-1]["frame"] <= max_gap:
            current.append(entry)
        else:
            runs.append(current)
            current = [entry]
    runs.append(current)
    return runs


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

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
    stride = max(1, args.temporal_stride)
    short_history_len = max(2, args.short_history_len)
    long_history_len = max(1, args.long_history_len)
    min_history_obs = 2 * stride + 1

    # Accumulated per-identity timelines: each entry is one frame snapshot
    identity_timeline: dict[int, list[dict]] = defaultdict(list)
    # All valid detections per frame: used later for negative sampling
    frame_all_dets: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    def _build_short_history(track):
        short_memory = getattr(track, "short_memory", [])
        if not short_memory:
            return None, 0
        valid_len = min(len(short_memory), short_history_len)
        items = [np.asarray(f, dtype=np.float32) for f in short_memory[-short_history_len:]][::-1]
        while len(items) < short_history_len:
            items.append(items[-1])
        return np.stack(items, axis=0), valid_len

    def _build_long_history(track):
        long_memory = getattr(track, "long_memory", [])
        if not long_memory:
            return None, 0
        valid_len = min(len(long_memory), long_history_len)
        items = [np.asarray(f, dtype=np.float32) for f in long_memory[-long_history_len:]]
        while len(items) < long_history_len:
            items.append(items[-1])
        return np.stack(items, axis=0), valid_len

    # ── Phase 1: collect per-frame data ──────────────────────────────────────
    for frame_idx in range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1):
        detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]
        if not detections:
            continue
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        det_gt_ids, det_gt_ious = infer_detection_gt_ids(
            detections, gt_by_frame.get(frame_idx, []), args.iou_threshold
        )

        # Determine best detection per GT identity (by IoU)
        best_det_by_gt: dict[int, tuple[float, int]] = {}
        for det_idx, gt_id in enumerate(det_gt_ids):
            if gt_id is None:
                continue
            if gt_id not in best_det_by_gt or det_gt_ious[det_idx] > best_det_by_gt[gt_id][0]:
                best_det_by_gt[gt_id] = (det_gt_ious[det_idx], det_idx)

        # Record all valid detections for negative sampling (after NMS)
        for gt_id, (_, det_idx) in best_det_by_gt.items():
            feat = np.asarray(detections[det_idx].feature, dtype=np.float32)
            norm = np.linalg.norm(feat)
            if norm > 1e-12:
                feat = feat / norm
            frame_all_dets[frame_idx].append((gt_id, feat))

        # Snapshot memory state for each identity that:
        #   (a) has enough history, and (b) has a detection this frame
        for gt_id, memory in sorted(memories.items()):
            if len(getattr(memory, "det_feat_history", [])) < min_history_obs:
                continue
            if gt_id not in best_det_by_gt:
                continue
            short_hist, short_len = _build_short_history(memory)
            long_hist, long_len = _build_long_history(memory)
            if short_hist is None or long_hist is None:
                continue
            _, det_idx = best_det_by_gt[gt_id]
            det_feat = np.asarray(detections[det_idx].feature, dtype=np.float32)
            norm = np.linalg.norm(det_feat)
            if norm > 1e-12:
                det_feat = det_feat / norm
            identity_timeline[gt_id].append({
                "frame": frame_idx,
                "det_feat": det_feat,
                "short_hist": short_hist,
                "long_hist": long_hist,
                "short_hist_len": short_len,
                "long_hist_len": long_len,
            })

        # Update memories with this frame's detections
        for gt_id, (_, det_idx) in best_det_by_gt.items():
            detection = detections[det_idx]
            if gt_id not in memories:
                memories[gt_id] = Track(
                    detection.to_xyah(), gt_id, n_init=1, max_age=10 ** 9,
                    feature=detection.feature, score=detection.confidence,
                )
                memories[gt_id].state = TrackState.Confirmed
            else:
                memories[gt_id].update(detection)

    # ── Phase 2: build K-frame sliding windows ────────────────────────────────
    K = args.seq_len
    M = args.num_negatives
    feat_dim: int | None = None

    pos_det_seqs: list[np.ndarray] = []
    hist_short_seqs: list[np.ndarray] = []
    hist_long_seqs: list[np.ndarray] = []
    hist_slen_seqs: list[np.ndarray] = []
    hist_llen_seqs: list[np.ndarray] = []
    neg_det_seqs: list[np.ndarray] = []
    track_id_seqs: list[int] = []
    start_frame_seqs: list[int] = []

    skipped_no_neg = 0
    skipped_short_run = 0

    for gt_id, timeline in sorted(identity_timeline.items()):
        if not timeline:
            continue
        if feat_dim is None:
            feat_dim = timeline[0]["det_feat"].shape[0]

        runs = _split_into_runs(timeline, args.max_gap)
        for run in runs:
            if len(run) < K:
                skipped_short_run += 1
                continue

            # Build a negative pool from all frames covered by this run.
            # Using cross-frame negatives is valid because they belong to
            # different GT identities, and provides coverage even when
            # sequences have few co-visible identities.
            run_frames = {entry["frame"] for entry in run}
            run_neg_pool: list[np.ndarray] = [
                feat
                for frame in run_frames
                for gid, feat in frame_all_dets[frame]
                if gid != gt_id
            ]

            for start in range(0, len(run) - K + 1, args.seq_stride):
                window = run[start: start + K]

                pos_det = np.stack([w["det_feat"] for w in window])          # (K, D)
                sh = np.stack([w["short_hist"] for w in window])             # (K, Ls, D)
                lh = np.stack([w["long_hist"] for w in window])              # (K, Ll, D)
                sl = np.array([w["short_hist_len"] for w in window], dtype=np.int32)
                ll = np.array([w["long_hist_len"] for w in window], dtype=np.int32)

                # Prefer same-frame negatives; fall back to run-level pool.
                neg_det = np.zeros((K, M, feat_dim), dtype=np.float32)
                valid = True
                for step_i, step_data in enumerate(window):
                    frame = step_data["frame"]
                    same_frame = [
                        feat for gid, feat in frame_all_dets[frame] if gid != gt_id
                    ]
                    pool = same_frame if same_frame else run_neg_pool
                    if len(pool) == 0:
                        valid = False
                        break
                    idxs = rng.choice(len(pool), size=M, replace=len(pool) < M)
                    for neg_i, idx in enumerate(idxs):
                        neg_det[step_i, neg_i] = pool[idx]

                if not valid:
                    skipped_no_neg += 1
                    continue

                pos_det_seqs.append(pos_det)
                hist_short_seqs.append(sh)
                hist_long_seqs.append(lh)
                hist_slen_seqs.append(sl)
                hist_llen_seqs.append(ll)
                neg_det_seqs.append(neg_det)
                track_id_seqs.append(gt_id)
                start_frame_seqs.append(window[0]["frame"])

    if not pos_det_seqs:
        print("No sequences found — check --seq_len, --max_gap, and --num_negatives.")
        return

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        pos_det=np.stack(pos_det_seqs, axis=0),           # (N, K, D)
        hist_short=np.stack(hist_short_seqs, axis=0),     # (N, K, Ls, D)
        hist_long=np.stack(hist_long_seqs, axis=0),       # (N, K, Ll, D)
        hist_slen=np.stack(hist_slen_seqs, axis=0),       # (N, K)
        hist_llen=np.stack(hist_llen_seqs, axis=0),       # (N, K)
        neg_det=np.stack(neg_det_seqs, axis=0),           # (N, K, M, D)
        track_id=np.array(track_id_seqs, dtype=np.int32),
        start_frame=np.array(start_frame_seqs, dtype=np.int32),
        seq_len=np.array([K], dtype=np.int32),
        num_negatives=np.array([M], dtype=np.int32),
        short_history_len=np.array([short_history_len], dtype=np.int32),
        long_history_len=np.array([long_history_len], dtype=np.int32),
    )

    N = len(pos_det_seqs)
    print(f"Saved {N} sequences (K={K}, M={M}) → {output_path}")
    print(
        f"GT identities: {len(identity_timeline)} | "
        f"feat_dim={feat_dim} | "
        f"short_len={short_history_len} | long_len={long_history_len}"
    )
    print(f"Skipped: {skipped_short_run} runs too short, {skipped_no_neg} windows with no negatives")


if __name__ == "__main__":
    main()
