#!/usr/bin/env python3
"""Build frame/bbox temporal pairs for the patch-token matcher.

Each sample contains:
  - initial template frame index and bbox
  - current detection frame index and bbox
  - ordered online-template history frame indices and bboxes
  - binary label

Pseudo labels are derived from a tracking result txt by matching detections
to result boxes with IoU. This builder intentionally avoids any feature-level
memory logic so the patch-token experiments depend only on:
  - raw frames
  - detection boxes
  - temporal ordering
  - pseudo labels
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
import os
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build patch-token temporal pairs from a sequence."
    )
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--detection_file", required=True)
    parser.add_argument(
        "--result_txt",
        required=True,
        help="Tracking result txt used as pseudo labels",
    )
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--min_detection_height", type=int, default=0)
    parser.add_argument("--temporal_stride", type=int, default=2)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument(
        "--require_multi_identity_frame",
        action="store_true",
        help="Keep only frames with at least two visible pseudo identities. "
             "Useful when the downstream task is pairwise matching.",
    )
    return parser.parse_args()


def gather_sequence_info(sequence_dir, detection_file):
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
    }

    detections = np.load(detection_file) if detection_file is not None else None

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        image_size = None
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    return {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
    }


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
            tlwh = np.asarray(
                [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])],
                dtype=np.float32,
            )
            by_frame[frame].append({"track_id": track_id, "tlwh": tlwh})
    return by_frame


def iou_tlwh(a: np.ndarray, b: np.ndarray) -> float:
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


def infer_detection_target_ids(det_boxes, result_rows, iou_threshold):
    target_ids = []
    for det_tlwh in det_boxes:
        best_iou = 0.0
        best_id = None
        for row in result_rows:
            iou = iou_tlwh(det_tlwh, row["tlwh"])
            if iou > best_iou:
                best_iou = iou
                best_id = row["track_id"]
        if best_iou >= iou_threshold:
            target_ids.append(best_id)
        else:
            target_ids.append(None)
    return target_ids


def load_frame_detections(detection_mat, frame_idx, min_confidence, min_height):
    frame_mask = detection_mat[:, 0].astype(int) == frame_idx
    rows = detection_mat[frame_mask]
    if rows.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = rows[:, 2:6].astype(np.float32)
    scores = rows[:, 6].astype(np.float32)
    keep = (scores >= min_confidence) & (boxes[:, 3] >= min_height)
    return boxes[keep], scores[keep]


def select_one_detection_per_track(det_boxes, det_scores, det_target_ids):
    """Pick one positive detection per pseudo track in the current frame.

    Multiple detections may overlap the same pseudo result box. For history
    construction, keep the highest-confidence one.
    """
    selected = {}
    for det_idx, target_id in enumerate(det_target_ids):
        if target_id is None:
            continue
        score = float(det_scores[det_idx])
        if target_id not in selected or score > selected[target_id]["score"]:
            selected[target_id] = {
                "score": score,
                "bbox": det_boxes[det_idx].copy(),
            }
    return selected


def main():
    args = parse_args()
    seq_info = gather_sequence_info(args.sequence_dir, args.detection_file)
    detections = seq_info["detections"]
    if detections is None:
        raise ValueError("Detection file did not contain any detections")

    result_by_frame = load_result_rows(args.result_txt)
    stride = max(1, int(args.temporal_stride))

    init_frame_list = []
    init_bbox_list = []
    det_frame_list = []
    det_bbox_list = []
    hist_frame_list = []
    hist_bbox_list = []
    label_list = []
    frame_list = []
    track_id_list = []
    det_index_list = []

    # External history cache keyed by pseudo track id.
    # Each item is {"frame": int, "bbox": np.ndarray(4,)}
    track_histories: dict[int, list[dict[str, np.ndarray | int]]] = defaultdict(list)

    min_frame = seq_info["min_frame_idx"]
    max_frame = seq_info["max_frame_idx"]
    for frame_idx in range(min_frame, max_frame + 1):
        det_boxes, det_scores = load_frame_detections(
            detections,
            frame_idx,
            min_confidence=args.min_confidence,
            min_height=args.min_detection_height,
        )
        if det_boxes.shape[0] == 0:
            continue

        result_rows = result_by_frame.get(frame_idx, [])
        det_target_ids = infer_detection_target_ids(
            det_boxes, result_rows, args.iou_threshold
        )
        valid_target_ids = sorted(
            {track_id for track_id in det_target_ids if track_id is not None}
        )

        if args.require_multi_identity_frame and len(valid_target_ids) <= 1:
            selected = select_one_detection_per_track(det_boxes, det_scores, det_target_ids)
            for track_id, item in selected.items():
                track_histories[track_id].append(
                    {"frame": frame_idx, "bbox": item["bbox"]}
                )
            continue

        valid_det_indices = [
            det_idx
            for det_idx, target_id in enumerate(det_target_ids)
            if target_id is not None
        ]

        for track_id in valid_target_ids:
            history = track_histories.get(track_id, [])
            if len(history) < 2 * stride + 1:
                continue

            init_item = history[0]
            hist_items = [
                history[-1],
                history[-1 - stride],
                history[-1 - 2 * stride],
            ]
            hist_frames = np.asarray(
                [int(item["frame"]) for item in hist_items],
                dtype=np.int32,
            )
            hist_bboxes = np.stack(
                [np.asarray(item["bbox"], dtype=np.float32) for item in hist_items],
                axis=0,
            )

            for det_idx in valid_det_indices:
                init_frame_list.append(int(init_item["frame"]))
                init_bbox_list.append(np.asarray(init_item["bbox"], dtype=np.float32))
                det_frame_list.append(frame_idx)
                det_bbox_list.append(det_boxes[det_idx].copy())
                hist_frame_list.append(hist_frames.copy())
                hist_bbox_list.append(hist_bboxes.copy())
                label_list.append(
                    1.0 if det_target_ids[det_idx] == track_id else 0.0
                )
                frame_list.append(frame_idx)
                track_id_list.append(track_id)
                det_index_list.append(det_idx)

        selected = select_one_detection_per_track(det_boxes, det_scores, det_target_ids)
        for track_id, item in selected.items():
            track_histories[track_id].append(
                {"frame": frame_idx, "bbox": item["bbox"]}
            )

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sequence_dir=np.asarray(str(Path(args.sequence_dir).resolve())),
        init_frame=np.asarray(init_frame_list, dtype=np.int32),
        init_bbox=np.asarray(init_bbox_list, dtype=np.float32),
        det_frame=np.asarray(det_frame_list, dtype=np.int32),
        det_bbox=np.asarray(det_bbox_list, dtype=np.float32),
        hist_frame=np.asarray(hist_frame_list, dtype=np.int32),
        hist_bbox=np.asarray(hist_bbox_list, dtype=np.float32),
        label=np.asarray(label_list, dtype=np.float32),
        frame=np.asarray(frame_list, dtype=np.int32),
        track_id=np.asarray(track_id_list, dtype=np.int32),
        det_index=np.asarray(det_index_list, dtype=np.int32),
        temporal_stride=np.asarray([stride], dtype=np.int32),
    )

    num_samples = len(label_list)
    num_pos = int(np.sum(label_list))
    num_neg = num_samples - num_pos
    print(f"Saved patch-token pair dataset to: {output_path}")
    print(f"Samples: {num_samples}, positives: {num_pos}, negatives: {num_neg}")


if __name__ == "__main__":
    main()
