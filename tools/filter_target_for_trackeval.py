#!/usr/bin/env python3
"""Create target-only MOT files for standard TrackEval evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter GT and tracker MOT files to selected target GT identities."
    )
    parser.add_argument("--gt_txt", required=True)
    parser.add_argument("--result_txt", required=True)
    parser.add_argument("--target_gt_id", type=int, nargs="+", required=True)
    parser.add_argument("--output_gt_txt", required=True)
    parser.add_argument("--output_result_txt", required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    return parser.parse_args()


def parse_rows(path: Path):
    rows = []
    with path.open("r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(f"Invalid MOT row at {path}:{lineno}: {line}")
            rows.append(
                {
                    "frame": int(float(parts[0])),
                    "id": int(float(parts[1])),
                    "bbox": tuple(float(v) for v in parts[2:6]),
                    "line": line,
                }
            )
    return rows


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
    return inter / union if union > 0.0 else 0.0


def main():
    args = parse_args()
    gt_rows = parse_rows(Path(args.gt_txt))
    result_rows = parse_rows(Path(args.result_txt))

    target_gt_ids = set(args.target_gt_id)
    target_gt_rows = [row for row in gt_rows if row["id"] in target_gt_ids]
    if not target_gt_rows:
        raise ValueError(f"No GT rows found for target_gt_id={sorted(target_gt_ids)}")

    gt_by_frame = defaultdict(list)
    for row in target_gt_rows:
        gt_by_frame[row["frame"]].append(row)
    result_by_frame = defaultdict(list)
    for row in result_rows:
        result_by_frame[row["frame"]].append(row)

    kept_result_rows = []
    unmatched_target_boxes = 0
    for frame in sorted(gt_by_frame):
        frame_gt_rows = gt_by_frame[frame]
        candidates = result_by_frame.get(frame, [])
        if not candidates:
            unmatched_target_boxes += len(frame_gt_rows)
            continue

        iou_matrix = np.asarray(
            [
                [iou_tlwh(gt_row["bbox"], candidate["bbox"]) for candidate in candidates]
                for gt_row in frame_gt_rows
            ],
            dtype=np.float32,
        )
        row_indices, col_indices = linear_sum_assignment(1.0 - iou_matrix)
        matched_gt_indices = set()
        for row_idx, col_idx in zip(row_indices.tolist(), col_indices.tolist()):
            if iou_matrix[row_idx, col_idx] >= args.iou_threshold:
                kept_result_rows.append(candidates[col_idx])
                matched_gt_indices.add(row_idx)
        unmatched_target_boxes += len(frame_gt_rows) - len(matched_gt_indices)

    output_gt_txt = Path(args.output_gt_txt)
    output_result_txt = Path(args.output_result_txt)
    output_gt_txt.parent.mkdir(parents=True, exist_ok=True)
    output_result_txt.parent.mkdir(parents=True, exist_ok=True)

    output_gt_txt.write_text(
        "\n".join(row["line"] for row in target_gt_rows) + "\n"
    )
    output_result_txt.write_text(
        "\n".join(row["line"] for row in kept_result_rows) + ("\n" if kept_result_rows else "")
    )

    kept_ids = sorted({row["id"] for row in kept_result_rows})
    print(f"Target GT ids: {sorted(target_gt_ids)}")
    print(f"Target GT boxes: {len(target_gt_rows)}")
    print(f"Matched target boxes kept in result: {len(kept_result_rows)}")
    print(f"Unmatched target boxes: {unmatched_target_boxes}")
    print(f"Predicted ids retained: {kept_ids}")
    print(f"Saved target-only GT: {output_gt_txt}")
    print(f"Saved target-only result: {output_result_txt}")


if __name__ == "__main__":
    main()
