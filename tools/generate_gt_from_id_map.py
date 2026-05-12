#!/usr/bin/env python3
"""Generate a pseudo GT MOT file by remapping tracker ids to canonical ids.

This is useful for quick internal evaluation when a result txt already exists
and the user knows which predicted track ids belong to the same real person.

Important:
- This does NOT create true hand-labeled GT.
- Bounding boxes, misses, and false positives remain those of the source result.
- It only merges source track ids into user-defined canonical ids.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pseudo GT MOT txt from a tracking result and an id mapping json."
    )
    parser.add_argument("--result_txt", required=True)
    parser.add_argument("--id_map_json", required=True)
    parser.add_argument("--output_gt_txt", required=True)
    parser.add_argument(
        "--keep_unmapped",
        action="store_true",
        help="Keep unmapped source ids as their own canonical ids.",
    )
    parser.add_argument(
        "--default_conf",
        type=float,
        default=1.0,
        help="Used when the source result row does not contain a confidence column.",
    )
    return parser.parse_args()


def load_id_map(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "id_map" in data:
        data = data["id_map"]

    if not isinstance(data, dict):
        raise ValueError("id_map_json must be a dict or contain a top-level 'id_map' dict.")

    source_to_canonical: dict[int, int] = {}
    for canonical_id_raw, source_ids in data.items():
        canonical_id = int(canonical_id_raw)
        if not isinstance(source_ids, list):
            raise ValueError(f"Mapping for canonical id {canonical_id} must be a list.")
        for source_id in source_ids:
            source_id = int(source_id)
            if source_id in source_to_canonical and source_to_canonical[source_id] != canonical_id:
                raise ValueError(
                    f"Source id {source_id} is assigned to multiple canonical ids: "
                    f"{source_to_canonical[source_id]} and {canonical_id}"
                )
            source_to_canonical[source_id] = canonical_id
    return source_to_canonical


def parse_result_rows(path: Path, default_conf: float) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(f"Invalid MOT row at line {lineno}: {line}")
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) >= 7 else float(default_conf)
            rows.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "bbox": (x, y, w, h),
                    "conf": conf,
                }
            )
    return rows


def main():
    args = parse_args()
    result_txt = Path(args.result_txt)
    id_map_json = Path(args.id_map_json)
    output_gt_txt = Path(args.output_gt_txt)

    source_to_canonical = load_id_map(id_map_json)
    rows = parse_result_rows(result_txt, args.default_conf)

    by_frame_and_id: dict[tuple[int, int], dict] = {}
    dropped_unmapped = 0
    merged_duplicates = 0

    for row in rows:
        source_id = row["track_id"]
        canonical_id = source_to_canonical.get(source_id)
        if canonical_id is None:
            if args.keep_unmapped:
                canonical_id = source_id
            else:
                dropped_unmapped += 1
                continue

        key = (row["frame"], canonical_id)
        kept = by_frame_and_id.get(key)
        candidate = {
            "frame": row["frame"],
            "id": canonical_id,
            "bbox": row["bbox"],
            "conf": row["conf"],
        }
        if kept is None:
            by_frame_and_id[key] = candidate
        else:
            merged_duplicates += 1
            if candidate["conf"] > kept["conf"]:
                by_frame_and_id[key] = candidate

    output_gt_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_gt_txt.open("w") as f:
        for _, row in sorted(by_frame_and_id.items()):
            x, y, w, h = row["bbox"]
            # MOT GT-style row: frame,id,x,y,w,h,conf,class,visibility
            f.write(
                f"{row['frame']},{row['id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n"
            )

    print(f"Saved pseudo GT to: {output_gt_txt}")
    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(by_frame_and_id)}")
    print(f"Dropped unmapped rows: {dropped_unmapped}")
    print(f"Merged same-frame duplicates: {merged_duplicates}")


if __name__ == "__main__":
    main()
