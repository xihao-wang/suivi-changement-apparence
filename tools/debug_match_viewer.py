#!/usr/bin/env python3
"""GUI viewer for frame-by-frame inspection of tracking matches and distances."""

from __future__ import annotations

import argparse
import colorsys
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font as tkfont
from tkinter import ttk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def create_unique_color_uchar(tag: int, hue_step: float = 0.41) -> tuple[int, int, int]:
    h, v = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
    return int(255 * r), int(255 * g), int(255 * b)


@dataclass
class FrameReport:
    frame: int
    image_path: str
    detections: list[dict[str, Any]]
    tracks: list[dict[str, Any]]
    appearance_cost_matrix: list[list[float]]
    trend_cost_matrix: list[list[float]]
    final_cost_matrix: list[list[float]]
    raw_cost_matrix: list[list[float]]
    gated_cost_matrix: list[list[float]]
    matches: list[dict[str, int]]
    unmatched_track_ids: list[int]
    unmatched_detection_indices: list[int]
    ambiguous_track_ids: list[int]
    ambiguous_info: dict[int, dict[str, Any]]


def _format_bbox(tlwh) -> str:
    return f"[{tlwh[0]:.1f}, {tlwh[1]:.1f}, {tlwh[2]:.1f}, {tlwh[3]:.1f}]"


def build_reports(
    sequence_dir: str,
    detection_file: str,
    min_confidence: float,
    nms_max_overlap: float,
    min_detection_height: int,
    max_cosine_distance: float,
    nn_budget: int | None,
) -> tuple[list[FrameReport], int, int]:
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

    reports: list[FrameReport] = []
    min_frame = seq_info["min_frame_idx"]
    max_frame = seq_info["max_frame_idx"]

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
            appearance_cost, trend_cost, final_cost = \
                tracker.metric.distance_components_with_memory(features, candidate_tracks)
            raw_cost = final_cost
            gated_cost = linear_assignment.gate_cost_matrix(
                raw_cost.copy(),
                tracker.tracks,
                detections,
                confirmed_track_indices,
                detection_indices,
            )
        else:
            appearance_cost = np.zeros((len(candidate_tracks), len(detections)))
            trend_cost = np.zeros((len(candidate_tracks), len(detections)))
            final_cost = np.zeros((len(candidate_tracks), len(detections)))
            raw_cost = np.zeros((len(candidate_tracks), len(detections)))
            gated_cost = raw_cost.copy()

        matches, unmatched_tracks, unmatched_detections = tracker._match(detections)

        report = FrameReport(
            frame=frame_idx,
            image_path=seq_info["image_filenames"][frame_idx],
            detections=[
                {
                    "index": det_idx,
                    "confidence": float(det.confidence),
                    "bbox_tlwh": [float(x) for x in det.tlwh],
                }
                for det_idx, det in enumerate(detections)
            ],
            tracks=[
                {
                    "track_id": track.track_id,
                    "hits": track.hits,
                    "age": track.age,
                    "time_since_update": track.time_since_update,
                    "bbox_tlwh": [float(x) for x in track.to_tlwh()],
                }
                for track in candidate_tracks
            ],
            appearance_cost_matrix=appearance_cost.tolist(),
            trend_cost_matrix=trend_cost.tolist(),
            final_cost_matrix=final_cost.tolist(),
            raw_cost_matrix=raw_cost.tolist(),
            gated_cost_matrix=gated_cost.tolist(),
            matches=[
                {
                    "track_id": tracker.tracks[track_idx].track_id,
                    "detection_index": detection_idx,
                }
                for track_idx, detection_idx in matches
            ],
            unmatched_track_ids=[tracker.tracks[idx].track_id for idx in unmatched_tracks],
            unmatched_detection_indices=list(unmatched_detections),
            ambiguous_track_ids=list(tracker.last_ambiguous_tracks),
            ambiguous_info=dict(tracker.last_ambiguous_info),
        )
        reports.append(report)

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
        tracker.metric.partial_fit(
            np.asarray(feat_list), np.asarray(target_list), active_targets
        )

    return reports, min_frame, max_frame


class MatchViewerApp:
    def __init__(self, root: tk.Tk, reports: list[FrameReport], min_frame: int, max_frame: int):
        self.root = root
        self.reports = reports
        self.min_frame = min_frame
        self.max_frame = max_frame
        self.report_by_frame = {r.frame: r for r in reports}
        self.current_frame = min_frame
        self.photo = None

        self.root.title("Tracking Match Debug Viewer")
        self.root.geometry("1800x1200")
        self.root.minsize(1400, 900)

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=17)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=17)
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(size=16)
        self.root.option_add("*Font", default_font)
        self.root.bind("<KeyPress-a>", lambda _e: self.prev_frame())
        self.root.bind("<KeyPress-d>", lambda _e: self.next_frame())

        top = ttk.Frame(root)
        top.pack(fill="x", padx=12, pady=12)

        ttk.Button(top, text="<< Prev", command=self.prev_frame).pack(side="left")
        ttk.Button(top, text="Next >>", command=self.next_frame).pack(side="left", padx=(10, 0))
        ttk.Button(top, text="-10", command=lambda: self.step(-10)).pack(side="left", padx=(20, 0))
        ttk.Button(top, text="+10", command=lambda: self.step(10)).pack(side="left", padx=(10, 0))

        self.frame_label = ttk.Label(top, text="")
        self.frame_label.pack(side="left", padx=(24, 0))

        ttk.Label(top, text="Jump:").pack(side="left", padx=(20, 6))
        self.jump_var = tk.StringVar()
        self.jump_entry = ttk.Entry(top, textvariable=self.jump_var, width=8)
        self.jump_entry.pack(side="left")
        self.jump_entry.bind("<Return>", self.jump_to_frame)
        ttk.Button(top, text="Go", command=self.jump_to_frame).pack(side="left", padx=(6, 0))

        self.scale = ttk.Scale(
            top,
            from_=self.min_frame,
            to=self.max_frame,
            orient="horizontal",
            command=self.on_scale,
            length=900,
        )
        self.scale.pack(side="right", fill="x", expand=True, padx=(20, 0))

        main = ttk.Panedwindow(root, orient="vertical")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        upper = ttk.Frame(main)
        lower = ttk.Frame(main)
        main.add(upper, weight=5)
        main.add(lower, weight=3)

        self.image_label = ttk.Label(upper)
        self.image_label.pack(fill="both", expand=True)

        info_pane = ttk.Panedwindow(lower, orient="horizontal")
        info_pane.pack(fill="both", expand=True)

        left = ttk.Frame(info_pane)
        right = ttk.Frame(info_pane)
        info_pane.add(left, weight=2)
        info_pane.add(right, weight=3)

        self.summary_text = tk.Text(left, wrap="word", height=20, font=("TkDefaultFont", 21))
        self.summary_text.pack(fill="both", expand=True)
        self.summary_text.tag_configure("alert", foreground="red")

        self.matrix_text = tk.Text(right, wrap="none", height=20, font=("TkFixedFont", 20))
        self.matrix_text.pack(fill="both", expand=True)

        self.scale.set(self.current_frame)
        self.render()

    def on_scale(self, value):
        self.current_frame = int(float(value))
        self.render()

    def step(self, delta: int):
        self.current_frame = min(self.max_frame, max(self.min_frame, self.current_frame + delta))
        self.scale.set(self.current_frame)
        self.render()

    def jump_to_frame(self, _event=None):
        try:
            frame = int(self.jump_var.get().strip())
        except ValueError:
            return
        self.current_frame = min(self.max_frame, max(self.min_frame, frame))
        self.scale.set(self.current_frame)
        self.render()

    def prev_frame(self):
        self.step(-1)

    def next_frame(self):
        self.step(1)

    def render(self):
        report = self.report_by_frame.get(self.current_frame)
        if report is None:
            return
        self.frame_label.config(text=f"Frame {report.frame}")
        self.render_image(report)
        self.render_text(report)

    def render_image(self, report: FrameReport):
        image = cv2.imread(report.image_path, cv2.IMREAD_COLOR)
        if image is None:
            return

        for det in report.detections:
            x, y, w, h = [int(v) for v in det["bbox_tlwh"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                image,
                f"D{det['index']} {det['confidence']:.2f}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        for tr in report.tracks:
            x, y, w, h = [int(v) for v in tr["bbox_tlwh"]]
            color = create_unique_color_uchar(tr["track_id"])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                image,
                f"T{tr['track_id']}",
                (x, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            if tr["track_id"] in report.ambiguous_track_ids:
                cv2.putText(
                    image,
                    f"T{tr['track_id']} AMBIGUOUS",
                    (x, max(30, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        max_w, max_h = 2400,1120
        h, w = image.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h))
        pil_image = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.photo)

    def render_text(self, report: FrameReport):
        self.summary_text.delete("1.0", tk.END)
        self.matrix_text.delete("1.0", tk.END)

        self.summary_text.insert(tk.END, f"Frame: {report.frame}\n")
        self.summary_text.insert(tk.END, f"Detections: {len(report.detections)}\n")
        self.summary_text.insert(tk.END, f"Tracks: {len(report.tracks)}\n\n")

        self.summary_text.insert(tk.END, "Detections\n")
        for det in report.detections:
            self.summary_text.insert(
                tk.END,
                f"  D{det['index']}: conf={det['confidence']:.3f}, bbox={_format_bbox(det['bbox_tlwh'])}\n",
            )

        self.summary_text.insert(tk.END, "\nTracks\n")
        for tr in report.tracks:
            self.summary_text.insert(
                tk.END,
                f"  T{tr['track_id']}: hits={tr['hits']}, age={tr['age']}, time_since_update={tr['time_since_update']}, "
                f"bbox={_format_bbox(tr['bbox_tlwh'])}\n",
            )

        self.summary_text.insert(tk.END, "\nMatches\n")
        for match in report.matches:
            self.summary_text.insert(
                tk.END, f"  T{match['track_id']} <-> D{match['detection_index']}\n"
            )
        if not report.matches:
            self.summary_text.insert(tk.END, "  none\n")

        self.summary_text.insert(
            tk.END, f"\nUnmatched tracks: {report.unmatched_track_ids or 'none'}\n"
        )
        self.summary_text.insert(
            tk.END,
            f"Unmatched detections: {report.unmatched_detection_indices or 'none'}\n",
        )

        if report.ambiguous_track_ids:
            self.summary_text.insert(tk.END, "\nAmbiguous split warning\n", "alert")
            for track_id in report.ambiguous_track_ids:
                info = report.ambiguous_info.get(track_id, {})
                candidates = info.get("candidates", [])
                distances = info.get("distances", [])
                if len(candidates) >= 2 and len(distances) >= 2:
                    self.summary_text.insert(
                        tk.END,
                        f"  T{track_id}: D{candidates[0]} ({distances[0]:.4f}) vs "
                        f"D{candidates[1]} ({distances[1]:.4f})\n",
                        "alert",
                    )
                else:
                    self.summary_text.insert(
                        tk.END, f"  T{track_id}\n", "alert"
                    )

        self.matrix_text.insert(tk.END, "Appearance cost matrix\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.appearance_cost_matrix, report))
        self.matrix_text.insert(tk.END, "\nTrend cost matrix\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.trend_cost_matrix, report))
        self.matrix_text.insert(tk.END, "\nFinal cost matrix(before gating)\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.final_cost_matrix, report))
        self.matrix_text.insert(tk.END, "\nGated distance matrix(motion / Kalman gating)\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.gated_cost_matrix, report))

    def format_matrix(self, matrix: list[list[float]], report: FrameReport) -> str:
        if not report.tracks or not report.detections:
            return "  no matrix\n"
        det_headers = [f"D{d['index']}" for d in report.detections]
        lines = []
        header = ["track"] + det_headers
        widths = [10] + [12] * len(det_headers)
        lines.append("".join(str(h).ljust(w) for h, w in zip(header, widths)))
        for i, tr in enumerate(report.tracks):
            row = [f"T{tr['track_id']}"]
            for j in range(len(report.detections)):
                value = matrix[i][j]
                if np.isinf(value):
                    row.append("inf")
                else:
                    row.append(f"{value:.4f}")
            lines.append("".join(str(v).ljust(w) for v, w in zip(row, widths)))
        return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive viewer for track/detection matching.")
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--detection_file", required=True)
    parser.add_argument("--min_confidence", type=float, default=0.8)
    parser.add_argument("--min_detection_height", type=int, default=0)
    parser.add_argument("--nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.argv = [sys.argv[0], "CustomDemo", "test"]
    print("Preparing frame reports. This may take a moment...")
    reports, min_frame, max_frame = build_reports(
        sequence_dir=args.sequence_dir,
        detection_file=args.detection_file,
        min_confidence=args.min_confidence,
        nms_max_overlap=args.nms_max_overlap,
        min_detection_height=args.min_detection_height,
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=args.nn_budget,
    )
    print(f"Loaded {len(reports)} frames.")
    root = tk.Tk()
    MatchViewerApp(root, reports, min_frame, max_frame)
    root.mainloop()


if __name__ == "__main__":
    main()

"""
python3 tools/debug_match_viewer.py \
  --sequence_dir data/CustomDemo/test/YT-03 \
  --detection_file data/StrongSORT_data/CustomDemo_test_YOLOX+BoT/YT-03.npy
"""
