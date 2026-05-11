#!/usr/bin/env python3
"""GUI viewer for frame-by-frame inspection of tracking matches and distances."""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
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
    learned_temporal_score_matrix: list[list[float]]
    learned_short_attn_matrices: list[list[list[float]]]
    learned_short_attn_top_matrix: list[list[float]]
    learned_short_attn_entropy_matrix: list[list[float]]
    learned_short_similarity_matrix: list[list[float]]
    learned_long_gate_matrix: list[list[float]]
    learned_long_similarity_matrix: list[list[float]]
    learned_long_attn_recent_matrix: list[list[float]]
    learned_long_attn_top_matrix: list[list[float]]
    learned_long_attn_entropy_matrix: list[list[float]]
    final_cost_matrix: list[list[float]]
    raw_cost_matrix: list[list[float]]
    gated_cost_matrix: list[list[float]]
    matches: list[dict[str, int]]
    unmatched_track_ids: list[int]
    unmatched_detection_indices: list[int]
    ambiguous_track_ids: list[int]
    ambiguous_info: dict[int, dict[str, Any]]


def _load_temporal_scores_jsonl(path: str | None) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    score_by_frame: dict[int, dict[str, Any]] = {}
    with open(path, "r") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            obj = json.loads(row)
            frame = int(obj["frame"])
            score_by_frame[frame] = obj
    return score_by_frame


def _align_precomputed_score_matrix(
    precomputed: dict[str, Any],
    expected_track_ids: list[int],
    expected_detection_indices: list[int],
) -> tuple[np.ndarray | None, bool]:
    file_track_ids = [int(x) for x in precomputed.get("track_ids", [])]
    file_detection_indices = [int(x) for x in precomputed.get("detection_indices", [])]
    source_matrix = np.asarray(precomputed.get("scores", []), dtype=np.float32)
    if source_matrix.shape != (len(file_track_ids), len(file_detection_indices)):
        return None, False

    track_to_row = {track_id: row for row, track_id in enumerate(file_track_ids)}
    det_to_col = {det_idx: col for col, det_idx in enumerate(file_detection_indices)}
    aligned = np.zeros((len(expected_track_ids), len(expected_detection_indices)), dtype=np.float32)
    for row, track_id in enumerate(expected_track_ids):
        source_row = track_to_row.get(track_id)
        if source_row is None:
            return None, False
        for col, det_idx in enumerate(expected_detection_indices):
            source_col = det_to_col.get(det_idx)
            if source_col is None:
                return None, False
            aligned[row, col] = source_matrix[source_row, source_col]
    return aligned, True


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
    enable_learned_temporal: bool = False,
    temporal_model_ckpt: str | None = None,
    temporal_hidden_dim: int = 256,
    temporal_num_heads: int = 4,
    temporal_stride: int = 2,
    temporal_alpha: float = 1.0,
    fuse_learned_temporal: bool = False,
    temporal_max_correction: float = 0.02,
    temporal_min_scale: float = 0.02,
    temporal_scores_file: str | None = None,
) -> tuple[list[FrameReport], int, int]:
    from application_util import preprocessing
    from deep_sort import linear_assignment, nn_matching
    from deep_sort.temporal_model import TemporalAttentionScorer
    from deep_sort.tracker import Tracker
    from deep_sort_app import create_detections, gather_sequence_info
    from opts import opt

    seq_info = gather_sequence_info(sequence_dir, detection_file)
    precomputed_scores_by_frame = _load_temporal_scores_jsonl(temporal_scores_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    learned_temporal_model = None
    if enable_learned_temporal and not precomputed_scores_by_frame:
        if temporal_model_ckpt is None:
            raise ValueError(
                "--temporal_model_ckpt or --temporal_scores_file is required with --learned_temporal"
            )
        feature_dim = seq_info["detections"].shape[1] - 10
        ckpt_long_history_len = 3
        ckpt_use_long_memory = True
        if temporal_model_ckpt:
            ckpt = torch.load(temporal_model_ckpt, map_location="cpu")
            if isinstance(ckpt, dict):
                ckpt_long_history_len = int(ckpt.get("long_history_len", 3))
                ckpt_use_long_memory = bool(ckpt.get("use_long_memory", True))
            else:
                ckpt = {"state_dict": ckpt}
        learned_temporal_model = TemporalAttentionScorer(
            feature_dim=feature_dim,
            hidden_dim=temporal_hidden_dim,
            num_heads=temporal_num_heads,
            long_history_len=ckpt_long_history_len,
            use_long_memory=ckpt_use_long_memory,
        )
        if temporal_model_ckpt:
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            learned_temporal_model.load_state_dict(state_dict, strict=False)
        learned_temporal_model.eval()
    tracker = Tracker(
        metric,
        temporal_model=learned_temporal_model if enable_learned_temporal else None,
        temporal_alpha=temporal_alpha,
        fuse_temporal_model=fuse_learned_temporal,
        temporal_max_correction=temporal_max_correction,
        temporal_min_scale=temporal_min_scale,
    )

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
            appearance_cost, final_cost = \
                tracker.metric.distance_components_with_memory(features, candidate_tracks)
            if enable_learned_temporal:
                precomputed = precomputed_scores_by_frame.get(frame_idx)
                precomputed_ok = False
                if precomputed is not None:
                    expected_track_ids = [int(track.track_id) for track in candidate_tracks]
                    expected_detection_indices = list(range(len(detections)))
                    learned_score_candidate, precomputed_ok = _align_precomputed_score_matrix(
                        precomputed,
                        expected_track_ids,
                        expected_detection_indices,
                    )
                    if precomputed_ok:
                        learned_score = learned_score_candidate
                        learned_short_attn = []
                        learned_diag = _empty_learned_diagnostics(len(candidate_tracks), len(detections))
                if not precomputed_ok:
                    if learned_temporal_model is None:
                        raise ValueError(
                            f"Precomputed temporal scores do not match viewer state at frame {frame_idx}. "
                            "Provide the matching --temporal_model_ckpt to allow fallback recomputation, "
                            "or regenerate the jsonl with the same tracker settings."
                        )
                    else:
                        learned_score, learned_short_attn, learned_diag = \
                            _compute_learned_temporal_matrices(
                                candidate_tracks,
                                detections,
                                learned_temporal_model,
                                stride=max(1, int(temporal_stride)),
                            )
            else:
                learned_score = np.zeros((len(candidate_tracks), len(detections)))
                learned_short_attn = []
                learned_diag = _empty_learned_diagnostics(len(candidate_tracks), len(detections))
            if enable_learned_temporal and fuse_learned_temporal:
                learned_prob = 1.0 / (1.0 + np.exp(-learned_score))
                temporal_cost = 1.0 - learned_prob
                final_cost = tracker._fuse_temporal_cost(final_cost, temporal_cost)
                tracker.set_temporal_cost_override(
                    confirmed_track_indices,
                    detection_indices,
                    temporal_cost,
                )
            else:
                tracker.clear_temporal_cost_override()
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
            learned_score = np.zeros((len(candidate_tracks), len(detections)))
            learned_short_attn = []
            learned_diag = _empty_learned_diagnostics(len(candidate_tracks), len(detections))
            final_cost = np.zeros((len(candidate_tracks), len(detections)))
            raw_cost = np.zeros((len(candidate_tracks), len(detections)))
            gated_cost = raw_cost.copy()
            tracker.clear_temporal_cost_override()

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
                    "match_confidence": (
                        None if getattr(track, "match_confidence", None) is None
                        else float(track.match_confidence)
                    ),
                    "bbox_tlwh": [float(x) for x in track.to_tlwh()],
                }
                for track in candidate_tracks
            ],
            appearance_cost_matrix=appearance_cost.tolist(),
            learned_temporal_score_matrix=learned_score.tolist(),
            learned_short_attn_matrices=[matrix.tolist() for matrix in learned_short_attn],
            learned_short_attn_top_matrix=learned_diag["short_attn_top"].tolist(),
            learned_short_attn_entropy_matrix=learned_diag["short_attn_entropy"].tolist(),
            learned_short_similarity_matrix=learned_diag["short_similarity"].tolist(),
            learned_long_gate_matrix=learned_diag["long_gate"].tolist(),
            learned_long_similarity_matrix=learned_diag["long_similarity"].tolist(),
            learned_long_attn_recent_matrix=learned_diag["long_attn_recent"].tolist(),
            learned_long_attn_top_matrix=learned_diag["long_attn_top"].tolist(),
            learned_long_attn_entropy_matrix=learned_diag["long_attn_entropy"].tolist(),
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


def _empty_learned_diagnostics(num_tracks, num_dets):
    return {
        "short_attn_top": np.zeros((num_tracks, num_dets), dtype=np.float32),
        "short_attn_entropy": np.zeros((num_tracks, num_dets), dtype=np.float32),
        "short_similarity": np.zeros((num_tracks, num_dets), dtype=np.float32),
        "long_gate": np.zeros((num_tracks, num_dets), dtype=np.float32),
        "long_similarity": np.zeros((num_tracks, num_dets), dtype=np.float32),
        "long_attn_recent": np.zeros((num_tracks, num_dets), dtype=np.float32),
        "long_attn_top": np.zeros((num_tracks, num_dets), dtype=np.float32),
        "long_attn_entropy": np.zeros((num_tracks, num_dets), dtype=np.float32),
    }


def _normalized_entropy(weights, valid_len):
    valid_len = max(1, int(valid_len))
    weights = np.asarray(weights[:valid_len], dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    if valid_len <= 1:
        return 0.0
    return entropy / np.log(valid_len)


def _compute_learned_temporal_matrices(candidate_tracks, detections, model, stride):
    num_tracks = len(candidate_tracks)
    num_dets = len(detections)
    score_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    diag = _empty_learned_diagnostics(num_tracks, num_dets)
    history_len = getattr(model, "history_len", 5)
    short_attn_matrices = [
        np.zeros((num_tracks, num_dets), dtype=np.float32)
        for _ in range(history_len)
    ]

    det_batch = []
    hist_batch = []
    long_hist_batch = []
    short_len_batch = []
    long_len_batch = []
    pair_indices = []

    def _build_short_history(track):
        short_memory = getattr(track, "short_memory", [])
        if len(short_memory) > 0:
            valid_len = min(len(short_memory), history_len)
            recent = [
                np.asarray(feat, dtype=np.float32)
                for feat in short_memory[-history_len:]
            ][::-1]
            while len(recent) < history_len:
                recent.append(recent[-1])
            return np.stack(recent, axis=0), valid_len

        history = getattr(track, "det_feat_history", [])
        if len(history) >= history_len:
            valid_len = min(len(history), history_len)
            recent = [
                np.asarray(feat, dtype=np.float32)
                for feat in history[-history_len:]
            ][::-1]
            return np.stack(recent, axis=0), valid_len
        return None

    def _build_long_history(track):
        long_history_len = getattr(model, "long_history_len", 30)
        long_memory = getattr(track, "long_memory", [])
        if len(long_memory) > 0:
            valid_len = min(len(long_memory), long_history_len)
            long_items = [np.asarray(feat, dtype=np.float32) for feat in long_memory[-long_history_len:]]
            while len(long_items) < long_history_len:
                long_items.append(long_items[-1])
            return np.stack(long_items, axis=0), valid_len
        history = getattr(track, "det_feat_history", [])
        if len(history) > 0:
            valid_len = min(len(history), long_history_len)
            long_items = [np.asarray(feat, dtype=np.float32) for feat in history[-long_history_len:]]
            while len(long_items) < long_history_len:
                long_items.append(long_items[-1])
            return np.stack(long_items, axis=0), valid_len
        return None

    for i, track in enumerate(candidate_tracks):
        short_result = _build_short_history(track)
        if short_result is None:
            continue
        short_hist, short_hist_len = short_result
        long_result = _build_long_history(track)
        if long_result is None:
            long_hist = np.repeat(short_hist[-1:, :], getattr(model, "long_history_len", 30), axis=0)
            long_hist_len = 1
        else:
            long_hist, long_hist_len = long_result
        for j, det in enumerate(detections):
            det_feat = np.asarray(det.feature, dtype=np.float32)
            det_norm = np.linalg.norm(det_feat)
            if det_norm > 1e-12:
                det_feat = det_feat / det_norm
            det_batch.append(det_feat)
            hist_batch.append(short_hist)
            long_hist_batch.append(long_hist)
            short_len_batch.append(short_hist_len)
            long_len_batch.append(long_hist_len)
            pair_indices.append((i, j))

    if not pair_indices:
        return score_matrix, short_attn_matrices, diag

    det_tensor = torch.from_numpy(np.stack(det_batch, axis=0))
    hist_tensor = torch.from_numpy(np.stack(hist_batch, axis=0))
    long_hist_tensor = torch.from_numpy(np.stack(long_hist_batch, axis=0))
    short_len_tensor = torch.from_numpy(np.asarray(short_len_batch, dtype=np.int64))
    long_len_tensor = torch.from_numpy(np.asarray(long_len_batch, dtype=np.int64))

    with torch.no_grad():
        score_tensor, attn_tensor = model(
            det_tensor,
            hist_tensor,
            long_hist_feat=long_hist_tensor,
            short_hist_len=short_len_tensor,
            long_hist_len=long_len_tensor,
            return_attention=True,
        )

    scores = score_tensor.detach().cpu().numpy()
    attn = attn_tensor.detach().cpu().numpy().mean(axis=1).squeeze(1)
    short_similarity = model.last_short_similarity.detach().cpu().numpy().squeeze(1)
    long_gate = model.last_long_gate.detach().cpu().numpy().squeeze(1)
    long_similarity = model.last_long_similarity.detach().cpu().numpy().squeeze(1)
    long_attn_tensor = model.last_long_attention
    if long_attn_tensor is None:
        long_attn = None
    else:
        long_attn = long_attn_tensor.detach().cpu().numpy().mean(axis=1).squeeze(1)
    short_lens = np.asarray(short_len_batch, dtype=np.int64)
    long_lens = np.asarray(long_len_batch, dtype=np.int64)

    for idx, (i, j) in enumerate(pair_indices):
        score_matrix[i, j] = float(scores[idx])
        for token_idx in range(min(len(short_attn_matrices), attn.shape[1])):
            short_attn_matrices[token_idx][i, j] = float(attn[idx, token_idx])
        short_len = int(short_lens[idx])
        short_weights = attn[idx, :short_len]
        diag["short_attn_top"][i, j] = float(np.max(short_weights)) if short_weights.size else 0.0
        diag["short_attn_entropy"][i, j] = _normalized_entropy(attn[idx], short_len)
        diag["short_similarity"][i, j] = float(short_similarity[idx])
        diag["long_gate"][i, j] = float(long_gate[idx])
        diag["long_similarity"][i, j] = float(long_similarity[idx])
        if long_attn is not None:
            long_len = int(long_lens[idx])
            long_weights = long_attn[idx, :long_len]
            recent_count = min(5, long_len)
            recent_weights = long_attn[idx, long_len - recent_count: long_len]
            diag["long_attn_recent"][i, j] = float(np.sum(recent_weights)) if recent_weights.size else 0.0
            diag["long_attn_top"][i, j] = float(np.max(long_weights)) if long_weights.size else 0.0
            diag["long_attn_entropy"][i, j] = _normalized_entropy(long_attn[idx], long_len)

    return score_matrix, short_attn_matrices, diag


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
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(
                image,
                f"D{det['index']} {det['confidence']:.2f}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        for tr in report.tracks:
            x, y, w, h = [int(v) for v in tr["bbox_tlwh"]]
            color = create_unique_color_uchar(tr["track_id"])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            match_conf = tr.get("match_confidence")
            if match_conf is None:
                label = f"T{tr['track_id']}"
            else:
                label = f"T{tr['track_id']} {match_conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            text_x = min(max(0, x + w - text_w), max(0, image.shape[1] - text_w - 1))
            text_y = max(text_h + 2, y - 6)
            cv2.putText(
                image,
                label,
                (text_x, text_y),
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
            match_conf = tr.get("match_confidence")
            match_conf_text = "none" if match_conf is None else f"{match_conf:.3f}"
            self.summary_text.insert(
                tk.END,
                f"  T{tr['track_id']}: hits={tr['hits']}, age={tr['age']}, time_since_update={tr['time_since_update']}, "
                f"match_conf={match_conf_text}, bbox={_format_bbox(tr['bbox_tlwh'])}\n",
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
        self.matrix_text.insert(tk.END, "\nLearned temporal score matrix (sigmoid)\n")
        self.matrix_text.insert(
            tk.END,
            self.format_matrix(
                self._sigmoid_matrix(report.learned_temporal_score_matrix),
                report,
            ),
        )
        self.matrix_text.insert(tk.END, "\nLearned short summary: top attention\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_short_attn_top_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned short summary: attention entropy\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_short_attn_entropy_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned short summary: det/short similarity\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_short_similarity_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned long summary: gate weight\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_long_gate_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned long summary: det/long similarity\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_long_similarity_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned long summary: attention mass on recent 5 valid tokens\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_long_attn_recent_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned long summary: top attention\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_long_attn_top_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned long summary: attention entropy\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_long_attn_entropy_matrix, report))
        for token_idx, matrix in enumerate(report.learned_short_attn_matrices):
            if token_idx == 0:
                label = "short_memory[0] latest"
            else:
                label = f"short_memory[{token_idx}] older"
            self.matrix_text.insert(
                tk.END,
                f"\nLearned attention from det_feat to {label}\n",
            )
            self.matrix_text.insert(tk.END, self.format_matrix(matrix, report))
        self.matrix_text.insert(tk.END, "\nFinal cost matrix(before gating)\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.final_cost_matrix, report))
        self.matrix_text.insert(tk.END, "\nGated distance matrix(motion / Kalman gating)\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.gated_cost_matrix, report))

    @staticmethod
    def _sigmoid_matrix(matrix: list[list[float]]) -> list[list[float]]:
        arr = np.asarray(matrix, dtype=np.float64)
        arr = 1.0 / (1.0 + np.exp(-arr))
        return arr.tolist()

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
    parser.add_argument("--BoT", action="store_true", help="Use BoT configuration")
    parser.add_argument("--ECC", action="store_true", help="Enable ECC")
    parser.add_argument("--NSA", action="store_true", help="Enable NSA")
    parser.add_argument("--EMA", action="store_true", help="Enable EMA")
    parser.add_argument("--MC", action="store_true", help="Enable MC")
    parser.add_argument("--woC", action="store_true", help="Enable woC")
    parser.add_argument("--ltm_stm", action="store_true", help="Enable STM + LTM")
    parser.add_argument("--memory_init", action="store_true", help="Enable delayed long-memory initialization")
    parser.add_argument("--memory_aware", action="store_true", help="Enable memory-aware matching")
    parser.add_argument("--topk", action="store_true", help="Enable top-k matching")
    parser.add_argument("--full", action="store_true", help="Enable full modified pipeline")
    parser.add_argument("--learned_temporal", action="store_true", help="Show learned temporal score matrix using TemporalAttentionScorer")
    parser.add_argument("--fuse_learned_temporal", action="store_true", help="Use learned temporal score in the online tracker association.")
    parser.add_argument("--temporal_model_ckpt", type=str, default=None, help="Optional checkpoint path for the learned temporal scorer")
    parser.add_argument("--temporal_scores_file", type=str, default=None, help="Optional precomputed jsonl from deep_sort_app. If provided, viewer reads learned scores from this file instead of recomputing them.")
    parser.add_argument("--temporal_hidden_dim", type=int, default=256, help="Hidden dimension for the learned temporal scorer")
    parser.add_argument("--temporal_num_heads", type=int, default=4, help="Number of attention heads for the learned temporal scorer")
    parser.add_argument("--learned_temporal_stride", type=int, default=2, help="Deprecated; kept for old commands. Current scorer uses short_memory tokens directly.")
    parser.add_argument("--learned_temporal_alpha", type=float, default=1.0, help="Baseline-vs-learned cost fusion weight used by the online tracker.")
    parser.add_argument("--learned_temporal_max_correction", type=float, default=0.02, help="Maximum absolute cost correction applied by learned temporal fusion.")
    parser.add_argument("--learned_temporal_min_scale", type=float, default=0.02, help="Minimum baseline row scale used by adaptive temporal fusion.")
    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument("--min_detection_height", type=int, default=None)
    parser.add_argument("--nms_max_overlap", type=float, default=None)
    parser.add_argument("--max_cosine_distance", type=float, default=None)
    parser.add_argument("--nn_budget", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    opt_argv = [sys.argv[0], "CustomDemo", "test"]
    if args.BoT:
        opt_argv.append("--BoT")
    if args.ECC:
        opt_argv.append("--ECC")
    if args.NSA:
        opt_argv.append("--NSA")
    if args.EMA:
        opt_argv.append("--EMA")
    if args.MC:
        opt_argv.append("--MC")
    if args.woC:
        opt_argv.append("--woC")
    if args.ltm_stm:
        opt_argv.append("--ltm_stm")
    if args.memory_init:
        opt_argv.append("--memory_init")
    if args.memory_aware:
        opt_argv.append("--memory_aware")
    if args.topk:
        opt_argv.append("--topk")
    if args.full:
        opt_argv.append("--full")
    sys.argv = opt_argv
    from opts import opt

    print("Preparing frame reports. This may take a moment...")
    min_confidence = opt.min_confidence if args.min_confidence is None else args.min_confidence
    min_detection_height = (
        opt.min_detection_height if args.min_detection_height is None else args.min_detection_height
    )
    nms_max_overlap = opt.nms_max_overlap if args.nms_max_overlap is None else args.nms_max_overlap
    max_cosine_distance = (
        opt.max_cosine_distance if args.max_cosine_distance is None else args.max_cosine_distance
    )
    nn_budget = opt.nn_budget if args.nn_budget is None else args.nn_budget
    reports, min_frame, max_frame = build_reports(
        sequence_dir=args.sequence_dir,
        detection_file=args.detection_file,
        min_confidence=min_confidence,
        nms_max_overlap=nms_max_overlap,
        min_detection_height=min_detection_height,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget,
        enable_learned_temporal=args.learned_temporal,
        temporal_model_ckpt=args.temporal_model_ckpt,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_num_heads=args.temporal_num_heads,
        temporal_stride=args.learned_temporal_stride,
        temporal_alpha=args.learned_temporal_alpha,
        fuse_learned_temporal=args.fuse_learned_temporal,
        temporal_max_correction=args.learned_temporal_max_correction,
        temporal_min_scale=args.learned_temporal_min_scale,
        temporal_scores_file=args.temporal_scores_file,
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
