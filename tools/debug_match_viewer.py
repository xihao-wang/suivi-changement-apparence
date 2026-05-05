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
    learned_temporal_score_matrix: list[list[float]]
    learned_attn_init_matrix: list[list[float]]
    learned_attn_t_matrix: list[list[float]]
    learned_attn_t_i_matrix: list[list[float]]
    learned_attn_t_2i_matrix: list[list[float]]
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
    result_file: str | None,
    temporal_scores_file: str | None,
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
) -> tuple[list[FrameReport], int, int, dict[str, Any] | None]:
    from application_util import preprocessing
    from deep_sort import linear_assignment, nn_matching
    from deep_sort.temporal_model import TemporalAttentionScorer
    from deep_sort.tracker import Tracker
    from deep_sort_app import create_detections, gather_sequence_info
    from opts import opt
    from tools.train_temporal_model import (
        crop_and_resize,
        frame_to_image_path,
        image_to_tensor,
    )

    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)
    learned_temporal_model = None
    patch_runtime = None
    if enable_learned_temporal and temporal_scores_file is None:
        if not temporal_model_ckpt:
            raise ValueError("--temporal_model_ckpt is required with --learned_temporal when --temporal_scores_file is not provided")
        if temporal_model_ckpt:
            ckpt = torch.load(temporal_model_ckpt, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(ckpt, dict) or "image_height" not in ckpt:
                raise ValueError(
                    "This viewer now expects a patch-token temporal checkpoint "
                    "with image_height/image_width metadata."
                )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            history_indices = ckpt.get("history_indices", [0, 1, 2])
            learned_temporal_model = TemporalAttentionScorer(
                image_height=int(ckpt["image_height"]),
                image_width=int(ckpt["image_width"]),
                patch_size=int(ckpt["patch_size"]),
                hidden_dim=int(ckpt["hidden_dim"]),
                num_heads=int(ckpt["num_heads"]),
                history_len=int(ckpt["history_len"]),
                num_stages=int(ckpt.get("num_stages", 1)),
                stage_dims=ckpt.get("stage_dims"),
                stage_heads=ckpt.get("stage_heads"),
                stage_depths=ckpt.get("stage_depths"),
                stage_kernels=ckpt.get("stage_kernels"),
                stage_strides=ckpt.get("stage_strides"),
            ).to(device)
            learned_temporal_model.load_state_dict(state_dict, strict=False)
            learned_temporal_model.eval()
            patch_runtime = {
                "device": device,
                "history_indices": history_indices,
                "history_stride": max(1, int(temporal_stride)),
                "image_height": int(ckpt["image_height"]),
                "image_width": int(ckpt["image_width"]),
                "keep_aspect_ratio": bool(ckpt.get("keep_aspect_ratio", True)),
                "num_patches": learned_temporal_model.num_patches,
                "crop_and_resize": crop_and_resize,
                "frame_to_image_path": frame_to_image_path,
                "image_to_tensor": image_to_tensor,
                "frame_cache": {},
            }
            tracker.set_patch_temporal_matcher(learned_temporal_model, patch_runtime)

    reports: list[FrameReport] = []
    min_frame = seq_info["min_frame_idx"]
    max_frame = seq_info["max_frame_idx"]
    result_by_frame: dict[int, list[dict[str, Any]]] = {}
    temporal_scores_by_frame: dict[int, dict[str, Any]] = {}
    track_crop_state: dict[int, dict[str, Any]] = {}
    if result_file:
        result_path = Path(result_file)
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_file}")
        for line in result_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(float(parts[0]))
            track_id = int(float(parts[1]))
            tlwh = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            result_by_frame.setdefault(frame_idx, []).append(
                {
                    "track_id": track_id,
                    "hits": -1,
                    "age": -1,
                    "time_since_update": 0,
                    "bbox_tlwh": tlwh,
                }
            )
    if temporal_scores_file:
        score_path = Path(temporal_scores_file)
        if not score_path.exists():
            raise FileNotFoundError(f"Temporal scores file not found: {temporal_scores_file}")
        for line in score_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            temporal_scores_by_frame[int(item["frame"])] = item

    for frame_idx in range(min_frame, max_frame + 1):
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height
        )
        detections = [d for d in detections if d.confidence >= min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        detection_display_indices = list(range(len(detections)))

        if result_file is None:
            if opt.ECC:
                tracker.camera_update(Path(sequence_dir).name, frame_idx)

            tracker.predict()

            confirmed_track_indices = [
                i for i, t in enumerate(tracker.tracks) if t.is_confirmed()
            ]
            candidate_tracks = [tracker.tracks[i] for i in confirmed_track_indices]

            if len(candidate_tracks) > 0 and len(detections) > 0:
                if enable_learned_temporal:
                    learned_score, learned_attn_init, learned_attn_t, learned_attn_t_i, learned_attn_t_2i = \
                        _compute_learned_temporal_matrices(
                        frame_idx,
                        sequence_dir,
                        candidate_tracks,
                        detections,
                        learned_temporal_model,
                        track_crop_state,
                        patch_runtime,
                    )
                else:
                    learned_score = np.zeros((len(candidate_tracks), len(detections)))
                    learned_attn_init = np.zeros((len(candidate_tracks), len(detections)))
                    learned_attn_t = np.zeros((len(candidate_tracks), len(detections)))
                    learned_attn_t_i = np.zeros((len(candidate_tracks), len(detections)))
                    learned_attn_t_2i = np.zeros((len(candidate_tracks), len(detections)))
            else:
                learned_score = np.zeros((len(candidate_tracks), len(detections)))
                learned_attn_init = np.zeros((len(candidate_tracks), len(detections)))
                learned_attn_t = np.zeros((len(candidate_tracks), len(detections)))
                learned_attn_t_i = np.zeros((len(candidate_tracks), len(detections)))
                learned_attn_t_2i = np.zeros((len(candidate_tracks), len(detections)))
            matches, unmatched_tracks, unmatched_detections = tracker._match(detections)
            tracks_payload = [
                {
                    "track_id": track.track_id,
                    "hits": track.hits,
                    "age": track.age,
                    "time_since_update": track.time_since_update,
                    "bbox_tlwh": [float(x) for x in track.to_tlwh()],
                }
                for track in candidate_tracks
            ]
            unmatched_track_ids = [tracker.tracks[idx].track_id for idx in unmatched_tracks]
            ambiguous_track_ids = list(tracker.last_ambiguous_tracks)
            ambiguous_info = dict(tracker.last_ambiguous_info)
            matches_payload = [
                {
                    "track_id": tracker.tracks[track_idx].track_id,
                    "detection_index": detection_idx,
                }
                for track_idx, detection_idx in matches
            ]
        else:
            score_item = temporal_scores_by_frame.get(frame_idx)
            all_tracks_payload = list(result_by_frame.get(frame_idx, []))
            if score_item is not None:
                score_track_ids = [int(x) for x in score_item.get("track_ids", [])]
                score_det_indices = [int(x) for x in score_item.get("detection_indices", [])]
                score_det_index_set = set(score_det_indices)
                track_lookup = {
                    int(tr["track_id"]): tr for tr in all_tracks_payload
                }
                tracks_payload = [
                    track_lookup[track_id]
                    for track_id in score_track_ids
                    if track_id in track_lookup
                ]
                selected_detections = [
                    (det_idx, det)
                    for det_idx, det in enumerate(detections)
                    if det_idx in score_det_index_set
                ]
                detections = [det for _, det in selected_detections]
                detection_display_indices = [det_idx for det_idx, _ in selected_detections]
            else:
                tracks_payload = all_tracks_payload
            candidate_tracks = []
            if score_item is not None:
                learned_score = np.asarray(score_item["score_matrix"], dtype=np.float32)
                learned_attn_init = np.asarray(score_item["attn_init_matrix"], dtype=np.float32)
                learned_attn_t = np.asarray(score_item["attn_t_matrix"], dtype=np.float32)
                learned_attn_t_i = np.asarray(score_item["attn_t_i_matrix"], dtype=np.float32)
                learned_attn_t_2i = np.asarray(score_item["attn_t_2i_matrix"], dtype=np.float32)
            else:
                learned_score = np.zeros((len(tracks_payload), len(detections)))
                learned_attn_init = np.zeros((len(tracks_payload), len(detections)))
                learned_attn_t = np.zeros((len(tracks_payload), len(detections)))
                learned_attn_t_i = np.zeros((len(tracks_payload), len(detections)))
                learned_attn_t_2i = np.zeros((len(tracks_payload), len(detections)))
            matches_payload = []
            matched_track_ids = set()
            unmatched_track_ids = [tr["track_id"] for tr in tracks_payload]
            unmatched_detections = list(range(len(detections)))
            ambiguous_track_ids = []
            ambiguous_info = {}

        report = FrameReport(
            frame=frame_idx,
            image_path=seq_info["image_filenames"][frame_idx],
            detections=[
                {
                    "index": display_idx,
                    "confidence": float(det.confidence),
                    "bbox_tlwh": [float(x) for x in det.tlwh],
                }
                for display_idx, det in zip(detection_display_indices, detections)
            ],
            tracks=tracks_payload,
            learned_temporal_score_matrix=learned_score.tolist(),
            learned_attn_init_matrix=learned_attn_init.tolist(),
            learned_attn_t_matrix=learned_attn_t.tolist(),
            learned_attn_t_i_matrix=learned_attn_t_i.tolist(),
            learned_attn_t_2i_matrix=learned_attn_t_2i.tolist(),
            matches=matches_payload,
            unmatched_track_ids=unmatched_track_ids,
            unmatched_detection_indices=list(unmatched_detections),
            ambiguous_track_ids=ambiguous_track_ids,
            ambiguous_info=ambiguous_info,
        )
        reports.append(report)

        if result_file is None:
            for track_idx, detection_idx in matches:
                track = tracker.tracks[track_idx]
                det = detections[detection_idx]
                state = track_crop_state.setdefault(
                    track.track_id,
                    {
                        "init": (frame_idx, np.asarray(det.tlwh, dtype=np.float32).copy()),
                        "history": [],
                    },
                )
                state["history"].append(
                    (frame_idx, np.asarray(det.tlwh, dtype=np.float32).copy())
                )
                tracker.tracks[track_idx].update(detections[detection_idx])
            for track_idx in unmatched_tracks:
                tracker.tracks[track_idx].mark_missed()
            for detection_idx in unmatched_detections:
                det = detections[detection_idx]
                tracker._initiate_track(detections[detection_idx])
                new_track = tracker.tracks[-1]
                track_crop_state[new_track.track_id] = {
                    "init": (frame_idx, np.asarray(det.tlwh, dtype=np.float32).copy()),
                    "history": [(frame_idx, np.asarray(det.tlwh, dtype=np.float32).copy())],
                }
            deleted_track_ids = {t.track_id for t in tracker.tracks if t.is_deleted()}
            tracker.tracks = [t for t in tracker.tracks if not t.is_deleted()]
            for track_id in deleted_track_ids:
                track_crop_state.pop(track_id, None)

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

    return reports, min_frame, max_frame, None


def _get_cached_image(runtime, sequence_dir, frame_idx):
    image_path = runtime["frame_to_image_path"](sequence_dir, int(frame_idx))
    cache = runtime["frame_cache"]
    image = cache.get(image_path)
    if image is None:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        cache[image_path] = image
    return image


def _make_crop_tensor(runtime, sequence_dir, frame_idx, tlwh):
    image = _get_cached_image(runtime, sequence_dir, frame_idx)
    crop = runtime["crop_and_resize"](
        image,
        tlwh,
        runtime["image_height"],
        runtime["image_width"],
        keep_aspect_ratio=runtime["keep_aspect_ratio"],
    )
    return runtime["image_to_tensor"](crop)


def _compute_learned_temporal_matrices(
    frame_idx,
    sequence_dir,
    candidate_tracks,
    detections,
    model,
    track_crop_state,
    runtime,
):
    num_tracks = len(candidate_tracks)
    num_dets = len(detections)
    score_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_init_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_i_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_2i_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

    init_batch = []
    det_batch = []
    hist_batch = []
    pair_indices = []
    history_indices = runtime["history_indices"]
    stride = runtime["history_stride"]

    for i, track in enumerate(candidate_tracks):
        state = track_crop_state.get(track.track_id)
        if state is None:
            continue
        history = state["history"]
        max_required = max(history_indices) * stride
        if len(history) < max_required + 1:
            continue
        init_frame, init_bbox = state["init"]
        hist_items = []
        for hist_idx in history_indices:
            hist_frame, hist_bbox = history[-1 - hist_idx * stride]
            hist_items.append(
                _make_crop_tensor(runtime, sequence_dir, hist_frame, hist_bbox)
            )
        init_tensor = _make_crop_tensor(runtime, sequence_dir, init_frame, init_bbox)
        for j, det in enumerate(detections):
            init_batch.append(init_tensor)
            det_batch.append(
                _make_crop_tensor(runtime, sequence_dir, frame_idx, det.tlwh)
            )
            hist_batch.append(torch.stack(hist_items, dim=0))
            pair_indices.append((i, j))

    if not pair_indices:
        return (
            score_matrix,
            attn_init_matrix,
            attn_t_matrix,
            attn_t_i_matrix,
            attn_t_2i_matrix,
        )

    device = runtime["device"]
    init_tensor = torch.stack(init_batch, dim=0).to(device)
    det_tensor = torch.stack(det_batch, dim=0).to(device)
    hist_tensor = torch.stack(hist_batch, dim=0).to(device)

    with torch.no_grad():
        score_tensor, attn_dict = model(
            init_tensor,
            det_tensor,
            hist_tensor,
            return_attention=True,
        )

    scores = score_tensor.detach().cpu().numpy()
    search_attn = attn_dict["search_attn"].detach().cpu().numpy()
    attn = search_attn.mean(axis=1).mean(axis=1)  # (B, K)
    num_patches = runtime["num_patches"]

    for idx, (i, j) in enumerate(pair_indices):
        score_matrix[i, j] = float(scores[idx])
        attn_init_matrix[i, j] = float(attn[idx, :num_patches].sum())
        cursor = num_patches
        group_masses = []
        for _ in history_indices:
            group_masses.append(float(attn[idx, cursor:cursor + num_patches].sum()))
            cursor += num_patches
        if len(group_masses) > 0:
            attn_t_matrix[i, j] = group_masses[0]
        if len(group_masses) > 1:
            attn_t_i_matrix[i, j] = group_masses[1]
        if len(group_masses) > 2:
            attn_t_2i_matrix[i, j] = group_masses[2]

    return (
        score_matrix,
        attn_init_matrix,
        attn_t_matrix,
        attn_t_i_matrix,
        attn_t_2i_matrix,
    )


def _compute_learned_temporal_matrices_from_report(
    report: FrameReport,
    sequence_dir: str,
    model,
    runtime,
):
    num_tracks = len(report.tracks)
    num_dets = len(report.detections)
    score_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_init_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_i_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_2i_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    if num_tracks == 0 or num_dets == 0:
        return (
            score_matrix,
            attn_init_matrix,
            attn_t_matrix,
            attn_t_i_matrix,
            attn_t_2i_matrix,
        )

    init_batch = []
    det_batch = []
    hist_batch = []
    pair_indices = []
    history_indices = runtime["history_indices"]
    stride = runtime["history_stride"]

    for i, track in enumerate(report.tracks):
        state = report.track_states.get(track["track_id"])
        if state is None:
            continue
        history = state["history"]
        max_required = max(history_indices) * stride
        if len(history) < max_required + 1:
            continue
        init_frame, init_bbox = state["init"]
        hist_items = []
        for hist_idx in history_indices:
            hist_frame, hist_bbox = history[-1 - hist_idx * stride]
            hist_items.append(
                _make_crop_tensor(runtime, sequence_dir, hist_frame, hist_bbox)
            )
        init_tensor = _make_crop_tensor(runtime, sequence_dir, init_frame, init_bbox)
        for j, det in enumerate(report.detections):
            init_batch.append(init_tensor)
            det_batch.append(
                _make_crop_tensor(runtime, sequence_dir, report.frame, det["bbox_tlwh"])
            )
            hist_batch.append(torch.stack(hist_items, dim=0))
            pair_indices.append((i, j))

    if not pair_indices:
        return (
            score_matrix,
            attn_init_matrix,
            attn_t_matrix,
            attn_t_i_matrix,
            attn_t_2i_matrix,
        )

    device = runtime["device"]
    init_tensor = torch.stack(init_batch, dim=0).to(device)
    det_tensor = torch.stack(det_batch, dim=0).to(device)
    hist_tensor = torch.stack(hist_batch, dim=0).to(device)

    with torch.no_grad():
        score_tensor, attn_dict = model(
            init_tensor,
            det_tensor,
            hist_tensor,
            return_attention=True,
        )

    scores = score_tensor.detach().cpu().numpy()
    search_attn = attn_dict["search_attn"].detach().cpu().numpy()
    attn = search_attn.mean(axis=1).mean(axis=1)
    num_patches = runtime["num_patches"]

    for idx, (i, j) in enumerate(pair_indices):
        score_matrix[i, j] = float(scores[idx])
        attn_init_matrix[i, j] = float(attn[idx, num_patches:2 * num_patches].sum())
        cursor = 2 * num_patches
        group_masses = []
        for _ in history_indices:
            group_masses.append(float(attn[idx, cursor:cursor + num_patches].sum()))
            cursor += num_patches
        if len(group_masses) > 0:
            attn_t_matrix[i, j] = group_masses[0]
        if len(group_masses) > 1:
            attn_t_i_matrix[i, j] = group_masses[1]
        if len(group_masses) > 2:
            attn_t_2i_matrix[i, j] = group_masses[2]

    return (
        score_matrix,
        attn_init_matrix,
        attn_t_matrix,
        attn_t_i_matrix,
        attn_t_2i_matrix,
    )


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
            label = f"T{tr['track_id']}"
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

        self.matrix_text.insert(tk.END, "\nLearned temporal score matrix (sigmoid)\n")
        self.matrix_text.insert(
            tk.END,
            self.format_matrix(
                self._sigmoid_matrix(report.learned_temporal_score_matrix),
                report,
            ),
        )
        self.matrix_text.insert(tk.END, "\nLearned attention mass on initial template\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_attn_init_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned attention mass on df_t\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_attn_t_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned attention mass on df_t-i\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_attn_t_i_matrix, report))
        self.matrix_text.insert(tk.END, "\nLearned attention mass on df_t-2i\n")
        self.matrix_text.insert(tk.END, self.format_matrix(report.learned_attn_t_2i_matrix, report))

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
    parser.add_argument("--result_file", type=str, default=None, help="Optional tracking result txt. If provided, viewer reads track boxes directly from this file instead of rerunning tracker.")
    parser.add_argument("--temporal_scores_file", type=str, default=None, help="Optional jsonl sidecar written by deep_sort_app.py containing per-frame learned score matrices")
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
    parser.add_argument("--learned_temporal", action="store_true", help="Show learned temporal score matrix using the patch-token TemporalAttentionScorer")
    parser.add_argument("--temporal_model_ckpt", type=str, default=None, help="Checkpoint path for the patch-token learned temporal scorer")
    parser.add_argument("--temporal_hidden_dim", type=int, default=256, help="Unused for patch-token checkpoints; kept for compatibility")
    parser.add_argument("--temporal_num_heads", type=int, default=4, help="Unused for patch-token checkpoints; kept for compatibility")
    parser.add_argument("--learned_temporal_stride", type=int, default=2, help="Temporal stride used to sample online templates [t, t-i, t-2i]")
    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument("--min_detection_height", type=int, default=None)
    parser.add_argument("--nms_max_overlap", type=float, default=None)
    parser.add_argument("--max_cosine_distance", type=float, default=None)
    parser.add_argument("--nn_budget", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.result_file and args.learned_temporal and not args.temporal_scores_file:
        raise ValueError(
            "--result_file and --learned_temporal require --temporal_scores_file. "
            "Otherwise remove --result_file to run baseline tracker and compute learned temporal matrices on its live candidate tracks."
        )
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
    reports, min_frame, max_frame, lazy_context = build_reports(
        sequence_dir=args.sequence_dir,
        detection_file=args.detection_file,
        result_file=args.result_file,
        temporal_scores_file=args.temporal_scores_file,
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
