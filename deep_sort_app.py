# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

from application_util import preprocessing
from application_util import visualization
from deep_sort.detection import Detection


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def _build_short_history(track, history_len):
    short_memory = getattr(track, "short_memory", [])
    if len(short_memory) > 0:
        valid_len = min(len(short_memory), history_len)
        items = [np.asarray(feat, dtype=np.float32) for feat in short_memory[-history_len:]][::-1]
        while len(items) < history_len:
            items.append(items[-1])
        return np.stack(items, axis=0), valid_len

    history = getattr(track, "det_feat_history", [])
    if len(history) > 0:
        valid_len = min(len(history), history_len)
        items = [np.asarray(feat, dtype=np.float32) for feat in history[-history_len:]][::-1]
        while len(items) < history_len:
            items.append(items[-1])
        return np.stack(items, axis=0), valid_len
    return None


def _build_long_history(track, long_history_len):
    long_memory = getattr(track, "long_memory", [])
    if len(long_memory) > 0:
        valid_len = min(len(long_memory), long_history_len)
        items = [np.asarray(feat, dtype=np.float32) for feat in long_memory[-long_history_len:]]
        while len(items) < long_history_len:
            items.append(items[-1])
        return np.stack(items, axis=0), valid_len

    history = getattr(track, "det_feat_history", [])
    if len(history) > 0:
        valid_len = min(len(history), long_history_len)
        items = [np.asarray(feat, dtype=np.float32) for feat in history[-long_history_len:]]
        while len(items) < long_history_len:
            items.append(items[-1])
        return np.stack(items, axis=0), valid_len
    return None


def _load_temporal_model(temporal_model_ckpt, feature_dim):
    from deep_sort.temporal_model import TemporalAttentionScorer

    if temporal_model_ckpt is None:
        raise ValueError("--temporal_model_ckpt is required with --learned_temporal")
    ckpt = torch.load(temporal_model_ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    hidden_dim = int(ckpt.get("hidden_dim", 256)) if isinstance(ckpt, dict) else 256
    num_heads = int(ckpt.get("num_heads", 4)) if isinstance(ckpt, dict) else 4
    history_len = int(ckpt.get("history_len", 5)) if isinstance(ckpt, dict) else 5
    long_history_len = int(ckpt.get("long_history_len", 30)) if isinstance(ckpt, dict) else 30
    use_long_memory = bool(ckpt.get("use_long_memory", True)) if isinstance(ckpt, dict) else True
    model = TemporalAttentionScorer(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        history_len=history_len,
        long_history_len=long_history_len,
        use_long_memory=use_long_memory,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _compute_temporal_scores(candidate_tracks, detections, model):
    score_matrix = np.zeros((len(candidate_tracks), len(detections)), dtype=np.float32)
    det_batch, short_batch, long_batch = [], [], []
    short_len_batch, long_len_batch, pair_indices = [], [], []
    history_len = getattr(model, "history_len", 5)
    long_history_len = getattr(model, "long_history_len", 30)

    for track_idx, track in enumerate(candidate_tracks):
        short_result = _build_short_history(track, history_len)
        if short_result is None:
            continue
        short_hist, short_len = short_result
        long_result = _build_long_history(track, long_history_len)
        if long_result is None:
            long_hist = np.repeat(short_hist[-1:, :], long_history_len, axis=0)
            long_len = 1
        else:
            long_hist, long_len = long_result
        for det_idx, det in enumerate(detections):
            det_feat = np.asarray(det.feature, dtype=np.float32)
            norm = np.linalg.norm(det_feat)
            if norm > 1e-12:
                det_feat = det_feat / norm
            det_batch.append(det_feat)
            short_batch.append(short_hist)
            long_batch.append(long_hist)
            short_len_batch.append(short_len)
            long_len_batch.append(long_len)
            pair_indices.append((track_idx, det_idx))

    if not pair_indices:
        return score_matrix

    with torch.no_grad():
        scores = model(
            torch.from_numpy(np.stack(det_batch, axis=0)),
            torch.from_numpy(np.stack(short_batch, axis=0)),
            long_hist_feat=torch.from_numpy(np.stack(long_batch, axis=0)),
            short_hist_len=torch.from_numpy(np.asarray(short_len_batch, dtype=np.int64)),
            long_hist_len=torch.from_numpy(np.asarray(long_len_batch, dtype=np.int64)),
            return_attention=False,
        ).detach().cpu().numpy()

    for idx, (track_idx, det_idx) in enumerate(pair_indices):
        score_matrix[track_idx, det_idx] = float(scores[idx])
    return score_matrix


def _compute_debug_matrices(tracker, detections, learned_score_matrix=None):
    from deep_sort import linear_assignment

    confirmed_track_indices = [
        idx for idx, track in enumerate(tracker.tracks) if track.is_confirmed()
    ]
    detection_indices = list(range(len(detections)))
    candidate_tracks = [tracker.tracks[idx] for idx in confirmed_track_indices]
    if not candidate_tracks or not detections:
        empty = np.zeros((len(candidate_tracks), len(detections)), dtype=np.float32)
        return confirmed_track_indices, empty, empty.copy(), empty.copy()

    features = np.array([detections[i].feature for i in detection_indices])
    appearance_cost, final_cost = tracker.metric.distance_components_with_memory(
        features, candidate_tracks
    )
    if (
        learned_score_matrix is not None
        and tracker.temporal_model is not None
        and tracker.fuse_temporal_model
    ):
        learned_prob = 1.0 / (1.0 + np.exp(-learned_score_matrix))
        temporal_cost = 1.0 - learned_prob
        final_cost = tracker._fuse_temporal_cost(final_cost, temporal_cost)
    gated_cost = linear_assignment.gate_cost_matrix(
        final_cost.copy(),
        tracker.tracks,
        detections,
        confirmed_track_indices,
        detection_indices,
    )
    return confirmed_track_indices, appearance_cost, final_cost, gated_cost


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, learned_temporal=False, temporal_model_ckpt=None,
        temporal_scores_file=None, learned_temporal_alpha=1.0,
        fuse_learned_temporal=False, learned_temporal_max_correction=0.02,
        learned_temporal_min_scale=0.02):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    from deep_sort import nn_matching
    from deep_sort.tracker import Tracker
    from opts import opt

    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )
    temporal_model = None
    temporal_score_rows = []
    if learned_temporal:
        temporal_model = _load_temporal_model(
            temporal_model_ckpt, seq_info["feature_dim"]
        )
    tracker = Tracker(
        metric,
        temporal_model=temporal_model if learned_temporal else None,
        temporal_alpha=learned_temporal_alpha,
        fuse_temporal_model=fuse_learned_temporal,
        temporal_max_correction=learned_temporal_max_correction,
        temporal_min_scale=learned_temporal_min_scale,
    )
    results = []

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        if opt.ECC:
            tracker.camera_update(sequence_dir.split('/')[-1], frame_idx)

        tracker.predict()
        temporal_score_row = None
        if temporal_model is not None:
            candidate_tracks = [track for track in tracker.tracks if track.is_confirmed()]
            score_matrix = _compute_temporal_scores(candidate_tracks, detections, temporal_model)
            temporal_score_row = {
                "frame": int(frame_idx),
                "track_ids": [int(track.track_id) for track in candidate_tracks],
                "detection_indices": list(range(len(detections))),
                "scores": score_matrix.tolist(),
            }
        elif temporal_scores_file:
            temporal_score_row = {
                "frame": int(frame_idx),
                "track_ids": [],
                "detection_indices": list(range(len(detections))),
                "scores": [],
            }
        if temporal_score_row is not None:
            matrix_track_indices, appearance_cost, final_cost, gated_cost = _compute_debug_matrices(
                tracker,
                detections,
                learned_score_matrix=score_matrix if temporal_model is not None else None,
            )
            temporal_score_row["matrix_track_ids"] = [
                int(tracker.tracks[idx].track_id) for idx in matrix_track_indices
            ]
            temporal_score_row["appearance_cost_matrix"] = appearance_cost.tolist()
            temporal_score_row["final_cost_matrix"] = final_cost.tolist()
            temporal_score_row["gated_cost_matrix"] = gated_cost.tolist()
        tracker.update(detections)
        if temporal_score_row is not None:
            temporal_score_rows.append(temporal_score_row)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            match_confidence = getattr(track, "match_confidence", None)
            if match_confidence is None:
                match_confidence = 0.0
            results.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3],
                    float(match_confidence)])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6]),file=f)
    f.close()

    if temporal_scores_file:
        score_dir = os.path.dirname(temporal_scores_file)
        if score_dir:
            os.makedirs(score_dir, exist_ok=True)
        with open(temporal_scores_file, "w") as score_f:
            for row in temporal_score_rows:
                score_f.write(json.dumps(row) + "\n")

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.6, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.4)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
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
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--phase_truncation", action="store_true")
    parser.add_argument("--learned_temporal", action="store_true")
    parser.add_argument("--fuse_learned_temporal", action="store_true")
    parser.add_argument("--temporal_model_ckpt", default=None)
    parser.add_argument("--temporal_scores_file", default=None)
    parser.add_argument("--learned_temporal_alpha", type=float, default=1.0)
    parser.add_argument("--learned_temporal_max_correction", type=float, default=0.02)
    parser.add_argument("--learned_temporal_min_scale", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    opt_argv = [sys.argv[0], "CustomDemo", "test"]
    for flag in [
        "BoT",
        "ECC",
        "NSA",
        "EMA",
        "MC",
        "woC",
        "ltm_stm",
        "memory_init",
        "memory_aware",
        "topk",
        "full",
        "phase_truncation",
    ]:
        if getattr(args, flag):
            opt_argv.append(f"--{flag}")
    sys.argv = opt_argv
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display,
        learned_temporal=args.learned_temporal,
        temporal_model_ckpt=args.temporal_model_ckpt,
        temporal_scores_file=args.temporal_scores_file,
        learned_temporal_alpha=args.learned_temporal_alpha,
        fuse_learned_temporal=args.fuse_learned_temporal,
        learned_temporal_max_correction=args.learned_temporal_max_correction,
        learned_temporal_min_scale=args.learned_temporal_min_scale)
