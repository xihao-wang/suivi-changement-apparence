# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import json
import os

import cv2
import numpy as np
import torch

from application_util import preprocessing
from application_util import visualization
from deep_sort.detection import Detection
from deep_sort.temporal_model import TemporalAttentionScorer
from tools.train_temporal_model import crop_and_resize, frame_to_image_path, image_to_tensor


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


def _compute_temporal_payload(frame_idx, sequence_dir, candidate_tracks, detections, model, track_crop_state, runtime):
    num_tracks = len(candidate_tracks)
    num_dets = len(detections)
    score_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_init_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_i_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    attn_t_2i_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
    history_indices = runtime["history_indices"]
    stride = runtime["history_stride"]
    init_batch = []
    det_batch = []
    hist_batch = []
    pair_indices = []

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
            hist_items.append(_make_crop_tensor(runtime, sequence_dir, hist_frame, hist_bbox))
        init_tensor = _make_crop_tensor(runtime, sequence_dir, init_frame, init_bbox)
        for j, det in enumerate(detections):
            init_batch.append(init_tensor)
            det_batch.append(_make_crop_tensor(runtime, sequence_dir, frame_idx, det.tlwh))
            hist_batch.append(torch.stack(hist_items, dim=0))
            pair_indices.append((i, j))

    if pair_indices:
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

    return {
        "frame": int(frame_idx),
        "track_ids": [int(t.track_id) for t in candidate_tracks],
        "detection_indices": [int(i) for i in range(len(detections))],
        "score_matrix": score_matrix.tolist(),
        "attn_init_matrix": attn_init_matrix.tolist(),
        "attn_t_matrix": attn_t_matrix.tolist(),
        "attn_t_i_matrix": attn_t_i_matrix.tolist(),
        "attn_t_2i_matrix": attn_t_2i_matrix.tolist(),
    }


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, learned_temporal=False, temporal_model_ckpt=None,
        learned_temporal_stride=2, temporal_scores_file=None):
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
    tracker = Tracker(metric)
    if learned_temporal:
        if not temporal_model_ckpt:
            raise ValueError("--temporal_model_ckpt is required with --learned_temporal")
        ckpt = torch.load(temporal_model_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(ckpt, dict) or "image_height" not in ckpt:
            raise ValueError(
                "Patch-token checkpoint with image_height/image_width metadata required."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TemporalAttentionScorer(
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
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        runtime = {
            "device": device,
            "history_indices": ckpt.get("history_indices", [0, 1, 2]),
            "history_stride": max(1, int(learned_temporal_stride)),
            "image_height": int(ckpt["image_height"]),
            "image_width": int(ckpt["image_width"]),
            "keep_aspect_ratio": bool(ckpt.get("keep_aspect_ratio", True)),
            "num_patches": model.num_patches,
            "crop_and_resize": crop_and_resize,
            "frame_to_image_path": frame_to_image_path,
            "image_to_tensor": image_to_tensor,
            "frame_cache": {},
            "cv2": cv2,
        }
        tracker.set_patch_temporal_matcher(model, runtime)
    results = []
    temporal_payloads = []

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

        if learned_temporal:
            tracker.set_frame_context(sequence_dir, seq_info["image_filenames"], frame_idx)
        tracker.predict()
        if learned_temporal and temporal_scores_file:
            confirmed_track_indices = [
                i for i, t in enumerate(tracker.tracks) if t.is_confirmed()
            ]
            candidate_tracks = [tracker.tracks[i] for i in confirmed_track_indices]
            temporal_payloads.append(
                _compute_temporal_payload(
                    frame_idx,
                    sequence_dir,
                    candidate_tracks,
                    detections,
                    model,
                    tracker.patch_track_state,
                    runtime,
                )
            )
        tracker.update(detections)

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
            results.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

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
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    f.close()
    if temporal_scores_file:
        temporal_dir = os.path.dirname(temporal_scores_file)
        if temporal_dir:
            os.makedirs(temporal_dir, exist_ok=True)
        with open(temporal_scores_file, "w") as tf:
            for item in temporal_payloads:
                tf.write(json.dumps(item) + "\n")

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
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
    parser.add_argument(
        "--learned_temporal", action="store_true",
        help="Use patch-token temporal matcher inside Tracker._match")
    parser.add_argument(
        "--temporal_model_ckpt", type=str, default=None,
        help="Checkpoint path for patch-token temporal matcher")
    parser.add_argument(
        "--learned_temporal_stride", type=int, default=2,
        help="Temporal stride used to sample online templates [t, t-i, t-2i]")
    parser.add_argument(
        "--temporal_scores_file", type=str, default=None,
        help="Optional jsonl sidecar file to save per-frame patch-token score matrices and attention masses")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys_argv_backup = list(os.sys.argv)
    opt_argv = [os.sys.argv[0], "CustomDemo", "test"]
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
    os.sys.argv = opt_argv
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display,
        learned_temporal=args.learned_temporal,
        temporal_model_ckpt=args.temporal_model_ckpt,
        learned_temporal_stride=args.learned_temporal_stride,
        temporal_scores_file=args.temporal_scores_file)
    os.sys.argv = sys_argv_backup
