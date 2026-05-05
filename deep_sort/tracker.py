# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import torch
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from opts import opt

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=10):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.tracks = []
        self._next_id = 1
        self.last_ambiguous_tracks = []
        self.last_ambiguous_info = {}
        self.patch_temporal_model = None
        self.patch_runtime = None
        self.patch_track_state = {}
        self.current_sequence_dir = None
        self.current_image_filenames = None
        self.current_frame_idx = None
        self.last_patch_score_matrix = None
        self.last_matches = []
        self.last_unmatched_tracks = []
        self.last_unmatched_detections = []

    def set_patch_temporal_matcher(self, model, runtime):
        self.patch_temporal_model = model
        self.patch_runtime = runtime
        self.patch_track_state = {}
        self.last_patch_score_matrix = None

    def set_frame_context(self, sequence_dir, image_filenames, frame_idx):
        self.current_sequence_dir = sequence_dir
        self.current_image_filenames = image_filenames
        self.current_frame_idx = int(frame_idx)

    def _get_cached_image(self, frame_idx):
        if self.patch_runtime is None or self.current_sequence_dir is None:
            return None
        image_path = self.patch_runtime["frame_to_image_path"](
            self.current_sequence_dir, int(frame_idx)
        )
        cache = self.patch_runtime["frame_cache"]
        image = cache.get(image_path)
        if image is None:
            image = self.patch_runtime["cv2"].imread(
                image_path, self.patch_runtime["cv2"].IMREAD_COLOR
            )
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            cache[image_path] = image
        return image

    def _make_crop_tensor(self, frame_idx, tlwh):
        image = self._get_cached_image(frame_idx)
        crop = self.patch_runtime["crop_and_resize"](
            image,
            tlwh,
            self.patch_runtime["image_height"],
            self.patch_runtime["image_width"],
            keep_aspect_ratio=self.patch_runtime["keep_aspect_ratio"],
        )
        return self.patch_runtime["image_to_tensor"](crop)

    def _patch_state_on_match(self, track, detection):
        if self.patch_temporal_model is None or self.current_frame_idx is None:
            return
        state = self.patch_track_state.setdefault(
            track.track_id,
            {
                "init": (
                    self.current_frame_idx,
                    np.asarray(detection.tlwh, dtype=np.float32).copy(),
                ),
                "history": [],
            },
        )
        state["history"].append(
            (self.current_frame_idx, np.asarray(detection.tlwh, dtype=np.float32).copy())
        )

    def _patch_state_on_initiate(self, track, detection):
        if self.patch_temporal_model is None or self.current_frame_idx is None:
            return
        self.patch_track_state[track.track_id] = {
            "init": (
                self.current_frame_idx,
                np.asarray(detection.tlwh, dtype=np.float32).copy(),
            ),
            "history": [
                (
                    self.current_frame_idx,
                    np.asarray(detection.tlwh, dtype=np.float32).copy(),
                )
            ],
        }

    def _remove_deleted_patch_states(self):
        if self.patch_temporal_model is None:
            return
        active_ids = {t.track_id for t in self.tracks if not t.is_deleted()}
        stale = [track_id for track_id in self.patch_track_state if track_id not in active_ids]
        for track_id in stale:
            self.patch_track_state.pop(track_id, None)

    def _patch_temporal_cost_matrix(self, tracks, dets, track_indices, detection_indices):
        num_tracks = len(track_indices)
        num_dets = len(detection_indices)
        cost_matrix = np.full((num_tracks, num_dets), np.nan, dtype=np.float32)
        score_matrix = np.full((num_tracks, num_dets), np.nan, dtype=np.float32)
        if (
            self.patch_temporal_model is None
            or self.current_frame_idx is None
            or self.patch_runtime is None
        ):
            return cost_matrix, score_matrix

        history_indices = self.patch_runtime["history_indices"]
        stride = self.patch_runtime["history_stride"]
        max_required = max(history_indices) * stride if history_indices else 0

        init_batch = []
        det_batch = []
        hist_batch = []
        pair_refs = []
        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            state = self.patch_track_state.get(track.track_id)
            if state is None:
                continue
            history = state["history"]
            if len(history) < max_required + 1:
                continue
            init_frame, init_bbox = state["init"]
            hist_items = []
            for hist_idx in history_indices:
                hist_frame, hist_bbox = history[-1 - hist_idx * stride]
                hist_items.append(self._make_crop_tensor(hist_frame, hist_bbox))
            init_tensor = self._make_crop_tensor(init_frame, init_bbox)
            for col, det_idx in enumerate(detection_indices):
                init_batch.append(init_tensor)
                det_batch.append(
                    self._make_crop_tensor(self.current_frame_idx, dets[det_idx].tlwh)
                )
                hist_batch.append(torch.stack(hist_items, dim=0))
                pair_refs.append((row, col))

        if not pair_refs:
            return cost_matrix, score_matrix

        device = self.patch_runtime["device"]
        init_tensor = torch.stack(init_batch, dim=0).to(device)
        det_tensor = torch.stack(det_batch, dim=0).to(device)
        hist_tensor = torch.stack(hist_batch, dim=0).to(device)

        with torch.no_grad():
            logits = self.patch_temporal_model(
                init_tensor,
                det_tensor,
                hist_tensor,
                return_attention=False,
            )
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        for idx, (row, col) in enumerate(pair_refs):
            score = float(scores[idx])
            score_matrix[row, col] = score
            cost_matrix[row, col] = 1.0 - score
        return cost_matrix, score_matrix

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def camera_update(self, video, frame):
        for track in self.tracks:
            track.camera_update(video, frame)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        self.last_matches = list(matches)
        self.last_unmatched_tracks = list(unmatched_tracks)
        self.last_unmatched_detections = list(unmatched_detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self._patch_state_on_match(self.tracks[track_idx], detections[detection_idx])
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        self._remove_deleted_patch_states()

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            if not opt.EMA:
                track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            if opt.enable_memory_matching:
                candidate_tracks = [tracks[i] for i in track_indices]
                cost_matrix = self.metric.distance_with_memory(features, candidate_tracks)
            else:
                targets = np.array([tracks[i].track_id for i in track_indices])
                cost_matrix = self.metric.distance(features, targets)
            self.last_patch_score_matrix = None
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Detect split ambiguity before the standard assignment step.
        detection_indices = list(range(len(detections)))
        ambiguous_tracks = []
        ambiguous_info = {}
        if confirmed_tracks and detection_indices:
            cost_matrix = gated_metric(
                self.tracks, detections, confirmed_tracks, detection_indices
            )
            big_cost = 1e5
            for row_idx, track_idx in enumerate(confirmed_tracks):
                row = cost_matrix[row_idx]
                valid = []
                for col_idx, dist in enumerate(row):
                    if dist < big_cost:
                        valid.append((detection_indices[col_idx], float(dist)))

                if len(valid) < 2:
                    continue

                valid.sort(key=lambda x: x[1])
                det_a, d1 = valid[0]
                det_b, d2 = valid[1]

                if (
                    d1 < opt.ambiguity_distance_threshold
                    and d2 < opt.ambiguity_distance_threshold
                    and abs(d1 - d2) < opt.ambiguity_margin
                ):
                    ambiguous_tracks.append(self.tracks[track_idx].track_id)
                    ambiguous_info[self.tracks[track_idx].track_id] = {
                        "candidates": [det_a, det_b],
                        "distances": [d1, d2],
                    }

        self.last_ambiguous_tracks = ambiguous_tracks
        self.last_ambiguous_info = ambiguous_info

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, self.n_init, self.max_age,
            detection.feature, detection.confidence))
        self._patch_state_on_initiate(self.tracks[-1], detection)
        self._next_id += 1
