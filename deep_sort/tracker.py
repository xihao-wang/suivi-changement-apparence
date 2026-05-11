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

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=10,
                 temporal_model=None, temporal_alpha=1.0,
                 fuse_temporal_model=False, temporal_max_correction=0.02,
                 temporal_min_scale=0.02):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.temporal_model = temporal_model
        self.temporal_alpha = float(temporal_alpha)
        self.fuse_temporal_model = bool(fuse_temporal_model)
        self.temporal_max_correction = float(temporal_max_correction)
        self.temporal_min_scale = float(temporal_min_scale)

        self.tracks = []
        self._next_id = 1
        self.last_ambiguous_tracks = []
        self.last_ambiguous_info = {}
        self.last_match_confidences = {}
        self.temporal_cost_override = None

    def set_temporal_cost_override(self, track_indices, detection_indices, temporal_cost):
        self.temporal_cost_override = {
            "track_indices": list(track_indices),
            "detection_indices": list(detection_indices),
            "temporal_cost": np.asarray(temporal_cost, dtype=np.float32),
        }

    def clear_temporal_cost_override(self):
        self.temporal_cost_override = None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _normalize_cost_rows(cost_matrix):
        normalized = cost_matrix.copy().astype(np.float32)
        big_cost = linear_assignment.INFTY_COST
        for row_idx in range(normalized.shape[0]):
            row = normalized[row_idx]
            valid_mask = row < big_cost
            if not np.any(valid_mask):
                continue
            valid = row[valid_mask]
            min_v = float(np.min(valid))
            max_v = float(np.max(valid))
            if max_v - min_v < 1e-12:
                normalized[row_idx, valid_mask] = 0.0
            else:
                normalized[row_idx, valid_mask] = (valid - min_v) / (max_v - min_v)
        return normalized

    def _temporal_cost_matrix(self, tracks, detections, track_indices, detection_indices):
        if not self.fuse_temporal_model:
            return None
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return None
        if self.temporal_cost_override is not None:
            override = self.temporal_cost_override
            source_tracks = override["track_indices"]
            source_dets = override["detection_indices"]
            source_cost = override["temporal_cost"]
            aligned = np.ones((len(track_indices), len(detection_indices)), dtype=np.float32)
            source_track_to_row = {track_idx: row for row, track_idx in enumerate(source_tracks)}
            source_det_to_col = {det_idx: col for col, det_idx in enumerate(source_dets)}
            for row, track_idx in enumerate(track_indices):
                source_row = source_track_to_row.get(track_idx)
                if source_row is None:
                    continue
                for col, detection_idx in enumerate(detection_indices):
                    source_col = source_det_to_col.get(detection_idx)
                    if source_col is not None:
                        aligned[row, col] = source_cost[source_row, source_col]
            return aligned
        if self.temporal_model is None:
            return None

        history_len = getattr(self.temporal_model, "history_len", 5)
        long_history_len = getattr(self.temporal_model, "long_history_len", 30)
        det_batch = []
        short_batch = []
        long_batch = []
        short_len_batch = []
        long_len_batch = []
        pair_indices = []
        temporal_cost = np.ones((len(track_indices), len(detection_indices)), dtype=np.float32)

        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            short_result = self._build_short_history(track, history_len)
            if short_result is None:
                continue
            short_hist, short_len = short_result
            long_result = self._build_long_history(track, long_history_len)
            if long_result is None:
                long_hist = np.repeat(short_hist[-1:, :], long_history_len, axis=0)
                long_len = 1
            else:
                long_hist, long_len = long_result
            for col, detection_idx in enumerate(detection_indices):
                det_feat = np.asarray(detections[detection_idx].feature, dtype=np.float32)
                norm = np.linalg.norm(det_feat)
                if norm > 1e-12:
                    det_feat = det_feat / norm
                det_batch.append(det_feat)
                short_batch.append(short_hist)
                long_batch.append(long_hist)
                short_len_batch.append(short_len)
                long_len_batch.append(long_len)
                pair_indices.append((row, col))

        if not pair_indices:
            return temporal_cost

        with torch.no_grad():
            logits = self.temporal_model(
                torch.from_numpy(np.stack(det_batch, axis=0)),
                torch.from_numpy(np.stack(short_batch, axis=0)),
                long_hist_feat=torch.from_numpy(np.stack(long_batch, axis=0)),
                short_hist_len=torch.from_numpy(np.asarray(short_len_batch, dtype=np.int64)),
                long_hist_len=torch.from_numpy(np.asarray(long_len_batch, dtype=np.int64)),
                return_attention=False,
            )
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        for idx, (row, col) in enumerate(pair_indices):
            temporal_cost[row, col] = 1.0 - float(probs[idx])
        return temporal_cost

    def _fuse_temporal_cost(self, base_cost, temporal_cost):
        if temporal_cost is None:
            return base_cost
        gamma = max(self.temporal_alpha, 0.0)
        fused = base_cost.copy()
        big_cost = linear_assignment.INFTY_COST
        min_scale = max(self.temporal_min_scale, 0.0)
        max_correction = max(self.temporal_max_correction, 0.0)
        eps = 1e-12

        for row_idx in range(base_cost.shape[0]):
            valid_mask = base_cost[row_idx] < big_cost
            if np.sum(valid_mask) < 2:
                continue
            learned_row = temporal_cost[row_idx, valid_mask].astype(np.float32)
            base_row = base_cost[row_idx, valid_mask].astype(np.float32)
            learned_delta = learned_row - float(np.mean(learned_row))
            learned_scale = float(np.std(learned_delta))
            if learned_scale < eps:
                continue
            base_scale = float(np.std(base_row))
            effective_scale = max(base_scale, min_scale)
            beta = gamma * effective_scale / max(learned_scale, eps)
            correction = beta * learned_delta
            correction = np.clip(correction, -max_correction, max_correction)
            fused[row_idx, valid_mask] = np.maximum(base_row + correction, 0.0)
        return fused

    @staticmethod
    def _cost_to_confidence(cost, max_cost):
        if max_cost <= 0:
            return 0.0
        value = 1.0 - float(cost) / float(max_cost)
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _assignment_cost(cost_matrix):
        if cost_matrix.size == 0:
            return None
        big_cost = linear_assignment.INFTY_COST
        indices = linear_assignment.linear_assignment(cost_matrix.copy())
        if len(indices) == 0:
            return None
        total = 0.0
        for row, col in indices:
            value = float(cost_matrix[row, col])
            if value >= big_cost:
                return None
            total += value
        return total

    def _combo_match_confidences(self, matches, cost_lookup, max_cost):
        if not matches:
            return {}

        matched_tracks = [track_idx for track_idx, _ in matches]
        matched_detections = [detection_idx for _, detection_idx in matches]
        detection_candidates = list(matched_detections)
        for track_idx in matched_tracks:
            for lookup_track_idx, detection_idx in cost_lookup.keys():
                if lookup_track_idx == track_idx and detection_idx not in detection_candidates:
                    detection_candidates.append(detection_idx)
        big_cost = linear_assignment.INFTY_COST
        cost_matrix = np.full(
            (len(matched_tracks), len(detection_candidates)),
            big_cost,
            dtype=np.float32,
        )

        for row, track_idx in enumerate(matched_tracks):
            for col, detection_idx in enumerate(detection_candidates):
                cost_matrix[row, col] = float(
                    cost_lookup.get((track_idx, detection_idx), big_cost)
                )

        current_total = 0.0
        for row, (track_idx, detection_idx) in enumerate(matches):
            col = detection_candidates.index(detection_idx)
            cost = float(cost_matrix[row, col])
            if cost >= big_cost:
                return {
                    pair: self._cost_to_confidence(
                        cost_lookup.get(pair, max_cost), max_cost
                    )
                    for pair in matches
                }
            current_total += cost

        margin_scale = max(float(getattr(opt, "match_conf_margin_scale", 0.02)), 1e-12)
        confidences = {}
        for row, (track_idx, detection_idx) in enumerate(matches):
            col = detection_candidates.index(detection_idx)
            matched_cost = float(cost_matrix[row, col])
            alternative_matrix = cost_matrix.copy()
            alternative_matrix[row, col] = big_cost
            alternative_total = self._assignment_cost(alternative_matrix)

            absolute_conf = self._cost_to_confidence(matched_cost, max_cost)
            if alternative_total is None:
                combo_conf = 1.0
            else:
                combo_margin = float(alternative_total - current_total)
                combo_conf = float(np.clip(combo_margin / margin_scale, 0.0, 1.0))
            confidences[(track_idx, detection_idx)] = absolute_conf * combo_conf
        return confidences

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

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

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
        self.last_match_confidences = {}
        for track in self.tracks:
            track.match_confidence = None
        match_costs = {}

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            if opt.enable_memory_matching:
                candidate_tracks = [tracks[i] for i in track_indices]
                cost_matrix = self.metric.distance_with_memory(features, candidate_tracks)
            else:
                targets = np.array([tracks[i].track_id for i in track_indices])
                cost_matrix = self.metric.distance(features, targets)
            temporal_cost = self._temporal_cost_matrix(
                tracks, dets, track_indices, detection_indices
            )
            cost_matrix = self._fuse_temporal_cost(cost_matrix, temporal_cost)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices,
                detection_indices)
            for row, track_idx in enumerate(track_indices):
                for col, detection_idx in enumerate(detection_indices):
                    match_costs[(track_idx, detection_idx)] = float(cost_matrix[row, col])

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
        appearance_confidences = self._combo_match_confidences(
            matches_a, match_costs, self.metric.matching_threshold
        )
        for track_idx, detection_idx in matches_a:
            confidence = appearance_confidences.get((track_idx, detection_idx), 0.0)
            self.tracks[track_idx].match_confidence = confidence
            self.last_match_confidences[(track_idx, detection_idx)] = confidence
        if matches_b:
            iou_costs = {}
            iou_track_indices = [track_idx for track_idx, _ in matches_b]
            iou_detection_indices = [detection_idx for _, detection_idx in matches_b]
            if iou_track_indices and iou_detection_indices:
                iou_matrix = iou_matching.iou_cost(
                    self.tracks,
                    detections,
                    iou_track_indices,
                    iou_detection_indices,
                )
                for row, track_idx in enumerate(iou_track_indices):
                    for col, detection_idx in enumerate(iou_detection_indices):
                        iou_costs[(track_idx, detection_idx)] = float(iou_matrix[row, col])
            iou_confidences = self._combo_match_confidences(
                matches_b, iou_costs, self.max_iou_distance
            )
            for track_idx, detection_idx in matches_b:
                confidence = iou_confidences.get((track_idx, detection_idx), 0.0)
                self.tracks[track_idx].match_confidence = confidence
                self.last_match_confidences[(track_idx, detection_idx)] = confidence
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, self.n_init, self.max_age,
            detection.feature, detection.confidence))
        self._next_id += 1
