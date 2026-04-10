# vim: expandtab:ts=4:sw=4
import numpy as np
from deep_sort.kalman_filter import KalmanFilter
from opts import opt

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, detection, track_id, n_init, max_age,
                 feature=None, score=None):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative

        # The matcher still reads `self.features`, but now this list stores
        # the memory-derived prototype instead of raw historical features.
        self.features = []
        if feature is not None:
            feature /= np.linalg.norm(feature)

        # Memory-based identity representation.
        self.short_memory = []
        self.prot_short_history = []
        self.long_memory = []
        self.prot_short = None
        self.prot_long = None
        self.prototype = None
        if feature is not None:
            if opt.enable_stm_ltm:
                self.short_memory.append(feature)
                self.prot_short = feature.copy()
                self.prot_short_history.append(self.prot_short.copy())
                self.prot_long = None
                self.prototype = feature.copy()
                self.features = [self.prototype.copy()]
            else:
                self.features = [feature.copy()]

        self.scores = []
        if score is not None:
            self.scores.append(score)

        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilter()

        self.mean, self.covariance = self.kf.initiate(detection)

    @staticmethod
    def _normalize_vector(vec):
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return None
        return vec / norm

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def get_matrix(dict_frame_matrix, frame):
        eye = np.eye(3)
        matrix = dict_frame_matrix[frame]
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def camera_update(self, video, frame):
        dict_frame_matrix = opt.ecc[video]
        frame = str(int(frame))
        if frame in dict_frame_matrix:
            matrix = self.get_matrix(dict_frame_matrix, frame)
            x1, y1, x2, y2 = self.to_tlbr()
            x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
            x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
            w, h = x2_ - x1_, y2_ - y1_
            cx, cy = x1_ + w / 2, y1_ + h / 2
            self.mean[:4] = [cx, cy, w / h, h]

    def update(self, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.to_xyah(), detection.confidence)

        feature = detection.feature / np.linalg.norm(detection.feature)

        if not opt.enable_stm_ltm:
            self.features = [feature.copy()]
            self.hits += 1
            self.time_since_update = 0
            if self.state == TrackState.Tentative and self.hits >= self._n_init:
                self.state = TrackState.Confirmed
            return

        # Update short-term memory with the latest raw observation.
        self.short_memory.append(feature)
        if len(self.short_memory) > opt.short_memory_size:
            self.short_memory.pop(0)

        self.prot_short = np.mean(self.short_memory, axis=0)
        self.prot_short /= np.linalg.norm(self.prot_short)
        self.prot_short_history.append(self.prot_short.copy())
        prot_history_cap = opt.short_memory_size * 4
        if len(self.prot_short_history) > prot_history_cap:
            self.prot_short_history.pop(0)

        self.hits += 1

        # Start writing long-term memory later than track confirmation, and
        # avoid writing every frame to reduce contamination during overlap.
        if opt.enable_memory_init_control:
            if self.hits >= opt.memory_init_hits:
                sim_long = np.dot(self.prot_long, feature) if self.prot_long is not None else 1.0
                sim_short = np.dot(self.prot_short, feature) if self.prot_short is not None else 1.0

                long_memory_write_gate = (
                    detection.confidence > opt.memory_min_confidence and
                    sim_long > opt.memory_sim_threshold and
                    sim_short > opt.short_memory_gate and
                    self.hits % opt.long_memory_stride == 0
                )

                if long_memory_write_gate:
                    self.long_memory.append(feature)
                    if len(self.long_memory) > opt.long_memory_size:
                        self.long_memory.pop(0)
        else:
            self.long_memory.append(feature)
            if len(self.long_memory) > opt.long_memory_size:
                self.long_memory.pop(0)

        if len(self.long_memory) > 0:
            self.prot_long = np.mean(self.long_memory, axis=0)
            self.prot_long /= np.linalg.norm(self.prot_long)
            sim_ls = np.dot(self.prot_long, feature)
            beta = 0.8 if sim_ls > 0.8 else 0.6
            self.prototype = beta * self.prot_long + (1 - beta) * self.prot_short
        else:
            self.prot_long = None
            self.prototype = self.prot_short.copy()

        self.prototype /= np.linalg.norm(self.prototype)
        self.features = [self.prototype.copy()]

        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
