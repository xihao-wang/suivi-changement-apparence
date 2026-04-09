# vim: expandtab:ts=4:sw=4
import numpy as np
from opts import opt


def _normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return None
    return vec / norm

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    # x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
    # y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
    
    def _cosine_distance_to_memory(self, feature, memory_bank):
        if len(memory_bank) == 0:
            return 1.0
        feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        memory_bank = np.asarray(memory_bank, dtype=np.float32)
        distances = _cosine_distance(memory_bank, feature).reshape(-1)
        if opt.enable_topk_matching:
            k = min(opt.k, len(distances))
            topk = np.sort(distances)[:k]
            return float(np.mean(topk))
        return float(np.min(distances))

    def _prototype_distance(self, feature, track):
        if track.prototype is not None:
            ref = track.prototype
        elif track.prot_short is not None:
            ref = track.prot_short
        elif len(track.features) > 0:
            ref = track.features[-1]
        else:
            return 1.0
        ref = np.asarray(ref, dtype=np.float32).reshape(1, -1)
        feat = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        return float(_cosine_distance(ref, feat).reshape(-1)[0])

    def _temporal_order_cost(self, feature, track):
        temporal_cost, _, _, _ = self._temporal_order_components(feature, track)
        return temporal_cost

    def _temporal_order_components(self, feature, track):
        if (
            not opt.enable_temporal_order
            or track.prot_short is None
            or track.temporal_delta_1 is None
            or track.temporal_delta_2 is None
        ):
            return 1.0, 0.0, 0.0, 0.0

        delta_now = _normalize_vector(
            np.asarray(feature, dtype=np.float32) - np.asarray(track.prot_short, dtype=np.float32)
        )
        if delta_now is None:
            return 1.0, 0.0, 0.0, 0.0

        s1 = float(np.clip(np.dot(delta_now, track.temporal_delta_1), -1.0, 1.0))
        s2 = float(np.clip(np.dot(delta_now, track.temporal_delta_2), -1.0, 1.0))
        continuity_now = _normalize_vector(delta_now - track.temporal_delta_1)
        continuity_hist = _normalize_vector(track.temporal_delta_1 - track.temporal_delta_2)
        if continuity_now is None or continuity_hist is None:
            c = 0.0
        else:
            c = float(np.clip(np.dot(continuity_now, continuity_hist), -1.0, 1.0))

        score_temp = (
            opt.temporal_app_weight * s1
            + opt.temporal_order_weight * (s1 - s2)
        )
        score_temp = float(np.clip(score_temp, -1.0, 1.0))
        temporal_cost = (1.0 - score_temp) / 2.0
        return temporal_cost, s1, s2, c

    def distance_with_memory(self, features, tracks):
        cost_matrix = np.zeros((len(tracks), len(features)))
        for i, track in enumerate(tracks):
            for j, feature in enumerate(features):
                d_short = self._cosine_distance_to_memory(feature, track.short_memory)
                d_long = self._cosine_distance_to_memory(feature, track.long_memory)

                appearance_cost = (
                    opt.short_distance_weight * d_short
                    + (1 - opt.short_distance_weight) * d_long
                )
                if opt.enable_temporal_order:
                    temporal_cost = self._temporal_order_cost(feature, track)
                    final_cost = (
                        (1 - opt.temporal_cost_weight) * appearance_cost
                        + opt.temporal_cost_weight * temporal_cost
                    )
                else:
                    final_cost = appearance_cost
                cost_matrix[i, j] = final_cost
        return cost_matrix

    def distance_components_with_memory(self, features, tracks):
        """Return appearance, temporal, temporal components and final cost matrices."""
        appearance_matrix = np.zeros((len(tracks), len(features)))
        temporal_matrix = np.zeros((len(tracks), len(features)))
        s1_matrix = np.zeros((len(tracks), len(features)))
        s2_matrix = np.zeros((len(tracks), len(features)))
        c_matrix = np.zeros((len(tracks), len(features)))
        final_matrix = np.zeros((len(tracks), len(features)))

        for i, track in enumerate(tracks):
            for j, feature in enumerate(features):
                d_short = self._cosine_distance_to_memory(feature, track.short_memory)
                d_long = self._cosine_distance_to_memory(feature, track.long_memory)

                appearance_cost = (
                    opt.short_distance_weight * d_short
                    + (1 - opt.short_distance_weight) * d_long
                )
                if opt.enable_temporal_order:
                    temporal_cost, s1, s2, c = self._temporal_order_components(feature, track)
                    final_cost = (
                        (1 - opt.temporal_cost_weight) * appearance_cost
                        + opt.temporal_cost_weight * temporal_cost
                    )
                else:
                    temporal_cost = 1.0
                    s1 = 0.0
                    s2 = 0.0
                    c = 0.0
                    final_cost = appearance_cost

                appearance_matrix[i, j] = appearance_cost
                temporal_matrix[i, j] = temporal_cost
                s1_matrix[i, j] = s1
                s2_matrix[i, j] = s2
                c_matrix[i, j] = c
                final_matrix[i, j] = final_cost

        return appearance_matrix, temporal_matrix, s1_matrix, s2_matrix, c_matrix, final_matrix
