# vim: expandtab:ts=4:sw=4
import numpy as np
from opts import opt

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

    def _trend_consistency_cost(self, feature, track):
        if not opt.enable_trend:
            return 0.0
        if track.prot_short is None or track.appearance_trend is None:
            return 1.0
        
        delta = feature - track.prot_short
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-6:
            return 0.0
        delta /= delta_norm  # Normalize the trend vector
        trend_sim = float(np.dot(delta, track.appearance_trend))

        return 1.0-max(-1.0, min(1.0, trend_sim))  # Convert similarity to distance

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

    def _lightweight_mam_cost(self, feature, track):
        """Attention-style memory cost for a track-detection pair.

        The candidate detection is used as a query over confirmed track memory.
        We only return a matching cost here; the track memory is not updated
        until the association is accepted by the tracker.
        """
        if not opt.enable_light_mam:
            return 0.0

        memory_tokens = []
        if track.prot_long is not None:
            memory_tokens.append(track.prot_long)
        if track.prot_short is not None:
            memory_tokens.append(track.prot_short)
        memory_tokens.extend(track.short_memory)

        if len(memory_tokens) == 0:
            return 1.0

        memory_tokens = np.asarray(memory_tokens, dtype=np.float32)
        query = np.asarray(feature, dtype=np.float32).reshape(1, -1)

        memory_norm = np.linalg.norm(memory_tokens, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        memory_tokens = memory_tokens / np.maximum(memory_norm, 1e-12)
        query = query / np.maximum(query_norm, 1e-12)

        temperature = max(float(opt.mam_temperature), 1e-6)
        scores = np.dot(memory_tokens, query.reshape(-1)) / temperature
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / np.maximum(np.sum(weights), 1e-12)

        context = np.sum(weights[:, None] * memory_tokens, axis=0, keepdims=True)
        context_norm = np.linalg.norm(context, axis=1, keepdims=True)
        context = context / np.maximum(context_norm, 1e-12)

        return float(_cosine_distance(context, query, data_is_normalized=True).reshape(-1)[0])

    def _combine_costs(self, appearance_cost, trend_cost, mam_cost):
        cost_terms = [(opt.appearance_cost_weight, appearance_cost)]
        if opt.enable_trend:
            cost_terms.append((opt.trend_cost_weight, trend_cost))
        if opt.enable_light_mam:
            cost_terms.append((opt.mam_cost_weight, mam_cost))

        weight_sum = sum(weight for weight, _ in cost_terms)
        if weight_sum <= 0:
            return appearance_cost
        return sum(weight * cost for weight, cost in cost_terms) / weight_sum

    def distance_with_memory(self, features, tracks):
        cost_matrix = np.zeros((len(tracks), len(features)))
        for i, track in enumerate(tracks):
            for j, feature in enumerate(features):
                d_short = self._cosine_distance_to_memory(feature, track.short_memory)
                d_long = self._cosine_distance_to_memory(feature, track.long_memory)

                appearance_cost = (opt.short_distance_weight * d_short + (1 - opt.short_distance_weight) * d_long)
                trend_cost = self._trend_consistency_cost(feature, track)
                trend_cost = trend_cost * opt.trend_scale
                mam_cost = self._lightweight_mam_cost(feature, track)
                cost_matrix[i, j] = self._combine_costs(
                    appearance_cost, trend_cost, mam_cost
                )
        return cost_matrix

    def distance_with_trend_only(self, features, tracks):
        """Use prototype-based appearance distance plus trend, without memory-aware matching."""
        cost_matrix = np.zeros((len(tracks), len(features)))
        for i, track in enumerate(tracks):
            for j, feature in enumerate(features):
                appearance_cost = self._prototype_distance(feature, track)
                trend_cost = self._trend_consistency_cost(feature, track)
                trend_cost = trend_cost * opt.trend_scale
                mam_cost = self._lightweight_mam_cost(feature, track)
                cost_matrix[i, j] = self._combine_costs(
                    appearance_cost, trend_cost, mam_cost
                )
        return cost_matrix

    def distance_components_with_memory(self, features, tracks):
        """Return appearance, trend, MAM, and final cost matrices for debugging."""
        appearance_matrix = np.zeros((len(tracks), len(features)))
        trend_matrix = np.zeros((len(tracks), len(features)))
        mam_matrix = np.zeros((len(tracks), len(features)))
        final_matrix = np.zeros((len(tracks), len(features)))

        for i, track in enumerate(tracks):
            for j, feature in enumerate(features):
                d_short = self._cosine_distance_to_memory(feature, track.short_memory)
                d_long = self._cosine_distance_to_memory(feature, track.long_memory)

                appearance_cost = (
                    opt.short_distance_weight * d_short
                    + (1 - opt.short_distance_weight) * d_long
                )
                trend_cost = self._trend_consistency_cost(feature, track)
                trend_cost = trend_cost * opt.trend_scale
                mam_cost = self._lightweight_mam_cost(feature, track)
                final_cost = self._combine_costs(
                    appearance_cost, trend_cost, mam_cost
                )

                appearance_matrix[i, j] = appearance_cost
                trend_matrix[i, j] = trend_cost
                mam_matrix[i, j] = mam_cost
                final_matrix[i, j] = final_cost

        return appearance_matrix, trend_matrix, mam_matrix, final_matrix
