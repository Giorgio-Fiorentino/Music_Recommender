"""Collaborative filtering (CF): item-item and user-user.

CF needs no content -- it learns purely from the interaction matrix, exploiting
the "people who behave alike, like alike" assumption. We implement both classic
neighbourhood variants so we can compare them (a chosen extension):

* **Item-item CF**: "you listened to A, and A is similar to B (because the same
  people listen to both), so here's B." Similarity is between *columns* of the
  user-item matrix. Usually more stable than user-user because item co-listening
  patterns change slowly and items have more ratings than users have items.

* **User-user CF**: "users similar to you also listen to B." Similarity is between
  *rows*. More intuitive but noisier and harder to scale as users grow.

Both use **cosine similarity** on the ``log1p(plays)`` weighted matrix and a
**top-k neighbourhood** (only the k most similar items/users contribute), which
removes weak, noisy correlations and is the standard memory-based CF formulation.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loading import build_user_item_matrix, get_seen_items

_EPS = 1e-8


def _topk_per_row(sim: np.ndarray, k: int) -> np.ndarray:
    """Keep only the k largest entries in each row, zero the rest.

    This restricts each item/user to its k nearest neighbours. The diagonal must
    already be zeroed by the caller so an item is never its own neighbour.
    """
    if k >= sim.shape[1]:
        return sim
    out = np.zeros_like(sim)
    # argpartition finds the top-k indices per row in O(n) without full sort.
    topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    rows = np.arange(sim.shape[0])[:, None]
    out[rows, topk_idx] = sim[rows, topk_idx]
    return out


class ItemItemCollaborativeFiltering:
    """Item-item neighbourhood CF."""

    def __init__(self, k=20, similarity="cosine"):
        self.k = k
        self.similarity = similarity
        self.matrix_ = None              # users x items (sparse)
        self.sim_full_ = None            # items x items (dense) - for similar_items
        self.sim_topk_ = None            # items x items, top-k per row kept
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings):
        (self.matrix_, self.user_ids_, self.item_ids_,
         self.user_id_to_index_, self.item_id_to_index_) = build_user_item_matrix(ratings)

        # Cosine between item columns. cosine_similarity on the transposed matrix
        # gives an (items x items) matrix; it handles the L2 normalization.
        sim = cosine_similarity(self.matrix_.T).astype(np.float32)
        np.fill_diagonal(sim, 0.0)       # an item is not its own neighbour
        self.sim_full_ = sim
        self.sim_topk_ = _topk_per_row(sim, self.k)
        return self

    def _scores_for_user(self, user_id):
        """Predicted score for *every* item for one user (vectorized).

        score(u, i) = sum_j sim_k(i, j) * r(u, j) / sum_j |sim_k(i, j)|
        where j ranges over the item's top-k neighbours that the user has played.
        """
        u_idx = self.user_id_to_index_.get(user_id)
        if u_idx is None:
            return None
        r_u = self.matrix_[u_idx].toarray().ravel()          # (n_items,)
        rated_mask = (r_u != 0).astype(np.float32)
        numerator = self.sim_topk_ @ r_u                     # weighted sum of neighbour plays
        denominator = np.abs(self.sim_topk_) @ rated_mask    # normaliser
        return numerator / (denominator + _EPS)

    def predict_score(self, user_id, item_id):
        scores = self._scores_for_user(user_id)
        j = self.item_id_to_index_.get(item_id)
        if scores is None or j is None:
            return 0.0
        return float(scores[j])

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        scores = self._scores_for_user(user_id)
        if scores is None:
            return []
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        out = []
        for idx in np.argsort(-scores):
            item_id = self.item_ids_[idx]
            if item_id in seen or scores[idx] <= 0:
                continue
            out.append((item_id, float(scores[idx])))
            if len(out) >= n:
                break
        return out

    def similar_items(self, item_id, n=10):
        j = self.item_id_to_index_.get(item_id)
        if j is None:
            return []
        row = self.sim_full_[j]
        out = []
        for idx in np.argsort(-row):
            if idx == j:
                continue
            out.append((self.item_ids_[idx], float(row[idx])))
            if len(out) >= n:
                break
        return out


class UserUserCollaborativeFiltering:
    """User-user neighbourhood CF."""

    def __init__(self, k=20, similarity="cosine"):
        self.k = k
        self.similarity = similarity
        self.matrix_ = None              # users x items (sparse)
        self.sim_topk_ = None            # users x users, top-k per row kept
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings):
        (self.matrix_, self.user_ids_, self.item_ids_,
         self.user_id_to_index_, self.item_id_to_index_) = build_user_item_matrix(ratings)

        sim = cosine_similarity(self.matrix_).astype(np.float32)   # users x users
        np.fill_diagonal(sim, 0.0)
        self.sim_topk_ = _topk_per_row(sim, self.k)
        return self

    def _scores_for_user(self, user_id):
        """Predicted score for every item for one user.

        score(u, i) = sum_v sim_k(u, v) * r(v, i) / sum_v |sim_k(u, v)| over
        neighbours v who have played item i.
        """
        u_idx = self.user_id_to_index_.get(user_id)
        if u_idx is None:
            return None
        sim_row = self.sim_topk_[u_idx]                       # (n_users,)
        # numerator over all items: neighbours' plays weighted by similarity.
        numerator = self.matrix_.T @ sim_row                 # (n_items,)
        # denominator per item: total |sim| of neighbours who actually played it.
        mask = (self.matrix_ != 0).astype(np.float32)
        denominator = mask.T @ np.abs(sim_row)               # (n_items,)
        return np.asarray(numerator).ravel() / (np.asarray(denominator).ravel() + _EPS)

    def predict_score(self, user_id, item_id):
        scores = self._scores_for_user(user_id)
        j = self.item_id_to_index_.get(item_id)
        if scores is None or j is None:
            return 0.0
        return float(scores[j])

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        scores = self._scores_for_user(user_id)
        if scores is None:
            return []
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        out = []
        for idx in np.argsort(-scores):
            item_id = self.item_ids_[idx]
            if item_id in seen or scores[idx] <= 0:
                continue
            out.append((item_id, float(scores[idx])))
            if len(out) >= n:
                break
        return out
