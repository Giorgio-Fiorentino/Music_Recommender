"""Content-based recommender.

Idea: describe each *item* by its content (here, the tags users applied to the
artist), describe each *user* by the content of the items they liked, then
recommend items whose content is closest to the user's taste profile.

Pipeline:
1. **Item vectors** -- TF-IDF over each artist's tag document. TF-IDF up-weights
   tags that are frequent for an artist but rare across the catalogue, so a
   distinctive tag like "norwegian_black_metal" counts for more than a generic
   "rock".
2. **User profile** -- a weighted, mean-centered sum of the vectors of artists
   the user listened to:
        profile(u) = sum_i (w(u,i) - mean_w(u)) * vector(i)
   Centering by the user's own mean listening weight turns absolute play counts
   into *relative* preference: artists played more than the user's average pull
   the profile toward their tags; below-average artists push away.
3. **Scoring** -- cosine similarity between the profile and every item vector.

Strengths: works for cold/new items (needs only content, no interactions) and
gives explainable recs ("because you listen to gothic/darkwave artists").
Weakness: it can only recommend more of what the user already likes (low
serendipity) and is blind to quality signals beyond tags.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loading import get_seen_items


class ContentBasedRecommender:
    """Content-based recommender using TF-IDF over artist tags."""

    def __init__(self, feature_col=config.GENRES_COL):
        self.feature_col = feature_col
        self.vectorizer = None
        self.item_features_ = None       # sparse (n_items x n_terms), L2-normalized
        self.item_ids_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings, items):
        """Build the TF-IDF item-feature matrix.

        We restrict the catalogue to artists that appear in ``ratings`` so every
        model recommends over the same item space (fair evaluation).
        """
        catalog = set(ratings[config.ITEM_COL].unique())
        items = items[items[config.ITEM_ID_COL].isin(catalog)].copy()
        items[self.feature_col] = items[self.feature_col].fillna("")
        items = items.sort_values(config.ITEM_ID_COL).reset_index(drop=True)

        # token_pattern keeps our underscore tokens ("alternative_metal") intact.
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+", min_df=2)
        self.item_features_ = self.vectorizer.fit_transform(items[self.feature_col])
        self.item_ids_ = items[config.ITEM_ID_COL].to_numpy()
        self.item_id_to_index_ = {it: i for i, it in enumerate(self.item_ids_)}
        return self

    def build_user_profile(self, user_id, ratings_train):
        """Return the user's taste vector (1 x n_terms), or None if empty."""
        user_rows = ratings_train[ratings_train[config.USER_COL] == user_id]
        if user_rows.empty:
            return None
        mean_w = user_rows[config.WEIGHT_COL].mean()

        profile = np.zeros(self.item_features_.shape[1], dtype=float)
        contributed = False
        for item_id, w in zip(user_rows[config.ITEM_COL], user_rows[config.WEIGHT_COL]):
            idx = self.item_id_to_index_.get(item_id)
            if idx is None:
                continue
            centered = w - mean_w
            profile += centered * self.item_features_[idx].toarray().ravel()
            contributed = True
        if not contributed or not np.any(profile):
            return None
        return profile.reshape(1, -1)

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        profile = self.build_user_profile(user_id, ratings_train)
        if profile is None:
            return []
        # Cosine similarity of the profile against every item vector.
        scores = cosine_similarity(profile, self.item_features_).ravel()

        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        order = np.argsort(-scores)
        out = []
        for idx in order:
            item_id = self.item_ids_[idx]
            if item_id in seen or scores[idx] <= 0:
                continue
            out.append((item_id, float(scores[idx])))
            if len(out) >= n:
                break
        return out

    def similar_items(self, item_id, n=10):
        """Return the n artists most content-similar to a given artist."""
        idx = self.item_id_to_index_.get(item_id)
        if idx is None:
            return []
        scores = cosine_similarity(self.item_features_[idx], self.item_features_).ravel()
        order = np.argsort(-scores)
        out = []
        for j in order:
            if j == idx:
                continue
            out.append((self.item_ids_[j], float(scores[j])))
            if len(out) >= n:
                break
        return out
