"""Non-personalized baseline recommenders.

These ignore *who* the user is -- they recommend the same global ranking to
everyone (minus already-seen items). They matter for two reasons:

1. They are the **floor** any personalized model must beat. If item-item CF can't
   outperform "most popular", the personalization isn't adding value.
2. "Most popular" is a notoriously strong accuracy baseline on skewed catalogues
   *and* the textbook example of **popularity bias** -- great Precision@K, terrible
   novelty/coverage. That tension is exactly what the beyond-accuracy metrics
   later expose.

All recommenders in this project share the same interface:
    fit(ratings, items=None) -> self
    recommend(user_id, ratings_train, n=10, exclude_seen=True) -> [(item_id, score), ...]
"""

import numpy as np
import pandas as pd

from . import config
from .data_loading import get_seen_items


class MostPopularRecommender:
    """Recommend the most popular artists (by number of distinct listeners)."""

    def __init__(self):
        self.ranking_ = None        # item ids, most popular first
        self.scores_ = None         # aligned popularity (listener counts)

    def fit(self, ratings, items=None):
        # Popularity = how many users interacted with the item. We count distinct
        # listeners rather than total plays, so one super-fan can't inflate an
        # artist's popularity.
        counts = ratings.groupby(config.ITEM_COL)[config.USER_COL].nunique()
        counts = counts.sort_values(ascending=False)
        self.ranking_ = counts.index.to_numpy()
        self.scores_ = counts.to_numpy(dtype=float)
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        out = []
        for item_id, score in zip(self.ranking_, self.scores_):
            if item_id in seen:
                continue
            out.append((item_id, float(score)))
            if len(out) >= n:
                break
        return out


class HighestAverageRatingRecommender:
    """Recommend artists with the highest average listening weight.

    The implicit-feedback analogue of "highest average rating". We rank by the
    mean ``log1p(plays)`` across listeners, but only among artists with at least
    ``min_ratings`` listeners. The minimum-support filter is essential: without
    it, an obscure artist played heavily by a single user would top the chart on
    a sample size of one.
    """

    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings
        self.ranking_ = None
        self.scores_ = None

    def fit(self, ratings, items=None):
        grp = ratings.groupby(config.ITEM_COL)[config.WEIGHT_COL]
        stats = pd.DataFrame({"mean": grp.mean(), "count": grp.count()})
        stats = stats[stats["count"] >= self.min_ratings]
        stats = stats.sort_values("mean", ascending=False)
        self.ranking_ = stats.index.to_numpy()
        self.scores_ = stats["mean"].to_numpy(dtype=float)
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        out = []
        for item_id, score in zip(self.ranking_, self.scores_):
            if item_id in seen:
                continue
            out.append((item_id, float(score)))
            if len(out) >= n:
                break
        return out


class RandomRecommender:
    """Recommend random unseen artists -- the absolute floor / sanity check.

    Any model that cannot beat random is broken. Random also tends to have the
    *highest* catalog coverage and novelty, a useful contrast to "most popular".
    """

    def __init__(self, random_state=config.RANDOM_STATE):
        self.random_state = random_state
        self.items_ = None

    def fit(self, ratings, items=None):
        self.items_ = ratings[config.ITEM_COL].unique()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        # Seed per user so recommendations are reproducible but user-specific.
        rng = np.random.default_rng(self.random_state + int(user_id))
        candidates = np.array([it for it in self.items_ if it not in seen])
        if len(candidates) == 0:
            return []
        chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(it, 0.0) for it in chosen]
