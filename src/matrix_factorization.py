"""Matrix factorization recommenders.

Where neighbourhood CF compares whole rows/columns, matrix factorization learns
a small **latent vector** for every user and every item such that their dot
product reproduces the observed interactions:

        score(u, i) = x_u . y_i

The latent dimensions are discovered automatically and often correspond to fuzzy
"taste axes" (e.g. acoustic-vs-electronic, mainstream-vs-niche). MF generalises
better than neighbourhood CF on sparse data and scales well at prediction time.

We provide two models:

* ``ImplicitALSRecommender`` -- a from-scratch implementation of the Hu, Koren &
  Volinsky (2008) implicit-feedback ALS. This is the model we explain in depth.
* ``SurpriseSVDRecommender`` -- the classic biased SVD from the scikit-surprise
  library, trained on log-plays as pseudo-ratings, used as a library comparison.
"""

import numpy as np

from . import config
from .data_loading import build_user_item_matrix, get_seen_items


class ImplicitALSRecommender:
    """Implicit-feedback ALS (Hu-Koren-Volinsky 2008).

    The insight for implicit data: we never observe "dislike", only presence and
    absence. So we split each observation into:

    * **preference** p(u,i) = 1 if the user played the artist at all, else 0;
    * **confidence** c(u,i) = 1 + alpha * r(u,i), where r = log1p(plays).
      We are *more* confident that a heavily-played artist is a true positive,
      and only weakly confident that an unplayed one is a true negative.

    We then minimize the confidence-weighted squared error
        sum_{u,i} c(u,i) (p(u,i) - x_u . y_i)^2 + lambda(||x_u||^2 + ||y_i||^2)
    by Alternating Least Squares: fix item vectors, solve a ridge regression for
    each user in closed form, then swap, and repeat. The HKV speed trick is to
    precompute Y^T Y once per iteration so each user solve only adds the small
    correction from that user's observed items.
    """

    def __init__(self, n_factors=32, n_epochs=15, alpha=40.0, reg=0.1,
                 random_state=config.RANDOM_STATE):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.reg = reg
        self.random_state = random_state
        self.user_factors_ = None
        self.item_factors_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def _als_step(self, solve_factors, fixed_factors, Cui_rows):
        """One half-iteration: update ``solve_factors`` given ``fixed_factors``.

        ``Cui_rows`` is a CSR matrix whose row e holds the confidence weights
        r(e, ·) for the entity being solved (users when solving users).
        """
        f = self.n_factors
        YtY = fixed_factors.T @ fixed_factors                 # (f, f), shared by all
        lambdaI = self.reg * np.eye(f)
        new = np.zeros_like(solve_factors)

        for e in range(solve_factors.shape[0]):
            start, end = Cui_rows.indptr[e], Cui_rows.indptr[e + 1]
            idx = Cui_rows.indices[start:end]                 # observed columns
            if len(idx) == 0:
                continue
            r = Cui_rows.data[start:end]                      # log1p plays
            c = 1.0 + self.alpha * r                          # confidence
            Y = fixed_factors[idx]                            # (n_obs, f)
            # A = Y^T C Y + lambda I = Y^T Y + Y^T (C - I) Y + lambda I
            A = YtY + (Y.T * (c - 1.0)) @ Y + lambdaI
            # b = Y^T C p, and p = 1 on observed items -> b = sum_i c_i * y_i
            b = (Y.T * c) @ np.ones(len(idx))
            new[e] = np.linalg.solve(A, b)
        return new

    def fit(self, ratings):
        (matrix, self.user_ids_, self.item_ids_,
         self.user_id_to_index_, self.item_id_to_index_) = build_user_item_matrix(ratings)
        n_users, n_items = matrix.shape

        rng = np.random.default_rng(self.random_state)
        # Small random init; scaling keeps initial dot products near zero.
        self.user_factors_ = 0.01 * rng.standard_normal((n_users, self.n_factors))
        self.item_factors_ = 0.01 * rng.standard_normal((n_items, self.n_factors))

        user_rows = matrix.tocsr()        # user e -> observed items + weights
        item_rows = matrix.T.tocsr()      # item e -> observing users + weights

        for _ in range(self.n_epochs):
            self.user_factors_ = self._als_step(self.user_factors_, self.item_factors_, user_rows)
            self.item_factors_ = self._als_step(self.item_factors_, self.user_factors_, item_rows)
        return self

    def predict_score(self, user_id, item_id):
        u = self.user_id_to_index_.get(user_id)
        i = self.item_id_to_index_.get(item_id)
        if u is None or i is None:
            return 0.0
        return float(self.user_factors_[u] @ self.item_factors_[i])

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        u = self.user_id_to_index_.get(user_id)
        if u is None:
            return []
        scores = self.item_factors_ @ self.user_factors_[u]   # (n_items,)
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        out = []
        for idx in np.argsort(-scores):
            item_id = self.item_ids_[idx]
            if item_id in seen:
                continue
            out.append((item_id, float(scores[idx])))
            if len(out) >= n:
                break
        return out


class SurpriseSVDRecommender:
    """Biased SVD from scikit-surprise, trained on log-plays as pseudo-ratings.

        score(u, i) = mu + b_u + b_i + p_u . q_i

    Surprise's SVD is built for *explicit* ratings, so we treat ``log1p(plays)``
    as a continuous rating on a [0, max] scale. It is a useful contrast to the
    implicit ALS: same latent-factor idea, but it models observed interactions
    only (no confidence weighting of the unobserved). Imported lazily so the rest
    of the project runs even if Surprise is not installed.
    """

    def __init__(self, n_factors=50, n_epochs=20, random_state=config.RANDOM_STATE):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.model_ = None
        self.item_ids_ = None
        self.trainset_ = None

    def fit(self, ratings):
        from surprise import SVD, Dataset, Reader

        lo = float(ratings[config.WEIGHT_COL].min())
        hi = float(ratings[config.WEIGHT_COL].max())
        reader = Reader(rating_scale=(lo, hi))
        data = Dataset.load_from_df(
            ratings[[config.USER_COL, config.ITEM_COL, config.WEIGHT_COL]], reader
        )
        self.trainset_ = data.build_full_trainset()
        self.model_ = SVD(
            n_factors=self.n_factors, n_epochs=self.n_epochs,
            random_state=self.random_state,
        )
        self.model_.fit(self.trainset_)
        self.item_ids_ = ratings[config.ITEM_COL].unique()
        return self

    def predict_score(self, user_id, item_id):
        return float(self.model_.predict(user_id, item_id).est)

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        preds = []
        for item_id in self.item_ids_:
            if exclude_seen and item_id in seen:
                continue
            preds.append((item_id, float(self.model_.predict(user_id, item_id).est)))
        preds.sort(key=lambda t: t[1], reverse=True)
        return preds[:n]


# Backwards-compatible alias for the name used in the placeholder main.py.
MatrixFactorizationRecommender = ImplicitALSRecommender
