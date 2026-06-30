"""Evaluation metrics and the model-comparison loop.

We use a **top-N ranking** evaluation with **binary relevance**: for each user we
hid some of their artists (the test set); a recommendation is "relevant" if it is
one of those held-out artists. This mirrors the real task ("did we recommend
things the user actually goes on to listen to?") better than rating prediction.

Two families of metrics, because -- per the brief -- accuracy is not enough:

Accuracy / ranking
    precision@k   fraction of the top-k that are relevant (how clean is the list)
    recall@k      fraction of relevant items we surfaced (how complete)
    ndcg@k        rewards putting relevant items *higher* (rank-sensitive)
    mrr           1 / rank of the first hit (how soon a hit appears)
    hit_rate@k    did we get at least one hit?

Beyond-accuracy
    catalog_coverage  how much of the catalogue the model ever recommends
    novelty           how non-obvious / niche the recommendations are
    diversity         how varied (by content) a single user's list is
    avg_popularity    mean popularity of recommended items -> popularity bias
"""

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Accuracy / ranking metrics (per user)
# ---------------------------------------------------------------------------
def precision_at_k(recommended_items, relevant_items, k=10):
    if k == 0:
        return 0.0
    topk = recommended_items[:k]
    hits = sum(1 for it in topk if it in relevant_items)
    return hits / k


def recall_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return 0.0
    topk = recommended_items[:k]
    hits = sum(1 for it in topk if it in relevant_items)
    return hits / len(relevant_items)


def hit_rate_at_k(recommended_items, relevant_items, k=10):
    return 1.0 if any(it in relevant_items for it in recommended_items[:k]) else 0.0


def dcg_at_k(relevance_scores, k=10):
    """Discounted Cumulative Gain. Rank i starts at 1; discount = 1/log2(i+1)."""
    rel = np.asarray(relevance_scores[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def ndcg_at_k(recommended_items, relevant_items, k=10):
    """Normalized DCG with binary relevance.

    DCG of our ranking divided by the best achievable DCG (all relevant items at
    the top). 1.0 means perfect ordering; 0.0 means no relevant items in top-k.
    """
    gains = [1.0 if it in relevant_items else 0.0 for it in recommended_items[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = dcg_at_k([1.0] * min(len(relevant_items), k), k)
    return dcg / ideal if ideal > 0 else 0.0


def mean_reciprocal_rank(recommended_items, relevant_items, k=10):
    for rank, it in enumerate(recommended_items[:k], start=1):
        if it in relevant_items:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------
def catalog_coverage(all_recommendations, all_items):
    """Share of the catalogue that appears in at least one user's recommendations.

    Low coverage => the model funnels everyone toward the same few items
    (popularity bias); high coverage => it exploits the long tail.
    """
    recommended = set(all_recommendations)
    catalog = set(all_items)
    return len(recommended & catalog) / len(catalog) if catalog else 0.0


def item_popularity(ratings_train):
    """Series: itemId -> number of distinct listeners in the training set."""
    return ratings_train.groupby(config.ITEM_COL)[config.USER_COL].nunique()


def novelty(recommended_items, popularity, n_users):
    """Mean self-information of the recommended items.

    self-information(i) = -log2( listeners(i) / n_users ). A blockbuster everyone
    knows carries little information (low novelty); a niche artist carries a lot.
    We average over the list.
    """
    if not recommended_items or n_users == 0:
        return 0.0
    vals = []
    for it in recommended_items:
        listeners = popularity.get(it, 0)
        if listeners <= 0:
            continue
        prob = listeners / n_users
        vals.append(-np.log2(prob))
    return float(np.mean(vals)) if vals else 0.0


def intra_list_diversity(recommended_items, cb_model):
    """1 - average pairwise content cosine similarity within the list.

    Uses the content-based model's TF-IDF tag vectors. 0 => every recommended
    artist is content-identical; 1 => maximally varied. Captures whether a model
    gives a one-note list or a varied one.
    """
    if cb_model is None or len(recommended_items) < 2:
        return 0.0
    idxs = [cb_model.item_id_to_index_.get(it) for it in recommended_items]
    idxs = [i for i in idxs if i is not None]
    if len(idxs) < 2:
        return 0.0
    vectors = cb_model.item_features_[idxs]
    sims = (vectors @ vectors.T).toarray()        # TF-IDF rows are L2-normalized
    n = sims.shape[0]
    iu = np.triu_indices(n, k=1)                   # unique pairs, exclude diagonal
    mean_sim = float(np.mean(sims[iu])) if iu[0].size else 0.0
    return 1.0 - mean_sim


def average_popularity(recommended_items, popularity):
    """Mean listener-count of recommended items -> a popularity-bias indicator."""
    if not recommended_items:
        return 0.0
    return float(np.mean([popularity.get(it, 0) for it in recommended_items]))


# ---------------------------------------------------------------------------
# Model comparison loop
# ---------------------------------------------------------------------------
def evaluate_model(model, ratings_train, ratings_test, users, k=10,
                   popularity=None, n_train_users=None, cb_model=None,
                   catalog_items=None):
    """Evaluate one recommender over a set of users and return mean metrics.

    Parameters
    ----------
    model        : a fitted recommender exposing ``recommend(user, train, n)``
    ratings_train: training interactions (recommendations drawn from here)
    ratings_test : held-out interactions (define relevance)
    users        : iterable of user ids to evaluate
    k            : cutoff for all @k metrics
    popularity   : Series itemId->listeners (computed from train if None)
    n_train_users: number of training users (for novelty; inferred if None)
    cb_model     : fitted ContentBasedRecommender (for diversity); optional
    catalog_items: full catalogue for coverage (train items if None)
    """
    if popularity is None:
        popularity = item_popularity(ratings_train)
    if n_train_users is None:
        n_train_users = ratings_train[config.USER_COL].nunique()
    if catalog_items is None:
        catalog_items = ratings_train[config.ITEM_COL].unique()

    relevant_by_user = ratings_test.groupby(config.USER_COL)[config.ITEM_COL].apply(set).to_dict()

    acc = {m: [] for m in ["precision", "recall", "ndcg", "mrr", "hit_rate",
                           "novelty", "diversity", "avg_popularity"]}
    all_recommended = set()
    n_eval = 0

    for u in users:
        relevant = relevant_by_user.get(u, set())
        if not relevant:                      # only score users with held-out items
            continue
        recs = [it for it, _ in model.recommend(u, ratings_train, n=k)]
        if not recs:
            # Model couldn't serve this user (e.g. no friends): counts as a miss.
            for key in ["precision", "recall", "ndcg", "mrr", "hit_rate",
                        "novelty", "diversity", "avg_popularity"]:
                acc[key].append(0.0)
            n_eval += 1
            continue

        acc["precision"].append(precision_at_k(recs, relevant, k))
        acc["recall"].append(recall_at_k(recs, relevant, k))
        acc["ndcg"].append(ndcg_at_k(recs, relevant, k))
        acc["mrr"].append(mean_reciprocal_rank(recs, relevant, k))
        acc["hit_rate"].append(hit_rate_at_k(recs, relevant, k))
        acc["novelty"].append(novelty(recs, popularity, n_train_users))
        acc["diversity"].append(intra_list_diversity(recs, cb_model))
        acc["avg_popularity"].append(average_popularity(recs, popularity))
        all_recommended.update(recs)
        n_eval += 1

    result = {m: (float(np.mean(v)) if v else 0.0) for m, v in acc.items()}
    result["coverage"] = catalog_coverage(all_recommended, catalog_items)
    result["n_eval_users"] = n_eval
    return result


def evaluate_all(models: dict, ratings_train, ratings_test, users, k=10,
                 cb_model=None) -> pd.DataFrame:
    """Run ``evaluate_model`` for every model and return a comparison DataFrame."""
    popularity = item_popularity(ratings_train)
    n_train_users = ratings_train[config.USER_COL].nunique()
    catalog_items = ratings_train[config.ITEM_COL].unique()

    rows = {}
    for name, model in models.items():
        rows[name] = evaluate_model(
            model, ratings_train, ratings_test, users, k=k,
            popularity=popularity, n_train_users=n_train_users,
            cb_model=cb_model, catalog_items=catalog_items,
        )
    df = pd.DataFrame(rows).T
    ordered = ["precision", "recall", "ndcg", "mrr", "hit_rate",
               "coverage", "novelty", "diversity", "avg_popularity", "n_eval_users"]
    return df[[c for c in ordered if c in df.columns]]
