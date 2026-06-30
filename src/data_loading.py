"""Data loading and preprocessing for the Last.fm HetRec 2011 dataset.

Why this module matters
-----------------------
Everything downstream (EDA, every recommender, evaluation) consumes the output
of this module, so the design decisions here shape the whole project:

1. **Implicit feedback.** Last.fm records *play counts*, not 1-5 ratings. We keep
   the raw count (`plays`) for EDA and derive a modelling signal
   ``weight = log1p(plays)``. The log compresses an extreme range (a user can
   play one artist 100,000 times and another 5 times); without it, a handful of
   obsessions would dominate every similarity and factorization.

2. **Generic column names.** We rename Last.fm's ``userID/artistID/weight`` to
   the generic ``userId/itemId/weight`` from ``config`` so model code never
   hard-codes Last.fm specifics.

3. **Cold-start filtering.** Users/artists with very few interactions add noise
   and make neighbourhoods unreliable, so we iteratively drop them.

4. **Per-user leave-out split.** We hold out a fraction of *each user's* artists
   for testing (not a global random split), guaranteeing every test user is also
   present in training -- otherwise user-based methods would have nothing to work
   with for held-out users.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from . import config


def _read_dat(path) -> pd.DataFrame:
    """Read a tab-separated Last.fm .dat file with robust encoding handling.

    The dataset mixes encodings: artists.dat is UTF-8 ("Björk"), but tags.dat is
    latin-1. We try UTF-8 first (the stricter codec) and fall back to latin-1,
    so each file is decoded with whatever actually works.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep="\t", encoding=enc)
        except UnicodeDecodeError:
            continue
    # Last resort: never crash, replace undecodable bytes.
    return pd.read_csv(path, sep="\t", encoding="utf-8", encoding_errors="replace")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_ratings(path=config.RATINGS_PATH) -> pd.DataFrame:
    """Load user-artist listening counts (the implicit-feedback interactions).

    Raw Last.fm columns: ``userID, artistID, weight`` (weight = total plays).

    Returns a DataFrame with generic columns:
        userId, itemId, plays (raw int), weight (= log1p(plays))
    """
    df = _read_dat(path)
    required = {"userID", "artistID", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.rename(
        columns={
            "userID": config.USER_COL,
            "artistID": config.ITEM_COL,
            "weight": config.RAW_WEIGHT_COL,
        }
    )
    # Drop non-positive / missing play counts (data hygiene).
    df = df[df[config.RAW_WEIGHT_COL] > 0].copy()
    # The modelling signal: log1p damps the heavy tail of play counts.
    df[config.WEIGHT_COL] = np.log1p(df[config.RAW_WEIGHT_COL].astype(float))
    return df[[config.USER_COL, config.ITEM_COL, config.RAW_WEIGHT_COL, config.WEIGHT_COL]]


def load_items(path=config.ITEMS_PATH, attach_tags: bool = True) -> pd.DataFrame:
    """Load artist metadata, optionally enriched with a tag "document".

    Raw Last.fm columns: ``id, name, url, pictureURL``.

    Returns:
        DataFrame with columns: itemId, name, url, (tags if attach_tags)
        where ``tags`` is a space-joined string of the artist's tags, used as
        the content-based feature text.
    """
    df = _read_dat(path)
    if "id" not in df.columns or "name" not in df.columns:
        raise ValueError(f"{path} missing required columns id/name")
    df = df.rename(columns={"id": config.ITEM_ID_COL, "name": config.TITLE_COL})
    keep = [c for c in [config.ITEM_ID_COL, config.TITLE_COL, "url"] if c in df.columns]
    df = df[keep].copy()

    if attach_tags:
        tag_docs = build_item_tag_documents()
        df = df.merge(tag_docs, on=config.ITEM_ID_COL, how="left")
        df[config.GENRES_COL] = df[config.GENRES_COL].fillna("")
    return df


def build_item_tag_documents(
    tags_path=config.TAGS_PATH,
    user_tagged_path=config.USER_TAGGED_ARTISTS_PATH,
) -> pd.DataFrame:
    """Build one tag "document" per artist for content-based features.

    Logic:
    - ``user_taggedartists.dat`` gives (user, artist, tagID) assignments.
    - ``tags.dat`` maps tagID -> human tag string ("alternative metal").
    - We join them, collapse multi-word tags into single tokens
      ("alternative metal" -> "alternative_metal") so TF-IDF treats them as one
      term, and join all of an artist's tag tokens (with repetition) into a
      single text. Repetition is deliberate: a tag applied by many users gets a
      higher term frequency, which is exactly the signal we want.

    Returns:
        DataFrame with columns: itemId, tags (space-joined token string)
    """
    tags = _read_dat(tags_path)
    assigns = _read_dat(user_tagged_path)
    merged = assigns.merge(tags, on="tagID", how="inner")
    merged["token"] = (
        merged["tagValue"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    )
    docs = (
        merged.groupby("artistID")["token"]
        .apply(lambda toks: " ".join(toks))
        .reset_index()
        .rename(columns={"artistID": config.ITEM_ID_COL, "token": config.GENRES_COL})
    )
    return docs


def load_friends(path=config.USER_FRIENDS_PATH) -> pd.DataFrame:
    """Load the bidirectional social graph.

    Raw Last.fm columns: ``userID, friendID``. Returns generic columns
    ``userId, friendId``. Each friendship appears as two directed rows already.
    """
    df = _read_dat(path)
    df = df.rename(columns={"userID": config.USER_COL, "friendID": "friendId"})
    return df[[config.USER_COL, "friendId"]]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_active(
    ratings: pd.DataFrame,
    min_user=config.MIN_USER_INTERACTIONS,
    min_item=config.MIN_ITEM_INTERACTIONS,
    max_passes: int = 5,
) -> pd.DataFrame:
    """Iteratively drop cold users/items.

    Removing rare items can make some users fall below the threshold (and vice
    versa), so we repeat until the set is stable or we hit ``max_passes``.
    """
    df = ratings
    for _ in range(max_passes):
        before = len(df)
        item_counts = df[config.ITEM_COL].value_counts()
        keep_items = item_counts[item_counts >= min_item].index
        df = df[df[config.ITEM_COL].isin(keep_items)]

        user_counts = df[config.USER_COL].value_counts()
        keep_users = user_counts[user_counts >= min_user].index
        df = df[df[config.USER_COL].isin(keep_users)]

        if len(df) == before:
            break
    return df.copy()


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def describe_dataset(ratings: pd.DataFrame, items: pd.DataFrame | None = None,
                     verbose: bool = True) -> dict:
    """Compute core dataset statistics for the EDA section.

    Returns a dict (so the app/notebook can reuse it) and optionally prints it.
    """
    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_interactions = len(ratings)
    # Sparsity = 1 - (observed cells / all possible cells).
    sparsity = 1.0 - n_interactions / (n_users * n_items)

    per_user = ratings.groupby(config.USER_COL).size()
    per_item = ratings.groupby(config.ITEM_COL).size()

    most_active = per_user.sort_values(ascending=False).head(10)
    most_popular = per_item.sort_values(ascending=False).head(10)

    stats = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_interactions),
        "sparsity": float(sparsity),
        "avg_interactions_per_user": float(per_user.mean()),
        "avg_interactions_per_item": float(per_item.mean()),
        "median_plays": float(ratings[config.RAW_WEIGHT_COL].median()),
        "max_plays": int(ratings[config.RAW_WEIGHT_COL].max()),
        "most_active_users": most_active,
        "most_popular_items": most_popular,
    }

    if verbose:
        print("=== Dataset summary ===")
        print(f"Users           : {stats['n_users']:,}")
        print(f"Items (artists) : {stats['n_items']:,}")
        print(f"Interactions    : {stats['n_interactions']:,}")
        print(f"Sparsity        : {stats['sparsity']*100:.3f}% empty")
        print(f"Avg artists/user: {stats['avg_interactions_per_user']:.1f}")
        print(f"Avg users/artist: {stats['avg_interactions_per_item']:.1f}")
        print(f"Plays: median={stats['median_plays']:.0f}, max={stats['max_plays']:,}")
        if items is not None:
            named = most_popular.rename("listeners").reset_index().merge(
                items[[config.ITEM_ID_COL, config.TITLE_COL]],
                left_on=config.ITEM_COL, right_on=config.ITEM_ID_COL, how="left",
            )
            print("\nTop popular artists:")
            for _, r in named.iterrows():
                print(f"  {r[config.TITLE_COL]!s:<30} {int(r['listeners'])} listeners")
    return stats


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
def train_test_split_ratings(
    ratings: pd.DataFrame,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
):
    """Per-user leave-out split.

    For each user we move a ``test_size`` fraction of their interactions to the
    test set, keeping the rest for training. Users keep at least one training
    item (so models can learn a profile) and contribute at least one test item
    when they have >= 2 interactions (so they are evaluable).

    Returns:
        (train_df, test_df)
    """
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []

    for _, group in ratings.groupby(config.USER_COL):
        idx = group.index.to_numpy()
        n = len(idx)
        if n < 2:
            train_idx.extend(idx)          # too few to hold anything out
            continue
        idx = rng.permutation(idx)         # returns a fresh, writable array
        n_test = max(1, int(round(test_size * n)))
        n_test = min(n_test, n - 1)        # always leave >= 1 for training
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train = ratings.loc[train_idx].copy()
    test = ratings.loc[test_idx].copy()
    return train, test


def get_seen_items(ratings: pd.DataFrame, user_id) -> set:
    """Set of item IDs a user has already interacted with (to exclude from recs)."""
    return set(ratings.loc[ratings[config.USER_COL] == user_id, config.ITEM_COL])


# ---------------------------------------------------------------------------
# Shared sparse matrix builder (used by CF and MF)
# ---------------------------------------------------------------------------
def build_user_item_matrix(ratings: pd.DataFrame, value_col=config.WEIGHT_COL):
    """Build a sparse user-item matrix plus id<->index lookups.

    Returns:
        matrix          : scipy.sparse.csr_matrix, shape (n_users, n_items)
        user_ids        : np.ndarray of user ids (row order)
        item_ids        : np.ndarray of item ids (column order)
        user_id_to_index: dict
        item_id_to_index: dict
    """
    user_ids = np.sort(ratings[config.USER_COL].unique())
    item_ids = np.sort(ratings[config.ITEM_COL].unique())
    user_id_to_index = {u: i for i, u in enumerate(user_ids)}
    item_id_to_index = {it: j for j, it in enumerate(item_ids)}

    rows = ratings[config.USER_COL].map(user_id_to_index).to_numpy()
    cols = ratings[config.ITEM_COL].map(item_id_to_index).to_numpy()
    vals = ratings[value_col].to_numpy(dtype=float)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return matrix, user_ids, item_ids, user_id_to_index, item_id_to_index
