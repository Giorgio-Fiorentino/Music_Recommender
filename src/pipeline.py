"""Shared pipeline: load data and build the full set of fitted models.

Both the command-line runner (``main.py``) and the Streamlit app use these
helpers so the data preparation and model zoo are defined in exactly one place.
"""

from . import data_loading as dl
from .baselines import (
    HighestAverageRatingRecommender,
    MostPopularRecommender,
    RandomRecommender,
)
from .collaborative_filtering import (
    ItemItemCollaborativeFiltering,
    UserUserCollaborativeFiltering,
)
from .content_based import ContentBasedRecommender
from .matrix_factorization import ImplicitALSRecommender, SurpriseSVDRecommender
from .social import FriendBasedRecommender


def prepare_data():
    """Load, filter, and split the Last.fm data.

    Returns:
        dict with keys: ratings, items, friends, train, test
    """
    dl.ensure_raw_data()  # download the dataset on a fresh checkout (e.g. cloud)
    ratings = dl.filter_active(dl.load_ratings())
    items = dl.load_items()
    friends = dl.load_friends()
    train, test = dl.train_test_split_ratings(ratings)
    return {
        "ratings": ratings,
        "items": items,
        "friends": friends,
        "train": train,
        "test": test,
    }


def build_models(train, items, friends, include_surprise=True):
    """Fit every recommender on the training split.

    Returns:
        (models, cb_model) where ``models`` is an ordered dict-like mapping name
        -> fitted model, and ``cb_model`` is the content-based model (reused by
        the evaluation diversity metric).
    """
    content = ContentBasedRecommender().fit(train, items)

    models = {
        "most_popular": MostPopularRecommender().fit(train),
        "highest_average": HighestAverageRatingRecommender(min_ratings=20).fit(train),
        "random": RandomRecommender().fit(train),
        "content_based": content,
        "item_item_cf": ItemItemCollaborativeFiltering(k=20).fit(train),
        "user_user_cf": UserUserCollaborativeFiltering(k=20).fit(train),
        "implicit_als": ImplicitALSRecommender().fit(train),
        "friends_social": FriendBasedRecommender().fit(train, friends),
    }

    if include_surprise:
        try:
            models["surprise_svd"] = SurpriseSVDRecommender().fit(train)
        except Exception as exc:  # noqa: BLE001 - keep pipeline alive without surprise
            print(f"[pipeline] scikit-surprise SVD skipped: {exc}")

    return models, content
