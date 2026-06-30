"""Social / friend-based recommender (uses user_friends.dat).

Last.fm gives us something most recommender datasets don't: an explicit **social
graph**. This is a form of trust-based collaborative filtering -- instead of
finding "similar" users by their listening overlap (as user-user CF does), we
use the users the person *declared* as friends as the neighbourhood.

    score(u, i) = sum over friends f of u of  weight(f, i)

i.e. an artist is recommended if the user's friends collectively listen to it a
lot (and the user hasn't already). The premise -- "your friends' taste predicts
yours" (homophily) -- is exactly what social music platforms exploit.

This complements the other models: it can surface artists that pure listening
overlap would miss, and it lets us ask in the evaluation whether the social
signal actually helps. Its obvious limit is coverage: a user with no friends, or
friends with little listening history, gets nothing -- so we report how many
users it can serve.
"""

from collections import defaultdict

from . import config
from .data_loading import get_seen_items, load_friends


class FriendBasedRecommender:
    """Recommend artists popular among a user's declared friends."""

    def __init__(self):
        self.friends_of_ = None      # userId -> list of friendIds
        self.user_items_ = None      # userId -> list of (itemId, weight)

    def fit(self, ratings, friends=None):
        if friends is None:
            friends = load_friends()
        self.friends_of_ = (
            friends.groupby(config.USER_COL)["friendId"].apply(list).to_dict()
        )
        # Pre-index each user's (item, weight) pairs for fast friend aggregation.
        self.user_items_ = {
            uid: list(zip(grp[config.ITEM_COL], grp[config.WEIGHT_COL]))
            for uid, grp in ratings.groupby(config.USER_COL)
        }
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        friends = self.friends_of_.get(user_id, [])
        if not friends:
            return []

        scores = defaultdict(float)
        for f in friends:
            for item_id, w in self.user_items_.get(f, []):
                scores[item_id] += w
        if not scores:
            return []

        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for item_id, score in ranked:
            if item_id in seen:
                continue
            out.append((item_id, float(score)))
            if len(out) >= n:
                break
        return out

    def coverage_fraction(self, users) -> float:
        """Fraction of the given users this model can actually serve (have friends)."""
        servable = sum(1 for u in users if self.friends_of_.get(u))
        return servable / len(users) if len(users) else 0.0
