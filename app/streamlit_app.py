"""Interactive prototype for the Last.fm music recommender.

Run from the project root:
    streamlit run app/streamlit_app.py

Three sections (sidebar):
    * Recommend  -- pick a user, compare algorithms side by side
    * Explore    -- dataset EDA dashboard
    * Evaluate   -- metric comparison table + charts
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data_loading import describe_dataset
from src.pipeline import build_models, prepare_data

st.set_page_config(page_title="Last.fm Recommender", page_icon="🎧", layout="wide")

ALGO_LABELS = {
    "most_popular": "Most Popular (baseline)",
    "highest_average": "Highest Average (baseline)",
    "random": "Random (baseline)",
    "content_based": "Content-based (tags)",
    "item_item_cf": "Item-Item CF",
    "user_user_cf": "User-User CF",
    "implicit_als": "Matrix Factorization (ALS)",
    "surprise_svd": "Matrix Factorization (SVD)",
    "friends_social": "Friends / Social",
}


# ---------------------------------------------------------------------------
# Cached heavy objects
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Last.fm data ...")
def get_data():
    return prepare_data()


@st.cache_resource(show_spinner="Training recommenders ...")
def get_models(_train, _items, _friends):
    return build_models(_train, _items, _friends)


@st.cache_data
def get_stats(_ratings):
    return describe_dataset(_ratings, verbose=False)


def name_map(items):
    return dict(zip(items[config.ITEM_ID_COL], items[config.TITLE_COL]))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_recommend(data, models, cb_model):
    items, train = data["items"], data["train"]
    names = name_map(items)
    tags = dict(zip(items[config.ITEM_ID_COL], items[config.GENRES_COL]))

    st.header("🎧 Recommendations")
    user_ids = sorted(train[config.USER_COL].unique())

    c1, c2 = st.columns([1, 2])
    with c1:
        user_id = st.selectbox("User", user_ids, index=0)
        n = st.slider("How many recommendations", 3, 20, 10)
        chosen = st.multiselect(
            "Algorithms to compare",
            list(models.keys()),
            default=["content_based", "item_item_cf", "implicit_als"],
            format_func=lambda k: ALGO_LABELS.get(k, k),
        )

    # User's listening profile
    played = (
        train[train[config.USER_COL] == user_id]
        .sort_values(config.RAW_WEIGHT_COL, ascending=False)
    )
    with c2:
        st.subheader(f"User {user_id} listens to")
        prof = played.head(10).copy()
        prof["artist"] = prof[config.ITEM_COL].map(names)
        st.dataframe(
            prof[["artist", config.RAW_WEIGHT_COL]].rename(columns={config.RAW_WEIGHT_COL: "plays"}),
            hide_index=True, use_container_width=True,
        )

    st.divider()
    if not chosen:
        st.info("Pick at least one algorithm.")
        return

    cols = st.columns(len(chosen))
    for col, algo in zip(cols, chosen):
        with col:
            st.markdown(f"**{ALGO_LABELS.get(algo, algo)}**")
            recs = models[algo].recommend(user_id, train, n=n)
            if not recs:
                st.caption("No recommendations (e.g. user has no friends).")
                continue
            rows = []
            for item_id, score in recs:
                t = tags.get(item_id, "") or ""
                rows.append({
                    "artist": names.get(item_id, str(item_id)),
                    "score": round(score, 3),
                    "top tags": " · ".join(t.split()[:3]),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_explore(data):
    ratings, items = data["ratings"], data["items"]
    names = name_map(items)
    stats = get_stats(ratings)

    st.header("📊 Dataset exploration")
    a, b, c, d = st.columns(4)
    a.metric("Users", f"{stats['n_users']:,}")
    b.metric("Artists", f"{stats['n_items']:,}")
    c.metric("Interactions", f"{stats['n_interactions']:,}")
    d.metric("Sparsity", f"{stats['sparsity']*100:.2f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Most popular artists (by listeners)")
        pop = stats["most_popular_items"].rename("listeners").reset_index()
        pop["artist"] = pop[config.ITEM_COL].map(names)
        st.bar_chart(pop.set_index("artist")["listeners"])

    with right:
        st.subheader("Play-count distribution (log scale)")
        logplays = np.log1p(ratings[config.RAW_WEIGHT_COL])
        hist = np.histogram(logplays, bins=40)
        chart_df = pd.DataFrame({"log1p(plays)": hist[1][:-1], "count": hist[0]})
        st.bar_chart(chart_df.set_index("log1p(plays)"))

    st.subheader("The long tail: artist popularity, ranked")
    per_item = ratings.groupby(config.ITEM_COL)[config.USER_COL].nunique().sort_values(ascending=False)
    st.line_chart(per_item.reset_index(drop=True).rename("listeners"))
    st.caption(
        "A few artists have very many listeners; the vast majority have very few. "
        "This long-tail skew is exactly why 'most popular' is a strong but biased baseline."
    )


def page_evaluate():
    st.header("📈 Model evaluation")
    csv = config.RESULTS_DIR / "metrics.csv"
    if not csv.exists():
        st.warning("No metrics yet. Run `python main.py` to generate results/metrics.csv.")
        return

    metrics = pd.read_csv(csv, index_col=0)
    metrics.index = [ALGO_LABELS.get(i, i) for i in metrics.index]

    st.subheader(f"Metrics @{config.TOP_K}")
    acc = [c for c in ["precision", "recall", "ndcg", "mrr", "hit_rate"] if c in metrics.columns]
    beyond = [c for c in ["coverage", "novelty", "diversity", "avg_popularity"] if c in metrics.columns]
    st.dataframe(
        metrics.style.background_gradient(cmap="Greens", subset=acc)
                     .background_gradient(cmap="Blues", subset=beyond)
                     .format("{:.3f}"),
        use_container_width=True,
    )

    st.divider()
    figs = [
        ("accuracy_metrics.png", "Accuracy metrics"),
        ("beyond_accuracy_metrics.png", "Beyond-accuracy metrics"),
        ("accuracy_vs_popularity.png", "Accuracy vs popularity bias"),
    ]
    for fname, caption in figs:
        path = config.FIGURES_DIR / fname
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("🎵 Last.fm Recommender")
    st.sidebar.caption("Music recommender prototype — HetRec 2011")
    page = st.sidebar.radio("Section", ["Recommend", "Explore", "Evaluate"])

    data = get_data()
    if page == "Recommend":
        models, cb_model = get_models(data["train"], data["items"], data["friends"])
        page_recommend(data, models, cb_model)
    elif page == "Explore":
        page_explore(data)
    else:
        page_evaluate()


if __name__ == "__main__":
    main()
