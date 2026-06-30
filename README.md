# 🎧 Last.fm Music Recommender

Individual project — ESADE *Recommender Systems* (Prof. Marc Torrens), **Music track**.

A from-scratch music recommender prototype built on the **Last.fm HetRec 2011**
dataset. It implements non-personalized, content-based, collaborative-filtering,
matrix-factorization, and social recommenders behind one shared interface,
evaluates them with accuracy **and** beyond-accuracy metrics, and ships an
interactive Streamlit app.

## Dataset

[Last.fm HetRec 2011](https://grouplens.org/datasets/hetrec-2011/) (GroupLens):
1 892 users, 17 632 artists, ~92 800 user–artist play counts, ~11 900 tags, and a
12 717-edge friend graph. **Implicit feedback** (play counts), not star ratings.
Non-commercial research use — cite Last.fm and the HetRec'11 workshop.

The raw `.dat` files are *not* committed (license). Download them with:

```bash
python scripts/download_data.py     # -> data/raw/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # or: pip install -r requirements-lock.txt
python scripts/download_data.py
```

## Run

```bash
python main.py                          # full pipeline -> results/metrics.csv + figures
streamlit run app/streamlit_app.py      # interactive prototype
python tests/test_metrics.py            # metric sanity tests
```

## Methods implemented (`src/`)

| Module | Models |
| --- | --- |
| `baselines.py` | Most-Popular, Highest-Average (min listeners), Random |
| `content_based.py` | TF-IDF over artist tags + centered user profiles (cosine) |
| `collaborative_filtering.py` | Item-Item CF, User-User CF (cosine, top-k) |
| `matrix_factorization.py` | Implicit ALS (from scratch, Hu-Koren-Volinsky) + scikit-surprise SVD |
| `social.py` | Friend/trust-based recommender (uses `user_friends.dat`) |

All recommenders share one interface:

```python
model.fit(train, ...)
model.recommend(user_id, train, n=10, exclude_seen=True)  # -> [(item_id, score), ...]
```

## Evaluation (`src/evaluation.py`)

Top-N ranking with binary relevance from a **per-user leave-out** split.

- **Accuracy:** Precision@K, Recall@K, NDCG@K, MRR, Hit-Rate@K
- **Beyond-accuracy:** catalog coverage, novelty (self-information), intra-list
  diversity (content), average popularity (popularity-bias indicator)

## Design decisions

The full design rationale is in
[`docs/superpowers/specs/`](docs/superpowers/specs/). Key choices:

- **`weight = log1p(plays)`** to tame the extreme play-count skew.
- **Per-user 80/20 split** so every test user is also in training.
- **Implicit ALS** (confidence-weighted) as the principled MF for play counts;
  surprise SVD as a library comparison.

## Project layout

```
src/            recommender modules (one method family per file)
app/            Streamlit prototype
scripts/        dataset download
tests/          metric sanity tests
notebooks/      analysis / EDA
results/        metrics.csv + figures (generated)
docs/           design spec
```
