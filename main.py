"""End-to-end runner for the Last.fm music recommender prototype.

Pipeline:
    1. Load + filter + describe the dataset (EDA stats)
    2. Per-user train/test split
    3. Fit every recommender
    4. Show top-N recommendations for 3 example users (qualitative comparison)
    5. Evaluate all models -> results/metrics.csv  (quantitative comparison)
    6. Save comparison figures -> results/figures/

Run:  python main.py
"""

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

from src import config
from src.data_loading import describe_dataset
from src.evaluation import evaluate_all
from src.pipeline import build_models, prepare_data

N_EXAMPLE_USERS = 3
TOP_N = config.TOP_K


def show_examples(models, items, train, users):
    """Print top-N recommendations from each model for a few users."""
    name_of = dict(zip(items[config.ITEM_ID_COL], items[config.TITLE_COL]))
    for u in users:
        played = (
            train[train[config.USER_COL] == u]
            .sort_values(config.RAW_WEIGHT_COL, ascending=False)
            .head(5)
        )
        print(f"\n{'='*70}\nUser {u} -- top played artists: "
              f"{', '.join(name_of.get(i, str(i)) for i in played[config.ITEM_COL])}")
        for mname, model in models.items():
            recs = model.recommend(u, train, n=5)
            labels = [name_of.get(i, str(i)) for i, _ in recs] or ["(none)"]
            print(f"  {mname:<16}: {', '.join(labels)}")


def save_figures(metrics):
    """Write comparison charts to results/figures/."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Accuracy metrics side by side.
    acc_cols = [c for c in ["precision", "recall", "ndcg", "mrr"] if c in metrics.columns]
    ax = metrics[acc_cols].plot(kind="bar", figsize=(11, 5))
    ax.set_title(f"Accuracy metrics @{TOP_N}")
    ax.set_ylabel("score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "accuracy_metrics.png", dpi=120)
    plt.close()

    # 2) Beyond-accuracy: coverage, novelty, diversity (normalized for shape).
    beyond = [c for c in ["coverage", "novelty", "diversity"] if c in metrics.columns]
    norm = metrics[beyond] / metrics[beyond].max()
    ax = norm.plot(kind="bar", figsize=(11, 5))
    ax.set_title("Beyond-accuracy metrics (normalized to max=1 per metric)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "beyond_accuracy_metrics.png", dpi=120)
    plt.close()

    # 3) The accuracy-vs-popularity-bias tradeoff.
    if {"precision", "avg_popularity"} <= set(metrics.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(metrics["avg_popularity"], metrics["precision"])
        for name, row in metrics.iterrows():
            ax.annotate(name, (row["avg_popularity"], row["precision"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("avg popularity of recommendations (popularity bias ->)")
        ax.set_ylabel(f"precision@{TOP_N}")
        ax.set_title("Accuracy vs popularity bias")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "accuracy_vs_popularity.png", dpi=120)
        plt.close()

    print(f"Saved figures to {config.FIGURES_DIR}")


def main():
    print("Last.fm music recommender -- full pipeline\n")

    data = prepare_data()
    train, test, items, friends = data["train"], data["test"], data["items"], data["friends"]

    describe_dataset(data["ratings"], items)

    print("\nFitting models ...")
    models, cb_model = build_models(train, items, friends)
    print(f"Fitted: {', '.join(models)}")

    # Example users: a few with enough history to be interesting.
    example_users = (
        train.groupby(config.USER_COL).size().sort_values(ascending=False)
        .head(50).sample(N_EXAMPLE_USERS, random_state=config.RANDOM_STATE).index.tolist()
    )
    show_examples(models, items, train, example_users)

    print("\nEvaluating all models (this can take a couple of minutes) ...")
    eval_users = sorted(test[config.USER_COL].unique())
    metrics = evaluate_all(models, train, test, eval_users, k=TOP_N, cb_model=cb_model)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.RESULTS_DIR / "metrics.csv"
    metrics.round(4).to_csv(out_csv)
    print(f"\n=== Metrics @{TOP_N} (saved to {out_csv}) ===")
    print(metrics.round(4).to_string())

    save_figures(metrics)
    print("\nDone.")


if __name__ == "__main__":
    main()
