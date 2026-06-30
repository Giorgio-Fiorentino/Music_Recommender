"""Sanity tests for the evaluation metrics on hand-computable examples.

Run:  python -m pytest tests/  (or)  python tests/test_metrics.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev


REC = ["A", "B", "C", "D", "E"]
REL = {"B", "D", "X"}            # X is relevant but not recommended


def test_precision_at_k():
    # 2 of the top-5 (B, D) are relevant -> 2/5
    assert math.isclose(ev.precision_at_k(REC, REL, 5), 0.4)


def test_recall_at_k():
    # 2 of the 3 relevant items surfaced -> 2/3
    assert math.isclose(ev.recall_at_k(REC, REL, 5), 2 / 3)


def test_hit_rate():
    assert ev.hit_rate_at_k(REC, REL, 5) == 1.0
    assert ev.hit_rate_at_k(["A", "C"], REL, 5) == 0.0


def test_mrr():
    # first relevant item (B) is at rank 2 -> 1/2
    assert math.isclose(ev.mean_reciprocal_rank(REC, REL, 5), 0.5)


def test_dcg():
    # gains [1,1] -> 1/log2(2) + 1/log2(3) = 1 + 0.63093
    assert math.isclose(ev.dcg_at_k([1, 1], 2), 1 + 1 / math.log2(3), rel_tol=1e-9)


def test_ndcg():
    # our hits land at ranks 2 and 4.
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    # ideal: there are 3 relevant items, so the best ranking puts 3 hits at the
    # top (ranks 1, 2, 3) -- IDCG uses min(|relevant|, k) = 3 ones.
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(ev.ndcg_at_k(REC, REL, 5), dcg / idcg, rel_tol=1e-9)


def test_coverage():
    # 2 unique recommended items out of a 4-item catalogue -> 0.5
    assert math.isclose(ev.catalog_coverage(["A", "B", "A"], ["A", "B", "C", "D"]), 0.5)


def test_novelty_monotonic():
    import pandas as pd
    pop = pd.Series({"hit": 100, "niche": 1})
    # a niche item (lower popularity) must be more novel than a hit
    assert ev.novelty(["niche"], pop, 100) > ev.novelty(["hit"], pop, 100)


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} metric tests passed.")


if __name__ == "__main__":
    _run_all()
