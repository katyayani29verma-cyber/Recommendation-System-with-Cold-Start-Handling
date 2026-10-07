from src.evaluation.metrics import average_precision_at_k, ndcg_at_k


def test_ndcg_perfect_order_is_one():
    relevance = {1: 3.0, 2: 2.0, 3: 1.0}
    assert ndcg_at_k([1, 2, 3], relevance, k=3) == 1.0


def test_ndcg_worst_order_is_low():
    relevance = {1: 3.0, 2: 2.0, 3: 1.0}
    best = ndcg_at_k([1, 2, 3], relevance, k=3)
    worst = ndcg_at_k([3, 2, 1], relevance, k=3)
    assert worst < best


def test_ndcg_empty_relevance_is_zero():
    assert ndcg_at_k([1, 2, 3], {}, k=3) == 0.0


def test_average_precision_perfect_hit_first():
    assert average_precision_at_k([1, 2, 3], {1}, k=3) == 1.0


def test_average_precision_no_relevant_is_zero():
    assert average_precision_at_k([1, 2, 3], set(), k=3) == 0.0


def test_average_precision_partial_hits():
    # relevant {1, 3}; hit at rank 1 (prec 1.0) and rank 3 (prec 2/3)
    ap = average_precision_at_k([1, 2, 3], {1, 3}, k=3)
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-9
