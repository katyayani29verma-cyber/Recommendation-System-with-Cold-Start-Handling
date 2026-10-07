import pandas as pd

from src.evaluation.run_comparison import summarize, target_summary


def make_results() -> pd.DataFrame:
    rows = []
    for seed in (0, 1):
        for mode in ("fixed", "adaptive"):
            for user_id, niche in ((1, True), (2, False)):
                for step in (0, 1):
                    rows.append(
                        {
                            "seed": seed, "mode": mode, "user_id": user_id, "niche": niche,
                            "step": step, "n_skips": 0, "ndcg@5": 0.1 * (seed + 1),
                            "map@5": 0.0, "reached_target": step == 1, "questions_to_target": 1 if step == 1 else None,
                        }
                    )
    return pd.DataFrame(rows)


def test_summarize_reports_all_and_niche_subsets():
    summary = summarize(make_results(), [5])
    assert set(summary["subset"]) == {"all", "niche"}
    niche = summary[(summary["subset"] == "niche") & (summary["mode"] == "fixed") & (summary["step"] == 1)]
    assert niche["n_users"].iloc[0] == 1
    assert abs(niche["ndcg@5_mean"].iloc[0] - 0.15) < 1e-9
    assert niche["ndcg@5_std"].iloc[0] > 0


def test_target_summary_counts_reached_users():
    targets = target_summary(make_results())
    assert (targets["reached_rate_mean"] == 1.0).all()
    assert (targets["median_questions_mean"] == 1.0).all()
