import numpy as np

from uncertainty.uq_study import (
    CostModel,
    binary_entropy,
    classwise_conformal_thresholds,
    expected_calibration_error,
    make_synthetic,
    policy_cost,
    prediction_sets,
    risk_coverage,
    run_study,
    split_users,
)


def test_ece_is_zero_for_perfect_bins():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    p = np.where(y == 1, 1.0, 0.0)
    assert expected_calibration_error(y, p) == 0.0


def test_ece_detects_overconfidence():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.95)) > 0.4


def test_entropy_peaks_at_half():
    assert binary_entropy(np.array([0.5]))[0] > binary_entropy(np.array([0.9]))[0]


def test_split_users_has_no_user_overlap():
    X, y, g = make_synthetic(n_users=20, seed=1)
    tr, ca, te = split_users(g, y, seed=0)
    users = [set(g[i]) for i in (tr, ca, te)]
    assert not (users[0] & users[1]) and not (users[0] & users[2]) and not (users[1] & users[2])


def test_conformal_threshold_gives_target_coverage_on_calibration_scores():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 2000)
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.2, 2000), 0.01, 0.99)
    q = classwise_conformal_thresholds(p, y, alpha=0.1)
    sets = prediction_sets(p, q)
    cov = sets[np.arange(len(y)), y].mean()
    assert cov >= 0.9


def test_conformal_is_infinite_when_class_too_rare():
    p = np.array([0.1] * 50 + [0.9] * 3)
    y = np.array([0] * 50 + [1] * 3)
    q = classwise_conformal_thresholds(p, y, alpha=0.1)
    assert np.isinf(q[1])


def test_risk_coverage_ends_at_overall_error():
    u = np.array([0.1, 0.2, 0.3, 0.4])
    err = np.array([0, 0, 1, 1])
    cov, risk = risk_coverage(u, err)
    assert cov[-1] == 1.0 and risk[-1] == 0.5 and risk[0] == 0.0


def test_policy_cost_counts_defer_cost_only():
    costs = CostModel(c_fn=10, c_fp=1, c_defer=0.5)
    y = np.array([1, 0])
    cost = policy_cost(y, act=np.array([False, True]), defer=np.array([True, False]), costs=costs)
    assert abs(cost - (0.5 + 1.0) / 2) < 1e-9


def test_run_study_smoke_and_coverage(tmp_path):
    X, y, g = make_synthetic(n_users=30, seed=3)
    summary, per_repeat = run_study(X, y, g, alpha=0.1, n_repeats=2, n_members=2,
                                    out_dir=str(tmp_path), fast=True)
    assert len(per_repeat) == 2
    assert (tmp_path / "reliability.png").exists()
    assert summary.loc["coverage_all", "mean"] > 0.8
    assert summary.loc["frac_empty", "mean"] < 0.2