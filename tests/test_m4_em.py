# M4: ExposureRelevanceEM.
# Acceptance: under (E1) with Plackett-Luce logging and n=64000 the ratio
# theta_hat_k / theta_hat_1 converges to the true ratio (rel. error < 5%);
# the observed likelihood is non-decreasing; near-deterministic logging fails
# to recover the ratios (negative control — non-identification).
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

from conftest import make_exposure_dataset
from conftest import simulate_log_fast
from exposure_model import ExposureRelevanceEM
from utils import greedy_ranking_from_scores


def _fit_em_on_e1_log(n_rounds, seed=0, deterministic_logging=False, **em_kwargs):
    ds = make_exposure_dataset(
        exposure_structure="pbm",
        exposure_decay_rate=1.0,
        n_unique_action=8,
        len_list=4,
        dim_context=3,
    )
    if deterministic_logging:
        rng = np.random.RandomState(seed)
        context = rng.normal(size=(n_rounds, ds.dim_context))
        logits = ds.behavior_policy_function(
            context=context,
            action_context=ds.action_context,
            random_state=ds.random_state,
        )
        ranking = greedy_ranking_from_scores(logits, ds.len_list)  # one slate per x
        relevance = ds.base_expected_reward(context)[
            np.arange(n_rounds)[:, None], ranking
        ]
        exposure_p = ds.calc_expected_exposure(context, ranking)
        reward, _ = ds.sample_exposure_and_reward(exposure_p, relevance, rng)
        log = dict(
            context=context,
            action=ranking.flatten(),
            reward=reward.flatten(),
            position=np.tile(np.arange(ds.len_list), n_rounds),
        )
    else:
        log = simulate_log_fast(ds, n_rounds, seed)
    params = dict(
        len_list=ds.len_list,
        n_unique_action=ds.n_unique_action,
        relevance_model=LogisticRegression(max_iter=1000, C=100.0),
        exposure_model_class="pbm",
        n_em_iter=30,
        warm_start=True,
        monotone_position_tower=True,
    )
    params.update(em_kwargs)
    em = ExposureRelevanceEM(**params)
    em.fit(
        context=log["context"],
        action=log["action"],
        reward=log["reward"],
        position=log["position"],
    )
    true_theta = ds._theta()
    ratio_err = np.abs(
        em.theta_ / em.theta_[0] - true_theta / true_theta[0]
    ) / (true_theta / true_theta[0])
    return em, ratio_err


def test_m4_theta_ratio_recovery_fast():
    em, ratio_err = _fit_em_on_e1_log(n_rounds=16000, seed=0)
    assert ratio_err.max() < 0.15, f"theta ratio rel. error {ratio_err}"


@pytest.mark.slow
def test_m4_theta_ratio_recovery_full():
    """Plan acceptance: (E1), PL logging, n=64000 — theta_hat_k / theta_hat_1
    within 5% relative error of the true ratio."""
    em, ratio_err = _fit_em_on_e1_log(n_rounds=64000, seed=0)
    assert ratio_err.max() < 0.05, f"theta ratio rel. error {ratio_err}"


def test_m4_likelihood_monotone_nondecreasing():
    em, _ = _fit_em_on_e1_log(n_rounds=4000, seed=1)
    hist = np.asarray(em.likelihood_history_)
    assert len(hist) >= 2
    assert np.all(np.diff(hist) >= -1e-8), f"likelihood not monotone: {hist}"


def test_m4_deterministic_logging_fails_negative_control():
    """With a deterministic logging policy, e and r are fully confounded: the ratio recovery must be clearly worse than under PL
    logging on the same budget."""
    _, err_pl = _fit_em_on_e1_log(n_rounds=12000, seed=2)
    _, err_det = _fit_em_on_e1_log(n_rounds=12000, seed=2, deterministic_logging=True)
    assert err_det.max() > 2 * err_pl.max(), (
        f"deterministic {err_det.max():.3f} vs PL {err_pl.max():.3f}"
    )


def test_m4_relevance_model_must_be_classifier():
    """Reversed validation vs SlateRegressionModel (the copy-paste hazard in
    the plan's pitfalls memo): a regressor must be rejected."""
    with pytest.raises(ValueError):
        ExposureRelevanceEM(
            len_list=3, n_unique_action=5, relevance_model=Ridge()
        )


def test_m4_monotone_position_tower():
    em, _ = _fit_em_on_e1_log(n_rounds=4000, seed=3)
    assert np.all(np.diff(em.theta_) <= 1e-12), f"theta not non-increasing: {em.theta_}"


def test_m4_warm_start_initialization_reasonable():
    ds = make_exposure_dataset(
        exposure_structure="pbm", n_unique_action=8, len_list=4
    )
    log = simulate_log_fast(ds, 8000, seed=4)
    em = ExposureRelevanceEM(
        len_list=4,
        n_unique_action=8,
        relevance_model=LogisticRegression(max_iter=1000),
        exposure_model_class="pbm",
    )
    theta_init = em._warm_start_by_intervention_harvesting(
        action=log["action"], reward=log["reward"], position=log["position"]
    )
    assert theta_init.shape == (4,)
    assert np.all(theta_init > 0) and np.all(theta_init <= 1.0)
    assert np.all(np.diff(theta_init) <= 1e-12)  # monotone non-increasing
    # the harvested ratios should be in the right ballpark of (1/k)
    true_ratio = ds._theta()
    assert np.abs(theta_init - true_ratio).max() < 0.3
