# (A2)-violation appendix experiment: DBN-satisfaction variant of the
# exposure DGP.
#
# What is pinned here:
#   1. or_correlation = 0 reproduces the exposure DGP to MACHINE PRECISION
#      (same convention as test_aips_equivalence.py): identical click logs
#      for the same RNG state, identical ground truths, and an identical
#      end-to-end pipeline replicate.
#   2. The closed-form ground truth E[Y_k] = e_k r_k prod_{j<k}(1 - rho e_j r_j)
#      is EXACT — verified against a full enumeration of all (B, R, S) noise
#      outcomes, so the obp-style "plug expected values into the recursion"
#      overestimation cannot sneak in.
#   3. The sampler and the closed form agree (large-sample z-test, the
#      test_cascade_ground_truth_matches_sampler_mean convention).
#   4. E[Y_k] degrades monotonically in rho (the sweep axis is doing what the
#      appendix figure claims).
#   5. The Hydra wiring: app_a2_robustness composes, runs end to end without
#      oracle rows, and `or_correlation` is part of every aggregation key set
#      (CONFIG_KEYS / flatten_setting) so
#      sweep cells cannot silently blend.
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose
from hydra import initialize_config_dir

from obp.dataset import logistic_reward_function

import main as main_mod
from dataset import DBNClickBanditDataset
from utils import scaled_linear_behavior_policy

from conftest import DGP_SEED
from conftest import make_epsilon_greedy_policy
from conftest import make_exposure_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_dbn_dataset(
    or_correlation,
    exposure_structure="ranking_dependent",
    n_unique_action=6,
    len_list=3,
    dim_context=3,
    exposure_decay_rate=1.0,
    attention_spillover=1.0,
    tau0=1.0,
    random_state=DGP_SEED,
    **kwargs,
) -> DBNClickBanditDataset:
    """Mirror of conftest.make_exposure_dataset for the DBN variant (same DGP
    coefficients for the same random_state, so the two are comparable)."""
    params = dict(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type="binary",
        reward_structure="independent",
        click_model=None,
        base_reward_function=logistic_reward_function,
        behavior_policy_function=scaled_linear_behavior_policy(tau0),
        random_state=random_state,
        exposure_structure=exposure_structure,
        exposure_decay_rate=exposure_decay_rate,
        attention_spillover=attention_spillover,
        return_exposure=True,
        or_correlation=or_correlation,
    )
    params.update(kwargs)
    return DBNClickBanditDataset(**params)


def _random_slates(dataset, rng, n):
    context = rng.normal(size=(n, dataset.dim_context))
    action_2d = np.array(
        [
            rng.permutation(dataset.n_unique_action)[: dataset.len_list]
            for _ in range(n)
        ]
    )
    return context, action_2d


# ---------------------------------------------------------------------------
# 1. rho = 0: bitwise match with the exposure DGP
# ---------------------------------------------------------------------------
def test_dbn_rho_zero_matches_exposure_dgp_bitwise():
    """Same RNG state -> identical logged data. The satisfaction draws come
    after the O / R draws in the stream and multiply in as exact zeros, so
    every array of the bandit feedback must match exactly."""
    ds_exp = make_exposure_dataset(
        exposure_structure="ranking_dependent", attention_spillover=1.0
    )
    ds_dbn = make_dbn_dataset(0.0)
    for seed in (0, 1):
        ds_exp.random_ = np.random.RandomState(seed)
        ds_dbn.random_ = np.random.RandomState(seed)
        bf_e = ds_exp.obtain_batch_bandit_feedback(n_rounds=300)
        bf_d = ds_dbn.obtain_batch_bandit_feedback(n_rounds=300)
        for key in (
            "context",
            "action",
            "position",
            "pscore",
            "pscore_item_position",
            "pscore_cascade",
            "reward",
            "exposure",
            "expected_reward_factual",
            "expected_exposure_factual",
            "expected_relevance_factual",
        ):
            np.testing.assert_array_equal(bf_e[key], bf_d[key], err_msg=key)


def test_dbn_rho_zero_ground_truth_matches_exposure_dgp():
    """(E3): identical code path -> exact equality. (E1) pbm: the exposure DGP
    takes the separable closed form for the uniform part while the DBN variant
    must not (`_uniform_value_is_separable` is False even at rho = 0 by class,
    exercising the enumeration path) -> agreement to float roundoff."""
    rng = np.random.RandomState(0)
    # (E3) ranking_dependent — both sides enumerate
    ds_exp = make_exposure_dataset(
        exposure_structure="ranking_dependent", attention_spillover=1.0
    )
    ds_dbn = make_dbn_dataset(0.0)
    context = rng.normal(size=(40, ds_exp.dim_context))
    policy = make_epsilon_greedy_policy(ds_exp, context, epsilon=0.3)
    common = dict(
        context=context,
        greedy_ranking=policy.greedy_ranking,
        epsilon=0.3,
        method="exact",
    )
    v_exp = ds_exp.calc_ground_truth_policy_value_epsilon_greedy(**common)
    v_dbn = ds_dbn.calc_ground_truth_policy_value_epsilon_greedy(**common)
    assert v_exp == v_dbn
    # (E1) pbm — separable closed form vs the DBN enumeration path
    ds_exp_pbm = make_exposure_dataset(exposure_structure="pbm")
    ds_dbn_pbm = make_dbn_dataset(0.0, exposure_structure="pbm", attention_spillover=0.0)
    v_exp_pbm = ds_exp_pbm.calc_ground_truth_policy_value_epsilon_greedy(**common)
    v_dbn_pbm = ds_dbn_pbm.calc_ground_truth_policy_value_epsilon_greedy(**common)
    np.testing.assert_allclose(v_dbn_pbm, v_exp_pbm, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. closed-form ground truth is exact (full noise-outcome enumeration)
# ---------------------------------------------------------------------------
def test_dbn_expected_slot_rewards_exact_by_enumeration():
    """Enumerate all 2^(3K) outcomes of the (B, R, S) noise triple and average
    the sequential DBN click process exactly. The closed form must match to
    machine precision — this is the guard against replicating obp's
    expected-value-in-the-recursion ground-truth bug."""
    ds = make_dbn_dataset(0.7)
    rho = ds.or_correlation
    K = ds.len_list
    rng = np.random.RandomState(3)
    context, action_2d = _random_slates(ds, rng, n=8)
    n = context.shape[0]
    e = ds.calc_expected_exposure(context, action_2d)
    r = ds.base_expected_reward(context)[np.arange(n)[:, None], action_2d]

    brute_y = np.zeros((n, K))
    brute_o = np.zeros((n, K))
    for b, rr, s in product(product((0, 1), repeat=K), repeat=3):
        prob = np.ones(n)
        for k in range(K):
            prob = prob * (e[:, k] if b[k] else 1.0 - e[:, k])
            prob = prob * (r[:, k] if rr[k] else 1.0 - r[:, k])
            prob = prob * (rho if s[k] else 1.0 - rho)
        active = 1
        y_pat = np.zeros(K)
        o_pat = np.zeros(K)
        for k in range(K):
            o_pat[k] = active * b[k]
            y_pat[k] = o_pat[k] * rr[k]
            if y_pat[k] and s[k]:
                active = 0
        brute_y += prob[:, None] * y_pat[None, :]
        brute_o += prob[:, None] * o_pat[None, :]

    np.testing.assert_allclose(
        ds.expected_slot_rewards(context, action_2d), brute_y, rtol=1e-10, atol=1e-15
    )
    np.testing.assert_allclose(
        ds._marginal_exposure_from_er(e, r), brute_o, rtol=1e-10, atol=1e-15
    )


def test_dbn_sampler_mean_matches_closed_form():
    """Large-sample slot means of the actual sampler vs the closed form
    (the test_cascade_ground_truth_matches_sampler_mean convention)."""
    ds = make_dbn_dataset(0.6)
    rng = np.random.RandomState(5)
    n_ctx = 20
    reps = 4000
    context, action_2d = _random_slates(ds, rng, n=n_ctx)
    e = ds.calc_expected_exposure(context, action_2d)
    r = ds.base_expected_reward(context)[np.arange(n_ctx)[:, None], action_2d]
    expected_y = ds.expected_slot_rewards(context, action_2d)
    expected_o = ds._marginal_exposure_from_er(e, r)

    y, o = ds.sample_exposure_and_reward(
        np.repeat(e, reps, axis=0), np.repeat(r, reps, axis=0), np.random.RandomState(7)
    )
    mc_y = y.reshape((n_ctx, reps, ds.len_list)).mean(axis=(0, 1))
    mc_o = o.reshape((n_ctx, reps, ds.len_list)).mean(axis=(0, 1))
    def _assert_z(expected, mc, name):
        se = np.sqrt((expected * (1 - expected)).mean(axis=0) / (n_ctx * reps))
        diff = expected.mean(axis=0) - mc
        # slot 1 exposure is exactly 1.0 under (E3) (theta_1 = 1, no items
        # above) -> zero sampling variance, so require exact agreement there
        np.testing.assert_array_equal(diff[se == 0], 0.0, err_msg=name)
        z = diff[se > 0] / se[se > 0]
        assert np.abs(z).max() < 4, f"{name} vs sampler: z={z}"

    _assert_z(expected_y, mc_y, "E[Y_k]")
    _assert_z(expected_o, mc_o, "E[O_k]")


def test_dbn_epsilon_greedy_ground_truth_matches_policy_mc():
    """The epsilon-greedy ground-truth mixture (exact enumeration of the
    uniform part) vs a direct Monte Carlo over sampled evaluation-policy
    rankings, with the click process integrated in closed form per ranking
    (the closed form itself is pinned by the enumeration test above)."""
    ds = make_dbn_dataset(0.7)
    rng = np.random.RandomState(0)
    context = rng.normal(size=(60, ds.dim_context))
    policy = make_epsilon_greedy_policy(ds, context, epsilon=0.4)
    v_exact = ds.calc_ground_truth_policy_value_epsilon_greedy(
        context=context,
        greedy_ranking=policy.greedy_ranking,
        epsilon=0.4,
        method="exact",
    )
    rng_mc = np.random.RandomState(11)
    draws = 600
    vals = np.empty(draws)
    for i in range(draws):
        ranking = policy.sample_rankings(rng_mc)
        q = ds.expected_slot_rewards(context, ranking)
        vals[i] = (ds.position_weight[None, :] * q).sum(axis=1).mean()
    se = vals.std(ddof=1) / np.sqrt(draws)
    assert abs(vals.mean() - v_exact) < 4 * se, (vals.mean(), v_exact, se)


# ---------------------------------------------------------------------------
# 3. the sweep axis is monotone
# ---------------------------------------------------------------------------
def test_dbn_expected_rewards_monotone_in_rho():
    """Slot 1 is untouched by construction; every later slot's E[Y_k] is
    non-increasing in rho (strictly decreasing in aggregate), so a stronger
    O-R correlation moves the truth monotonically away from the e * r
    decomposition that LE-IIPS / ED-DR assume."""
    rng = np.random.RandomState(9)
    rhos = [0.0, 0.3, 0.6, 1.0]
    datasets = [make_dbn_dataset(rho) for rho in rhos]
    context, action_2d = _random_slates(datasets[0], rng, n=50)
    qs = [ds.expected_slot_rewards(context, action_2d) for ds in datasets]
    for q_lo, q_hi in zip(qs[:-1], qs[1:]):
        np.testing.assert_array_equal(q_lo[:, 0], q_hi[:, 0])
        assert np.all(q_hi[:, 1:] <= q_lo[:, 1:] + 1e-15)
        assert q_hi[:, 1:].sum() < q_lo[:, 1:].sum()


# ---------------------------------------------------------------------------
# 4. Hydra wiring / aggregation keys / end-to-end pipeline
# ---------------------------------------------------------------------------
TINY_OVERRIDES = [
    "setting.n_rounds=400",
    "setting.n_unique_action=6",
    "setting.len_list=3",
    "setting.dim_context=3",
    "setting.n_mc_samples=20",
    "setting.n_em_iter=5",
    "setting.n_folds=2",
    "setting.cascade_dr_base_model=ridge",
    "setting.ground_truth.n_mc_samples=20000",
    "n_seeds=2",
    "n_jobs=1",
]


def compose_cfg(overrides):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base="1.3"
    ):
        return compose(config_name="config", overrides=overrides)


def test_a2_robustness_setting_is_dbn():
    cfg = compose_cfg(["setting=app_a2_robustness"])
    assert cfg.setting.dgp == "dbn"
    assert not cfg.setting.include_oracle
    assert float(cfg.setting.or_correlation) == 0.0  # sweep axis, rho=0 default


def test_or_correlation_is_an_aggregation_key_everywhere():
    """Sweep-axis keys missing from an aggregation key set silently blend
    cells from different runs."""
    assert "or_correlation" in main_mod.CONFIG_KEYS
    assert "or_correlation" in main_mod.flatten_setting_defaults()
    cfg = compose_cfg(["setting=app_a2_robustness", "setting.or_correlation=0.6"])
    assert main_mod.flatten_setting(cfg.setting)["or_correlation"] == 0.6


def test_dbn_pipeline_runs_without_oracle_rows():
    cfg = compose_cfg(
        TINY_OVERRIDES + ["setting=app_a2_robustness", "setting.or_correlation=0.6"]
    )
    rows = main_mod.run_one_replicate(cfg, seed=0)
    df = pd.DataFrame(rows)
    assert "le-iips (oracle)" not in set(df["estimator"])
    assert "ed-dr (oracle)" not in set(df["estimator"])
    assert np.isfinite(df["estimate"].astype(float)).all()


def test_dbn_pipeline_rho_zero_equals_exposure_pipeline():
    """End-to-end rho = 0 consistency: every estimator's estimate and the
    ground truth of a full replicate must be bitwise identical between
    dgp=dbn (rho=0) and dgp=exposure on the same seed.

    AIPS is excluded from the run: its bootstrap sampler consumes ONE shared
    RNG stream across bootstrap draws, and the DBN sampler's satisfaction
    binomials (all zero at rho=0) shift that stream, so AIPS is
    distribution-identical but not bitwise. Everything else draws from
    per-purpose seeded streams and must match exactly."""
    estimators = "[sips,iips,rips,cascade-dr,dm,dr-iips,le-iips,ed-dr]"
    cfg_dbn = compose_cfg(
        TINY_OVERRIDES
        + [
            "setting=app_a2_robustness",
            "setting.or_correlation=0.0",
            f"setting.estimators={estimators}",
        ]
    )
    cfg_exp = compose_cfg(
        TINY_OVERRIDES
        + ["setting.include_oracle=false", f"setting.estimators={estimators}"]
    )
    df_dbn = pd.DataFrame(main_mod.run_one_replicate(cfg_dbn, seed=0))
    df_exp = pd.DataFrame(main_mod.run_one_replicate(cfg_exp, seed=0))
    merged = df_dbn.merge(df_exp, on="estimator", suffixes=("_dbn", "_exp"))
    assert len(merged) == len(df_dbn) == len(df_exp) == 8
    np.testing.assert_array_equal(
        merged["estimate_dbn"].values, merged["estimate_exp"].values
    )
    np.testing.assert_array_equal(
        merged["ground_truth_dbn"].values, merged["ground_truth_exp"].values
    )
