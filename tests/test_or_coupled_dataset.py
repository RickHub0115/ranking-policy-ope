# Slot-level (A2)/(A3)-violation appendix experiment A2': Frechet-mixture
# variant of the exposure DGP.
#
# What is pinned here (mirrors tests/test_a2_dbn_dataset.py):
#   1. or_coupling = 0 reproduces the exposure DGP to MACHINE PRECISION:
#      identical click logs for the same RNG state, identical ground truths,
#      and an identical end-to-end replicate for every estimator EXCEPT AIPS
#      (whose shared bootstrap stream shifts — same disclosed behavior as
#      the DBN variant).
#   2. The closed form E[Y_k] = (1-delta) e r + delta min(e, r) is EXACT
#      (full enumeration over the coupling mask + exact integration of the
#      shared uniform).
#   3. The sampler agrees with the closed form AND preserves the exposure /
#      relevance marginals at every delta.
#   4. E[Y_k] is monotone in delta (min(e, r) >= e r), so the sweep axis
#      moves the truth away from the product decomposition monotonically.
#   5. Hydra wiring: app_a2_slot_coupling composes with dgp=or_coupled and
#      include_oracle=true (the oracle is well-defined — the marginals stay
#      true — and its failure IS the experiment), and `or_coupling` is part
#      of every aggregation key set.
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose
from hydra import initialize_config_dir

from obp.dataset import logistic_reward_function

import main as main_mod
from dataset import FrechetCoupledExposureDataset
from utils import scaled_linear_behavior_policy

from conftest import DGP_SEED
from conftest import make_epsilon_greedy_policy
from conftest import make_exposure_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_or_coupled_dataset(
    or_coupling,
    exposure_structure="ranking_dependent",
    n_unique_action=6,
    len_list=3,
    dim_context=3,
    exposure_decay_rate=1.0,
    attention_spillover=1.0,
    tau0=1.0,
    random_state=DGP_SEED,
    **kwargs,
) -> FrechetCoupledExposureDataset:
    """Mirror of conftest.make_exposure_dataset for the coupled variant (same
    DGP coefficients for the same random_state, so the two are comparable)."""
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
        or_coupling=or_coupling,
    )
    params.update(kwargs)
    return FrechetCoupledExposureDataset(**params)


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
# 1. delta = 0: bitwise match with the exposure DGP
# ---------------------------------------------------------------------------
def test_or_coupled_delta_zero_matches_exposure_dgp_bitwise():
    """Same RNG state -> identical logged data. The coupling mask / shared
    uniforms are drawn after the base O / R draws and substitute nothing at
    delta = 0, so every array of the bandit feedback must match exactly."""
    ds_exp = make_exposure_dataset(
        exposure_structure="ranking_dependent", attention_spillover=1.0
    )
    ds_cpl = make_or_coupled_dataset(0.0)
    for seed in (0, 1):
        ds_exp.random_ = np.random.RandomState(seed)
        ds_cpl.random_ = np.random.RandomState(seed)
        bf_e = ds_exp.obtain_batch_bandit_feedback(n_rounds=300)
        bf_c = ds_cpl.obtain_batch_bandit_feedback(n_rounds=300)
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
            np.testing.assert_array_equal(bf_e[key], bf_c[key], err_msg=key)


def test_or_coupled_delta_zero_ground_truth_matches_exposure_dgp():
    """(E3): identical code path -> exact equality. (E1) pbm: unlike the DBN
    variant, delta = 0 keeps the separable closed form of the parent (the
    coupled slot value degenerates to the exact parent expression), so both
    structures must agree EXACTLY."""
    rng = np.random.RandomState(0)
    for structure, spill in (("ranking_dependent", 1.0), ("pbm", 0.0)):
        ds_exp = make_exposure_dataset(
            exposure_structure=structure, attention_spillover=spill
        )
        ds_cpl = make_or_coupled_dataset(
            0.0, exposure_structure=structure, attention_spillover=spill
        )
        context = rng.normal(size=(40, ds_exp.dim_context))
        policy = make_epsilon_greedy_policy(ds_exp, context, epsilon=0.3)
        common = dict(
            context=context,
            greedy_ranking=policy.greedy_ranking,
            epsilon=0.3,
            method="exact",
        )
        v_exp = ds_exp.calc_ground_truth_policy_value_epsilon_greedy(**common)
        v_cpl = ds_cpl.calc_ground_truth_policy_value_epsilon_greedy(**common)
        assert v_exp == v_cpl, structure


# ---------------------------------------------------------------------------
# 2. closed form is exact (mask enumeration x exact uniform integration)
# ---------------------------------------------------------------------------
def test_or_coupled_expected_slot_rewards_exact_by_enumeration():
    """Per slot: E[Y_k] = sum over the coupling indicator c of
    P(c) * E[O R | c], with E[O R | c=0] = e r (independent draws) and
    E[O R | c=1] = P(U < min(e, r)) = min(e, r) (the shared uniform,
    integrated exactly). Slots couple independently, so the slot-wise
    enumeration is the full enumeration."""
    ds = make_or_coupled_dataset(0.7)
    delta = ds.or_coupling
    rng = np.random.RandomState(3)
    context, action_2d = _random_slates(ds, rng, n=8)
    n = context.shape[0]
    e = ds.calc_expected_exposure(context, action_2d)
    r = ds.base_expected_reward(context)[np.arange(n)[:, None], action_2d]

    brute_y = np.zeros_like(e)
    for c in (0, 1):
        p_c = delta if c else 1.0 - delta
        e_or = np.minimum(e, r) if c else e * r
        brute_y += p_c * e_or

    np.testing.assert_allclose(
        ds.expected_slot_rewards(context, action_2d), brute_y, rtol=1e-12, atol=1e-15
    )
    # the exposure marginal is NOT deflated: P(O=1) = (1-d) e + d P(U<e) = e
    np.testing.assert_array_equal(ds._marginal_exposure_from_er(e, r), e)


def test_or_coupled_sampler_matches_closed_form_and_preserves_marginals():
    """Large-sample slot means of the actual sampler vs the closed form for
    Y, and vs the UNCHANGED marginals for O (the
    coupling preserves the marginal distributions at every delta)."""
    ds = make_or_coupled_dataset(0.6)
    rng = np.random.RandomState(5)
    n_ctx = 20
    reps = 4000
    context, action_2d = _random_slates(ds, rng, n=n_ctx)
    e = ds.calc_expected_exposure(context, action_2d)
    r = ds.base_expected_reward(context)[np.arange(n_ctx)[:, None], action_2d]
    expected_y = ds.expected_slot_rewards(context, action_2d)

    y, o = ds.sample_exposure_and_reward(
        np.repeat(e, reps, axis=0), np.repeat(r, reps, axis=0), np.random.RandomState(7)
    )
    mc_y = y.reshape((n_ctx, reps, ds.len_list)).mean(axis=(0, 1))
    mc_o = o.reshape((n_ctx, reps, ds.len_list)).mean(axis=(0, 1))

    def _assert_z(expected, mc, name):
        se = np.sqrt((expected * (1 - expected)).mean(axis=0) / (n_ctx * reps))
        diff = expected.mean(axis=0) - mc
        # slot 1 exposure is exactly 1.0 under (E3) -> zero sampling variance
        np.testing.assert_array_equal(diff[se == 0], 0.0, err_msg=name)
        z = diff[se > 0] / se[se > 0]
        assert np.abs(z).max() < 4, f"{name} vs sampler: z={z}"

    _assert_z(expected_y, mc_y, "E[Y_k]")
    _assert_z(e, mc_o, "E[O_k] (marginal preserved)")


# ---------------------------------------------------------------------------
# 3. the sweep axis is monotone
# ---------------------------------------------------------------------------
def test_or_coupled_expected_rewards_monotone_in_delta():
    """min(e, r) >= e r for e, r in [0, 1], so E[Y_k] is non-decreasing in
    delta everywhere and strictly increasing in aggregate — the coupling
    moves the truth monotonically away from the e * r decomposition."""
    rng = np.random.RandomState(9)
    deltas = [0.0, 0.3, 0.6, 1.0]
    datasets = [make_or_coupled_dataset(d) for d in deltas]
    context, action_2d = _random_slates(datasets[0], rng, n=50)
    qs = [ds.expected_slot_rewards(context, action_2d) for ds in datasets]
    for q_lo, q_hi in zip(qs[:-1], qs[1:]):
        assert np.all(q_hi >= q_lo - 1e-15)
        assert q_hi.sum() > q_lo.sum()


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


def test_a2_slot_setting_is_or_coupled_with_oracle():
    cfg = compose_cfg(["setting=app_a2_slot_coupling"])
    assert cfg.setting.dgp == "or_coupled"
    # the oracle stays ON: the marginals are true, and the oracle's failure
    # under delta > 0 is the point of the experiment
    assert cfg.setting.include_oracle
    assert float(cfg.setting.or_coupling) == 0.0  # sweep axis default


def test_or_coupling_is_an_aggregation_key_everywhere():
    """Sweep-axis keys missing from an aggregation key set silently blend
    cells from different runs."""
    assert "or_coupling" in main_mod.CONFIG_KEYS
    # both appended AFTER every pre-existing key, in the order they were
    # introduced (or_coupling, then relevance_scale),
    # so the pre-existing aggregate columns keep their relative order
    assert main_mod.CONFIG_KEYS[-2:] == ["or_coupling", "relevance_scale"]
    assert "or_coupling" in main_mod.flatten_setting_defaults()
    cfg = compose_cfg(["setting=app_a2_slot_coupling", "setting.or_coupling=0.6"])
    assert main_mod.flatten_setting(cfg.setting)["or_coupling"] == 0.6


def test_or_coupled_pipeline_runs_with_oracle_rows():
    cfg = compose_cfg(
        TINY_OVERRIDES + ["setting=app_a2_slot_coupling", "setting.or_coupling=0.6"]
    )
    rows = main_mod.run_one_replicate(cfg, seed=0)
    df = pd.DataFrame(rows)
    assert {"le-iips (oracle)", "ed-dr (oracle)"} <= set(df["estimator"])
    assert np.isfinite(df["estimate"].astype(float)).all()
    assert np.isfinite(df["oracle_ratio_error_mean"].astype(float)).all()


@pytest.mark.parametrize("relevance_scale", [1.0, 0.25])
def test_or_coupled_pipeline_delta_zero_equals_exposure_pipeline(relevance_scale):
    """End-to-end delta = 0 consistency:
    every estimator's estimate — INCLUDING the oracle rows — and the ground
    truth of a full replicate must be bitwise identical between
    dgp=or_coupled (delta=0) and dgp=exposure on the same seed.

    Parameterized over relevance_scale: the a2slot
    redesign runs E3 with relevance_scale=0.25, and the delta = 0 guarantee
    must hold under the SAME lambda / beta / relevance_scale as the
    production command.

    AIPS is excluded from the run: its bootstrap sampler consumes ONE shared
    RNG stream across bootstrap draws, and the coupled sampler's mask /
    shared-uniform draws (no-ops at delta=0) shift that stream, so AIPS is
    distribution-identical but not bitwise. Everything else draws from
    per-purpose seeded streams and must match exactly."""
    estimators = "[sips,iips,rips,cascade-dr,dm,dr-iips,le-iips,ed-dr]"
    scale_override = f"setting.relevance_scale={relevance_scale}"
    cfg_cpl = compose_cfg(
        TINY_OVERRIDES
        + [
            "setting=app_a2_slot_coupling",
            "setting.or_coupling=0.0",
            scale_override,
            f"setting.estimators={estimators}",
        ]
    )
    cfg_exp = compose_cfg(
        TINY_OVERRIDES + [scale_override, f"setting.estimators={estimators}"]
    )
    df_cpl = pd.DataFrame(main_mod.run_one_replicate(cfg_cpl, seed=0))
    df_exp = pd.DataFrame(main_mod.run_one_replicate(cfg_exp, seed=0))
    merged = df_cpl.merge(df_exp, on="estimator", suffixes=("_cpl", "_exp"))
    # 8 base estimators + the 2 oracle rows (include_oracle on both sides)
    assert len(merged) == len(df_cpl) == len(df_exp) == 10
    np.testing.assert_array_equal(
        merged["estimate_cpl"].values, merged["estimate_exp"].values
    )
    np.testing.assert_array_equal(
        merged["ground_truth_cpl"].values, merged["ground_truth_exp"].values
    )
    np.testing.assert_array_equal(
        merged["oracle_ratio_error_mean_cpl"].values,
        merged["oracle_ratio_error_mean_exp"].values,
    )
