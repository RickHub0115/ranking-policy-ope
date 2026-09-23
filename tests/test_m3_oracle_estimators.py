# M3: oracle LE-IIPS / ED-DR — the core numerical verification of the theory
#. True e, r, e_bar^pi are injected.
# Acceptance:
# (i)  under (E1)/(E2) the correction ratio is identically 1 and LE-IIPS
#      numerically equals IIPS (np.allclose);
# (ii) under (E3) LE-IIPS / ED-DR are unbiased (Prop. 1 / Cor. 2a) while
#      IIPS has significantly non-zero bias;
# (iii) Cor. 2a: ED-DR stays unbiased with a deliberately broken r_hat (kept
#      inside the (x, a(k)) function class); Cor. 2b: with true (E2) exposure,
#      r_hat true and e_hat mis-valued within the position x context class,
#      the bias is ~ 0.
import numpy as np
import pytest
from obp.ope import SlateIndependentIPS

from conftest import compute_oracle_inputs
from conftest import make_epsilon_greedy_policy
from conftest import make_exposure_dataset
from estimators_slate import SlateExposureDecomposedDR
from estimators_slate import SlateLatentExposureIPS


def _one_replicate_estimates(
    ds,
    seed,
    epsilon=0.3,
    n_rounds=1500,
    corrupt_exposure=None,
    corrupt_relevance=None,
    corrupt_exposure_ratio=None,
    rao_blackwell=False,
):
    """One replication of oracle-injected estimates.

    rao_blackwell=True replaces the sampled clicks Y by their conditional
    expectation q = e * r inside the estimators. This leaves every estimator's
    expectation (and hence its bias) unchanged while removing the click-noise
    variance, so bias tests reach usable power at fast-test scale.
    """
    ds.random_ = np.random.RandomState(seed)
    bf = ds.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    if rao_blackwell:
        bf = dict(bf)
        bf["reward"] = bf["expected_reward_factual"]
    policy = make_epsilon_greedy_policy(ds, bf["context"], epsilon=epsilon)
    _, eval_pscore_item_pos, _ = policy.pscores(bf["action"])
    oracle_inputs = compute_oracle_inputs(
        ds,
        bf,
        policy,
        n_mc_samples=100,
        random_state=seed + 1,
        corrupt_exposure=corrupt_exposure,
        corrupt_relevance=corrupt_relevance,
    )
    if corrupt_exposure_ratio is not None:
        # Figure 3 weight-channel corruption (mirrors src/main.py): mis-value
        # e_bar^pi, which enters only the importance-weight ratio
        oracle_inputs = dict(oracle_inputs)
        oracle_inputs["expected_exposure_eval_hat"] = np.clip(
            corrupt_exposure_ratio(oracle_inputs["expected_exposure_eval_hat"]),
            1e-6,
            1.0,
        )
    common = dict(
        slate_id=bf["slate_id"],
        reward=bf["reward"],
        position=bf["position"],
        pscore_item_position=bf["pscore_item_position"],
        evaluation_policy_pscore_item_position=eval_pscore_item_pos,
    )
    iips = SlateIndependentIPS(len_list=ds.len_list).estimate_policy_value(**common)
    le = SlateLatentExposureIPS(len_list=ds.len_list).estimate_policy_value(
        **common,
        exposure_factual_hat=oracle_inputs["exposure_factual_hat"],
        expected_exposure_eval_hat=oracle_inputs["expected_exposure_eval_hat"],
    )
    eddr = SlateExposureDecomposedDR(len_list=ds.len_list).estimate_policy_value(
        **common, **oracle_inputs
    )
    ground_truth = ds.calc_ground_truth_policy_value_epsilon_greedy(
        context=bf["context"],
        greedy_ranking=policy.greedy_ranking,
        epsilon=epsilon,
        method="exact",
    )
    return dict(iips=iips, le_iips=le, ed_dr=eddr, ground_truth=ground_truth)


# ---------------------------------------------------------------------------
# (i) exact IIPS reduction under (E1)/(E2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("structure", ["pbm", "contextual_pbm"])
def test_m3_i_le_iips_reduces_to_iips_under_e1_e2(structure):
    ds = make_exposure_dataset(exposure_structure=structure)
    res = _one_replicate_estimates(ds, seed=0, n_rounds=400)
    # correction ratio == 1 identically (consistent construction, eq. (2.5a))
    # => LE-IIPS coincides with IIPS to numerical precision
    np.testing.assert_allclose(res["le_iips"], res["iips"], rtol=1e-10)


# ---------------------------------------------------------------------------
# (ii) unbiasedness under (E3) + significant IIPS bias
# ---------------------------------------------------------------------------
def _bias_summary(
    n_reps, n_rounds, corrupt_exposure=None, corrupt_relevance=None,
    corrupt_exposure_ratio=None,
    structure="ranking_dependent", spillover=2.0, rao_blackwell=False,
    epsilon=0.3,
):
    ds = make_exposure_dataset(
        exposure_structure=structure,
        attention_spillover=spillover,
        n_unique_action=6,
        len_list=3,
    )
    errs = {"iips": [], "le_iips": [], "ed_dr": []}
    for rep in range(n_reps):
        res = _one_replicate_estimates(
            ds,
            seed=100 + rep,
            n_rounds=n_rounds,
            epsilon=epsilon,
            corrupt_exposure=corrupt_exposure,
            corrupt_relevance=corrupt_relevance,
            corrupt_exposure_ratio=corrupt_exposure_ratio,
            rao_blackwell=rao_blackwell,
        )
        for k in errs:
            errs[k].append(res[k] - res["ground_truth"])
    return {k: np.asarray(v) for k, v in errs.items()}


def _mean_and_se(errors):
    return errors.mean(), errors.std(ddof=1) / np.sqrt(len(errors))


def test_m3_ii_oracle_unbiased_and_iips_biased_under_e3_fast():
    errs = _bias_summary(n_reps=60, n_rounds=1500, rao_blackwell=True)
    for name in ("le_iips", "ed_dr"):
        mean, se = _mean_and_se(errs[name])
        assert abs(mean) < 4 * se, f"{name}: bias {mean:.4g} (se {se:.4g})"
    mean_iips, se_iips = _mean_and_se(errs["iips"])
    assert abs(mean_iips) > 4 * se_iips, (
        f"IIPS bias not detected: {mean_iips:.4g} (se {se_iips:.4g})"
    )
    # the proposal's bias is much smaller than IIPS's systematic bias
    assert abs(errs["ed_dr"].mean()) < abs(mean_iips)


@pytest.mark.slow
def test_m3_ii_oracle_unbiased_and_iips_biased_under_e3_full():
    """Plan acceptance: (E3), M=500 — LE-IIPS / ED-DR bias ~ 0 (Prop. 1 /
    Cor. 2a) and IIPS bias significantly non-zero."""
    errs = _bias_summary(n_reps=500, n_rounds=2000)
    for name in ("le_iips", "ed_dr"):
        mean, se = _mean_and_se(errs[name])
        assert abs(mean) < 3.5 * se, f"{name}: bias {mean:.4g} (se {se:.4g})"
    mean_iips, se_iips = _mean_and_se(errs["iips"])
    assert abs(mean_iips) > 5 * se_iips


# ---------------------------------------------------------------------------
# (iii) the two corollaries — the heart of the theory verification
# ---------------------------------------------------------------------------
def _corrupt_r(r):
    # pointwise map of r(x, a(k)) — stays inside the (x, a(k)) function class
    return np.clip(0.5 * r + 0.3, 1e-4, 1 - 1e-4)


def _corrupt_e(e):
    # pointwise map of theta_k(x): stays position x context only under (E1)/(E2)
    return np.clip(e, 1e-4, 1.0) ** 1.6


def test_m3_iii_corollary_2a_exposure_side_robustness():
    """e_hat = e true, r_hat deliberately broken (within class) -> unbiased."""
    errs = _bias_summary(
        n_reps=60, n_rounds=1500, corrupt_relevance=_corrupt_r, rao_blackwell=True
    )
    mean, se = _mean_and_se(errs["ed_dr"])
    assert abs(mean) < 4 * se, f"Cor 2a violated: bias {mean:.4g} (se {se:.4g})"


def test_m3_iii_corollary_2b_relevance_side_robustness():
    """True exposure (E2); r_hat = r true; e_hat mis-valued within the
    position x context class -> unbiased (Cor. 2b)."""
    errs = _bias_summary(
        n_reps=60,
        n_rounds=1500,
        corrupt_exposure=_corrupt_e,
        structure="contextual_pbm",
        spillover=0.0,
        rao_blackwell=True,
    )
    mean, se = _mean_and_se(errs["ed_dr"])
    assert abs(mean) < 4 * se, f"Cor 2b violated: bias {mean:.4g} (se {se:.4g})"


# ---------------------------------------------------------------------------
# (iv) Prop. 4: variance comparison against IIPS under (E1)
# ---------------------------------------------------------------------------
def test_m3_iv_prop4_variance_reduction_under_e1():
    """Prop. 4: under (E1) with an oracle q_hat = q (perfectly correlated with
    the reward), Var(ED-DR) <= Var(IIPS). Sampled clicks are kept (NO
    Rao-Blackwellization — the click noise is exactly what the DR residual
    term reduces). Both estimators see the same logs per replication, so the
    variance comparison is paired; seeds are fixed, so the check is
    deterministic. Empirical ratio at this scale is ~ 0.6."""
    errs = _bias_summary(
        n_reps=40, n_rounds=800, structure="pbm", spillover=0.0,
        rao_blackwell=False,
    )
    var_iips = errs["iips"].var(ddof=1)
    var_eddr = errs["ed_dr"].var(ddof=1)
    assert var_eddr < 0.9 * var_iips, (
        f"Prop. 4 violated: Var(ED-DR)={var_eddr:.3e} "
        f"vs Var(IIPS)={var_iips:.3e} (ratio {var_eddr / var_iips:.2f})"
    )
    # under (E1) LE-IIPS == IIPS record-by-record (ratio-1 shortcut), so its
    # variance must coincide — the reduction is attributable to the DR terms
    np.testing.assert_allclose(
        errs["le_iips"].var(ddof=1), var_iips, rtol=1e-8
    )


# ---------------------------------------------------------------------------
# (v) Figure 3 grid pattern (channel-wise corruption; redesign 2026-07-18)
# ---------------------------------------------------------------------------
# The grid axes corrupt ED-DR's two nuisance CHANNELS: the weight channel
# (e_bar^pi -> importance-weight ratio only) and the outcome channel (r_hat).
# The old design corrupted e_hat pointwise, which keeps the weights correct
# (Corollary 2b: an in-class relative error cancels in e_bar^pi_hat / e_hat)
# so ALL cells were unbiased and the both-wrong cell could not show bias.
def _corrupt_power(v):
    # the "power" map of src/main.py make_corruption
    return np.clip(v, 1e-4, 1.0) ** 1.5


def test_m3_v_fig3_weight_channel_alone_is_rescued():
    """Weights wrong (e_bar^pi corrupted) but r_hat = r true -> the residual
    baseline equals the true q, so ED-DR stays unbiased (DR guarantee)."""
    errs = _bias_summary(
        n_reps=60,
        n_rounds=1500,
        corrupt_exposure_ratio=_corrupt_power,
        structure="contextual_pbm",
        spillover=0.0,
        rao_blackwell=True,
    )
    mean, se = _mean_and_se(errs["ed_dr"])
    assert abs(mean) < 4 * se, f"weight-channel cell biased: {mean:.4g} (se {se:.4g})"
    # LE-IIPS has no DM correction, so the same corruption must bias it —
    # this pins that the corruption actually reaches the weights
    mean_le, se_le = _mean_and_se(errs["le_iips"])
    assert abs(mean_le) > 4 * se_le, (
        f"corruption did not reach the weights: le-iips bias {mean_le:.4g} "
        f"(se {se_le:.4g})"
    )


def test_m3_v_fig3_both_channels_wrong_bias_appears():
    """Both channels corrupted -> bias E_pi[(f(e_bar)/e_bar - 1) e (r - r^1.5)]
    has a fixed (negative) sign for the power maps, so it must be significant
    and dominate the one-sided cells."""
    kwargs = dict(
        n_reps=60,
        n_rounds=1500,
        structure="contextual_pbm",
        spillover=0.0,
        rao_blackwell=True,
    )
    errs_both = _bias_summary(
        corrupt_exposure_ratio=_corrupt_power,
        corrupt_relevance=_corrupt_power,
        **kwargs,
    )
    mean_b, se_b = _mean_and_se(errs_both["ed_dr"])
    assert mean_b < 0 and abs(mean_b) > 4 * se_b, (
        f"both-wrong bias not detected: {mean_b:.4g} (se {se_b:.4g})"
    )
    errs_r = _bias_summary(corrupt_relevance=_corrupt_power, **kwargs)
    mean_r, _ = _mean_and_se(errs_r["ed_dr"])
    assert abs(mean_b) > 3 * abs(mean_r), (
        f"both-wrong ({mean_b:.4g}) does not dominate r-only ({mean_r:.4g})"
    )


@pytest.mark.slow
def test_m3_v_fig3_grid_pattern_full():
    """Plan acceptance for Figure 3 at scale: one-sided cells ~ 0, both-wrong
    cell significantly negative."""
    kwargs = dict(
        n_reps=200,
        n_rounds=1500,
        structure="contextual_pbm",
        spillover=0.0,
        rao_blackwell=True,
    )
    errs_w = _bias_summary(corrupt_exposure_ratio=_corrupt_power, **kwargs)
    errs_r = _bias_summary(corrupt_relevance=_corrupt_power, **kwargs)
    errs_both = _bias_summary(
        corrupt_exposure_ratio=_corrupt_power,
        corrupt_relevance=_corrupt_power,
        **kwargs,
    )
    for name, errs in (("weights-only", errs_w), ("r-only", errs_r)):
        mean, se = _mean_and_se(errs["ed_dr"])
        assert abs(mean) < 4 * se, f"{name}: bias {mean:.4g} (se {se:.4g})"
    mean_b, se_b = _mean_and_se(errs_both["ed_dr"])
    assert mean_b < 0 and abs(mean_b) > 5 * se_b
