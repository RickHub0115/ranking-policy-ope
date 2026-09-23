# M2: ExposureSlateBanditDataset.
# Acceptance: (i) under (E1) the position-wise reward means match the parent
# with click_model='pbm' (same Y distribution); (ii) (E3) with beta=0 equals
# (E1); (iii) the overridden ground truth matches on-policy MC to 3 digits.
import numpy as np
import pytest
from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticSlateBanditDataset

from conftest import DGP_SEED
from conftest import make_epsilon_greedy_policy
from conftest import make_exposure_dataset
from dataset import CascadeClickBanditDataset
from utils import cascade_expected_slate_reward
from utils import scaled_linear_behavior_policy


def test_feedback_keys_shapes_and_product_decomposition():
    ds = make_exposure_dataset(exposure_structure="ranking_dependent",
                               attention_spillover=1.0)
    bf = ds.obtain_batch_bandit_feedback(n_rounds=200)
    n_records = 200 * ds.len_list
    for key in [
        "reward",
        "expected_reward_factual",
        "expected_exposure_factual",
        "expected_relevance_factual",
        "exposure",
        "pscore",
        "pscore_item_position",
        "pscore_cascade",
    ]:
        assert key in bf and bf[key].shape == (n_records,), key
    # E[Y|x,a] = e * r stored consistently
    np.testing.assert_allclose(
        bf["expected_reward_factual"],
        bf["expected_exposure_factual"] * bf["expected_relevance_factual"],
    )
    assert set(np.unique(bf["reward"])) <= {0, 1}
    # position is 0-indexed long format (off-by-one pitfall)
    assert bf["position"].min() == 0 and bf["position"].max() == ds.len_list - 1


def test_theta_uses_one_indexed_positions():
    ds = make_exposure_dataset(exposure_decay_rate=1.0, len_list=3)
    theta = ds._theta()
    np.testing.assert_allclose(theta, [1.0, 0.5, 1.0 / 3.0])


def test_m2_i_e1_reward_distribution_equals_pbm_thinning():
    """(i): under (E1), O independent of R implies Y ~ Bern(theta_k * r), i.e.
    the explicit O * R sampling has the same distribution as the parent-style
    expected-value thinning ("click_model='pbm'" semantics).

    NOTE: the plan's original formulation ("position-wise means match the
    parent dataset with click_model='pbm'") is unattainable against obp<=0.5.7
    verbatim, because the parent's `action_interaction_reward_function` builds
    the context row index as `len(action) // n_rounds` (off by len_list) and
    therefore attaches context floor(i/K)'s rewards to round i (see
    test_known_obp_reward_function_context_bug). The distributional claim is
    tested here on the corrected DGP instead.
    """
    n_rounds = 8000
    ds = make_exposure_dataset(exposure_structure="pbm", exposure_decay_rate=1.0)
    bf = ds.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=False
    )
    # (a) explicit O * R sampling (the logged rewards)
    # (b) expected-value thinning: Y ~ Bern(e * r) on the same factuals
    rng = np.random.RandomState(999)
    thinned = rng.binomial(n=1, p=bf["expected_reward_factual"])
    for pos in range(ds.len_list):
        idx = bf["position"] == pos
        p_ours = bf["reward"][idx].mean()
        p_thin = thinned[idx].mean()
        p_true = bf["expected_reward_factual"][idx].mean()
        se = np.sqrt(p_true * (1 - p_true) / n_rounds)
        assert abs(p_ours - p_true) < 4 * se, f"pos {pos}: O*R vs E[Y]"
        assert abs(p_ours - p_thin) < 6 * se, f"pos {pos}: O*R vs thinning"
    # theta ratio appears in the position-wise CTR profile
    ctr = np.array(
        [bf["reward"][bf["position"] == k].mean() for k in range(ds.len_list)]
    )
    assert np.all(np.diff(ctr) < 0), f"CTR must decay over positions: {ctr}"


def test_known_obp_reward_function_context_bug():
    """Regression guard documenting the upstream bug we work around:
    obp<=0.5.7's `action_interaction_reward_function` attaches the rewards of
    context floor(i / len_list) to round i (context index divisor is
    `len(action) // n_rounds` = len_list * the intended value). Our dataset
    classes recompute the relevance from the base reward matrix instead.
    If a future obp release fixes this, this test fails — then remove the
    workarounds in dataset.py / utils.cascade_expected_slate_reward.
    """
    ds = make_exposure_dataset(
        exposure_structure="pbm", n_unique_action=5, len_list=3
    )
    rng = np.random.RandomState(0)
    context = rng.normal(size=(6, ds.dim_context))
    base = ds.base_expected_reward(context)
    action = np.array([[0, 1, 2], [3, 4, 0], [1, 2, 3], [4, 0, 1], [2, 3, 4], [0, 2, 4]])
    rel = ds.reward_function(
        context=context,
        action_context=ds.action_context,
        action=action.flatten(),
        action_interaction_weight_matrix=ds.action_interaction_weight_matrix,
        base_reward_function=ds.base_reward_function,
        reward_type=ds.reward_type,
        reward_structure=ds.reward_structure,
        len_list=ds.len_list,
        is_enumerated=False,
        random_state=ds.random_state,
    ).astype(float)
    correct = base[np.arange(6)[:, None], action]
    buggy = base[np.arange(6)[:, None] // ds.len_list, action]
    assert np.abs(rel - correct).max() > 0.05, (
        "obp's reward_function now matches the correct per-round gather — "
        "the upstream bug appears fixed; remove the workarounds"
    )
    np.testing.assert_allclose(rel, buggy, atol=2e-3)  # float16 quantization
    # and our dataset stores the CORRECT factual relevance
    ds.random_ = np.random.RandomState(1)
    bf = ds.obtain_batch_bandit_feedback(n_rounds=50, return_pscore_item_position=False)
    action_2d = bf["action"].reshape((50, 3))
    expected = ds.base_expected_reward(bf["context"])[
        np.arange(50)[:, None], action_2d
    ].flatten()
    np.testing.assert_allclose(bf["expected_relevance_factual"], expected)


def test_m2_ii_e3_with_beta_zero_equals_e1():
    """(ii): beta = 0 degenerates (E3) to (E1) — exposures identical, and with
    the same RNG stream the sampled rewards are bitwise identical."""
    ds_e1 = make_exposure_dataset(exposure_structure="pbm")
    ds_e3 = make_exposure_dataset(
        exposure_structure="ranking_dependent", attention_spillover=0.0
    )
    bf_e1 = ds_e1.obtain_batch_bandit_feedback(n_rounds=300)
    bf_e3 = ds_e3.obtain_batch_bandit_feedback(n_rounds=300)
    np.testing.assert_allclose(
        bf_e1["expected_exposure_factual"], bf_e3["expected_exposure_factual"]
    )
    np.testing.assert_array_equal(bf_e1["reward"], bf_e3["reward"])
    np.testing.assert_array_equal(bf_e1["exposure"], bf_e3["exposure"])


def _ground_truth_pair(n_mc_samples: int):
    ds = make_exposure_dataset(
        exposure_structure="ranking_dependent",
        n_unique_action=5,
        len_list=3,
        attention_spillover=1.5,
    )
    rng = np.random.RandomState(0)
    context = rng.normal(size=(100, ds.dim_context))
    base_reward = ds.base_expected_reward(context)
    from scipy.special import logit

    eval_logits = logit(np.clip(base_reward, 1e-6, 1 - 1e-6))
    v_exact = ds.calc_ground_truth_policy_value(
        context=context, evaluation_policy_logit_=eval_logits, method="exact"
    )
    v_mc = ds.calc_ground_truth_policy_value(
        context=context,
        evaluation_policy_logit_=eval_logits,
        method="mc",
        n_mc_samples=n_mc_samples,
        random_state=1,
    )
    return v_exact, v_mc


def test_m2_iii_ground_truth_exact_vs_mc_fast():
    v_exact, v_mc = _ground_truth_pair(n_mc_samples=2 * 10**5)
    assert abs(v_exact - v_mc) / v_exact < 5e-3


@pytest.mark.slow
def test_m2_iii_ground_truth_exact_vs_mc_full():
    """Plan acceptance: overridden exact ground truth == on-policy MC (10^6)
    to 3 digits."""
    v_exact, v_mc = _ground_truth_pair(n_mc_samples=10**6)
    assert abs(v_exact - v_mc) / v_exact < 1e-3


def test_epsilon_greedy_ground_truth_matches_brute_force_mc():
    """The mixture-form ground truth ((1-eps) greedy + eps uniform) equals a
    brute-force MC that samples the mixture policy end-to-end."""
    ds = make_exposure_dataset(
        exposure_structure="ranking_dependent",
        n_unique_action=5,
        len_list=3,
        attention_spillover=1.0,
    )
    rng = np.random.RandomState(3)
    context = rng.normal(size=(80, ds.dim_context))
    policy = make_epsilon_greedy_policy(ds, context, epsilon=0.4)
    v = ds.calc_ground_truth_policy_value_epsilon_greedy(
        context=context,
        greedy_ranking=policy.greedy_ranking,
        epsilon=0.4,
        method="exact",
    )
    relevance_all = ds.base_expected_reward(context)
    mc_rng = np.random.RandomState(11)
    total, n_draws = 0.0, 3000
    for _ in range(n_draws):
        ranking = policy.sample_rankings(mc_rng)
        e = ds.calc_expected_exposure(context, ranking)
        r = relevance_all[np.arange(len(context))[:, None], ranking]
        total += (e * r).sum(axis=1).mean()
    v_mc = total / n_draws
    assert abs(v - v_mc) / v < 2e-2


def test_cascade_ground_truth_matches_sampler_mean():
    """Regression guard for the second obp-inherited ground-truth pitfall
    (cf. test_known_obp_reward_function_context_bug for the first):
    obp's `calc_ground_truth_policy_value` (L999-L1013) plugs EXPECTED
    previous-slot rewards into the cascade discount recursion, while the log
    rewards come from `sample_reward_given_expected_reward`, whose discount
    depends on the REALIZED clicks Y_{1:k-1}. Because the discount is a
    nonlinear (product) function of the click prefix, the recursion
    overestimates E[Y_k] for k >= 3. `cascade_expected_slate_reward` computes
    the exact expectation by a DP over the discount distribution; this test
    pins it to large-sample slot means of the actual sampler, and documents
    that the obp-style recursion does NOT match (if obp fixes its ground
    truth, the second assertion fails — then this workaround can be revisited).
    """
    ds = CascadeClickBanditDataset(
        n_unique_action=6,
        len_list=4,
        dim_context=3,
        reward_type="binary",
        reward_structure="independent",
        click_model="cascade",
        eta=1.0,
        base_reward_function=logistic_reward_function,
        behavior_policy_function=scaled_linear_behavior_policy(1.0),
        random_state=DGP_SEED,
    )
    rng = np.random.RandomState(0)
    n_ctx = 25
    context = rng.normal(size=(n_ctx, ds.dim_context))
    action_2d = np.array(
        [rng.permutation(ds.n_unique_action)[: ds.len_list] for _ in range(n_ctx)]
    )
    expected = cascade_expected_slate_reward(ds, context, action_2d)

    # the obp-style expected-value recursion (what the ground truth used to be)
    base_q = ds.base_expected_reward(context)[
        np.arange(n_ctx)[:, None], action_2d
    ].astype(float)
    q = base_q * ds.exam_weight
    disc = np.ones(n_ctx)
    prev = np.zeros(n_ctx)
    recursion = np.empty_like(q)
    for pos_ in range(ds.len_list):
        disc = disc * (prev * ds.attractiveness[pos_] + (1 - prev))
        recursion[:, pos_] = disc * q[:, pos_]
        prev = recursion[:, pos_]

    # large-sample slot means of the actual sampler on the same (x, a) pairs
    reps = 8000
    ds.random_ = np.random.RandomState(7)
    reward = ds.sample_reward_given_expected_reward(
        expected_reward_factual=np.repeat(base_q, reps, axis=0).copy()
    )
    mc = reward.reshape((n_ctx, reps, ds.len_list)).mean(axis=(0, 1))
    se = np.sqrt((expected * (1 - expected)).mean(axis=0) / (n_ctx * reps))

    z_exact = (expected.mean(axis=0) - mc) / se
    z_recursion = (recursion.mean(axis=0) - mc) / se
    assert np.abs(z_exact).max() < 4, f"exact E[Y_k] vs sampler: z={z_exact}"
    # slots 1-2 agree by construction; the recursion bias appears at k >= 3
    np.testing.assert_allclose(recursion[:, :2], expected[:, :2], rtol=1e-12)
    assert np.all(z_recursion[2:] > 6), (
        "the obp-style expected-value recursion now matches the sampler — "
        f"upstream may have changed (z={z_recursion})"
    )


def test_dataset_rejects_wrong_reward_structure():
    with pytest.raises(ValueError):
        make_exposure_dataset(reward_structure="cascade_additive")
    with pytest.raises(ValueError):
        make_exposure_dataset(click_model="pbm")
