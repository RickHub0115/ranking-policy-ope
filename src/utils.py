# Contains code derived from Open Bandit Pipeline (zr-obp) v0.5.5,
# https://github.com/st-tech/zr-obp, obp/dataset/synthetic_slate.py:
# `EpsilonGreedyEvaluationPolicy.pscores` follows
# `generate_evaluation_policy_pscore` / `_calc_epsilon_greedy_pscore`
# (modified).
# Copyright 2020 Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc.
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). See the NOTICE file at the
# repository root.
#
# Utility functions for the ED-DR / LE-IIPS experiments.
#
# This module hosts everything that is shared across dataset / nuisance /
# estimator / experiment code but does not belong to any of them:
#   - position weights (alpha_k)
#   - Plackett-Luce (Gumbel argsort) ranking sampler
#   - epsilon-greedy evaluation policy (pscores / sampler / action_dist / truth)
#   - temperature-scaled linear behavior policy wrapper
#   - input validation for the new estimators (reusing obp.utils)
#
# NOTE on OBP's `evaluation_policy_type="optimal"`:
# `SyntheticSlateBanditDataset.generate_evaluation_policy_pscore` in OBP uses
# `base_expected_reward.argsort(axis=1)[:, :len_list]` for
# `evaluation_policy_type="optimal"`, i.e., an ASCENDING argsort, which selects
# the len_list actions with the SMALLEST expected reward ("anti-optimal" takes
# the largest). We therefore implement the epsilon-greedy evaluation policy
# ourselves with a descending sort so that "optimal" means the largest
# expected rewards. See `greedy_ranking_from_scores`.
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.special import perm as n_permutations
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from obp.dataset.synthetic_slate import linear_behavior_policy_logit
from obp.utils import check_array
from obp.utils import _check_slate_ope_inputs


EPS = 1e-10


# --------------------------------------------------------------------------
# position weights alpha_k
# --------------------------------------------------------------------------
def make_position_weight(len_list: int, kind: str = "uniform") -> np.ndarray:
    """Return the (known, deterministic) position weights alpha_k, shape (len_list,).

    kind="uniform": alpha_k = 1 (OBP default; alpha does not appear in OBP).
    kind="dcg":     alpha_k = 1 / log2(k + 1), k = 1, ..., len_list.
    """
    check_scalar(len_list, "len_list", int, min_val=1)
    if kind == "uniform":
        return np.ones(len_list)
    elif kind == "dcg":
        return 1.0 / np.log2(np.arange(1, len_list + 1) + 1)
    else:
        raise ValueError(f"`kind` must be 'uniform' or 'dcg', but {kind} is given")


def position_weight_for(
    position: np.ndarray, position_weight: Optional[np.ndarray]
) -> np.ndarray:
    """Map the slot-level weight vector alpha (len_list,) to per-record weights.

    `position` follows the OBP long format and is 0-indexed
    (position = k - 1 for the k-th slot; watch out for the off-by-one).
    """
    if position_weight is None:
        return np.ones_like(position, dtype=float)
    position_weight = np.asarray(position_weight, dtype=float)
    return position_weight[position]


def apply_position_weight_to_reward(
    reward: np.ndarray, position: np.ndarray, position_weight: Optional[np.ndarray]
) -> np.ndarray:
    """Multiply slot-level rewards by alpha_k.

    For pure IPS-type estimators (SIPS/IIPS/RIPS) and for Cascade-DR
    (whose Q_l is defined as E[sum_{l'>=l} alpha_{l'} r(l')]), feeding
    alpha-weighted rewards is exactly equivalent to a position-weighted value.
    This lets us run the unmodified OBP baselines under alpha_k != 1.
    """
    return reward * position_weight_for(position, position_weight)


# --------------------------------------------------------------------------
# Plackett-Luce sampling (Gumbel argsort)
# --------------------------------------------------------------------------
def sample_ranking_from_logits(
    policy_logit_: np.ndarray,
    len_list: int,
    random_: np.random.RandomState,
    idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample rankings from a Plackett-Luce policy defined by logits.

    Sequential softmax sampling without replacement (what OBP's
    `sample_action_and_obtain_pscore` does row by row) is distributionally
    equivalent to perturbing the logits with Gumbel noise and taking the
    top-`len_list` argsort, which is what we do here (vectorized).

    Parameters
    ----------
    policy_logit_: (n_rounds, n_unique_action)
    idx: optional subset of rounds to sample for.

    Returns
    -------
    ranking: (n_rounds_or_len(idx), len_list) int array
    """
    check_array(policy_logit_, name="policy_logit_", expected_dim=2)
    logits = policy_logit_ if idx is None else policy_logit_[idx]
    gumbel_noise = random_.gumbel(size=logits.shape)
    return np.argsort(-(logits + gumbel_noise), axis=1)[:, :len_list].astype(int)


# --------------------------------------------------------------------------
# epsilon-greedy evaluation policy
# --------------------------------------------------------------------------
def greedy_ranking_from_scores(
    scores: np.ndarray, len_list: int, optimal: bool = True
) -> np.ndarray:
    """Deterministic ranking sorted by scores.

    optimal=True sorts DESCENDING (largest expected reward first). This
    deliberately differs from OBP's `generate_evaluation_policy_pscore`
    ("optimal" there is an ascending argsort; see module docstring).
    """
    check_array(scores, name="scores", expected_dim=2)
    if optimal:
        return np.argsort(-scores, axis=1)[:, :len_list].astype(int)
    return np.argsort(scores, axis=1)[:, :len_list].astype(int)


@dataclass
class EpsilonGreedyEvaluationPolicy:
    """Slate-level epsilon-greedy mixture policy.

    With probability (1 - epsilon) the whole greedy slate `greedy_ranking[i]`
    is shown; with probability epsilon a uniformly random permutation of
    `len_list` distinct actions is shown. This matches the mixture whose
    pscores OBP computes in `_calc_epsilon_greedy_pscore`.
    """

    greedy_ranking: np.ndarray  # (n_rounds, len_list)
    epsilon: float
    n_unique_action: int
    len_list: int = field(init=False)

    def __post_init__(self) -> None:
        check_array(self.greedy_ranking, name="greedy_ranking", expected_dim=2)
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0, max_val=1.0)
        self.len_list = self.greedy_ranking.shape[1]

    # -- propensities -----------------------------------------------------
    def pscores(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Three pscore variants of the logged actions under this policy.

        Returns (pscore, pscore_item_position, pscore_cascade), each of
        shape (n_rounds * len_list,), the same convention as
        `SyntheticSlateBanditDataset.generate_evaluation_policy_pscore`.
        """
        n_rounds = self.greedy_ranking.shape[0]
        m, K = self.n_unique_action, self.len_list
        action_2d = action.reshape((n_rounds, K))
        random_pscore = np.full(
            n_rounds * K, 1.0 / n_permutations(m, K, exact=True)
        )
        random_pscore_item_position = np.full(n_rounds * K, 1.0 / m)
        random_pscore_cascade = (
            1.0
            / np.tile(np.arange(m, m - K, -1), (n_rounds, 1)).cumprod(axis=1)
        ).flatten()
        action_match_flg = self.greedy_ranking == action_2d
        pscore_flg = np.repeat(action_match_flg.all(axis=1), K)
        pscore_item_position_flg = action_match_flg.flatten()
        pscore_cascade_flg = action_match_flg.cumprod(axis=1).flatten()
        pscore = pscore_flg * (1 - self.epsilon) + self.epsilon * random_pscore
        pscore_item_position = (
            pscore_item_position_flg * (1 - self.epsilon)
            + self.epsilon * random_pscore_item_position
        )
        pscore_cascade = (
            pscore_cascade_flg * (1 - self.epsilon)
            + self.epsilon * random_pscore_cascade
        )
        return pscore, pscore_item_position, pscore_cascade

    # -- sampler -----------------------------------------------------------
    def sample_rankings(
        self, random_: np.random.RandomState, idx: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Sample one ranking per round from the mixture. Returns (n, len_list)."""
        greedy = self.greedy_ranking if idx is None else self.greedy_ranking[idx]
        n = greedy.shape[0]
        sampled = greedy.copy()
        explore = random_.uniform(size=n) < self.epsilon
        n_explore = int(explore.sum())
        if n_explore > 0:
            # vectorized uniform K-permutations via random-key argsort
            keys = random_.uniform(size=(n_explore, self.n_unique_action))
            sampled[explore] = np.argsort(keys, axis=1)[:, : self.len_list]
        return sampled.astype(int)

    # -- conditional action distribution (for Cascade-DR / SlateRegressionModel)
    def action_dist(self, action: np.ndarray) -> np.ndarray:
        """Exact Plackett-Luce style conditional action distribution.

        Computes pi_e(a(l) = b | x_i, a_i(1), ..., a_i(l-1)) for all b, where
        the prefix is that of the LOGGED action (OBP's convention for
        `evaluation_policy_action_dist`). For the mixture policy,

          P(prefix a(1:l)) = (1-eps) 1{prefix == greedy prefix} + eps / P(m, l)

        so the conditional is a ratio of two such terms; actions already used
        in the prefix have probability zero.

        Returns flat array of shape (n_rounds * len_list * n_unique_action,).
        """
        m, K = self.n_unique_action, self.len_list
        n_rounds = self.greedy_ranking.shape[0]
        action_2d = action.reshape((n_rounds, K))
        # prefix_match[i, l] = 1{a_i(1:l) == greedy_i(1:l)}; l = 0 means empty prefix
        match = (action_2d == self.greedy_ranking).cumprod(axis=1)  # (n, K)
        # number of ordered l-prefixes: P(m, l)
        n_prefix_permutations = np.array(
            [n_permutations(m, l, exact=True) for l in range(K + 1)], dtype=float
        )
        action_dist = np.zeros((n_rounds, K, m))
        used = np.zeros((n_rounds, m), dtype=bool)
        for pos_ in range(K):
            prefix_match = match[:, pos_ - 1] if pos_ > 0 else np.ones(n_rounds)
            # denominator: P(prefix of length pos_)
            denom = (1 - self.epsilon) * prefix_match + self.epsilon / n_prefix_permutations[pos_]
            # numerator for arbitrary unused b: eps / P(m, pos_ + 1); the greedy
            # continuation additionally receives the (1 - eps) mass when the
            # prefix so far matches greedy.
            numer = np.full((n_rounds, m), self.epsilon / n_prefix_permutations[pos_ + 1])
            numer[np.arange(n_rounds), self.greedy_ranking[:, pos_]] += (
                1 - self.epsilon
            ) * prefix_match
            numer[used] = 0.0
            action_dist[:, pos_, :] = numer / denom[:, None]
            used[np.arange(n_rounds), action_2d[:, pos_]] = True
        return action_dist.flatten()


def scaled_linear_behavior_policy(tau: float) -> Callable:
    """Return a behavior_policy_function with logits scaled by 1 / tau.

    OBP's `SyntheticSlateBanditDataset` calls
    `behavior_policy_function(context=..., action_context=..., random_state=...)`
    without a temperature argument, so we close over tau here
    (`linear_behavior_policy_logit` itself already accepts `tau`).
    tau -> 0 gives a near-deterministic logging policy.
    """
    check_scalar(tau, "tau", (int, float), min_val=0.0)

    def behavior_policy_function(
        context: np.ndarray, action_context: np.ndarray, random_state: Optional[int] = None
    ) -> np.ndarray:
        return linear_behavior_policy_logit(
            context=context,
            action_context=action_context,
            random_state=random_state,
            tau=tau,
        )

    return behavior_policy_function


# --------------------------------------------------------------------------
# expected slate reward for the cascade (misspecified-DGP) baseline dataset
# --------------------------------------------------------------------------
def cascade_expected_slate_reward(dataset, context: np.ndarray, action_2d: np.ndarray) -> np.ndarray:
    """Exact expected slot rewards E[Y_k | x, a] of a cascade click-model
    dataset for given factual slates; returns (n, len_list).

    The log rewards come from `sample_reward_given_expected_reward`, whose
    discount at slot k is a function of the REALIZED clicks Y_{1:k-1}
    (discount_k = prod_{j<k} [Y_j * attractiveness_{j+1} + (1 - Y_j)]), so
    E[Y_k] = q_k * E[discount_k] must be taken over click prefixes. obp's own
    `calc_ground_truth_policy_value` (L999-L1013) instead plugs the previous
    slot's EXPECTED reward into the same recursion; since the recursion is
    convex in Y_{k-1} through discount_{k-1}, that substitutes (E[D])^2 where
    E[D^2] is needed and systematically overestimates E[Y_k] for k >= 3
    (verified against 4e6-sample means: +2.3% at k=3, +5.9% at k=4 for
    q ~ 0.25-0.4, eta=1). We therefore do NOT replicate obp here but run an
    exact dynamic program over the discount-factor distribution: the discount
    is a Markov state taking at most 2^{k-1} values (products of
    attractiveness factors), so the DP is exact and cheap for K <= ~10.
    Regression guard: tests/test_m2_dataset.py::
    test_cascade_ground_truth_matches_sampler_mean.

    The base rewards are gathered per round directly: obp<=0.5.7's
    reward_function builds the context row index with `len(action) // n_rounds`
    (off by a factor of len_list), so it cannot be reused here.

    Requires reward_structure="independent".
    """
    if dataset.reward_structure != "independent":
        raise ValueError("cascade_expected_slate_reward assumes reward_structure='independent'")
    n = context.shape[0]
    base = dataset.base_reward_function(
        context=context,
        action_context=dataset.action_context,
        random_state=dataset.random_state,
    )
    q = base[np.arange(n)[:, None], action_2d].astype(float)
    q *= dataset.exam_weight
    len_list = dataset.len_list
    attractiveness = dataset.attractiveness
    expected = np.empty_like(q)
    # states: discount value -> probability array (n,); the value entering
    # slot k is determined by WHICH previous slots were clicked, and every
    # path to the same click subset multiplies the same factors in the same
    # order, so float keys merge exactly
    states = {1.0: np.ones(n)}
    for pos_ in range(len_list):
        click_prob_total = np.zeros(n)
        next_states = {}
        for disc, prob in states.items():
            p_click = disc * q[:, pos_]
            click_prob_total += prob * p_click
            if pos_ + 1 < len_list:
                # a click at slot pos_ multiplies the next slot's discount by
                # attractiveness[pos_ + 1] (sampler indexing)
                for disc_next, mass in (
                    (disc * attractiveness[pos_ + 1], prob * p_click),
                    (disc, prob * (1.0 - p_click)),
                ):
                    if disc_next in next_states:
                        next_states[disc_next] += mass
                    else:
                        next_states[disc_next] = mass
        expected[:, pos_] = click_prob_total
        states = next_states
    return expected


# --------------------------------------------------------------------------
# input checks for the new estimators (reusing obp.utils)
# --------------------------------------------------------------------------
def _validate_nuisance_array(
    array: np.ndarray, name: str, n_records: int, positive: bool = False
) -> None:
    check_array(array=array, name=name, expected_dim=1)
    if array.shape[0] != n_records:
        raise ValueError(
            f"`{name}` must have the same number of records as `reward`, "
            f"but {array.shape[0]} != {n_records}"
        )
    if positive and np.any(array <= 0):
        raise ValueError(f"`{name}` must be strictly positive (it appears in a denominator)")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"`{name}` must be in the range of [0, 1]")


def check_latent_exposure_inputs(
    slate_id: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    pscore_item_position: np.ndarray,
    evaluation_policy_pscore_item_position: np.ndarray,
    exposure_factual_hat: np.ndarray,
    expected_exposure_eval_hat: np.ndarray,
) -> None:
    """Check inputs of SlateLatentExposureIPS."""
    _check_slate_ope_inputs(
        slate_id=slate_id,
        reward=reward,
        position=position,
        pscore=pscore_item_position,
        evaluation_policy_pscore=evaluation_policy_pscore_item_position,
        pscore_type="pscore_item_position",
    )
    n_records = reward.shape[0]
    _validate_nuisance_array(
        exposure_factual_hat, "exposure_factual_hat", n_records, positive=True
    )
    _validate_nuisance_array(
        expected_exposure_eval_hat, "expected_exposure_eval_hat", n_records
    )


def check_exposure_decomposed_dr_inputs(
    slate_id: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    pscore_item_position: np.ndarray,
    evaluation_policy_pscore_item_position: np.ndarray,
    exposure_factual_hat: np.ndarray,
    relevance_factual_hat: np.ndarray,
    expected_exposure_eval_hat: np.ndarray,
    dm_slot_values: np.ndarray,
) -> None:
    """Check inputs of SlateExposureDecomposedDR."""
    check_latent_exposure_inputs(
        slate_id=slate_id,
        reward=reward,
        position=position,
        pscore_item_position=pscore_item_position,
        evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
        exposure_factual_hat=exposure_factual_hat,
        expected_exposure_eval_hat=expected_exposure_eval_hat,
    )
    n_records = reward.shape[0]
    _validate_nuisance_array(relevance_factual_hat, "relevance_factual_hat", n_records)
    _validate_nuisance_array(dm_slot_values, "dm_slot_values", n_records)
