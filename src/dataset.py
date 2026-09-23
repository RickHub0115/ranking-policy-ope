# Contains code derived from Open Bandit Pipeline (zr-obp) v0.5.5,
# https://github.com/st-tech/zr-obp, obp/dataset/synthetic_slate.py: the
# input validation and the permutation-enumeration / batching structure of
# `calc_ground_truth_policy_value` are reused in `_calc_ground_truth_exact`
# and `calc_ground_truth_policy_value_epsilon_greedy` (modified).
# Copyright 2020 Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc.
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). See the NOTICE file at the
# repository root.
#
# ExposureSlateBanditDataset: semi-synthetic DGP with a latent exposure variable.
#
# Subclasses `obp.dataset.SyntheticSlateBanditDataset`.
# Action sampling / pscore computation / evaluation-policy methods of the parent
# are used untouched; only reward generation and the ground-truth computation
# are replaced:
#
#   O_k ~ Bern(e_k(x, a)),  R_k ~ Bern(r(x, a(k))),  Y_k = O_k * R_k
#
# The relevance r(x, a) is exactly the parent's `base_reward_function` output
# under reward_structure="independent" & click_model=None, so all position /
# ranking dependence lives in the exposure side e_k(x, a) and the product
# decomposition (A3) of the research plan holds exactly in the DGP.
from dataclasses import dataclass
from itertools import permutations
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from scipy.special import perm as n_permutations
from scipy.stats import norm
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from tqdm import tqdm

from obp.dataset import SyntheticSlateBanditDataset
from obp.types import BanditFeedback
from obp.utils import check_array
from obp.utils import sigmoid

from utils import make_position_weight
from utils import sample_ranking_from_logits


EXPOSURE_STRUCTURES = ("pbm", "contextual_pbm", "ranking_dependent")


@dataclass
class ExposureSlateBanditDataset(SyntheticSlateBanditDataset):
    """Synthetic slate data with explicitly sampled latent exposure O_k.

    Additional Parameters (on top of the parent's)
    ----------------------------------------------
    exposure_structure: str, default="pbm"
        One of {"pbm", "contextual_pbm", "ranking_dependent"} = (E1)/(E2)/(E3).

    exposure_decay_rate: float, default=1.0
        lambda: theta_k = (1/k)^lambda (k is 1-indexed; `position` in the OBP
        long format is 0-indexed — mind the off-by-one).

    n_user_segments: int, default=2
        For (E2): the population is split into equal-probability segments by
        context[:, 0] and each segment gets its own decay rate.

    segment_decay_rates: Optional[Sequence[float]], default=None
        For (E2): decay rate per segment. When None, defaults to
        exposure_decay_rate * geomspace(0.5, 2.0, n_user_segments).

    attention_spillover: float, default=0.0
        beta: strength of exposure stealing in (E3):
        e_k(x, a) = theta_k * exp(-beta * sum_{j<k} attract(x, a(j))).
        beta = 0 makes (E3) collapse to (E1) (continuity check).

    attract_relevance_align: float, default=0.5
        Mixing weight in [0, 1] between an independent linear score and the
        relevance logit inside attract(x, a). With 0 the attractiveness is
        unrelated to the relevance, and a policy that ranks by relevance barely
        shifts the exposure profile relative to the logging policy — the IIPS
        bias under (E3) is then tiny by construction. Positive alignment
        ("attractive items are the relevant ones") makes the ranking-dependent
        exposure interact with the evaluation policy, which is the regime the
        (E3) experiments are about. attract stays a function of (x, a) only.

    relevance_scale: float, default=1.0
        s in (0, 1]: uniform scale on the relevance, r(x, a) = s *
        base_reward_function(x, a).
        Applied inside `base_expected_reward`, so the DGP, the ground truth,
        the oracle nuisances and the EM's training data all see the same
        scaled relevance, and the epsilon-greedy evaluation policy's ranking
        is unchanged (a uniform scale preserves the argsort). NOTE: with
        attract_relevance_align > 0 the scale also lowers attract(x, a)
        through the relevance logit, which RAISES the (E3) exposure e — the
        a2slot redesign gate includes this side effect.

    return_exposure: bool, default=False
        Whether to include the realized exposure O_k in the logged data
        (oracle diagnostics only; never available to estimators).

    position_weight_kind: str, default="uniform"
        alpha_k used in the ground-truth policy value ("uniform" or "dcg").
    """

    exposure_structure: str = "pbm"
    exposure_decay_rate: float = 1.0
    n_user_segments: int = 2
    segment_decay_rates: Optional[Sequence[float]] = None
    attention_spillover: float = 0.0
    attract_relevance_align: float = 0.5
    relevance_scale: float = 1.0
    return_exposure: bool = False
    position_weight_kind: str = "uniform"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.exposure_structure not in EXPOSURE_STRUCTURES:
            raise ValueError(
                f"`exposure_structure` must be one of {EXPOSURE_STRUCTURES}, "
                f"but {self.exposure_structure} is given"
            )
        # the exposure DGP requires the pure product decomposition; ranking
        # dependence must all live in e_k, so the parent's reward machinery is
        # pinned to the independent structure with no click model.
        if self.reward_structure != "independent":
            raise ValueError(
                "`reward_structure` must be 'independent' for "
                "ExposureSlateBanditDataset (the (A3) decomposition requires it); "
                "use the parent class directly for cascade-type DGPs"
            )
        if self.click_model is not None:
            raise ValueError(
                "`click_model` must be None for ExposureSlateBanditDataset; "
                "use the parent class directly for click-model DGPs"
            )
        if self.reward_type != "binary":
            raise ValueError("`reward_type` must be 'binary' (Y = O * R is binary)")
        if self.base_reward_function is None:
            raise ValueError(
                "`base_reward_function` must be given: the relevance r(x, a) is "
                "defined as its output (the context-free fallback of the parent "
                "is position-dependent and would break the (A3) decomposition)"
            )
        check_scalar(
            self.exposure_decay_rate, "exposure_decay_rate", (int, float), min_val=0.0
        )
        check_scalar(
            self.attention_spillover, "attention_spillover", (int, float), min_val=0.0
        )
        check_scalar(self.n_user_segments, "n_user_segments", int, min_val=1)
        check_scalar(
            self.attract_relevance_align,
            "attract_relevance_align",
            (int, float),
            min_val=0.0,
            max_val=1.0,
        )
        check_scalar(
            self.relevance_scale,
            "relevance_scale",
            (int, float),
            min_val=0.0,
            max_val=1.0,
        )
        if self.relevance_scale <= 0.0:
            raise ValueError("`relevance_scale` must be in (0, 1]")
        if self.segment_decay_rates is None:
            self.segment_decay_rates_ = self.exposure_decay_rate * np.geomspace(
                0.5, 2.0, self.n_user_segments
            )
        else:
            self.segment_decay_rates_ = np.asarray(self.segment_decay_rates, dtype=float)
            if self.segment_decay_rates_.shape[0] != self.n_user_segments:
                raise ValueError(
                    "`segment_decay_rates` must have length `n_user_segments`"
                )
        # equal-probability segment edges for context[:, 0] ~ N(0, 1)
        self._segment_edges = norm.ppf(
            np.linspace(0.0, 1.0, self.n_user_segments + 1)[1:-1]
        )
        # fixed coefficients of the attractiveness function of (E3); a dedicated
        # RNG keeps them independent of the sampling stream self.random_
        attract_rng = check_random_state(
            None if self.random_state is None else self.random_state + 771
        )
        self._attract_context_coef = attract_rng.uniform(
            -1.0, 1.0, size=self.dim_context
        ) / np.sqrt(self.dim_context)
        self._attract_action_coef = attract_rng.uniform(
            -1.0, 1.0, size=self.n_unique_action
        )
        self.position_weight = make_position_weight(
            self.len_list, self.position_weight_kind
        )

    # ------------------------------------------------------------------
    # slot-value hooks (overridden by DBNClickBanditDataset)
    # ------------------------------------------------------------------
    def _expected_slot_rewards_from_er(
        self, exposure: np.ndarray, relevance: np.ndarray
    ) -> np.ndarray:
        """E[Y_k | x, a] per slot from e and r arrays of shape (..., K).

        Under (A2)/(A3) this is the pure product e * r. The DBN variant
        overrides it with the satisfaction survival factor, so every
        ground-truth path below must go through this hook instead of
        multiplying e and r inline."""
        return exposure * relevance

    def _marginal_exposure_from_er(
        self, exposure: np.ndarray, relevance: np.ndarray
    ) -> np.ndarray:
        """E[O_k | x, a]: equals e_k here; the DBN variant deflates it by the
        probability that no satisfied click occurred above slot k."""
        return exposure

    def _uniform_value_is_separable(self) -> bool:
        """Whether E_unif[sum_k alpha_k E[Y_k]] factorizes into
        (position profile) x (mean relevance) under a uniform random slate.
        True for (E1)/(E2) where e does not depend on the ranking; the DBN
        variant returns False because the satisfaction survival couples the
        slots through the relevance of the items ranked above."""
        return self.exposure_structure in ("pbm", "contextual_pbm")

    def expected_slot_rewards(
        self, context: np.ndarray, action_2d: np.ndarray
    ) -> np.ndarray:
        """Exact expected slot rewards E[Y_k | x, a] for factual slates;
        shape (n, len_list)."""
        exposure = self.calc_expected_exposure(context, action_2d)
        relevance = self.base_expected_reward(context)[
            np.arange(context.shape[0])[:, None], action_2d
        ]
        return self._expected_slot_rewards_from_er(exposure, relevance)

    # ------------------------------------------------------------------
    # exposure DGP
    # ------------------------------------------------------------------
    def _theta(self) -> np.ndarray:
        """theta_k = (1/k)^lambda, k = 1..len_list (same convention as OBP's
        exam_weight, cf. synthetic_slate.py L256)."""
        return (1.0 / np.arange(1, self.len_list + 1)) ** self.exposure_decay_rate

    def _segment_of(self, context: np.ndarray) -> np.ndarray:
        """Deterministic segment id in [0, n_user_segments) from context."""
        if self.n_user_segments == 1:
            return np.zeros(context.shape[0], dtype=int)
        return np.digitize(context[:, 0], self._segment_edges)

    def _attractiveness_matrix(self, context: np.ndarray) -> np.ndarray:
        """attract(x, a) in (0, 1) for ALL items; shape (n, n_unique_action)."""
        x_term = context @ self._attract_context_coef  # (n,)
        linear = x_term[:, None] + self._attract_action_coef[None, :]  # (n, m)
        if self.attract_relevance_align > 0.0:
            relevance = np.clip(self.base_expected_reward(context), 1e-6, 1 - 1e-6)
            relevance_logit = np.log(relevance / (1 - relevance))
            linear = (
                1.0 - self.attract_relevance_align
            ) * linear + self.attract_relevance_align * relevance_logit
        return sigmoid(linear)

    def calc_attractiveness(self, context: np.ndarray, action_2d: np.ndarray) -> np.ndarray:
        """attract(x, a(j)) in (0, 1) for each slot of each round; shape (n, K)."""
        item_attractiveness = self._attractiveness_matrix(context)
        return item_attractiveness[np.arange(context.shape[0])[:, None], action_2d]

    def calc_expected_exposure(
        self, context: np.ndarray, action_2d: np.ndarray
    ) -> np.ndarray:
        """True exposure probability e_k(x, a); shape (n, K), (E1)-(E3)."""
        check_array(array=context, name="context", expected_dim=2)
        check_array(array=action_2d, name="action_2d", expected_dim=2)
        if action_2d.shape[1] != self.len_list:
            raise ValueError("`action_2d` must have len_list columns")
        n = context.shape[0]
        positions = np.arange(1, self.len_list + 1)
        if self.exposure_structure == "pbm":  # (E1)
            return np.tile(self._theta(), (n, 1))
        if self.exposure_structure == "contextual_pbm":  # (E2)
            lam = self.segment_decay_rates_[self._segment_of(context)]  # (n,)
            return (1.0 / positions)[None, :] ** lam[:, None]
        # (E3) ranking-dependent: upper attractive items steal exposure
        attract = self.calc_attractiveness(context, action_2d)  # (n, K)
        cum_above = np.concatenate(
            [np.zeros((n, 1)), np.cumsum(attract, axis=1)[:, :-1]], axis=1
        )  # sum over j < k
        return self._theta()[None, :] * np.exp(
            -self.attention_spillover * cum_above
        )

    def sample_exposure_and_reward(
        self,
        expected_exposure: np.ndarray,
        expected_relevance: np.ndarray,
        random_: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample O ~ Bern(e), R ~ Bern(r), and return (Y = O * R, O); each (n, K)."""
        exposure = random_.binomial(n=1, p=expected_exposure)
        relevance = random_.binomial(n=1, p=expected_relevance)
        return exposure * relevance, exposure

    # ------------------------------------------------------------------
    # logged data
    # ------------------------------------------------------------------
    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ) -> BanditFeedback:
        """Parent produces context / action / pscores / expected_reward_factual;
        we discard the parent's reward and regenerate Y = O * R explicitly.

        (The parent's `sample_reward_given_expected_reward` is *not* overridden
        but simply ignored: rebuilding the reward keeps the parent's cascade
        branches intact at the cost of one wasted sampling pass.)
        """
        bandit_feedback = super().obtain_batch_bandit_feedback(
            n_rounds=n_rounds,
            return_pscore_item_position=return_pscore_item_position,
            clip_logit_value=clip_logit_value,
        )
        action_2d = bandit_feedback["action"].reshape((n_rounds, self.len_list))
        # r(x_i, a_i(k)) recomputed here instead of taking the parent's
        # expected_reward_factual: obp<=0.5.7's action_interaction_reward_function
        # divides by `n_rounds` instead of `n_rounds * len_list` when building
        # the context row index (synthetic_slate.py L1430), so with a
        # base_reward_function the parent's factual rewards use context
        # floor(i / len_list) rather than context i.
        relevance_factual = self.base_expected_reward(bandit_feedback["context"])[
            np.arange(n_rounds)[:, None], action_2d
        ]
        exposure_factual = self.calc_expected_exposure(
            bandit_feedback["context"], action_2d
        )
        reward, exposure = self.sample_exposure_and_reward(
            expected_exposure=exposure_factual,
            expected_relevance=relevance_factual,
            random_=self.random_,
        )
        bandit_feedback["reward"] = reward.flatten()
        # E[Y | x, a] = e * r under (A2)/(A3); the DBN variant's hook adds the
        # satisfaction survival factor to both quantities
        bandit_feedback["expected_reward_factual"] = self._expected_slot_rewards_from_er(
            exposure_factual, relevance_factual
        ).flatten()
        bandit_feedback["expected_exposure_factual"] = self._marginal_exposure_from_er(
            exposure_factual, relevance_factual
        ).flatten()
        bandit_feedback["expected_relevance_factual"] = relevance_factual.flatten()
        if self.return_exposure:
            bandit_feedback["exposure"] = exposure.flatten()
        return bandit_feedback

    # ------------------------------------------------------------------
    # relevance access (used by evaluation policies / oracle nuisances)
    # ------------------------------------------------------------------
    def base_expected_reward(self, context: np.ndarray) -> np.ndarray:
        """r(x, a) = relevance_scale * base_reward_function(x, a) for all
        actions; shape (n, n_unique_action). The scale is applied HERE so the
        DGP, ground truth, oracle and attract matrix stay consistent."""
        if self.base_reward_function is None:
            raise ValueError(
                "`base_reward_function` must be set for ExposureSlateBanditDataset"
            )
        return self.relevance_scale * self.base_reward_function(
            context=context,
            action_context=self.action_context,
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------
    # ground-truth policy value
    # ------------------------------------------------------------------
    def calc_ground_truth_policy_value(
        self,
        context: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
        method: str = "auto",
        n_mc_samples: int = 10**6,
        max_enumeration: int = 10**5,
        random_state: Optional[int] = None,
    ) -> float:
        """V(pi) for a Plackett-Luce evaluation policy given by logits.

        Overrides the parent (synthetic_slate.py L870): the slot value is
        q_k(x, a) = e_k(x, a) * r(x, a(k)) (weighted by alpha_k), replacing the
        parent's `expected_slate_rewards_ *= self.exam_weight`.

        method:
          "exact" — full permutation enumeration (parent's structure);
          "mc"    — on-policy simulation with n_mc_samples total draws;
          "auto"  — exact when P(m, K) <= max_enumeration, else mc.
        """
        check_array(array=context, name="context", expected_dim=2)
        check_array(
            array=evaluation_policy_logit_,
            name="evaluation_policy_logit_",
            expected_dim=2,
        )
        if evaluation_policy_logit_.shape[1] != self.n_unique_action:
            raise ValueError(
                "Expected `evaluation_policy_logit_.shape[1] == self.n_unique_action`,"
                "but found it False"
            )
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )
        if evaluation_policy_logit_.shape[0] != context.shape[0]:
            raise ValueError(
                "Expected `evaluation_policy_logit_.shape[0] == context.shape[0]`,"
                "but found it False"
            )
        if self.is_factorizable:
            raise NotImplementedError(
                "is_factorizable=True is out of scope for the exposure DGP"
            )
        if method not in ("auto", "exact", "mc"):
            raise ValueError("`method` must be one of 'auto', 'exact', 'mc'")
        n_enum = n_permutations(self.n_unique_action, self.len_list, exact=True)
        if method == "auto":
            method = "exact" if n_enum <= max_enumeration else "mc"
        if method == "exact":
            return self._calc_ground_truth_exact(context, evaluation_policy_logit_)
        return self._calc_ground_truth_on_policy_mc(
            context,
            evaluation_policy_logit_,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
        )

    def _calc_ground_truth_exact(
        self, context: np.ndarray, evaluation_policy_logit_: np.ndarray
    ) -> float:
        """Full enumeration; copies the parent's batching structure with the
        exam_weight line swapped for `calc_expected_exposure` on each permutation."""
        enumerated_slate_actions = np.array(
            list(permutations(np.arange(self.n_unique_action), self.len_list))
        ).astype("int8")
        n_enum = len(enumerated_slate_actions)
        n_rounds = len(context)

        pscores = []
        for i in tqdm(
            np.arange(n_rounds),
            desc="[calc_ground_truth_policy_value (pscore)]",
            total=n_rounds,
        ):
            pscores.append(
                self._calc_pscore_given_policy_logit(
                    all_slate_actions=enumerated_slate_actions,
                    policy_logit_i_=evaluation_policy_logit_[i],
                )
            )
        pscores = np.array(pscores)  # (n_rounds, n_enum)

        n_batch = (n_rounds * n_enum * self.len_list - 1) // 10**7 + 1
        batch_size = (n_rounds - 1) // n_batch + 1
        n_batch = (n_rounds - 1) // batch_size + 1

        policy_value = 0.0
        for batch_idx in tqdm(
            np.arange(n_batch),
            desc=f"[calc_ground_truth_policy_value (expected reward), batch_size={batch_size}]",
            total=n_batch,
        ):
            context_ = context[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            pscores_ = pscores[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            n_batch_rounds = len(context_)
            # relevance r(x, a(k)) for every (round, permutation, slot),
            # gathered from the base reward matrix directly (full precision;
            # the parent's reward_function has the context-index bug noted in
            # obtain_batch_bandit_feedback and quantizes to float16)
            relevance_ = self.base_expected_reward(context_)[
                np.arange(n_batch_rounds)[:, None, None],
                enumerated_slate_actions[None, :, :],
            ].reshape((n_batch_rounds * n_enum, self.len_list))
            # exposure e_k(x, a) for every (round, permutation, slot)
            exposure_ = self._calc_expected_exposure_enumerated(
                context_, enumerated_slate_actions
            ).reshape((n_batch_rounds * n_enum, self.len_list))
            slate_values = (
                self.position_weight[None, :]
                * self._expected_slot_rewards_from_er(exposure_, relevance_)
            ).sum(axis=1)
            policy_value += (pscores_.flatten() * slate_values).sum()
        policy_value /= n_rounds
        return float(policy_value)

    def _calc_expected_exposure_enumerated(
        self, context: np.ndarray, enumerated_slate_actions: np.ndarray
    ) -> np.ndarray:
        """e_k(x, a) for all rounds x all enumerated slates; (n, n_enum, K)."""
        n = context.shape[0]
        n_enum = enumerated_slate_actions.shape[0]
        positions = np.arange(1, self.len_list + 1)
        if self.exposure_structure == "pbm":
            return np.broadcast_to(
                self._theta()[None, None, :], (n, n_enum, self.len_list)
            ).copy()
        if self.exposure_structure == "contextual_pbm":
            lam = self.segment_decay_rates_[self._segment_of(context)]  # (n,)
            theta_ctx = (1.0 / positions)[None, :] ** lam[:, None]  # (n, K)
            return np.broadcast_to(
                theta_ctx[:, None, :], (n, n_enum, self.len_list)
            ).copy()
        # (E3): attract per (round, item) then gather per enumerated slate
        item_attractiveness = self._attractiveness_matrix(context)  # (n, m)
        attract_slates = item_attractiveness[:, enumerated_slate_actions]  # (n, n_enum, K)
        cum_above = np.concatenate(
            [
                np.zeros((n, n_enum, 1)),
                np.cumsum(attract_slates, axis=2)[:, :, :-1],
            ],
            axis=2,
        )
        return self._theta()[None, None, :] * np.exp(
            -self.attention_spillover * cum_above
        )

    def _calc_ground_truth_on_policy_mc(
        self,
        context: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
        n_mc_samples: int = 10**6,
        random_state: Optional[int] = None,
    ) -> float:
        """On-policy simulation of V(pi) (expected rewards, no click noise);
        cf. `calc_on_policy_policy_value` (L842) but with q = alpha * e * r."""
        random_ = check_random_state(random_state)
        n = context.shape[0]
        n_draws = max(1, int(np.ceil(n_mc_samples / n)))
        relevance_all = self.base_expected_reward(context)  # (n, m)
        total = 0.0
        for _ in range(n_draws):
            ranking = sample_ranking_from_logits(
                evaluation_policy_logit_, self.len_list, random_
            )
            exposure = self.calc_expected_exposure(context, ranking)  # (n, K)
            relevance = relevance_all[np.arange(n)[:, None], ranking]  # (n, K)
            total += (
                self.position_weight[None, :]
                * self._expected_slot_rewards_from_er(exposure, relevance)
            ).sum(axis=1).mean()
        return float(total / n_draws)

    def calc_ground_truth_policy_value_epsilon_greedy(
        self,
        context: np.ndarray,
        greedy_ranking: np.ndarray,
        epsilon: float,
        method: str = "auto",
        n_mc_samples: int = 10**6,
        max_enumeration: int = 10**5,
        random_state: Optional[int] = None,
    ) -> float:
        """V(pi) for the slate-level epsilon-greedy mixture policy.

        V = (1 - eps) * V(greedy slate)  +  eps * V(uniform random slate).

        The greedy part is exact. The uniform part is exact in closed form for
        (E1)/(E2) (uniform position marginals), by enumeration when feasible,
        and by Monte Carlo otherwise.
        """
        check_scalar(epsilon, "epsilon", float, min_val=0.0, max_val=1.0)
        n = context.shape[0]
        relevance_all = self.base_expected_reward(context)  # (n, m)
        # --- greedy part (deterministic slate) ---
        exposure_g = self.calc_expected_exposure(context, greedy_ranking)
        relevance_g = relevance_all[np.arange(n)[:, None], greedy_ranking]
        v_greedy = float(
            (
                self.position_weight[None, :]
                * self._expected_slot_rewards_from_er(exposure_g, relevance_g)
            )
            .sum(axis=1)
            .mean()
        )
        if epsilon == 0.0:
            return v_greedy
        # --- uniform part ---
        if self._uniform_value_is_separable():
            # e does not depend on the ranking; uniform position marginal over items:
            # E_unif[sum_k alpha_k e_k r(x, a(k))] = (sum_k alpha_k e_k(x)) * mean_a r(x, a)
            positions = np.arange(1, self.len_list + 1)
            if self.exposure_structure == "pbm":
                theta_ctx = np.tile(self._theta(), (n, 1))
            else:
                lam = self.segment_decay_rates_[self._segment_of(context)]
                theta_ctx = (1.0 / positions)[None, :] ** lam[:, None]
            v_unif = float(
                (
                    (self.position_weight[None, :] * theta_ctx).sum(axis=1)
                    * relevance_all.mean(axis=1)
                ).mean()
            )
        else:
            n_enum = n_permutations(self.n_unique_action, self.len_list, exact=True)
            use_exact = method == "exact" or (
                method == "auto" and n_enum <= max_enumeration
            )
            if method not in ("auto", "exact", "mc"):
                raise ValueError("`method` must be one of 'auto', 'exact', 'mc'")
            if use_exact:
                enumerated = np.array(
                    list(permutations(np.arange(self.n_unique_action), self.len_list))
                ).astype("int8")
                # batch over rounds to bound memory (same 1e7 budget as the parent)
                n_batch = (n * n_enum * self.len_list - 1) // 10**7 + 1
                batch_size = (n - 1) // n_batch + 1
                v_unif_total = 0.0
                for b in range(0, n, batch_size):
                    ctx_ = context[b : b + batch_size]
                    rel_ = relevance_all[b : b + batch_size]  # (nb, m)
                    e_ = self._calc_expected_exposure_enumerated(ctx_, enumerated)
                    r_ = rel_[
                        np.arange(len(ctx_))[:, None, None], enumerated[None, :, :]
                    ]  # (nb, n_enum, K)
                    v_unif_total += (
                        (
                            self.position_weight[None, None, :]
                            * self._expected_slot_rewards_from_er(e_, r_)
                        )
                        .sum(axis=2)
                        .mean(axis=1)
                        .sum()
                    )
                v_unif = float(v_unif_total / n)
            else:
                random_ = check_random_state(random_state)
                n_draws = max(1, int(np.ceil(n_mc_samples / n)))
                total = 0.0
                for _ in range(n_draws):
                    keys = random_.uniform(size=(n, self.n_unique_action))
                    ranking = np.argsort(keys, axis=1)[:, : self.len_list]
                    e_ = self.calc_expected_exposure(context, ranking)
                    r_ = relevance_all[np.arange(n)[:, None], ranking]
                    total += (
                        (
                            self.position_weight[None, :]
                            * self._expected_slot_rewards_from_er(e_, r_)
                        )
                        .sum(axis=1)
                        .mean()
                    )
                v_unif = float(total / n_draws)
        return (1 - epsilon) * v_greedy + epsilon * v_unif


@dataclass
class CascadeClickBanditDataset(SyntheticSlateBanditDataset):
    """Parent cascade-click DGP with the per-round relevance recomputed.

    Used for the losing-condition experiments (Figure 4): the true
    click process is a (dependent) cascade model. Slot-wise the reward still
    factors into exposure x relevance ((A2)/(A3) hold); what breaks is the
    position-independence (A4), and the true exposure — the expected product
    of no-clicks above — lies outside the assumed exposure classes, so the
    proposal's exposure estimate is misspecified.

    Rewards are regenerated here because obp<=0.5.7's
    `action_interaction_reward_function` builds the context row index with
    `len(action) // n_rounds` (synthetic_slate.py L1430), off by a factor of
    `len_list`, so with a `base_reward_function` the parent's factual expected
    rewards use context floor(i / len_list) instead of context i. We recompute
    q(x_i, a_i(k)) from the base reward matrix and re-sample the cascade
    rewards with the parent's own sampler (which applies the position-dependent
    attractiveness discounting).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.click_model != "cascade":
            raise ValueError("`click_model` must be 'cascade' for this class")
        if self.reward_structure != "independent":
            raise ValueError(
                "`reward_structure` must be 'independent' (interactions beyond "
                "the click model would confound the (A4)/e_hat-misspecification study)"
            )
        if self.base_reward_function is None:
            raise ValueError("`base_reward_function` must be given")

    def base_expected_reward(self, context: np.ndarray) -> np.ndarray:
        """Attractiveness q(x, a) of each item; shape (n, n_unique_action)."""
        return self.base_reward_function(
            context=context,
            action_context=self.action_context,
            random_state=self.random_state,
        )

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ) -> BanditFeedback:
        bandit_feedback = super().obtain_batch_bandit_feedback(
            n_rounds=n_rounds,
            return_pscore_item_position=return_pscore_item_position,
            clip_logit_value=clip_logit_value,
        )
        action_2d = bandit_feedback["action"].reshape((n_rounds, self.len_list))
        base_q = self.base_expected_reward(bandit_feedback["context"])[
            np.arange(n_rounds)[:, None], action_2d
        ]
        # the parent's sampler multiplies its input by exam_weight IN PLACE and
        # runs the cascade (DCM) discounting with `attractiveness` — pass a copy
        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=base_q.copy()
        )
        bandit_feedback["reward"] = reward.flatten()
        bandit_feedback["expected_reward_factual"] = base_q.flatten()
        return bandit_feedback


@dataclass
class DBNClickBanditDataset(ExposureSlateBanditDataset):
    """DBN-style (A2)-violating variant of the exposure DGP.

    Same exposure machinery as the parent — base examination B_k ~ Bern(e_k(x, a))
    with any of (E1)/(E2)/(E3), relevance R_k ~ Bern(r(x, a(k))) — plus the
    satisfaction channel of the DBN click model (Chapelle & Zhang 2009): after
    each click the user is satisfied with probability `or_correlation` and
    abandons the list, zeroing the exposure of every later slot:

        O_k = B_k * prod_{j<k} (1 - S_j),   S_j = Y_j * Bern(or_correlation),
        Y_k = O_k * R_k.

    The exposure noise O_k thus depends on the REALIZED relevance noise
    R_{j<k} of the slots above, violating the (A2) conditional independence
    behind the E[Y] = e * r decomposition with controllable strength, while
    the logging side (actions, pscores, e_k, r) is untouched.

    or_correlation = 0 reproduces the parent exposure DGP *bitwise* for the
    same RNG state: the base B / R draws come first and in the parent's exact
    order, and the satisfaction draws (all zeros at rho = 0) only multiply in
    afterwards. Regression guard:
    tests/test_a2_dbn_dataset.py::test_dbn_rho_zero_matches_exposure_dgp_bitwise.

    Ground truth: the "still active" state entering slot k is binary and
    independent of slot k's own draws, so the recursion is exactly linear and

        E[Y_k | x, a] = e_k r_k * prod_{j<k} (1 - rho * e_j r_j)

    holds in closed form (no Jensen gap). We deliberately do NOT replicate
    obp's cascade ground-truth recursion, which substitutes expected rewards
    into a nonlinear realized-click discount and overestimates E[Y_k] for
    k >= 3 (utils.cascade_expected_slate_reward). Exactness is
    pinned by a full noise-outcome enumeration in
    tests/test_a2_dbn_dataset.py::test_dbn_expected_slot_rewards_exact_by_enumeration.

    Additional Parameters
    ---------------------
    or_correlation: float, default=0.0
        rho in [0, 1]: the DBN satisfaction probability = strength of the
        O-R noise coupling. 0 = (A2) holds (exposure DGP); 1 = every click
        ends the session.
    """

    or_correlation: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        check_scalar(
            self.or_correlation,
            "or_correlation",
            (int, float),
            min_val=0.0,
            max_val=1.0,
        )

    def _dbn_survival(self, slot_rewards: np.ndarray) -> np.ndarray:
        """P(no satisfied click above slot k) from q_j = e_j r_j; (..., K).
        Entry k is prod_{j<k} (1 - rho * q_j) (the k = 1 entry is exactly 1.0,
        so at rho = 0 multiplying by the survival is a bitwise no-op)."""
        survival = np.cumprod(1.0 - self.or_correlation * slot_rewards, axis=-1)
        head = np.ones_like(survival[..., :1])
        return np.concatenate([head, survival[..., :-1]], axis=-1)

    def _expected_slot_rewards_from_er(
        self, exposure: np.ndarray, relevance: np.ndarray
    ) -> np.ndarray:
        slot_rewards = exposure * relevance
        return slot_rewards * self._dbn_survival(slot_rewards)

    def _marginal_exposure_from_er(
        self, exposure: np.ndarray, relevance: np.ndarray
    ) -> np.ndarray:
        return exposure * self._dbn_survival(exposure * relevance)

    def _uniform_value_is_separable(self) -> bool:
        # the survival factor couples the slots through the relevance of the
        # items ranked above, even when e itself is ranking-independent
        return False

    def sample_exposure_and_reward(
        self,
        expected_exposure: np.ndarray,
        expected_relevance: np.ndarray,
        random_: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the DBN click process; returns (Y, O), each (n, K).

        The base examination / relevance draws replicate the parent's order
        exactly, and the satisfaction draws come AFTER them in the stream, so
        rho = 0 yields a bitwise-identical click log for the same RNG state.
        """
        base_exam = random_.binomial(n=1, p=expected_exposure)
        relevance = random_.binomial(n=1, p=expected_relevance)
        satisfied_draw = random_.binomial(
            n=1, p=self.or_correlation, size=base_exam.shape
        )
        n, len_list = base_exam.shape
        exposure = np.empty_like(base_exam)
        active = np.ones(n, dtype=base_exam.dtype)
        for pos_ in range(len_list):
            exposure[:, pos_] = active * base_exam[:, pos_]
            click = exposure[:, pos_] * relevance[:, pos_]
            active = active * (1 - click * satisfied_draw[:, pos_])
        return exposure * relevance, exposure


@dataclass
class FrechetCoupledExposureDataset(ExposureSlateBanditDataset):
    """Slot-level (A2)-violating variant of the exposure DGP via a Frechet
    mixture (appendix A2').

    Where the DBN variant couples O_k to the REALIZED relevance of the slots
    ABOVE (vector-level (A2) violation; slot-wise (A2)/(A3) survive), this DGP
    breaks the independence O_k ⟂ R_k INSIDE each slot: with probability
    `1 - or_coupling` the pair (O_k, R_k) is drawn independently exactly as in
    the parent, and with probability `or_coupling` both indicators are driven
    by one shared uniform U (the Frechet upper-bound coupling
    O = 1{U < e}, R = 1{U < r}). Hence

        E[Y_k | x, a] = (1 - delta) e_k r_k + delta min(e_k, r_k),

    in closed form, while the marginals stay intact (E[O_k] = e_k,
    E[R_k] = r_k) and the slots couple independently of each other, so the
    position-independence (A4) and the marginal exposure are preserved —
    only (A2)/(A3) break, and an oracle that serves the true marginals e / r
    cannot repair the product decomposition (the point of the experiment).

    or_coupling = 0 reproduces the parent exposure DGP *bitwise* for the same
    RNG state: the base O / R draws come first and in the parent's exact
    order, and the coupling mask / shared-uniform draws (a no-op mask at
    delta = 0) only substitute afterwards. As with the DBN variant, the extra
    draws shift the shared bootstrap stream of AIPS, so AIPS at delta = 0 is
    distribution-identical but not bitwise (disclosed; every other estimator
    matches exactly).

    Additional Parameters
    ---------------------
    or_coupling: float, default=0.0
        delta in [0, 1]: the mixture weight of the Frechet coupling =
        strength of the within-slot O-R dependence. 0 = (A2)/(A3) hold
        (exposure DGP); 1 = fully comonotone slots.
    """

    or_coupling: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        check_scalar(
            self.or_coupling,
            "or_coupling",
            (int, float),
            min_val=0.0,
            max_val=1.0,
        )

    def _expected_slot_rewards_from_er(
        self, exposure: np.ndarray, relevance: np.ndarray
    ) -> np.ndarray:
        if self.or_coupling == 0.0:
            # exact parent value (bitwise, not just numerically equal)
            return super()._expected_slot_rewards_from_er(exposure, relevance)
        return (1.0 - self.or_coupling) * exposure * relevance + (
            self.or_coupling * np.minimum(exposure, relevance)
        )

    # _marginal_exposure_from_er stays the parent's identity: the coupling
    # preserves the exposure marginal, P(O = 1) = (1 - delta) e + delta e = e.

    def _uniform_value_is_separable(self) -> bool:
        # min(e, r) does not factor into (position profile) x (mean relevance);
        # at delta = 0 the parent's shortcut is exact (and keeps the delta = 0
        # ground truth bitwise identical to the exposure DGP)
        if self.or_coupling == 0.0:
            return super()._uniform_value_is_separable()
        return False

    def sample_exposure_and_reward(
        self,
        expected_exposure: np.ndarray,
        expected_relevance: np.ndarray,
        random_: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the Frechet-mixture click process; returns (Y, O), each (n, K).

        The independent O / R draws replicate the parent's order exactly, and
        the coupling mask / shared-uniform draws come AFTER them in the
        stream, so delta = 0 yields a bitwise-identical click log for the
        same RNG state.
        """
        exposure = random_.binomial(n=1, p=expected_exposure)
        relevance = random_.binomial(n=1, p=expected_relevance)
        coupled = random_.binomial(
            n=1, p=self.or_coupling, size=exposure.shape
        ).astype(bool)
        shared_u = random_.uniform(size=exposure.shape)
        exposure = np.where(coupled, (shared_u < expected_exposure).astype(exposure.dtype), exposure)
        relevance = np.where(coupled, (shared_u < expected_relevance).astype(relevance.dtype), relevance)
        return exposure * relevance, exposure
