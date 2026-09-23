# Contains code derived from Open Bandit Pipeline (zr-obp) v0.5.5,
# https://github.com/st-tech/zr-obp, obp/ope/estimators_slate.py: the
# slate-level bootstrap `_estimate_slate_confidence_interval_by_bootstrap`
# (copied from SlateCascadeDoublyRobust) and the estimate_policy_value /
# estimate_interval structure of SlateStandardIPS / SlateCascadeDoublyRobust.
# Copyright 2020 Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc.
# `SlateAdaptiveIPS` is a port of kdd2023-aips,
# https://github.com/aiueola/kdd2023-aips (see src/aips.py).
# Both upstreams are licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0); the derived code is modified.
# See the NOTICE file at the repository root.
#
# Proposed slate OPE estimators: LE-IIPS and ED-DR, plus the small DM / DR
# baselines that OBP lacks.
#
# Pattern: do NOT fork zr-obp — import the base classes from the pip-installed
# obp and extend (the kdd2023-aips `SlateAdaptiveIPS` recipe).
#
# LE-IIPS needs almost no new code: the exposure correction ratio folds into
# the pscores, so the base class `_estimate_round_rewards` (estimators_slate.py
# L53) and its slate-level bootstrap CI are reused as-is.
# ED-DR mirrors `SlateCascadeDoublyRobust` (L578): the "residual + DM" DR
# skeleton with cascade weights / Q-hat replaced by exposure-corrected IIPS
# weights / the decomposed model e_hat * r_hat.
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np

from obp.ope.estimators_slate import BaseSlateInverseProbabilityWeighting
from obp.ope.estimators_slate import BaseSlateOffPolicyEstimator
from obp.utils import check_array
from obp.utils import estimate_confidence_interval_by_bootstrap

from utils import check_exposure_decomposed_dr_inputs
from utils import check_latent_exposure_inputs
from utils import position_weight_for


def clip_exposure_denominator(
    exposure_factual_hat: np.ndarray,
    expected_exposure_eval_hat: np.ndarray,
    exposure_ratio_clip: Optional[float],
) -> np.ndarray:
    """Floor the exposure denominator so the estimated exposure ratio
    e_bar_hat^pi / e_hat never exceeds `exposure_ratio_clip` (truncated-IPS
    style; None disables). A degenerate EM fit can collapse e_hat to the
    numeric floor (PROB_CLIP), which otherwise inflates the importance weight
    by up to 1 / PROB_CLIP. At small n this is the norm, not a rare guard:
    measured at ranking_dependent n=100 the floor binds on every seed for
    ~10% of records (max 17%) and changes the proposed estimators' rel-MSE
    by orders of magnitude — read the small-n curves as truncated estimators
    and disclose accordingly."""
    if exposure_ratio_clip is None:
        return exposure_factual_hat
    return np.maximum(
        exposure_factual_hat,
        expected_exposure_eval_hat / float(exposure_ratio_clip),
    )


@dataclass
class SlateLatentExposureIPS(BaseSlateInverseProbabilityWeighting):
    """LE-IIPS (Latent-Exposure IIPS).

    w_hat_{ik} = [pi_k(a_i(k)|x_i) * e_bar_hat^pi_k(x_i, a_i(k))]
               / [pi_{0,k}(a_i(k)|x_i) * e_hat_k(x_i, a_i)]

    i.e. a ratio of exposure-corrected pscores, so the exposure terms are
    folded into `behavior_policy_pscore` / `evaluation_policy_pscore` of the
    base class (the same trick AIPS uses with its own pscores).

    `exposure_ratio_clip` caps the estimated exposure ratio e_bar_hat / e_hat
    at a constant M (None = off); see `clip_exposure_denominator`.
    """

    estimator_name: str = "le-iips"
    exposure_ratio_clip: Optional[float] = None

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        exposure_factual_hat: np.ndarray,
        expected_exposure_eval_hat: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimated policy value.

        Parameters (all flat, length <= n_rounds * len_list)
        ----------
        exposure_factual_hat: e_hat_k(x_i, a_i) — exposure of the *realized*
            ranking (denominator).
        expected_exposure_eval_hat: e_bar_hat^pi_k(x_i, a_i(k)) — conditional
            expected exposure under the evaluation policy (numerator).
        position_weight: optional alpha (len_list,); defaults to alpha_k = 1.
        """
        check_latent_exposure_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=exposure_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
        )
        exposure_denominator = clip_exposure_denominator(
            exposure_factual_hat=exposure_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            exposure_ratio_clip=self.exposure_ratio_clip,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_item_position * exposure_denominator,
            evaluation_policy_pscore=evaluation_policy_pscore_item_position
            * expected_exposure_eval_hat,
        ) * position_weight_for(position, position_weight)
        # slate count from slate_id, as in the OBP estimators
        return estimated_rewards.sum() / np.unique(slate_id).shape[0]

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        exposure_factual_hat: np.ndarray,
        expected_exposure_eval_hat: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Slate-level bootstrap confidence interval (base-class machinery)."""
        check_latent_exposure_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=exposure_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
        )
        exposure_denominator = clip_exposure_denominator(
            exposure_factual_hat=exposure_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            exposure_ratio_clip=self.exposure_ratio_clip,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_item_position * exposure_denominator,
            evaluation_policy_pscore=evaluation_policy_pscore_item_position
            * expected_exposure_eval_hat,
        ) * position_weight_for(position, position_weight)
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateExposureDecomposedDR(BaseSlateOffPolicyEstimator):
    """ED-DR (Exposure-Decomposed Doubly Robust), the main proposal.

    Per-record contribution (theory note eq. (ED-DR), slot decomposition):

        alpha_k * [ w_hat_{ik} * (Y_{ik} - e_hat_k r_hat_k) + dm_slot_values_{ik} ]

    where w_hat is the exposure-corrected IIPS weight and dm_slot_values is
    the slot decomposition of E_{a~pi}[sum_k alpha-free e_hat_k(x,a) r_hat(x,a(k))]
    (alpha is applied here, not inside the MC).
    Skeleton: `SlateCascadeDoublyRobust._estimate_round_rewards` (L615) with
    "cascade weights + Q-hat" replaced by "corrected IIPS weights + e_hat r_hat".
    """

    len_list: int
    estimator_name: str = "ed-dr"
    exposure_ratio_clip: Optional[float] = None

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        exposure_factual_hat: np.ndarray,
        relevance_factual_hat: np.ndarray,
        expected_exposure_eval_hat: np.ndarray,
        dm_slot_values: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """DR terms per record: importance_weight * (Y - q_hat_factual) + DM, weighted by alpha."""
        # the DR residual baseline keeps the raw e_hat; only the weight
        # denominator is floored (clip binds -> smaller weight, bounded bias)
        q_hat_factual = exposure_factual_hat * relevance_factual_hat
        exposure_denominator = clip_exposure_denominator(
            exposure_factual_hat=exposure_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            exposure_ratio_clip=self.exposure_ratio_clip,
        )
        importance_weight = (evaluation_policy_pscore_item_position * expected_exposure_eval_hat) / (
            pscore_item_position * exposure_denominator
        )
        estimated_rewards = importance_weight * (reward - q_hat_factual) + dm_slot_values
        return estimated_rewards * position_weight_for(position, position_weight)

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        exposure_factual_hat: np.ndarray,
        relevance_factual_hat: np.ndarray,
        expected_exposure_eval_hat: np.ndarray,
        dm_slot_values: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        check_exposure_decomposed_dr_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=exposure_factual_hat,
            relevance_factual_hat=relevance_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
        )
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                pscore_item_position=pscore_item_position,
                evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
                exposure_factual_hat=exposure_factual_hat,
                relevance_factual_hat=relevance_factual_hat,
                expected_exposure_eval_hat=expected_exposure_eval_hat,
                dm_slot_values=dm_slot_values,
                position_weight=position_weight,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        exposure_factual_hat: np.ndarray,
        relevance_factual_hat: np.ndarray,
        expected_exposure_eval_hat: np.ndarray,
        dm_slot_values: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        check_exposure_decomposed_dr_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=exposure_factual_hat,
            relevance_factual_hat=relevance_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=exposure_factual_hat,
            relevance_factual_hat=relevance_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
            position_weight=position_weight,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_slate_confidence_interval_by_bootstrap(
        self,
        slate_id: np.ndarray,
        estimated_rewards: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Slate-level bootstrap (copied from SlateCascadeDoublyRobust L860)."""
        unique_slate = np.unique(slate_id)
        estimated_round_rewards = list()
        for slate in unique_slate:
            estimated_round_rewards.append(estimated_rewards[slate_id == slate].sum())
        estimated_round_rewards = np.array(estimated_round_rewards)
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateIndependentDR(SlateExposureDecomposedDR):
    """Vanilla DR with IIPS weights + direct click regression (the
    "normal DR" baseline; not in OBP).

    Implemented by *reusing the ED-DR code with different arguments*:
    exposure_factual_hat = expected_exposure_eval_hat = 1 and
    relevance_factual_hat = q_hat (the direct click regression), which reduces
    the ablation "decomposed vs direct reward model" to a difference of inputs.
    """

    estimator_name: str = "dr-iips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        q_hat_factual: np.ndarray,
        dm_slot_values_direct: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        unit_nuisance = np.ones_like(reward, dtype=float)
        return super().estimate_policy_value(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=unit_nuisance,
            relevance_factual_hat=q_hat_factual,
            expected_exposure_eval_hat=unit_nuisance,
            dm_slot_values=dm_slot_values_direct,
            position_weight=position_weight,
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        q_hat_factual: np.ndarray,
        dm_slot_values_direct: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        unit_nuisance = np.ones_like(reward, dtype=float)
        return super().estimate_interval(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=unit_nuisance,
            relevance_factual_hat=q_hat_factual,
            expected_exposure_eval_hat=unit_nuisance,
            dm_slot_values=dm_slot_values_direct,
            position_weight=position_weight,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateDirectMethod(BaseSlateOffPolicyEstimator):
    """DM baseline: V_hat = (1/n) sum_i sum_k alpha_k E_{a~pi}[q_hat_k(x_i, a(k))].

    `dm_slot_values_direct` is the slot decomposition of the inner expectation,
    computed with the same MC samples as the proposed estimators (shared in
    `run_eval_policy_monte_carlo`).
    """

    len_list: int
    estimator_name: str = "dm"

    def _estimate_round_rewards(
        self,
        position: np.ndarray,
        dm_slot_values_direct: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        return dm_slot_values_direct * position_weight_for(position, position_weight)

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        position: np.ndarray,
        dm_slot_values_direct: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        check_array(array=slate_id, name="slate_id", expected_dim=1)
        check_array(
            array=dm_slot_values_direct, name="dm_slot_values_direct", expected_dim=1
        )
        if dm_slot_values_direct.shape[0] != slate_id.shape[0]:
            raise ValueError(
                "`dm_slot_values_direct` and `slate_id` must have the same length"
            )
        return (
            self._estimate_round_rewards(
                position=position,
                dm_slot_values_direct=dm_slot_values_direct,
                position_weight=position_weight,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        position: np.ndarray,
        dm_slot_values_direct: np.ndarray,
        position_weight: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_rewards = self._estimate_round_rewards(
            position=position,
            dm_slot_values_direct=dm_slot_values_direct,
            position_weight=position_weight,
        )
        unique_slate = np.unique(slate_id)
        estimated_round_rewards = np.array(
            [estimated_rewards[slate_id == s].sum() for s in unique_slate]
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateAdaptiveIPS(BaseSlateInverseProbabilityWeighting):
    """AIPS (Kiyohara et al., KDD 2023); port of kdd2023-aips.

    IPS with per-record pscores conditioned on the user behavior model chosen
    by the `UserBehaviorTree` (see `aips.py` for the tree, the subset pscores
    and the deviations from the reference implementation). Under DCG-type
    position weights the caller passes alpha-weighted rewards, exactly like
    the unmodified OBP baselines (equivalent reformulation).
    """

    estimator_name: str = "aips"

    @staticmethod
    def _check_inputs(
        reward: np.ndarray,
        pscore_given_user_behavior_model: np.ndarray,
        evaluation_policy_pscore_given_user_behavior_model: np.ndarray,
    ) -> None:
        for name, arr in (
            ("pscore_given_user_behavior_model", pscore_given_user_behavior_model),
            (
                "evaluation_policy_pscore_given_user_behavior_model",
                evaluation_policy_pscore_given_user_behavior_model,
            ),
        ):
            check_array(array=arr, name=name, expected_dim=1)
            if arr.shape[0] != reward.shape[0]:
                raise ValueError(f"`{name}` must have the same length as `reward`")
        if np.any(pscore_given_user_behavior_model <= 0.0):
            raise ValueError(
                "`pscore_given_user_behavior_model` must be strictly positive"
            )

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_given_user_behavior_model: np.ndarray,
        evaluation_policy_pscore_given_user_behavior_model: np.ndarray,
        **kwargs,
    ) -> float:
        self._check_inputs(
            reward=reward,
            pscore_given_user_behavior_model=pscore_given_user_behavior_model,
            evaluation_policy_pscore_given_user_behavior_model=evaluation_policy_pscore_given_user_behavior_model,
        )
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore_given_user_behavior_model,
                evaluation_policy_pscore=evaluation_policy_pscore_given_user_behavior_model,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_given_user_behavior_model: np.ndarray,
        evaluation_policy_pscore_given_user_behavior_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        self._check_inputs(
            reward=reward,
            pscore_given_user_behavior_model=pscore_given_user_behavior_model,
            evaluation_policy_pscore_given_user_behavior_model=evaluation_policy_pscore_given_user_behavior_model,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_given_user_behavior_model,
            evaluation_policy_pscore=evaluation_policy_pscore_given_user_behavior_model,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
