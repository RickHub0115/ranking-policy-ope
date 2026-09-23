# This file is a modified copy of obp/ope/meta_slate.py from Open Bandit
# Pipeline (zr-obp) v0.5.5, https://github.com/st-tech/zr-obp
# Copyright 2020 Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc.
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). The modifications are
# described below and listed in the NOTICE file at the repository root.
#
# Off-Policy Evaluation meta class for slate/ranking policies.
#
# Copied from obp/ope/meta_slate.py (obp==0.5.5, identical to zr-obp master
# 8cbd5fa) and extended: the estimator-input
# routing (`_create_estimator_inputs`, L136 in the original) additionally
# carries the exposure-decomposition nuisances
#
#   exposure_factual_hat, relevance_factual_hat, expected_exposure_eval_hat,
#   dm_slot_values, q_hat_factual, dm_slot_values_direct, position_weight
#
# Every estimator accepts **kwargs, so extra keys are harmless for the OBP
# estimators (SIPS/IIPS/RIPS/Cascade-DR) — the same design AIPS used for
# `pscore_given_user_behavior_model` (kdd2023-aips src/meta_slate.py L128-230).
# `estimate_policy_values` / `estimate_intervals` /
# `evaluate_performance_of_estimators` / summaries only pass the new arguments
# through.
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns
from sklearn.utils import check_scalar

from obp.types import BanditFeedback
from obp.utils import check_confidence_interval_arguments
from obp.ope.estimators_slate import BaseSlateOffPolicyEstimator
from obp.ope.estimators_slate import SlateCascadeDoublyRobust as CascadeDR


logger = getLogger(__name__)


@dataclass
class SlateOffPolicyEvaluation:
    """Class to conduct slate OPE with multiple estimators simultaneously.

    Parameters
    -----------
    bandit_feedback: BanditFeedback
        Logged bandit data used in OPE of slate/ranking policies.

    ope_estimators: List[BaseSlateOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of the
        evaluation policy. Estimators must follow the interface of
        `obp.ope.BaseSlateOffPolicyEstimator`.
    """

    bandit_feedback: BanditFeedback
    ope_estimators: List[BaseSlateOffPolicyEstimator]

    def __post_init__(self) -> None:
        """Initialize class."""
        for key_ in [
            "slate_id",
            "context",
            "action",
            "reward",
            "position",
        ]:
            if key_ not in self.bandit_feedback:
                raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")

        self.ope_estimators_ = dict()
        self.use_cascade_dr = False
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator
            if isinstance(estimator, CascadeDR):
                self.use_cascade_dr = True

    def _create_estimator_inputs(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        exposure_factual_hat: Optional[np.ndarray] = None,
        relevance_factual_hat: Optional[np.ndarray] = None,
        expected_exposure_eval_hat: Optional[np.ndarray] = None,
        dm_slot_values: Optional[np.ndarray] = None,
        q_hat_factual: Optional[np.ndarray] = None,
        dm_slot_values_direct: Optional[np.ndarray] = None,
        position_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Create input dictionary to estimate policy value using subclasses of
        `BaseSlateOffPolicyEstimator` (extended routing)."""
        if (
            evaluation_policy_pscore is None
            and evaluation_policy_pscore_item_position is None
            and evaluation_policy_pscore_cascade is None
        ):
            raise ValueError(
                "one of `evaluation_policy_pscore`, `evaluation_policy_pscore_item_position`, or `evaluation_policy_pscore_cascade` must be given"
            )
        if self.use_cascade_dr and evaluation_policy_action_dist is None:
            raise ValueError(
                "`evaluation_policy_action_dist` must be given when using `SlateCascadeDoublyRobust`"
            )
        if self.use_cascade_dr and q_hat is None:
            raise ValueError(
                "`q_hat` must be given when using `SlateCascadeDoublyRobust`"
            )

        estimator_inputs = {
            input_: self.bandit_feedback[input_]
            for input_ in [
                "slate_id",
                "action",
                "reward",
                "position",
                "pscore",
                "pscore_item_position",
                "pscore_cascade",
            ]
            if input_ in self.bandit_feedback
        }
        estimator_inputs["evaluation_policy_pscore"] = evaluation_policy_pscore
        estimator_inputs[
            "evaluation_policy_pscore_item_position"
        ] = evaluation_policy_pscore_item_position
        estimator_inputs[
            "evaluation_policy_pscore_cascade"
        ] = evaluation_policy_pscore_cascade
        estimator_inputs[
            "evaluation_policy_action_dist"
        ] = evaluation_policy_action_dist
        estimator_inputs["q_hat"] = q_hat
        # ---- exposure-decomposition nuisances (new) ----
        estimator_inputs["exposure_factual_hat"] = exposure_factual_hat
        estimator_inputs["relevance_factual_hat"] = relevance_factual_hat
        estimator_inputs["expected_exposure_eval_hat"] = expected_exposure_eval_hat
        estimator_inputs["dm_slot_values"] = dm_slot_values
        estimator_inputs["q_hat_factual"] = q_hat_factual
        estimator_inputs["dm_slot_values_direct"] = dm_slot_values_direct
        estimator_inputs["position_weight"] = position_weight

        return estimator_inputs

    def estimate_policy_values(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        exposure_factual_hat: Optional[np.ndarray] = None,
        relevance_factual_hat: Optional[np.ndarray] = None,
        expected_exposure_eval_hat: Optional[np.ndarray] = None,
        dm_slot_values: Optional[np.ndarray] = None,
        q_hat_factual: Optional[np.ndarray] = None,
        dm_slot_values_direct: Optional[np.ndarray] = None,
        position_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Estimate the policy value of the evaluation policy with all estimators.

        Returns
        ----------
        policy_value_dict: Dict[str, float]
            Dictionary containing the policy values estimated by each estimator.
        """
        policy_value_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
            exposure_factual_hat=exposure_factual_hat,
            relevance_factual_hat=relevance_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
            q_hat_factual=q_hat_factual,
            dm_slot_values_direct=dm_slot_values_direct,
            position_weight=position_weight,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_dict[estimator_name] = estimator.estimate_policy_value(
                **estimator_inputs
            )

        return policy_value_dict

    def estimate_intervals(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        exposure_factual_hat: Optional[np.ndarray] = None,
        relevance_factual_hat: Optional[np.ndarray] = None,
        expected_exposure_eval_hat: Optional[np.ndarray] = None,
        dm_slot_values: Optional[np.ndarray] = None,
        q_hat_factual: Optional[np.ndarray] = None,
        dm_slot_values_direct: Optional[np.ndarray] = None,
        position_weight: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate confidence intervals of policy values using bootstrap."""
        check_confidence_interval_arguments(
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        policy_value_interval_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
            exposure_factual_hat=exposure_factual_hat,
            relevance_factual_hat=relevance_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
            q_hat_factual=q_hat_factual,
            dm_slot_values_direct=dm_slot_values_direct,
            position_weight=position_weight,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_interval_dict[estimator_name] = estimator.estimate_interval(
                **estimator_inputs,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        exposure_factual_hat: Optional[np.ndarray] = None,
        relevance_factual_hat: Optional[np.ndarray] = None,
        expected_exposure_eval_hat: Optional[np.ndarray] = None,
        dm_slot_values: Optional[np.ndarray] = None,
        q_hat_factual: Optional[np.ndarray] = None,
        dm_slot_values_direct: Optional[np.ndarray] = None,
        position_weight: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Summarize policy values and their confidence intervals."""
        common_kwargs = dict(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
            exposure_factual_hat=exposure_factual_hat,
            relevance_factual_hat=relevance_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
            q_hat_factual=q_hat_factual,
            dm_slot_values_direct=dm_slot_values_direct,
            position_weight=position_weight,
        )
        policy_value_df = DataFrame(
            self.estimate_policy_values(**common_kwargs),
            index=["estimated_policy_value"],
        )
        policy_value_interval_df = DataFrame(
            self.estimate_intervals(
                **common_kwargs,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
        )
        policy_value_of_behavior_policy = (
            self.bandit_feedback["reward"].sum()
            / np.unique(self.bandit_feedback["slate_id"]).shape[0]
        )
        policy_value_df = policy_value_df.T
        if policy_value_of_behavior_policy <= 0:
            logger.warning(
                f"Policy value of the behavior policy is {policy_value_of_behavior_policy} (<=0); relative estimated policy value is set to np.nan"
            )
            policy_value_df["relative_estimated_policy_value"] = np.nan
        else:
            policy_value_df["relative_estimated_policy_value"] = (
                policy_value_df.estimated_policy_value / policy_value_of_behavior_policy
            )
        return policy_value_df, policy_value_interval_df.T

    def visualize_off_policy_estimates(
        self,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        exposure_factual_hat: Optional[np.ndarray] = None,
        relevance_factual_hat: Optional[np.ndarray] = None,
        expected_exposure_eval_hat: Optional[np.ndarray] = None,
        dm_slot_values: Optional[np.ndarray] = None,
        q_hat_factual: Optional[np.ndarray] = None,
        dm_slot_values_direct: Optional[np.ndarray] = None,
        position_weight: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize the estimated policy values (bar plot with bootstrap CI)."""
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "`fig_dir` must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "`fig_dir` must be a string"

        _, estimated_interval_a = self.summarize_off_policy_estimates(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
            exposure_factual_hat=exposure_factual_hat,
            relevance_factual_hat=relevance_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
            q_hat_factual=q_hat_factual,
            dm_slot_values_direct=dm_slot_values_direct,
            position_weight=position_weight,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        estimated_interval_a["errbar_length"] = (
            estimated_interval_a.drop("mean", axis=1).diff(axis=1).iloc[:, -1].abs()
        )
        if is_relative:
            estimated_interval_a /= (
                self.bandit_feedback["reward"].sum()
                / np.unique(self.bandit_feedback["slate_id"]).shape[0]
            )

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=estimated_interval_a[["mean"]].reset_index(),
            x="index",
            y="mean",
            ax=ax,
            ci=None,
        )
        plt.xlabel("OPE Estimators", fontsize=25)
        plt.ylabel(
            f"Estimated Policy Value (± {np.int32(100*(1 - alpha))}% CI)", fontsize=20
        )
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=25 - 2 * len(self.ope_estimators))
        ax.errorbar(
            np.arange(estimated_interval_a.shape[0]),
            estimated_interval_a["mean"],
            yerr=estimated_interval_a["errbar_length"],
            fmt="o",
            color="black",
        )

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def evaluate_performance_of_estimators(
        self,
        ground_truth_policy_value: float,
        evaluation_policy_pscore: Optional[np.ndarray] = None,
        evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,
        evaluation_policy_pscore_cascade: Optional[np.ndarray] = None,
        evaluation_policy_action_dist: Optional[np.ndarray] = None,
        q_hat: Optional[np.ndarray] = None,
        exposure_factual_hat: Optional[np.ndarray] = None,
        relevance_factual_hat: Optional[np.ndarray] = None,
        expected_exposure_eval_hat: Optional[np.ndarray] = None,
        dm_slot_values: Optional[np.ndarray] = None,
        q_hat_factual: Optional[np.ndarray] = None,
        dm_slot_values_direct: Optional[np.ndarray] = None,
        position_weight: Optional[np.ndarray] = None,
        metric: str = "se",
    ) -> Dict[str, float]:
        """Evaluate the accuracy of OPE estimators with relative-EE or SE."""
        check_scalar(ground_truth_policy_value, "ground_truth_policy_value", float)
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"`metric` must be either 'relative-ee' or 'se', but {metric} is given"
            )
        if metric == "relative-ee" and ground_truth_policy_value == 0.0:
            raise ValueError(
                "`ground_truth_policy_value` must be non-zero when metric is relative-ee"
            )

        eval_metric_ope_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
            exposure_factual_hat=exposure_factual_hat,
            relevance_factual_hat=relevance_factual_hat,
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
            q_hat_factual=q_hat_factual,
            dm_slot_values_direct=dm_slot_values_direct,
            position_weight=position_weight,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_policy_value = estimator.estimate_policy_value(**estimator_inputs)
            if metric == "relative-ee":
                relative_ee_ = estimated_policy_value - ground_truth_policy_value
                relative_ee_ /= ground_truth_policy_value
                eval_metric_ope_dict[estimator_name] = np.abs(relative_ee_)
            elif metric == "se":
                se_ = (estimated_policy_value - ground_truth_policy_value) ** 2
                eval_metric_ope_dict[estimator_name] = se_
        return eval_metric_ope_dict

    def summarize_estimators_comparison(
        self,
        ground_truth_policy_value: float,
        metric: str = "se",
        **kwargs,
    ) -> DataFrame:
        """Summarize the performance comparison among OPE estimators."""
        eval_metric_ope_df = DataFrame(
            self.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth_policy_value,
                metric=metric,
                **kwargs,
            ),
            index=[metric],
        )
        return eval_metric_ope_df.T
