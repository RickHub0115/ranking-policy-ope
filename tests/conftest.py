# Shared fixtures / helpers for the tests.
#
# Convention: every milestone has a fast variant (runs on `pytest`, minutes)
# and, where the acceptance criteria demand scale (M=100..500,
# n=64000), a `@pytest.mark.slow` variant with the exact thresholds
# (run with `pytest -m slow`).
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from obp.dataset import logistic_reward_function  # noqa: E402

from dataset import ExposureSlateBanditDataset  # noqa: E402
from exposure_model import OracleNuisance  # noqa: E402
from exposure_model import run_eval_policy_monte_carlo  # noqa: E402
from utils import EpsilonGreedyEvaluationPolicy  # noqa: E402
from utils import greedy_ranking_from_scores  # noqa: E402
from utils import sample_ranking_from_logits  # noqa: E402
from utils import scaled_linear_behavior_policy  # noqa: E402

DGP_SEED = 12345


def make_exposure_dataset(
    exposure_structure="pbm",
    n_unique_action=6,
    len_list=3,
    dim_context=3,
    exposure_decay_rate=1.0,
    attention_spillover=0.0,
    tau0=1.0,
    random_state=DGP_SEED,
    **kwargs,
) -> ExposureSlateBanditDataset:
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
    )
    params.update(kwargs)
    return ExposureSlateBanditDataset(**params)


def make_epsilon_greedy_policy(dataset, context, epsilon=0.3):
    base_reward = dataset.base_expected_reward(context)
    greedy = greedy_ranking_from_scores(base_reward, dataset.len_list, optimal=True)
    return EpsilonGreedyEvaluationPolicy(
        greedy_ranking=greedy,
        epsilon=epsilon,
        n_unique_action=dataset.n_unique_action,
    )


def compute_oracle_inputs(
    dataset,
    bandit_feedback,
    policy,
    n_mc_samples=100,
    random_state=0,
    corrupt_exposure=None,
    corrupt_relevance=None,
):
    """Oracle nuisance arrays for LE-IIPS / ED-DR (true e, r, e_bar^pi)."""
    n = bandit_feedback["n_rounds"]
    len_list = dataset.len_list
    action = bandit_feedback["action"]
    action_2d = action.reshape((n, len_list))
    context = bandit_feedback["context"]
    oracle = OracleNuisance(
        dataset=dataset,
        corrupt_exposure=corrupt_exposure,
        corrupt_relevance=corrupt_relevance,
    )
    e_bar, dm_slot_values, _ = run_eval_policy_monte_carlo(
        nuisance=oracle,
        context=context,
        action_2d=action_2d,
        sample_ranking_fn=lambda rng: policy.sample_rankings(rng),
        n_mc_samples=n_mc_samples,
        random_state=random_state,
    )
    return dict(
        exposure_factual_hat=oracle.predict_exposure(context, action_2d).flatten(),
        relevance_factual_hat=oracle.predict_relevance(context, action),
        expected_exposure_eval_hat=e_bar,
        dm_slot_values=dm_slot_values,
    )


def simulate_log_fast(dataset, n_rounds, seed):
    """Vectorized log generation (PL Gumbel sampling; no pscores) for tests
    that only need (context, action, reward, position), e.g. the EM tests.
    Distributionally identical to the OBP sequential sampler."""
    rng = np.random.RandomState(seed)
    context = rng.normal(size=(n_rounds, dataset.dim_context))
    logits = dataset.behavior_policy_function(
        context=context,
        action_context=dataset.action_context,
        random_state=dataset.random_state,
    )
    ranking = sample_ranking_from_logits(logits, dataset.len_list, rng)
    relevance_all = dataset.base_expected_reward(context)
    relevance = relevance_all[np.arange(n_rounds)[:, None], ranking]
    exposure_p = dataset.calc_expected_exposure(context, ranking)
    reward, exposure = dataset.sample_exposure_and_reward(exposure_p, relevance, rng)
    position = np.tile(np.arange(dataset.len_list), n_rounds)
    return dict(
        context=context,
        action=ranking.flatten(),
        reward=reward.flatten(),
        position=position,
        exposure=exposure.flatten(),
        ranking=ranking,
        behavior_logits=logits,
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
