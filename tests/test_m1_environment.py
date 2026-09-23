# M1: environment reproduction.
# Acceptance: obp installed (0.5.x, slate modules identical to master 8cbd5fa),
# and the OBP quickstart-style run of SIPS / IIPS / RIPS on click_model="pbm"
# is unbiased within the Monte-Carlo confidence interval.
import numpy as np
import pytest
from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticSlateBanditDataset
from obp.ope import SlateIndependentIPS
from obp.ope import SlateRewardInteractionIPS
from obp.ope import SlateStandardIPS

from conftest import DGP_SEED
from meta_slate import SlateOffPolicyEvaluation
from utils import scaled_linear_behavior_policy


def test_obp_version_and_slate_api():
    import obp

    assert obp.__version__.startswith("0.5.")
    # the classes / methods the upstream line references in src/ point at
    from obp.ope.estimators_slate import BaseSlateInverseProbabilityWeighting
    from obp.ope.estimators_slate import BaseSlateOffPolicyEstimator  # noqa: F401
    from obp.ope import SlateCascadeDoublyRobust  # noqa: F401
    from obp.ope import SlateRegressionModel  # noqa: F401

    assert hasattr(BaseSlateInverseProbabilityWeighting, "_estimate_round_rewards")
    ds_methods = [
        "obtain_batch_bandit_feedback",
        "sample_action_and_obtain_pscore",
        "sample_reward_given_expected_reward",
        "calc_ground_truth_policy_value",
        "generate_evaluation_policy_pscore",
        "obtain_pscore_given_evaluation_policy_logit",
        "calc_evaluation_policy_action_dist",
    ]
    for m in ds_methods:
        assert hasattr(SyntheticSlateBanditDataset, m), m


def _pbm_quickstart_bias(n_rounds: int, n_reps: int, seed0: int = 0):
    """SIPS/IIPS/RIPS on the parent dataset with click_model='pbm' against a
    uniform-random evaluation policy (whose pscores OBP provides exactly)."""
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=4,
        len_list=3,
        dim_context=2,
        reward_type="binary",
        reward_structure="independent",
        click_model="pbm",
        eta=1.0,
        base_reward_function=logistic_reward_function,
        behavior_policy_function=scaled_linear_behavior_policy(1.0),
        random_state=DGP_SEED,
    )
    errors = {"sips": [], "iips": [], "rips": []}
    for rep in range(n_reps):
        dataset.random_ = np.random.RandomState(seed0 + rep)
        bf = dataset.obtain_batch_bandit_feedback(
            n_rounds=n_rounds, return_pscore_item_position=True
        )
        pscore, pscore_item_pos, pscore_cascade = (
            dataset.generate_evaluation_policy_pscore(
                evaluation_policy_type="random", context=bf["context"]
            )
        )
        # ground truth of the uniform-random PL policy: uniform logits
        uniform_logits = np.zeros((bf["n_rounds"], dataset.n_unique_action))
        ground_truth = dataset.calc_ground_truth_policy_value(
            context=bf["context"], evaluation_policy_logit_=uniform_logits
        )
        ope = SlateOffPolicyEvaluation(
            bandit_feedback=bf,
            ope_estimators=[
                SlateStandardIPS(len_list=3),
                SlateIndependentIPS(len_list=3),
                SlateRewardInteractionIPS(len_list=3),
            ],
        )
        values = ope.estimate_policy_values(
            evaluation_policy_pscore=pscore,
            evaluation_policy_pscore_item_position=pscore_item_pos,
            evaluation_policy_pscore_cascade=pscore_cascade,
        )
        for k in errors:
            errors[k].append(values[k] - ground_truth)
    return {k: np.asarray(v) for k, v in errors.items()}


def _assert_unbiased(errors: np.ndarray, name: str, z: float = 3.5):
    se = errors.std(ddof=1) / np.sqrt(len(errors))
    assert abs(errors.mean()) < z * se + 1e-12, (
        f"{name}: mean bias {errors.mean():.4g} exceeds {z} x SE {se:.4g}"
    )


def test_m1_quickstart_unbiasedness_fast():
    errors = _pbm_quickstart_bias(n_rounds=400, n_reps=30)
    for name, err in errors.items():
        _assert_unbiased(err, name)


@pytest.mark.slow
def test_m1_quickstart_unbiasedness_full():
    """Plan acceptance: click_model='pbm', M=100 — the three estimators'
    biases sit at 0 within the confidence interval."""
    errors = _pbm_quickstart_bias(n_rounds=1000, n_reps=100)
    for name, err in errors.items():
        _assert_unbiased(err, name, z=3.0)
