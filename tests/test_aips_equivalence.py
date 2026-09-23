# AIPS port validity: with the user
# behavior structure FORCED to standard / cascade / independent, the AIPS
# subset pscores and the resulting estimate must match OBP's SIPS / RIPS /
# IIPS exactly (machine precision). This pins down the equivalence that was
# verified by hand, so future edits to `aips.py`
# or the pscore paths cannot silently break the port.
#
# Candidate index convention (aips.candidate_reward_structures):
#   0 = standard, 1 = cascade_neighbor_1, 2 = cascade, 3 = independent.
import numpy as np
import pytest
from obp.ope import SlateIndependentIPS
from obp.ope import SlateRewardInteractionIPS
from obp.ope import SlateStandardIPS

from aips import candidate_reward_structures
from aips import epsilon_greedy_pscore_given_structures
from aips import pl_pscore_given_structures
from estimators_slate import SlateAdaptiveIPS

from conftest import make_epsilon_greedy_policy
from conftest import make_exposure_dataset

N_ROUNDS = 400
EPSILON = 0.2


@pytest.fixture(scope="module")
def logged():
    """One logged dataset in the default experiment regime (m=10, K=5,
    ranking_dependent) plus the structure-wise AIPS pscores of both policies."""
    dataset = make_exposure_dataset(
        exposure_structure="ranking_dependent",
        n_unique_action=10,
        len_list=5,
        dim_context=5,
        attention_spillover=1.0,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=N_ROUNDS, return_pscore_item_position=True
    )
    n = bandit_feedback["n_rounds"]
    len_list = dataset.len_list
    action = bandit_feedback["action"]
    action_2d = action.reshape((n, len_list))

    behavior_logits = dataset.behavior_policy_function(
        context=bandit_feedback["context"],
        action_context=dataset.action_context,
        random_state=dataset.random_state,
    )
    policy = make_epsilon_greedy_policy(
        dataset, bandit_feedback["context"], epsilon=EPSILON
    )
    eval_pscore, eval_pscore_item_position, eval_pscore_cascade = policy.pscores(
        action
    )

    structures = candidate_reward_structures(len_list)
    pscore_all_b = pl_pscore_given_structures(
        behavior_logits,
        action_2d,
        structures,
        bandit_feedback["pscore_item_position"].reshape((n, len_list)),
    )
    pscore_all_e = epsilon_greedy_pscore_given_structures(
        greedy_ranking=policy.greedy_ranking,
        action_2d=action_2d,
        epsilon=EPSILON,
        n_unique_action=dataset.n_unique_action,
        structures=structures,
    )
    return dict(
        dataset=dataset,
        bandit_feedback=bandit_feedback,
        len_list=len_list,
        pscore_all_b=pscore_all_b,
        pscore_all_e=pscore_all_e,
        eval_pscore=eval_pscore,
        eval_pscore_item_position=eval_pscore_item_position,
        eval_pscore_cascade=eval_pscore_cascade,
    )


# (candidate index, OBP behavior pscore key) — cascade_neighbor_1 has no OBP
# counterpart by construction, hence only 0 / 2 / 3
STRUCTURE_TO_OBP = [
    (0, "pscore"),
    (2, "pscore_cascade"),
    (3, "pscore_item_position"),
]


@pytest.mark.parametrize("candidate, obp_key", STRUCTURE_TO_OBP)
def test_behavior_pscores_match_obp(logged, candidate, obp_key):
    """Structure-forced behavior-side subset pscores == OBP's logged pscores."""
    np.testing.assert_allclose(
        logged["pscore_all_b"][candidate].flatten(),
        logged["bandit_feedback"][obp_key],
        rtol=1e-10,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "candidate, eval_key",
    [
        (0, "eval_pscore"),
        (2, "eval_pscore_cascade"),
        (3, "eval_pscore_item_position"),
    ],
)
def test_evaluation_pscores_match_obp_convention(logged, candidate, eval_key):
    """Structure-forced eval-side subset pscores == the epsilon-greedy pscores
    of `EpsilonGreedyEvaluationPolicy.pscores` (OBP's mixture convention).
    The closed form is exact, so the match must be to machine precision."""
    np.testing.assert_allclose(
        logged["pscore_all_e"][candidate].flatten(),
        logged[eval_key],
        rtol=1e-12,
        atol=0.0,
    )


def test_structure_forced_aips_equals_sips_rips_iips(logged):
    """SlateAdaptiveIPS with a forced structure == the matching OBP estimator."""
    bf = logged["bandit_feedback"]
    len_list = logged["len_list"]
    common = dict(
        slate_id=bf["slate_id"], reward=bf["reward"], position=bf["position"]
    )
    obp_values = {
        "pscore": SlateStandardIPS(len_list=len_list).estimate_policy_value(
            pscore=bf["pscore"],
            evaluation_policy_pscore=logged["eval_pscore"],
            **common,
        ),
        "pscore_cascade": SlateRewardInteractionIPS(
            len_list=len_list
        ).estimate_policy_value(
            pscore_cascade=bf["pscore_cascade"],
            evaluation_policy_pscore_cascade=logged["eval_pscore_cascade"],
            **common,
        ),
        "pscore_item_position": SlateIndependentIPS(
            len_list=len_list
        ).estimate_policy_value(
            pscore_item_position=bf["pscore_item_position"],
            evaluation_policy_pscore_item_position=logged[
                "eval_pscore_item_position"
            ],
            **common,
        ),
    }
    aips = SlateAdaptiveIPS(len_list=len_list)
    for candidate, obp_key in STRUCTURE_TO_OBP:
        value = aips.estimate_policy_value(
            pscore_given_user_behavior_model=logged["pscore_all_b"][
                candidate
            ].flatten(),
            evaluation_policy_pscore_given_user_behavior_model=logged[
                "pscore_all_e"
            ][candidate].flatten(),
            **common,
        )
        np.testing.assert_allclose(
            value, obp_values[obp_key], rtol=1e-10, atol=0.0,
            err_msg=f"candidate {candidate} vs OBP {obp_key}",
        )
