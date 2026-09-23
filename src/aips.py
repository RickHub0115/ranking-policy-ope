# This file is a port, with modifications, of code from kdd2023-aips
# (Kiyohara et al., KDD 2023), https://github.com/aiueola/kdd2023-aips
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). The deviations from the
# reference implementation are described in the module docstring below and
# listed in the NOTICE file at the repository root.
"""AIPS baseline (Kiyohara et al., KDD 2023) ported for the eddr benchmark.

Port of kdd2023-aips (https://github.com/aiueola/kdd2023-aips): the
`UserBehaviorTree` that adaptively selects a slot-dependence structure per
context x position, and the subset pscores it needs. The `SlateAdaptiveIPS`
estimator itself lives in `estimators_slate.py` next to the other estimators.

Deviations from the reference implementation (kept for the reproducibility
appendix):

1. EXACT subset pscores. The reference computes every subset pscore with a
   sequential-deletion formula (`calc_pscore_given_reward_structure`) that is
   exact only for prefix sets under Plackett-Luce; its own IIPS pscores use
   the same approximation (`calc_pscore_of_basic_user_behavior_assumption`),
   so their benchmark is internally consistent. This benchmark gives IIPS the
   exact positional marginals (OBP enumeration), so AIPS must get exact
   pscores too or it carries avoidable bias. We therefore (i) restrict the
   candidate structures to rows that are prefixes {0..j} (sequential formula
   exact) or singletons {k} (exact positional marginal: reuses the logged
   `pscore_item_position`; Monte-Carlo marginalization for fresh bootstrap
   draws), and (ii) drop the reference's non-prefix "neighbor"/"random"
   candidates. The `independent` candidate then reproduces IIPS's weights
   exactly, so the tree's hypothesis class CONTAINS IIPS.
2. Subset pscores under the EVALUATION policy use the exact closed form of
   this repo's slate-level epsilon-greedy mixture
   (`(1 - eps) * 1{match on the set} + eps / P(m, |set|)`); the reference's
   per-draw greedy form does not apply to a slate-level mixture. A PL
   evaluation policy (`pl_softmax`) uses the same treatment as the logging
   policy.
3. The tree consumes pre-drawn bootstrap datasets and a pre-computed
   position-wise oracle value; the reference draws both inside the tree from
   the dataset object (`obtain_batch_bandit_feedback`, ground-truth calls).
4. Candidate behavior structures are K-generic (the reference's fixed list
   hard-codes len_list = 8); the last candidate must be the lowest-variance
   one (smallest dependence sets) because the reference clips all variance
   estimates from below by the variance of the LAST candidate.
5. The published experiments run the in-sample `fit_predict` (their
   `cross_fitting` helper is unused there); we do the same.

Like the reference, the tree's split criterion has ORACLE access: the true
position-wise value of the evaluation policy (bias term of the MSE criterion)
and fresh draws from the environment (bootstrap datasets), with `noise_level`
controlling the relative corruption of the bias oracle (0.3 in the published
config). AIPS is therefore a favorably idealized baseline; comparisons
against it are conservative for the proposed estimators.

Known behavior in THIS benchmark (disclosed in the paper): under m=10 / K=5 with non-factorizable rewards, the prefix-type
structures the tree favors carry heavy-tailed weights whose risk the in-sample
variance term underestimates, so AIPS systematically underestimates the policy
value and loses to IIPS (n=400 x 16 seeds: bias -0.37, MSE 0.159 vs iips
0.021) — a harder regime than the published m=3 / K=8 experiments, not a
porting bug (structure-forced pscores match OBP's sips/rips/iips exactly).
Use the per-run `aips_mean_structure` diagnostic when reporting. Also, with
the published min_samples_leaf=100 the tree cannot split at n=100, so the
small-n end of fig1 shows a global (non-adaptive) structure choice.
"""
from dataclasses import dataclass
from dataclasses import field
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.utils import check_random_state


# ---------------------------------------------------------------------------
# candidate behavior structures (K x K binary; row k = the set of slots the
# reward at slot k is assumed to depend on)
# ---------------------------------------------------------------------------
def candidate_reward_structures(len_list: int) -> List[np.ndarray]:
    """Ordered [standard, cascade_neighbor_1, cascade, independent] with
    decreasing dependence-set size, so the LAST candidate has the smallest
    importance weights (see module docstring, deviation 4). Every row is a
    prefix {0..j} or a singleton {k}, keeping all subset pscores exact
    (deviation 1)."""
    standard = np.ones((len_list, len_list))
    cascade = np.tril(np.ones((len_list, len_list)))
    independent = np.eye(len_list)
    cascade_neighbor_1 = cascade.copy()  # row k -> prefix {0..k+1}
    for pos_ in range(len_list - 1):
        cascade_neighbor_1[pos_, pos_ + 1] = 1.0
    return [standard, cascade_neighbor_1, cascade, independent]


# ---------------------------------------------------------------------------
# subset pscores
# ---------------------------------------------------------------------------
def mc_positional_marginal_pscore(
    policy_logit_: np.ndarray,
    action_2d: np.ndarray,
    n_samples: int,
    random_: np.random.RandomState,
) -> np.ndarray:
    """Monte-Carlo estimate of the positional marginals pi_k(a_i(k) | x_i) of
    a Plackett-Luce policy (Gumbel top-K draws; Laplace-smoothed so no weight
    is divided by zero). Same construction as `main._mc_marginal_pscore`, used
    for the tree's bootstrap datasets where the exact enumerated marginals of
    the logged sample are not available. Returns shape (n, len_list)."""
    n, len_list = action_2d.shape
    match_count = np.zeros((n, len_list))
    for _ in range(n_samples):
        gumbel = random_.gumbel(size=policy_logit_.shape)
        ranking = np.argsort(-(policy_logit_ + gumbel), axis=1)[:, :len_list]
        match_count += ranking == action_2d
    m = policy_logit_.shape[1]
    return (match_count + 1.0) / (n_samples + m)


def pl_pscore_given_structures(
    policy_logit_: np.ndarray,
    action_2d: np.ndarray,
    structures: List[np.ndarray],
    pscore_item_position: np.ndarray,  # (n, len_list) exact/MC marginals
) -> np.ndarray:
    """Exact subset pscores of a Plackett-Luce policy for prefix / singleton
    dependence sets (module docstring, deviation 1).

    Prefix set C = {0, ..., c-1} (row k of a structure): the sequential
    without-replacement softmax product is the exact marginal

        p = prod_j exp(z_{a(j)}) / (Z - sum_{j' < j} exp(z_{a(j')})).

    Singleton set C = {k}: the positional marginal pi_k(a(k) | x), supplied
    via `pscore_item_position` (OBP's exact enumeration for the logged
    sample; `mc_positional_marginal_pscore` for bootstrap draws).
    Returns shape (n_structures, n_rounds, len_list).
    """
    n, len_list = action_2d.shape
    expz = np.exp(policy_logit_ - policy_logit_.max(axis=1, keepdims=True))
    z_total = expz.sum(axis=1)  # (n,)
    rows = np.arange(n)
    # prefix probabilities for every prefix length c = 1..K, shared across
    # structures: prefix_p[:, c-1] = P(items a(0..c-1) at slots 0..c-1)
    chosen_all = expz[rows[:, None], action_2d]  # (n, K)
    removed_before = np.concatenate(
        [np.zeros((n, 1)), np.cumsum(chosen_all, axis=1)[:, :-1]], axis=1
    )
    denom = np.clip(z_total[:, None] - removed_before, 1e-300, None)
    prefix_p = np.clip(np.cumprod(chosen_all / denom, axis=1), 1e-300, None)

    pscore = np.ones((len(structures), n, len_list))
    for c_idx, structure in enumerate(structures):
        for pos_ in range(len_list):
            in_set = np.where(structure[pos_] > 0)[0]  # ascending
            if len(in_set) == 1:  # singleton {k}
                pscore[c_idx, :, pos_] = np.clip(
                    pscore_item_position[:, in_set[0]], 1e-300, None
                )
            elif np.array_equal(in_set, np.arange(len(in_set))):  # prefix
                pscore[c_idx, :, pos_] = prefix_p[:, len(in_set) - 1]
            else:
                raise ValueError(
                    "dependence sets must be prefixes or singletons for exact "
                    f"PL pscores; got {in_set} (structure {c_idx}, row {pos_})"
                )
    return pscore


def epsilon_greedy_pscore_given_structures(
    greedy_ranking: np.ndarray,
    action_2d: np.ndarray,
    epsilon: float,
    n_unique_action: int,
    structures: List[np.ndarray],
) -> np.ndarray:
    """Exact subset pscores of the slate-level epsilon-greedy mixture.

    P(items observed at set C) = (1 - eps) * 1{match on C} + eps / P(m, |C|)
    (a uniform K-permutation puts |C| given distinct items at |C| given slots
    with probability 1 / (m (m-1) ... (m - |C| + 1))).
    Returns shape (n_structures, n_rounds, len_list).
    """
    n, len_list = action_2d.shape
    match = greedy_ranking == action_2d  # (n, K)
    m = n_unique_action
    # 1 / P(m, c) for c = 0..K
    inv_perm = np.ones(len_list + 1)
    for c in range(1, len_list + 1):
        inv_perm[c] = inv_perm[c - 1] / (m - c + 1)
    pscore = np.ones((len(structures), n, len_list))
    for c_idx, structure in enumerate(structures):
        for pos_ in range(len_list):
            in_set = np.where(structure[pos_] > 0)[0]
            match_on_set = match[:, in_set].all(axis=1)
            pscore[c_idx, :, pos_] = (
                (1.0 - epsilon) * match_on_set + epsilon * inv_perm[len(in_set)]
            )
    return pscore


def evaluation_policy_pscore_given_structures(
    policy,
    action_2d: np.ndarray,
    n_unique_action: int,
    structures: List[np.ndarray],
    pscore_item_position: Optional[np.ndarray] = None,  # (n, K); PL policies
    pscore_mc_samples: int = 1000,
    random_: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Dispatch on the evaluation-policy type (duck-typed to avoid importing
    the policy classes: epsilon-greedy has `greedy_ranking`, PL has `logits`).
    For a PL policy the singleton rows need positional marginals: pass the
    exact ones if available, otherwise they are MC-estimated here."""
    if hasattr(policy, "greedy_ranking"):
        return epsilon_greedy_pscore_given_structures(
            greedy_ranking=policy.greedy_ranking,
            action_2d=action_2d,
            epsilon=float(policy.epsilon),
            n_unique_action=n_unique_action,
            structures=structures,
        )
    if hasattr(policy, "logits"):
        if pscore_item_position is None:
            pscore_item_position = mc_positional_marginal_pscore(
                policy.logits,
                action_2d,
                n_samples=pscore_mc_samples,
                random_=random_ if random_ is not None else check_random_state(0),
            )
        return pl_pscore_given_structures(
            policy.logits, action_2d, structures, pscore_item_position
        )
    raise TypeError(f"unsupported evaluation policy: {type(policy)!r}")


# ---------------------------------------------------------------------------
# user behavior tree (adapted port of kdd2023-aips tree.py)
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    node_id: int
    user_behavior: int
    depth: int
    sample_id: Optional[np.ndarray] = None
    bootstrap_sample_ids: Optional[List[np.ndarray]] = None


@dataclass
class UserBehaviorTree:
    """Context-partitioning tree that picks, per leaf x position, the candidate
    dependence structure minimizing an MSE proxy = (oracle bias + noise)^2 +
    empirical variance. Faithful port of the reference algorithm; see the
    module docstring for what is supplied from outside."""

    len_list: int
    n_candidates: int
    position_wise_value: np.ndarray  # (len_list,) oracle per-position value
    n_partition: int = 10
    min_samples_leaf: int = 100
    max_depth: Optional[int] = 5
    noise_level: float = 0.3
    random_state: Optional[int] = None
    decision_boundary: List[Dict] = field(init=False)

    def __post_init__(self) -> None:
        self.decision_boundary = [dict() for _ in range(self.len_list)]
        self._max_depth = np.inf if self.max_depth is None else self.max_depth
        self.random_ = check_random_state(self.random_state)

    # -- public API ---------------------------------------------------------
    def fit_predict(
        self,
        context: np.ndarray,  # (n, d)
        importance_weight: np.ndarray,  # (n_candidates, n, len_list)
        reward: np.ndarray,  # (n, len_list)
        position: int,
        bootstrap_data: List[Dict],  # dicts w/ context, importance_weight, reward
    ) -> np.ndarray:
        self.fit(
            context=context,
            importance_weight=importance_weight,
            reward=reward,
            position=position,
            bootstrap_data=bootstrap_data,
        )
        return self.train_reward_structure

    def fit(
        self,
        context: np.ndarray,
        importance_weight: np.ndarray,
        reward: np.ndarray,
        position: int,
        bootstrap_data: List[Dict],
    ) -> None:
        self.train_n_samples = context.shape[0]
        self.n_bootstrap = len(bootstrap_data)
        # per-position slices; bootstrap weighted rewards start from the
        # root's structure and are updated as nodes are assigned
        self.bootstrap_dataset = [
            dict(
                context=b["context"],
                reward=b["reward"][:, position],
                importance_weight=b["importance_weight"][:, :, position],
                weighted_reward=np.zeros(b["context"].shape[0]),
            )
            for b in bootstrap_data
        ]
        self.train_context = context
        self.train_importance_weight = importance_weight[:, :, position]
        self.train_reward = reward[:, position]
        self.train_reward_structure = np.zeros(self.train_n_samples, dtype=int)
        self.train_weighted_reward = np.zeros(self.train_n_samples)

        base_user_behavior = self._select_base_user_behavior(position)

        initial_node = _Node(
            node_id=0,
            sample_id=np.arange(self.train_n_samples),
            user_behavior=base_user_behavior,
            depth=0,
            bootstrap_sample_ids=[
                np.arange(b["context"].shape[0]) for b in self.bootstrap_dataset
            ],
        )
        self.decision_boundary[position] = {
            0: dict(
                parent_user_behavior=base_user_behavior,
                split_exist=False,
                feature_dim=None,
                feature_value=None,
            )
        }
        self._update_global_pscore(initial_node)

        node_queue = deque([initial_node])
        node_id = 0
        while len(node_queue):
            parent_node = node_queue.pop()
            split_exist, split_outcome = self._search_split(
                parent_node=parent_node, position=position
            )
            if split_exist:
                (
                    left_sample_id,
                    right_sample_id,
                    left_user_behavior,
                    right_user_behavior,
                    left_sample_ids,
                    right_sample_ids,
                ) = split_outcome
                left_node = _Node(
                    node_id=node_id + 1,
                    sample_id=left_sample_id,
                    user_behavior=left_user_behavior,
                    depth=parent_node.depth + 1,
                    bootstrap_sample_ids=left_sample_ids,
                )
                right_node = _Node(
                    node_id=node_id + 2,
                    sample_id=right_sample_id,
                    user_behavior=right_user_behavior,
                    depth=parent_node.depth + 1,
                    bootstrap_sample_ids=right_sample_ids,
                )
                for child in (left_node, right_node):
                    self.decision_boundary[position][child.node_id] = dict(
                        parent_user_behavior=child.user_behavior,
                        split_exist=False,
                        feature_dim=None,
                        feature_value=None,
                    )
                    self._update_global_pscore(child)
                    node_queue.append(child)
                node_id += 2

    def predict(self, context: np.ndarray, position: int) -> np.ndarray:
        n = context.shape[0]
        user_behavior = np.zeros(n, dtype=int)
        boundary = self.decision_boundary[position]
        initial_node = _Node(
            node_id=0,
            sample_id=np.arange(n),
            user_behavior=boundary[0]["parent_user_behavior"],
            depth=0,
        )
        node_queue = deque([initial_node])
        node_id = 0
        while len(node_queue):
            parent_node = node_queue.pop()
            decision = boundary[parent_node.node_id]
            if decision["split_exist"]:
                dim, value = decision["feature_dim"], decision["feature_value"]
                sample_id = parent_node.sample_id
                left = sample_id[context[sample_id, dim] < value]
                right = sample_id[context[sample_id, dim] >= value]
                for child_id, child_sample_id in (
                    (node_id + 1, left),
                    (node_id + 2, right),
                ):
                    node_queue.append(
                        _Node(
                            node_id=child_id,
                            sample_id=child_sample_id,
                            user_behavior=boundary[child_id][
                                "parent_user_behavior"
                            ],
                            depth=parent_node.depth + 1,
                        )
                    )
                node_id += 2
            else:
                user_behavior[parent_node.sample_id] = parent_node.user_behavior
        return user_behavior

    # -- internals (reference algorithm) -------------------------------------
    def _select_base_user_behavior(self, position: int) -> int:
        estimate = np.zeros((self.n_candidates, self.n_bootstrap))
        for i, b in enumerate(self.bootstrap_dataset):
            for user_behavior in range(self.n_candidates):
                estimate[user_behavior, i] = (
                    b["importance_weight"][user_behavior] * b["reward"]
                ).mean()
        bias = estimate.mean(axis=1) - self.position_wise_value[position]
        bias = self.random_.normal(loc=bias, scale=np.abs(bias) * self.noise_level)

        variance = np.zeros(self.n_candidates)
        for user_behavior in range(self.n_candidates):
            est = self.train_importance_weight[user_behavior] * self.train_reward
            variance[user_behavior] = est.var(ddof=1) / self.train_n_samples
        # variance floor of the reference: the last candidate has the smallest
        # dependence sets, hence the smallest weights / variance
        self.minimum_variance = variance[-1]
        return int((bias**2 + variance).argmin())

    def _calc_mse_global(self, position: int) -> float:
        estimate = np.array(
            [b["weighted_reward"].mean() for b in self.bootstrap_dataset]
        )
        bias = estimate.mean() - self.position_wise_value[position]
        bias = float(
            self.random_.normal(loc=bias, scale=np.abs(bias) * self.noise_level)
        )
        variance = self.train_weighted_reward.var(ddof=1) / self.train_n_samples
        variance = max(variance, self.minimum_variance)
        return bias**2 + variance

    def _update_global_pscore(self, node: _Node) -> None:
        user_behavior = node.user_behavior
        self.train_reward_structure[node.sample_id] = user_behavior
        self.train_weighted_reward[node.sample_id] = (
            self.train_importance_weight[user_behavior, node.sample_id]
            * self.train_reward[node.sample_id]
        )
        for i, b in enumerate(self.bootstrap_dataset):
            sample_id = node.bootstrap_sample_ids[i]
            b["weighted_reward"][sample_id] = (
                b["importance_weight"][user_behavior, sample_id]
                * b["reward"][sample_id]
            )

    def _search_split(self, parent_node: _Node, position: int):
        parent_sample_id = parent_node.sample_id
        n_parent_samples = len(parent_sample_id)
        if n_parent_samples < 2 * self.min_samples_leaf:
            return False, None
        if parent_node.depth >= self._max_depth:
            return False, None
        parent_context = self.train_context[parent_sample_id]
        min_left_proportion = self.min_samples_leaf / n_parent_samples

        best_mse = self._calc_mse_global(position)
        best = None

        split_feature_dims = self.random_.choice(
            self.train_context.shape[1], size=self.n_partition, replace=True
        )
        split_left_proportions = self.random_.uniform(
            min_left_proportion, 1 - min_left_proportion, size=self.n_partition
        )
        order = np.argsort(split_feature_dims)
        split_feature_dims = split_feature_dims[order]
        split_left_proportions = split_left_proportions[order]

        sorted_sample_id = None
        for i in range(self.n_partition):
            feature_dim = split_feature_dims[i]
            if i == 0 or split_feature_dims[i] != split_feature_dims[i - 1]:
                sorted_sample_id = parent_sample_id[
                    np.argsort(parent_context[:, feature_dim])
                ]
            split_id = int(split_left_proportions[i] * n_parent_samples)
            left_sample_id = sorted_sample_id[:split_id]
            right_sample_id = sorted_sample_id[split_id:]
            feature_value = (
                self.train_context[left_sample_id[-1], feature_dim]
                + self.train_context[right_sample_id[0], feature_dim]
            ) / 2
            (
                left_user_behavior,
                right_user_behavior,
                left_sample_ids,
                right_sample_ids,
                split_mse,
            ) = self._find_best_user_behavior(
                position=position,
                left_sample_id=left_sample_id,
                right_sample_id=right_sample_id,
                parent_sample_ids=parent_node.bootstrap_sample_ids,
                split_feature_dim=feature_dim,
                split_feature_value=feature_value,
            )
            if split_mse <= best_mse:
                best_mse = split_mse
                best = dict(
                    feature_dim=feature_dim,
                    feature_value=feature_value,
                    left_sample_id=left_sample_id,
                    right_sample_id=right_sample_id,
                    left_sample_ids=left_sample_ids,
                    right_sample_ids=right_sample_ids,
                    left_user_behavior=left_user_behavior,
                    right_user_behavior=right_user_behavior,
                )

        if best is None:
            return False, None
        self.decision_boundary[position][parent_node.node_id] = dict(
            parent_user_behavior=parent_node.user_behavior,
            split_exist=True,
            feature_dim=best["feature_dim"],
            feature_value=best["feature_value"],
        )
        return True, (
            best["left_sample_id"],
            best["right_sample_id"],
            best["left_user_behavior"],
            best["right_user_behavior"],
            best["left_sample_ids"],
            best["right_sample_ids"],
        )

    def _find_best_user_behavior(
        self,
        position: int,
        left_sample_id: np.ndarray,
        right_sample_id: np.ndarray,
        parent_sample_ids: List[np.ndarray],
        split_feature_dim: int,
        split_feature_value: float,
    ):
        C = self.n_candidates
        estimate = np.zeros((C, C, self.n_bootstrap))
        left_sample_ids, right_sample_ids = [], []
        for i, b in enumerate(self.bootstrap_dataset):
            sample_id = parent_sample_ids[i]
            context = b["context"][sample_id]
            left_id_ = sample_id[
                context[:, split_feature_dim] < split_feature_value
            ]
            right_id_ = sample_id[
                context[:, split_feature_dim] >= split_feature_value
            ]
            left_sample_ids.append(left_id_)
            right_sample_ids.append(right_id_)

            weighted_reward_ = b["weighted_reward"].copy()
            for left_ub in range(C):
                weighted_reward_[left_id_] = (
                    b["importance_weight"][left_ub, left_id_] * b["reward"][left_id_]
                )
                for right_ub in range(C):
                    weighted_reward_[right_id_] = (
                        b["importance_weight"][right_ub, right_id_]
                        * b["reward"][right_id_]
                    )
                    estimate[left_ub, right_ub, i] = weighted_reward_.mean()

        bias = estimate.mean(axis=2) - self.position_wise_value[position]
        bias = self.random_.normal(loc=bias, scale=np.abs(bias) * self.noise_level)

        variance = np.zeros((C, C))
        weighted_reward_ = self.train_weighted_reward.copy()
        for left_ub in range(C):
            weighted_reward_[left_sample_id] = (
                self.train_importance_weight[left_ub, left_sample_id]
                * self.train_reward[left_sample_id]
            )
            for right_ub in range(C):
                weighted_reward_[right_sample_id] = (
                    self.train_importance_weight[right_ub, right_sample_id]
                    * self.train_reward[right_sample_id]
                )
                variance[left_ub, right_ub] = (
                    weighted_reward_.var(ddof=1) / self.train_n_samples
                )
        variance = np.clip(variance, self.minimum_variance, None)

        mse = bias**2 + variance
        best_left_ub, best_right_ub = np.unravel_index(mse.argmin(), mse.shape)
        return (
            int(best_left_ub),
            int(best_right_ub),
            left_sample_ids,
            right_sample_ids,
            float(mse.min()),
        )


# ---------------------------------------------------------------------------
# orchestration: nuisance fitting + per-record pscore assembly
# ---------------------------------------------------------------------------
def fit_aips_pscores(
    *,
    context: np.ndarray,  # (n, d)
    action_2d: np.ndarray,  # (n, K)
    reward_2d: np.ndarray,  # (n, K) RAW rewards (unweighted)
    behavior_logits: np.ndarray,  # (n, m)
    behavior_pscore_item_position: np.ndarray,  # (n, K) exact marginals (OBP)
    evaluation_policy,  # duck-typed; see evaluation_policy_pscore_given_structures
    evaluation_policy_pscore_item_position: Optional[np.ndarray] = None,  # (n, K)
    # exact eval marginals for PL policies (epsilon-greedy ignores them);
    # None -> MC fallback inside evaluation_policy_pscore_given_structures
    position_wise_value: np.ndarray,  # (K,) oracle per-position value
    bootstrap_sampler: Callable,  # rng -> (context, action_2d, reward_2d, logits, policy)
    n_unique_action: int,
    n_partition: int = 10,
    min_samples_leaf: int = 100,
    max_depth: Optional[int] = 5,
    n_bootstrap: int = 10,
    noise_level: float = 0.3,
    pscore_mc_samples: int = 1000,
    random_state: Optional[int] = None,
):
    """Fit the user behavior tree and return the AIPS per-record pscores.

    Returns (behavior_pscore, evaluation_pscore, chosen_structure), the first
    two flat of shape (n * K,), the last (n, K) candidate indices.
    """
    n, len_list = action_2d.shape
    rng = check_random_state(random_state)
    structures = candidate_reward_structures(len_list)

    pscore_all_b = pl_pscore_given_structures(
        behavior_logits, action_2d, structures, behavior_pscore_item_position
    )
    pscore_all_e = evaluation_policy_pscore_given_structures(
        evaluation_policy,
        action_2d,
        n_unique_action,
        structures,
        pscore_item_position=evaluation_policy_pscore_item_position,
        pscore_mc_samples=pscore_mc_samples,
        random_=rng,
    )
    importance_weight = pscore_all_e / pscore_all_b  # (C, n, K)

    bootstrap_data = []
    for _ in range(n_bootstrap):
        ctx_b, act_b, y_b, logits_b, policy_b = bootstrap_sampler(rng)
        marginal_b = mc_positional_marginal_pscore(
            logits_b, act_b, n_samples=pscore_mc_samples, random_=rng
        )
        pb = pl_pscore_given_structures(logits_b, act_b, structures, marginal_b)
        pe = evaluation_policy_pscore_given_structures(
            policy_b,
            act_b,
            n_unique_action,
            structures,
            pscore_mc_samples=pscore_mc_samples,
            random_=rng,
        )
        bootstrap_data.append(
            dict(context=ctx_b, reward=y_b, importance_weight=pe / pb)
        )

    tree = UserBehaviorTree(
        len_list=len_list,
        n_candidates=len(structures),
        position_wise_value=position_wise_value,
        n_partition=n_partition,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        noise_level=noise_level,
        # offset so the tree's stream (split noise) never replays the draws of
        # `rng` above (bootstrap / MC pscores), which starts from the same seed.
        # +50_000 rather than +1: the caller derives random_state from the
        # replication seed (main.py: seed + 600_000), so +1 would make seed s's
        # tree stream identical to seed s+1's `rng` stream; a large offset keeps
        # the streams disjoint across replications (any n_seeds <= 50_000)
        random_state=random_state if random_state is None else random_state + 50_000,
    )
    chosen = np.zeros((n, len_list), dtype=int)
    for position in range(len_list):
        chosen[:, position] = tree.fit_predict(
            context=context,
            importance_weight=importance_weight,
            reward=reward_2d,
            position=position,
            bootstrap_data=bootstrap_data,
        )

    rows = np.arange(n)[:, None]
    cols = np.arange(len_list)[None, :]
    behavior_pscore = pscore_all_b[chosen, rows, cols].flatten()
    evaluation_pscore = pscore_all_e[chosen, rows, cols].flatten()
    return behavior_pscore, evaluation_pscore, chosen
