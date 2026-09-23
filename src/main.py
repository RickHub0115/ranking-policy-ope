# Semi-synthetic experiment driver.
#
# One Hydra run = one experimental configuration x `n_seeds` Monte-Carlo
# replications ("resample the log with `obtain_batch_bandit_feedback` -> learn
# nuisances -> compute V_hat with all estimators -> compare with the ground
# truth"). Raw per-seed estimates are saved to results.csv (bias^2 / variance
# decomposition happens at aggregation time), and every run writes a
# quick summary figure to figs/.
#
# The full sweeps for Figures 1-5 / Table 1 are Hydra multiruns; see
# run_experiments.sh for the exact commands, e.g.
#
#   python src/main.py -m setting=fig1_data_size \
#       setting.exposure_structure=pbm,ranking_dependent \
#       setting.n_rounds=100,200,400,800,1600,3200,6400 n_seeds=200 n_jobs=8
#
# After the sweeps are done, `python src/main.py mode=plot` scans logs/ and
# renders figs/fig1 ... fig5, table1 and the appendix figures; the aggregate
# CSV tables are written to data/<date>/ instead (see run_plot_mode).
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import hydra
import matplotlib
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from omegaconf import DictConfig
from omegaconf import OmegaConf
from scipy.special import logit as logit_fn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from obp.dataset import logistic_reward_function
from obp.ope import SlateCascadeDoublyRobust
from obp.ope import SlateIndependentIPS
from obp.ope import SlateRegressionModel
from obp.ope import SlateRewardInteractionIPS
from obp.ope import SlateStandardIPS
from obp.ope import SelfNormalizedSlateIndependentIPS
from obp.ope import SelfNormalizedSlateRewardInteractionIPS
from obp.ope import SelfNormalizedSlateStandardIPS

from dataset import CascadeClickBanditDataset
from aips import fit_aips_pscores
from dataset import DBNClickBanditDataset
from dataset import ExposureSlateBanditDataset
from dataset import FrechetCoupledExposureDataset
from estimators_slate import SlateAdaptiveIPS
from estimators_slate import SlateDirectMethod
from estimators_slate import SlateExposureDecomposedDR
from estimators_slate import SlateIndependentDR
from estimators_slate import SlateLatentExposureIPS
from exposure_model import DirectClickRegression
from exposure_model import ExposureRelevanceEM
from exposure_model import fit_nuisances_cross_fitted
from exposure_model import OracleNuisance
from exposure_model import run_eval_policy_monte_carlo
from meta_slate import SlateOffPolicyEvaluation
from utils import apply_position_weight_to_reward
from utils import cascade_expected_slate_reward
from utils import EpsilonGreedyEvaluationPolicy
from utils import greedy_ranking_from_scores
from utils import make_position_weight
from utils import position_weight_for
from utils import sample_ranking_from_logits
from utils import scaled_linear_behavior_policy

warnings.filterwarnings("ignore", category=FutureWarning)

DGP_RANDOM_STATE = 12345  # fixed DGP seed: the reward / behavior-policy /
# attractiveness coefficients define the environment and must NOT change with
# the replication seed; only `dataset.random_` advances per replication
# (the dataset object is built once per replication; only its RNG advances).

BASELINE_ESTIMATORS = ("sips", "iips", "rips", "cascade-dr", "snsips", "sniips", "snrips")
PROPOSED_ESTIMATORS = ("dm", "dr-iips", "le-iips", "ed-dr")
# Retired: SIPS is no longer computed (dropped from
# the estimator list of conf/setting/default.yaml) and the rows the earlier
# cycles wrote for it are removed when the logs are merged in mode=plot
# (load_results_csvs), so the per-setting seed-count check stays even across
# cycles that did and did not run it. The classes stay available for an
# explicit `setting.estimators=[sips,...]`.
RETIRED_ESTIMATORS = ("sips", "snsips")

# ---------------------------------------------------------------------------
# figure styling: fixed method -> color assignment (never cycled), thin marks,
# recessive grid; palette values are the validated categorical slots of the
# dataviz reference palette (light mode).
# ---------------------------------------------------------------------------
METHOD_STYLE: Dict[str, Dict] = {
    "sips": dict(color="#eb6834", ls="-", marker="v"),
    "iips": dict(color="#2a78d6", ls="-", marker="o"),
    "rips": dict(color="#1baf7a", ls="-", marker="s"),
    "cascade-dr": dict(color="#eda100", ls="-", marker="D"),
    "aips": dict(color="#8c564b", ls="-", marker="h"),
    "dm": dict(color="#008300", ls="-", marker="^"),
    "dr-iips": dict(color="#4a3aa7", ls="-", marker="P"),
    "le-iips": dict(color="#e87ba4", ls="-", marker="X"),
    "ed-dr": dict(color="#e34948", ls="-", marker="*"),
    "le-iips (oracle)": dict(color="#e87ba4", ls="--", marker="X"),
    "ed-dr (oracle)": dict(color="#e34948", ls="--", marker="*"),
    "snsips": dict(color="#eb6834", ls=":", marker="v"),
    "sniips": dict(color="#2a78d6", ls=":", marker="o"),
    "snrips": dict(color="#1baf7a", ls=":", marker="s"),
}
METHOD_ORDER = list(METHOD_STYLE.keys())
GRID_COLOR = "#e1e0d9"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#898781"


# ===========================================================================
# construction helpers
# ===========================================================================
def build_dataset(setting: DictConfig):
    """Instantiate the DGP. `dgp="exposure"` is the proposed latent-exposure
    dataset; `dgp="dbn"` is its DBN-satisfaction variant ((A2)-violating,
    O-R noise coupling of strength setting.or_correlation — appendix (A2)
    robustness); `dgp="or_coupled"` is the slot-level (A2)/(A3)-violating
    Frechet-mixture variant (within-slot O-R coupling of strength
    setting.or_coupling; marginals and (A4) preserved — appendix A2');
    `dgp="cascade"` is the parent OBP dataset with a cascade
    click model (the losing setting: slot-wise (A2)/(A3) still hold,
    but position-independence (A4) breaks and the true exposure — the
    expected product of no-clicks above — lies outside every assumed
    exposure class, so e_hat is misspecified)."""
    common = dict(
        n_unique_action=int(setting.n_unique_action),
        len_list=int(setting.len_list),
        dim_context=int(setting.dim_context),
        reward_type="binary",
        reward_structure="independent",
        base_reward_function=logistic_reward_function,
        behavior_policy_function=scaled_linear_behavior_policy(float(setting.tau0)),
        random_state=DGP_RANDOM_STATE,
        is_factorizable=False,
    )
    if setting.dgp in ("exposure", "dbn", "or_coupled"):
        exposure_kwargs = dict(
            click_model=None,
            exposure_structure=str(setting.exposure_structure),
            exposure_decay_rate=float(setting.exposure_decay_rate),
            n_user_segments=int(setting.n_user_segments),
            attention_spillover=float(setting.attention_spillover),
            attract_relevance_align=float(setting.attract_relevance_align),
            relevance_scale=float(setting.relevance_scale),
            return_exposure=True,
            position_weight_kind=str(setting.position_weight),
            **common,
        )
        if setting.dgp == "exposure":
            return ExposureSlateBanditDataset(**exposure_kwargs)
        if setting.dgp == "or_coupled":
            return FrechetCoupledExposureDataset(
                or_coupling=float(setting.or_coupling), **exposure_kwargs
            )
        return DBNClickBanditDataset(
            or_correlation=float(setting.or_correlation), **exposure_kwargs
        )
    elif setting.dgp == "cascade":
        return CascadeClickBanditDataset(
            click_model="cascade",
            eta=float(setting.exposure_decay_rate),
            **common,
        )
    raise ValueError(f"unknown dgp: {setting.dgp}")


class PlackettLuceEvaluationPolicy:
    """Softmax (Plackett-Luce) evaluation policy over logit(r(x, a)) / tau1."""

    def __init__(self, dataset, context: np.ndarray, tau1: float):
        self.dataset = dataset
        base_reward = dataset.base_reward_function(
            context=context,
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
        self.logits = logit_fn(np.clip(base_reward, 1e-6, 1 - 1e-6)) / tau1
        self.len_list = dataset.len_list

    def pscores(self, action: np.ndarray):
        pscore, pscore_item_position, pscore_cascade = (
            self.dataset.obtain_pscore_given_evaluation_policy_logit(
                action=action,
                evaluation_policy_logit_=self.logits,
                return_pscore_item_position=True,
            )
        )
        return pscore, pscore_item_position, pscore_cascade

    def action_dist(self, action: np.ndarray) -> np.ndarray:
        return self.dataset.calc_evaluation_policy_action_dist(
            action=action, evaluation_policy_logit_=self.logits
        )

    def sample_rankings(
        self, random_: np.random.RandomState, idx: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return sample_ranking_from_logits(self.logits, self.len_list, random_, idx=idx)


def build_evaluation_policy(setting: DictConfig, dataset, context: np.ndarray):
    if setting.evaluation_policy.type == "epsilon_greedy":
        base_reward = dataset.base_reward_function(
            context=context,
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
        greedy = greedy_ranking_from_scores(
            base_reward, dataset.len_list, optimal=True
        )
        return EpsilonGreedyEvaluationPolicy(
            greedy_ranking=greedy,
            epsilon=float(setting.evaluation_policy.epsilon),
            n_unique_action=dataset.n_unique_action,
        )
    elif setting.evaluation_policy.type == "pl_softmax":
        return PlackettLuceEvaluationPolicy(
            dataset, context, tau1=float(setting.evaluation_policy.tau1)
        )
    raise ValueError(f"unknown evaluation policy: {setting.evaluation_policy.type}")


def calc_ground_truth(
    setting: DictConfig,
    dataset,
    policy,
    context: np.ndarray,
    position_weight: np.ndarray,
    seed: int,
) -> float:
    """Per-replication ground truth V(pi) on this replication's contexts
    (conditional on contexts, so the context-sampling variance cancels
    between estimators and the target)."""
    gt = setting.ground_truth
    if setting.dgp in ("exposure", "dbn", "or_coupled"):
        # the DBN / Frechet-coupled variants override the slot-value hooks of
        # the dataset, so their exact/MC ground-truth machinery already
        # carries the (A2) violation (closed forms in dataset.py)
        if isinstance(policy, EpsilonGreedyEvaluationPolicy):
            return dataset.calc_ground_truth_policy_value_epsilon_greedy(
                context=context,
                greedy_ranking=policy.greedy_ranking,
                epsilon=policy.epsilon,
                method=str(gt.method),
                n_mc_samples=int(gt.n_mc_samples),
                max_enumeration=int(gt.max_enumeration),
                random_state=seed + 5_000_000,
            )
        return dataset.calc_ground_truth_policy_value(
            context=context,
            evaluation_policy_logit_=policy.logits,
            method=str(gt.method),
            n_mc_samples=int(gt.n_mc_samples),
            max_enumeration=int(gt.max_enumeration),
            random_state=seed + 5_000_000,
        )
    # ---- cascade DGP: exact E[Y_k | x, a] per sampled ranking (MC only over
    # the policy's rankings; the click process itself is integrated exactly by
    # cascade_expected_slate_reward's prefix DP) ----
    rng = check_random_state(seed + 5_000_000)
    n = context.shape[0]
    if isinstance(policy, EpsilonGreedyEvaluationPolicy):
        q_greedy = cascade_expected_slate_reward(dataset, context, policy.greedy_ranking)
        v_greedy = float((position_weight[None, :] * q_greedy).sum(axis=1).mean())
        if policy.epsilon == 0.0:
            return v_greedy
        n_draws = max(1, int(np.ceil(int(gt.n_mc_samples) / n)))
        total = 0.0
        for _ in range(n_draws):
            keys = rng.uniform(size=(n, dataset.n_unique_action))
            ranking = np.argsort(keys, axis=1)[:, : dataset.len_list]
            q = cascade_expected_slate_reward(dataset, context, ranking)
            total += (position_weight[None, :] * q).sum(axis=1).mean()
        return (1 - policy.epsilon) * v_greedy + policy.epsilon * total / n_draws
    n_draws = max(1, int(np.ceil(int(gt.n_mc_samples) / n)))
    total = 0.0
    for _ in range(n_draws):
        ranking = policy.sample_rankings(rng)
        q = cascade_expected_slate_reward(dataset, context, ranking)
        total += (position_weight[None, :] * q).sum(axis=1).mean()
    return float(total / n_draws)


def calc_position_wise_ground_truth(
    setting: DictConfig, dataset, policy, context: np.ndarray, seed: int
) -> np.ndarray:
    """Oracle per-position value of the evaluation policy — the bias reference
    of the AIPS user-behavior tree (kdd2023-aips gives the tree the same
    oracle; see aips.py). Raw per-slot rewards, no alpha weighting: scaling a
    position by alpha_k scales bias and sqrt(variance) alike, so the tree's
    argmin per position is unchanged.

    Averaged over the LOGGED contexts, while the tree's bootstrap estimates
    draw fresh contexts — both unbiased for the same population value (contexts
    are iid N(0, I)), but at small n the reference carries a candidate-common
    MC offset of order 1/sqrt(n). The published implementation's oracle is a
    finite-round MC simulation with the same error scale, so we keep this."""
    rng = check_random_state(seed)
    n = context.shape[0]
    n_draws = int(setting.aips.n_value_mc_draws)

    def slot_values(ranking: np.ndarray) -> np.ndarray:
        if setting.dgp in ("exposure", "dbn", "or_coupled"):
            # e * r for the exposure DGP (identical computation to the old
            # inline product); the DBN / Frechet-coupled hooks adjust it
            return dataset.expected_slot_rewards(context, ranking)
        return cascade_expected_slate_reward(dataset, context, ranking)

    if isinstance(policy, EpsilonGreedyEvaluationPolicy):
        v_greedy = slot_values(policy.greedy_ranking).mean(axis=0)
        if policy.epsilon == 0.0:
            return v_greedy
        total = np.zeros(dataset.len_list)
        for _ in range(n_draws):
            keys = rng.uniform(size=(n, dataset.n_unique_action))
            ranking = np.argsort(keys, axis=1)[:, : dataset.len_list]
            total += slot_values(ranking).mean(axis=0)
        return (1 - policy.epsilon) * v_greedy + policy.epsilon * total / n_draws
    total = np.zeros(dataset.len_list)
    for _ in range(n_draws):
        ranking = policy.sample_rankings(rng)
        total += slot_values(ranking).mean(axis=0)
    return total / n_draws


def make_aips_bootstrap_sampler(setting: DictConfig, dataset, n_rounds: int):
    """Fresh logged datasets for the AIPS tree's bootstrap bias term (the
    reference calls `obtain_batch_bandit_feedback`; we resample the same law
    directly — context ~ N(0, I) as in OBP — skipping its pscore machinery,
    which the tree never uses)."""
    len_list = dataset.len_list

    def sampler(rng: np.random.RandomState):
        ctx = rng.normal(size=(n_rounds, dataset.dim_context))
        logits = dataset.behavior_policy_function(
            context=ctx,
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
        act = sample_ranking_from_logits(logits, len_list, rng)
        rows = np.arange(n_rounds)[:, None]
        if setting.dgp in ("exposure", "dbn", "or_coupled"):
            r = dataset.base_expected_reward(ctx)[rows, act]
            e = dataset.calc_expected_exposure(ctx, act)
            # Bern(e) then Bern(r), same draws/order as the old inline
            # binomials for the exposure DGP; the DBN / Frechet-coupled
            # overrides draw their extra noise after these base draws
            y, _ = dataset.sample_exposure_and_reward(e, r, rng)
        else:
            base_q = dataset.base_expected_reward(ctx)[rows, act]
            # the parent's cascade sampler draws from self.random_
            saved_random = dataset.random_
            dataset.random_ = rng
            try:
                y = np.asarray(
                    dataset.sample_reward_given_expected_reward(
                        expected_reward_factual=base_q.copy()
                    )
                )
            finally:
                dataset.random_ = saved_random
        policy_b = build_evaluation_policy(setting, dataset, ctx)
        return ctx, act, y, logits, policy_b

    return sampler


def make_base_classifier(name: str, random_state: Optional[int] = None):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, C=100.0)
    if name == "gradient_boosting":
        # explicit seed: sklearn's default random_state=None consumes the
        # global numpy RNG, breaking same-seed reproducibility across runs
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=random_state
        )
    raise ValueError(f"unknown classifier: {name}")


def make_corruption(kind: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Controlled nuisance corruptions for the oracle experiments (Figure 3).

    Three channels use these maps (Figure 3):
      - `corrupt_exposure` (METHOD E): the map is handed to
        OracleNuisance.corrupt_exposure, i.e. it corrupts the exposure MODEL
        itself. Because run_eval_policy_monte_carlo and the oracle block both
        read exposures through OracleNuisance.predict_exposure, the SAME
        corrupted model consistently feeds the numerator e_bar^pi (MC), the
        denominator e_hat, and the residual baseline e_hat * r_hat. This is
        the only way to test whether an exposure error cancels in the ratio
        (Corollaries 2a/2b vs the E3 regime where it cannot).
      - `corrupt_exposure_ratio` (METHOD R): the map is applied to e_bar^pi
        ONLY (numerator of the exposure ratio inside the importance weights);
        the factual e_hat, DM term and residual baseline stay true. The ratio
        error is then exactly known — used to measure the bias slope.
      - `corrupt_relevance`: pointwise map of r(x, a(k)), breaking the DM
        term and the residual baseline while the weights stay correct.

    Kinds: "none" | "power" (v^1.5, the historical fixed map kept for the M3
    tests) | "shift" | "power:<t>" with strength t > 0, mapping v -> v^(1+t)
    (t = 0 is NOT accepted: run the zero-strength cell as "none" so it is the
    uncorrupted configuration itself, bit for bit).
    """
    if kind in (None, "none"):
        return None
    if kind == "power":
        return lambda v: np.clip(v, 1e-4, 1.0) ** 1.5
    if kind == "shift":
        return lambda v: np.clip(0.5 * v + 0.25, 1e-4, 1.0)
    if isinstance(kind, str) and kind.startswith("power:"):
        try:
            strength = float(kind.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"malformed corruption strength: {kind!r}")
        if not np.isfinite(strength) or strength <= 0.0:
            raise ValueError(
                f"corruption strength must be a finite positive number, got {kind!r} "
                "(run strength 0 as 'none')"
            )
        return lambda v: np.clip(v, 1e-4, 1.0) ** (1.0 + strength)
    raise ValueError(f"unknown corruption: {kind}")


# ===========================================================================
# one Monte-Carlo replication
# ===========================================================================
def run_one_replicate(cfg: DictConfig, seed: int) -> List[Dict]:
    setting = cfg.setting
    dataset = build_dataset(setting)
    # identical DGP across replications; only the sampling stream advances
    dataset.random_ = check_random_state(seed)
    len_list = dataset.len_list
    n_unique_action = dataset.n_unique_action

    use_mc_pscore = int(setting.pscore_mc_samples) > 0
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=int(setting.n_rounds),
        return_pscore_item_position=not use_mc_pscore,
        clip_logit_value=(
            float(setting.clip_logit_value)
            if setting.clip_logit_value is not None
            else None
        ),
    )
    n_rounds = bandit_feedback["n_rounds"]
    context = bandit_feedback["context"]
    action = bandit_feedback["action"]
    action_2d = action.reshape((n_rounds, len_list))
    position = bandit_feedback["position"]
    reward = bandit_feedback["reward"]

    if use_mc_pscore:
        # marginalize the behavior policy by Monte Carlo instead of the
        # combinatorial enumeration (large m / K)
        behavior_logits = dataset.behavior_policy_function(
            context=context,
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
        bandit_feedback["pscore_item_position"] = _mc_marginal_pscore(
            behavior_logits,
            action_2d,
            len_list,
            n_samples=int(setting.pscore_mc_samples),
            random_state=seed + 9_000_000,
        )

    # ---- evaluation policy ------------------------------------------------
    policy = build_evaluation_policy(setting, dataset, context)
    (
        evaluation_policy_pscore,
        evaluation_policy_pscore_item_position,
        evaluation_policy_pscore_cascade,
    ) = policy.pscores(action)

    position_weight = make_position_weight(len_list, str(setting.position_weight))
    pass_position_weight = (
        position_weight if str(setting.position_weight) != "uniform" else None
    )

    ground_truth = calc_ground_truth(
        setting, dataset, policy, context, position_weight, seed
    )

    estimator_names = list(setting.estimators)
    common_columns = dict(
        seed=seed,
        ground_truth=float(ground_truth),
        behavior_value=float(reward.sum() / n_rounds),
    )
    diagnostics: Dict[str, float] = {}
    results: List[Dict] = []

    # exposure-ratio truncation level for LE-IIPS / ED-DR (default.yaml);
    # "auto" -> sqrt(n * K), null -> disabled, number -> fixed cap.
    # The key is REQUIRED (all setting yamls inherit it from default.yaml): a
    # silent fallback would change the estimator definition on a config typo
    if "exposure_ratio_clip" not in setting:
        raise KeyError(
            "setting.exposure_ratio_clip is required ('auto', null, or a number); "
            "settings must inherit it from default.yaml"
        )
    ratio_clip_cfg = setting.exposure_ratio_clip
    if ratio_clip_cfg is None:
        exposure_ratio_clip = None
    elif isinstance(ratio_clip_cfg, str):
        if ratio_clip_cfg != "auto":
            raise ValueError(
                f"exposure_ratio_clip must be 'auto', null, or a number, got {ratio_clip_cfg!r}"
            )
        exposure_ratio_clip = float(np.sqrt(n_rounds * len_list))
    else:
        exposure_ratio_clip = float(ratio_clip_cfg)

    # ---- baselines (unmodified OBP estimators) -----------------------------
    # under alpha_k != 1 the OBP baselines receive alpha-weighted rewards
    # (equivalent reformulation; see utils.apply_position_weight_to_reward)
    baseline_feedback = dict(bandit_feedback)
    baseline_feedback["reward"] = apply_position_weight_to_reward(
        reward, position, pass_position_weight
    )
    baseline_names = [e for e in estimator_names if e in BASELINE_ESTIMATORS]
    cascade_dr_feasible = (
        "cascade-dr" in baseline_names
        and float(np.min(evaluation_policy_pscore_cascade)) > 0.0
    )
    q_hat = None
    evaluation_policy_action_dist = None
    if cascade_dr_feasible:
        evaluation_policy_action_dist = policy.action_dist(action)
        regression_model = SlateRegressionModel(
            base_model=(
                GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, random_state=seed + 800_000
                )
                if setting.cascade_dr_base_model == "gradient_boosting"
                else Ridge(alpha=1.0)
            ),
            len_list=len_list,
            n_unique_action=n_unique_action,
            fitting_method="normal",  # as in Kiyohara+ 2022
        )
        q_hat = regression_model.fit_predict(
            context=context,
            action=action,
            reward=baseline_feedback["reward"],
            pscore_cascade=bandit_feedback["pscore_cascade"],
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
    baseline_estimator_factory = {
        "sips": lambda: SlateStandardIPS(len_list=len_list),
        "iips": lambda: SlateIndependentIPS(len_list=len_list),
        "rips": lambda: SlateRewardInteractionIPS(len_list=len_list),
        "cascade-dr": lambda: SlateCascadeDoublyRobust(
            len_list=len_list, n_unique_action=n_unique_action
        ),
        "snsips": lambda: SelfNormalizedSlateStandardIPS(len_list=len_list),
        "sniips": lambda: SelfNormalizedSlateIndependentIPS(len_list=len_list),
        "snrips": lambda: SelfNormalizedSlateRewardInteractionIPS(len_list=len_list),
    }
    baseline_estimators = [
        baseline_estimator_factory[name]()
        for name in baseline_names
        if name != "cascade-dr" or cascade_dr_feasible
    ]
    if baseline_estimators:
        ope_baseline = SlateOffPolicyEvaluation(
            bandit_feedback=baseline_feedback, ope_estimators=baseline_estimators
        )
        baseline_values = ope_baseline.estimate_policy_values(
            evaluation_policy_pscore=evaluation_policy_pscore,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            q_hat=q_hat,
        )
    else:
        baseline_values = {}
    if "cascade-dr" in baseline_names and not cascade_dr_feasible:
        baseline_values["cascade-dr"] = np.nan  # deterministic eval policy:
        # SlateRegressionModel requires eval pscore_cascade > 0

    # ---- AIPS baseline (KDD 2023 port; src/aips.py) -------------------------
    if "aips" in estimator_names:
        aips_cfg = setting.aips
        behavior_logits_aips = dataset.behavior_policy_function(
            context=context,
            action_context=dataset.action_context,
            random_state=dataset.random_state,
        )
        position_wise_value = calc_position_wise_ground_truth(
            setting, dataset, policy, context, seed + 400_000
        )
        aips_behavior_pscore, aips_eval_pscore, aips_structure = fit_aips_pscores(
            context=context,
            action_2d=action_2d,
            reward_2d=reward.reshape((n_rounds, len_list)),  # RAW rewards
            behavior_logits=behavior_logits_aips,
            behavior_pscore_item_position=bandit_feedback[
                "pscore_item_position"
            ].reshape((n_rounds, len_list)),
            evaluation_policy=policy,
            evaluation_policy_pscore_item_position=(
                evaluation_policy_pscore_item_position.reshape(
                    (n_rounds, len_list)
                )
            ),
            position_wise_value=position_wise_value,
            bootstrap_sampler=make_aips_bootstrap_sampler(
                setting, dataset, n_rounds
            ),
            n_unique_action=n_unique_action,
            n_partition=int(aips_cfg.n_partition),
            min_samples_leaf=int(aips_cfg.min_samples_leaf),
            max_depth=(
                int(aips_cfg.max_depth) if aips_cfg.max_depth is not None else None
            ),
            n_bootstrap=int(aips_cfg.n_bootstrap),
            noise_level=float(aips_cfg.noise_level),
            pscore_mc_samples=int(aips_cfg.pscore_mc_samples),
            random_state=seed + 600_000,
        )
        baseline_values["aips"] = SlateAdaptiveIPS(
            len_list=len_list
        ).estimate_policy_value(
            slate_id=bandit_feedback["slate_id"],
            reward=baseline_feedback["reward"],  # alpha-weighted like OBP baselines
            position=position,
            pscore_given_user_behavior_model=aips_behavior_pscore,
            evaluation_policy_pscore_given_user_behavior_model=aips_eval_pscore,
        )
        # mean chosen candidate index (0 = standard, 1 = cascade_neighbor_1,
        # 2 = cascade, 3 = independent; aips.candidate_reward_structures);
        # tracks which behavior class the tree infers per configuration
        diagnostics["aips_mean_structure"] = float(aips_structure.mean())

    # ---- proposed estimators with ESTIMATED nuisances ----------------------
    proposed_names = [e for e in estimator_names if e in PROPOSED_ESTIMATORS]
    proposed_values: Dict[str, float] = {}
    if proposed_names:
        em_seed = (
            int(setting.em_random_state)
            if setting.em_random_state is not None
            else seed + 7_000_000
        )

        def em_factory() -> ExposureRelevanceEM:
            return ExposureRelevanceEM(
                len_list=len_list,
                n_unique_action=n_unique_action,
                relevance_model=make_base_classifier(
                    str(setting.relevance_model), random_state=em_seed
                ),
                # +500_000 rather than +1: with em_random_state=null, em_seed
                # derives from the replication seed, so +1 would make seed s's
                # exposure-classifier stream identical to seed (s+1)'s
                # relevance/EM stream (same collision pattern avoided with the
                # +50_000 offset in aips.py). Harmless today — the logistic
                # towers ignore random_state — but keep the streams disjoint.
                exposure_model=make_base_classifier(
                    str(setting.exposure_base_model), random_state=em_seed + 500_000
                ),
                exposure_model_class=str(setting.exposure_model_class),
                n_em_iter=int(setting.n_em_iter),
                tol=float(setting.em_tol),
                warm_start=bool(setting.warm_start),
                monotone_position_tower=bool(setting.monotone_position_tower),
                # explicit init scheme of the init-sensitivity appendix:
                # null keeps the historical behavior. NOT an
                # aggregation column — each non-null value must live under
                # its own setting_name (app_em_init_*), or cells would blend.
                init_scheme=(
                    str(setting.em_init) if setting.em_init is not None else None
                ),
                random_state=em_seed,
            )

        def direct_q_factory() -> DirectClickRegression:
            return DirectClickRegression(
                len_list=len_list,
                n_unique_action=n_unique_action,
                base_model=make_base_classifier(
                    str(setting.relevance_model), random_state=seed + 900_000
                ),
            )

        fitted_nuisances = fit_nuisances_cross_fitted(
            em_factory=em_factory,
            context=context,
            action=action,
            reward=reward,
            position=position,
            n_folds=int(setting.n_folds),
            random_state=seed + 3_000_000,
            direct_q_factory=direct_q_factory,
        )
        # per-fold MC under the evaluation policy, fold model on its own rows
        expected_exposure_eval_hat = np.zeros(n_rounds * len_list)
        dm_slot_values = np.zeros(n_rounds * len_list)
        dm_slot_values_direct = np.zeros(n_rounds * len_list)
        for f, em in enumerate(fitted_nuisances["ems"]):
            rounds_f = np.where(fitted_nuisances["fold_of"] == f)[0]
            records_f = np.repeat(fitted_nuisances["fold_of"] == f, len_list)
            e_bar_f, dm_f, dm_direct_f = run_eval_policy_monte_carlo(
                nuisance=em,
                context=context[rounds_f],
                action_2d=action_2d[rounds_f],
                sample_ranking_fn=lambda rng, idx=rounds_f: policy.sample_rankings(
                    rng, idx=idx
                ),
                n_mc_samples=int(setting.n_mc_samples),
                random_state=seed + 100_000 + f,
                direct_q_model=fitted_nuisances["direct_q_models"][f],
            )
            expected_exposure_eval_hat[records_f] = e_bar_f
            dm_slot_values[records_f] = dm_f
            dm_slot_values_direct[records_f] = dm_direct_f

        proposed_estimator_factory = {
            "dm": lambda: SlateDirectMethod(len_list=len_list),
            "dr-iips": lambda: SlateIndependentDR(len_list=len_list),
            "le-iips": lambda: SlateLatentExposureIPS(
                len_list=len_list, exposure_ratio_clip=exposure_ratio_clip
            ),
            "ed-dr": lambda: SlateExposureDecomposedDR(
                len_list=len_list, exposure_ratio_clip=exposure_ratio_clip
            ),
        }
        ope_proposed = SlateOffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=[proposed_estimator_factory[n]() for n in proposed_names],
        )
        proposed_values = ope_proposed.estimate_policy_values(
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=fitted_nuisances["exposure_factual_hat"],
            relevance_factual_hat=fitted_nuisances["relevance_factual_hat"],
            expected_exposure_eval_hat=expected_exposure_eval_hat,
            dm_slot_values=dm_slot_values,
            q_hat_factual=fitted_nuisances["q_hat_factual"],
            dm_slot_values_direct=dm_slot_values_direct,
            position_weight=pass_position_weight,
        )
        # EM diagnostics
        loglik_histories = [em.likelihood_history_ for em in fitted_nuisances["ems"]]
        diagnostics["em_n_iter"] = float(np.mean([em.n_iter_ for em in fitted_nuisances["ems"]]))
        diagnostics["em_loglik_final"] = float(
            np.mean([h[-1] for h in loglik_histories if len(h) > 0])
        )
        diagnostics["em_loglik_monotone"] = float(
            all(np.all(np.diff(h) >= -1e-8) for h in loglik_histories)
        )
        # how often the exposure-ratio truncation binds on the estimated
        # nuisances (a proxy for degenerate EM fits)
        if exposure_ratio_clip is not None:
            est_ratio = expected_exposure_eval_hat / fitted_nuisances["exposure_factual_hat"]
            diagnostics["exposure_ratio_clip_frac"] = float(
                np.mean(est_ratio > exposure_ratio_clip)
            )
        else:
            diagnostics["exposure_ratio_clip_frac"] = np.nan

    # ---- oracle versions (true e, r, e_bar^pi injected) --------
    # under dgp="cascade" and dgp="dbn" the (x, k, a(k))-measurable marginal
    # that OracleNuisance serves is no longer the true exposure (it depends
    # on the ranking prefix — through the no-click product for cascade, the
    # satisfaction survival for DBN) — both settings ship include_oracle
    # false, and this gate keeps a stray override from injecting a fake
    # oracle. dgp="or_coupled" IS allowed: the coupling preserves the
    # marginals e / r, so the oracle is well-defined — what it cannot repair
    # is the broken product decomposition (A2)/(A3), which is exactly what
    # the appendix-A2' experiment measures.
    oracle_values: Dict[str, float] = {}
    if bool(setting.include_oracle) and setting.dgp in ("exposure", "or_coupled"):
        # Figure 3 weight-side channels: METHOD E corrupts the
        # exposure MODEL (OracleNuisance hook -> numerator, denominator and
        # residual baseline consistently), METHOD R corrupts e_bar^pi only.
        # They answer different questions, so a run must pick one.
        corrupt_exposure_fn = make_corruption(
            str(setting.oracle_corruption.corrupt_exposure)
        )
        corrupt_ratio = make_corruption(
            str(setting.oracle_corruption.corrupt_exposure_ratio)
        )
        if corrupt_exposure_fn is not None and corrupt_ratio is not None:
            raise ValueError(
                "oracle_corruption.corrupt_exposure (method E) and "
                "oracle_corruption.corrupt_exposure_ratio (method R) are both "
                "set: the weight-side channels are mutually exclusive"
            )
        oracle = OracleNuisance(
            dataset=dataset,
            corrupt_relevance=make_corruption(
                str(setting.oracle_corruption.corrupt_relevance)
            ),
            corrupt_exposure=corrupt_exposure_fn,
        )
        e_bar_oracle, dm_oracle, _ = run_eval_policy_monte_carlo(
            nuisance=oracle,
            context=context,
            action_2d=action_2d,
            sample_ranking_fn=lambda rng: policy.sample_rankings(rng),
            n_mc_samples=int(setting.n_mc_samples),
            random_state=seed + 200_000,
        )
        # METHOD R (weight channel): mis-value e_bar^pi, which enters ONLY
        # the importance-weight ratio e_bar^pi / e_hat (see make_corruption
        # for why corrupting e_hat itself cannot bias only the both-wrong
        # cell of the retired 2x2 grid)
        e_bar_before_ratio_map = e_bar_oracle
        if corrupt_ratio is not None:
            e_bar_oracle = np.clip(corrupt_ratio(e_bar_before_ratio_map), 1e-6, 1.0)
        exposure_oracle = oracle.predict_exposure(context, action_2d).flatten()
        relevance_oracle = oracle.predict_relevance(context, action)
        ope_oracle = SlateOffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=[
                SlateLatentExposureIPS(
                    len_list=len_list,
                    estimator_name="le-iips (oracle)",
                    exposure_ratio_clip=exposure_ratio_clip,
                ),
                SlateExposureDecomposedDR(
                    len_list=len_list,
                    estimator_name="ed-dr (oracle)",
                    exposure_ratio_clip=exposure_ratio_clip,
                ),
            ],
        )
        oracle_values = ope_oracle.estimate_policy_values(
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            exposure_factual_hat=exposure_oracle,
            relevance_factual_hat=relevance_oracle,
            expected_exposure_eval_hat=e_bar_oracle,
            dm_slot_values=dm_oracle,
            position_weight=pass_position_weight,
        )

        # true e_bar^pi via the uncorrupted oracle (MC with the true e).
        # Shared by the corrupted-oracle ratio-error diagnostic below and the
        # estimated-nuisance diagnostics; the stream (seed + 300_000) is the
        # one the estimated-nuisance block always used, so pre-existing
        # diagnostics stay bit-identical.
        e_true = bandit_feedback["expected_exposure_factual"]
        r_true = bandit_feedback["expected_relevance_factual"]
        oracle_clean = OracleNuisance(dataset=dataset)
        e_bar_true, _, _ = run_eval_policy_monte_carlo(
            nuisance=oracle_clean,
            context=context,
            action_2d=action_2d,
            sample_ranking_fn=lambda rng: policy.sample_rankings(rng),
            n_mc_samples=int(setting.n_mc_samples),
            random_state=seed + 300_000,
        )
        # corrupted-oracle ratio error (record mean of the weight-ratio
        # error; NEW aggregate column — kept apart from the
        # non-oracle `ratio_error_mean` below):
        #   method R — |f(e_bar) - e_bar| / e_hat on the SAME MC draw, so the
        #     error is exactly known and free of the MC noise floor;
        #   method E / relevance-only / uncorrupted — the corrupted ratio vs
        #     the true ratio from an independent clean MC, so the value
        #     includes the S-sample noise floor. The uncorrupted run's value
        #     IS the measured floor (two-stage level selection).
        if corrupt_ratio is not None:
            oracle_ratio_err = (
                np.abs(e_bar_oracle - e_bar_before_ratio_map) / exposure_oracle
            )
        else:
            oracle_ratio_err = np.abs(
                e_bar_oracle / exposure_oracle - e_bar_true / e_true
            )
        diagnostics["oracle_ratio_error_mean"] = float(oracle_ratio_err.mean())
        # corrupted-relevance error (fig3 condition (e)): mean
        # relative error of the corrupted r on the logged slates. Pointwise
        # and exact (no MC involved); 0 for uncorrupted runs. Used to check
        # the R x relevance calibration (relevance error ~ 0.5 x ratio error)
        # against the calibration target.
        diagnostics["oracle_relevance_error_mean"] = float(
            np.mean(np.abs(relevance_oracle - r_true) / r_true)
        )

        # exposure-recovery diagnostics (only defined when a true
        # exposure exists — NaN for the cascade DGP)
        if proposed_names:
            e_hat = fitted_nuisances["exposure_factual_hat"]
            if np.std(e_hat) > 1e-12 and np.std(e_true) > 1e-12:
                diagnostics["exposure_corr"] = float(
                    np.corrcoef(e_hat, e_true)[0, 1]
                )
            else:
                diagnostics["exposure_corr"] = np.nan
            ratio_hat = expected_exposure_eval_hat / e_hat
            ratio_true = e_bar_true / e_true
            ratio_err = np.abs(ratio_hat - ratio_true)
            diagnostics["ratio_error_mean"] = float(ratio_err.mean())
            # empirical evaluation of the Prop. 3 linear bias bound
            w_iips = (
                evaluation_policy_pscore_item_position
                / bandit_feedback["pscore_item_position"]
            )
            alpha_rec = position_weight_for(position, pass_position_weight)
            diagnostics["prop3_bound"] = float(
                (alpha_rec * w_iips * r_true * ratio_err * e_true).sum() / n_rounds
            )

    estimates = {**baseline_values, **proposed_values, **oracle_values}
    for name, value in estimates.items():
        results.append(
            dict(common_columns, estimator=name, estimate=float(value), **diagnostics)
        )
    return results


def _mc_marginal_pscore(
    logits: np.ndarray,
    action_2d: np.ndarray,
    len_list: int,
    n_samples: int,
    random_state: int,
) -> np.ndarray:
    """Monte-Carlo marginalization of pi_{0,k}(a_i(k) | x_i) for large (m, K)
    where OBP's exact permutation enumeration is infeasible.
    Laplace-smoothed relative frequency of {a_s(k) == a_i(k)} over PL draws.

    Known approximation error: the (match + 1) / (S + m) smoothing
    shrinks pi_hat toward 1/m, and together with the Jensen term of the
    inverse this puts an O(1 / (S * pi)) systematic error into the importance
    weights — about 0.1-0.5% relative at the recommended S = 20000, m = 30.
    Footnote this as a known approximation when interpreting the large-(m, K)
    appendix runs; for exact weights drop the smoothing and clip zeros
    instead."""
    rng = check_random_state(random_state)
    n, _ = action_2d.shape
    match_count = np.zeros((n, len_list))
    for _ in range(n_samples):
        ranking = sample_ranking_from_logits(logits, len_list, rng)
        match_count += ranking == action_2d
    m = logits.shape[1]
    return ((match_count + 1.0) / (n_samples + m)).flatten()


# ===========================================================================
# aggregation & figures
# ===========================================================================
def _exposure_ratio_clip_str(ratio_clip_cfg) -> str:
    """Canonical string form of setting.exposure_ratio_clip for results.csv
    ('auto' / 'none' / the numeric cap). Part of CONFIG_KEYS: without it a
    clipped main run and an unclipped appendix run under the same log_dir
    would collapse into one cell in mode=plot, and drop_duplicates would
    silently replace one with the other."""
    if ratio_clip_cfg is None:
        return "none"
    if isinstance(ratio_clip_cfg, str):
        return str(ratio_clip_cfg)
    return str(float(ratio_clip_cfg))


def _weight_channel_str(setting: DictConfig) -> str:
    """Canonical string of the weight-side oracle corruption for results.csv.

    Both weight-side channels are recorded in the SINGLE pre-existing
    `corrupt_exposure_ratio` column (adding a required config column would
    invalidate every existing results.csv; only or_coupling is backfilled):
    method R keeps its raw kind ('none' / 'power' / 'power:<t>'),
    method E is recorded as 'E:<kind>'. The channels are mutually exclusive
    (also enforced at run time in run_one_replicate)."""
    ce = str(setting.oracle_corruption.corrupt_exposure)
    cr = str(setting.oracle_corruption.corrupt_exposure_ratio)
    if ce != "none" and cr != "none":
        raise ValueError(
            "oracle_corruption.corrupt_exposure and .corrupt_exposure_ratio "
            "are both set: the weight-side channels are mutually exclusive"
        )
    return f"E:{ce}" if ce != "none" else cr


def flatten_setting(setting: DictConfig) -> Dict:
    flat = {
        "setting_name": str(setting.name),
        "dgp": str(setting.dgp),
        "exposure_structure": str(setting.exposure_structure),
        "exposure_decay_rate": float(setting.exposure_decay_rate),
        "attention_spillover": float(setting.attention_spillover),
        "attract_relevance_align": float(setting.attract_relevance_align),
        "or_correlation": float(setting.or_correlation),
        "n_user_segments": int(setting.n_user_segments),
        "n_rounds": int(setting.n_rounds),
        "len_list": int(setting.len_list),
        "n_unique_action": int(setting.n_unique_action),
        "dim_context": int(setting.dim_context),
        "tau0": float(setting.tau0),
        "eval_type": str(setting.evaluation_policy.type),
        "epsilon": float(setting.evaluation_policy.epsilon),
        "tau1": float(setting.evaluation_policy.tau1),
        "exposure_model_class": str(setting.exposure_model_class),
        "relevance_model": str(setting.relevance_model),
        "warm_start": bool(setting.warm_start),
        "monotone_position_tower": bool(setting.monotone_position_tower),
        "n_em_iter": int(setting.n_em_iter),
        "em_random_state": (
            int(setting.em_random_state)
            if setting.em_random_state is not None
            else -1
        ),
        "n_folds": int(setting.n_folds),
        "exposure_ratio_clip": _exposure_ratio_clip_str(setting.exposure_ratio_clip),
        "n_mc_samples": int(setting.n_mc_samples),
        "position_weight": str(setting.position_weight),
        "corrupt_exposure_ratio": _weight_channel_str(setting),
        "corrupt_relevance": str(setting.oracle_corruption.corrupt_relevance),
        # appended last so the pre-existing aggregate columns keep their
        # relative order; old results.csv lack it and are backfilled with 0.0
        # in load_results_csvs
        "or_coupling": float(setting.or_coupling),
        # appended after or_coupling for the same reason; pre-existing runs
        # all used the unscaled relevance, so the load_results_csvs backfill
        # with 1.0 is exact, not a guess
        "relevance_scale": float(setting.relevance_scale),
    }
    return flat


def flatten_setting_defaults() -> Dict:
    """The flat row of conf/setting/default.yaml (mirrors `flatten_setting`);
    used by the milestone tests to synthesize results frames."""
    return {
        "setting_name": "default",
        "dgp": "exposure",
        "exposure_structure": "ranking_dependent",
        "exposure_decay_rate": 1.0,
        "attention_spillover": 1.0,
        "attract_relevance_align": 0.5,
        "or_correlation": 0.0,
        "n_user_segments": 2,
        "n_rounds": 4000,
        "len_list": 5,
        "n_unique_action": 10,
        "dim_context": 5,
        "tau0": 1.0,
        "eval_type": "epsilon_greedy",
        "epsilon": 0.2,
        "tau1": 1.0,
        "exposure_model_class": "ranking_dependent",
        "relevance_model": "logistic_regression",
        "warm_start": True,
        "monotone_position_tower": True,
        "n_em_iter": 20,
        "em_random_state": -1,
        "n_folds": 2,
        "exposure_ratio_clip": "none",
        "n_mc_samples": 100,
        "position_weight": "uniform",
        "corrupt_exposure_ratio": "none",
        "corrupt_relevance": "none",
        "or_coupling": 0.0,
        "relevance_scale": 1.0,
    }


CONFIG_KEYS = [
    "setting_name",
    "dgp",
    "exposure_structure",
    "exposure_decay_rate",
    "attention_spillover",
    "attract_relevance_align",
    "or_correlation",
    "n_rounds",
    "len_list",
    "n_unique_action",
    "tau0",
    "eval_type",
    "epsilon",
    "tau1",
    "exposure_model_class",
    "warm_start",
    "em_random_state",
    "exposure_ratio_clip",
    "n_mc_samples",
    "position_weight",
    "corrupt_exposure_ratio",
    "corrupt_relevance",
    "or_coupling",
    "relevance_scale",
]


CI_Z = 1.96  # 95% normal-approximation interval
# Draw the 95% CI bands / whiskers on the rel-MSE / bias^2 / variance figures?
# Off by default: the paper reports point estimates only. The SE columns
# of aggregate_all.csv are still written, and `mode=plot show_ci=true` puts
# the bands back for the same logs. The signed-bias panels of Figure 3 are a
# different figure (their claim is "the CI covers 0") and keep their CI.
SHOW_CI = False
# figure typography. The manuscripts caption every figure
# themselves, so the in-image "Fig.N: ..." headline is off by default; tick
# labels, axis labels / panel headers and the estimator names (legend entries,
# bar-chart x ticks) are two matplotlib size steps (x1.2 each) larger than the
# original 9 / 10 / 8 / 7-8 pt so they stay legible at \columnwidth.
SHOW_FIG_TITLE = False
TICK_FONTSIZE = 13
LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12
ESTIMATOR_TICK_FONTSIZE = 12


def se_of_mean(values) -> float:
    """Standard error of the sample mean, std(ddof=1) / sqrt(n).

    rel_mse is itself the sample mean of the per-seed squared relative
    errors s^2, so this closed-form SE is what the figure CIs need — no
    resampling. Returns NaN below two values (one seed carries no spread
    information), which downstream plotting renders as "no band"."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return float("nan")
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def se_of_variance(values) -> float:
    """Asymptotic SE of the (ddof=0) sample variance via its influence
    function (x - mu)^2 - sigma^2: SE = sqrt((m4 - var^2) / n) with m4 the
    fourth central moment (no bootstrap needed).
    Returns NaN below two values, which downstream plotting renders as
    "no band"."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return float("nan")
    var = float(np.var(v))
    m4 = float(np.mean((v - v.mean()) ** 4))
    return float(np.sqrt(max(m4 - var**2, 0.0) / v.size))


def ci_bounds(mean, se, z=CI_Z):
    """Multiplicative (log-scale delta-method) CI for a positive quantity:
    the Wald interval is taken on log(rel_mse) and mapped back, i.e.
    mean * exp(-+ z*se/mean). rel_mse is nonnegative and right-skewed, so an
    interval symmetric on the raw scale can put the lower bound at or below 0
    — which the log panels cannot draw and which contradicts the support of
    s^2. This form keeps lo > 0 always, is symmetric on the log axis, and
    agrees with mean -+ z*se to O((se/mean)^2) when the SE is small.

    Cells that cannot be logged (mean <= 0, or a non-finite mean/SE) fall back
    to the linear interval clipped at 0; a NaN SE therefore still yields
    (NaN, NaN), which the plotting code renders as "no band"."""
    mean = np.asarray(mean, dtype=float)
    se = np.asarray(se, dtype=float)
    linear_lo = np.clip(mean - z * se, 0.0, None)
    linear_hi = mean + z * se
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        half = z * se / mean
        log_lo = mean * np.exp(-half)
        log_hi = mean * np.exp(half)
    loggable = np.isfinite(mean) & np.isfinite(se) & (mean > 0.0)
    return (
        np.where(loggable, log_lo, linear_lo),
        np.where(loggable, log_hi, linear_hi),
    )


def bias_sq_ci_bounds(bias, se, z=CI_Z):
    """95% CI for bias^2, obtained by squaring the Wald interval of bias.
    When the bias interval covers 0 the lower
    bound is 0 — on a log axis the band then extends to the panel bottom,
    which is the intended reading ("bias^2 not distinguishable from 0").
    A NaN bias or SE yields (NaN, NaN) = no band."""
    bias = np.asarray(bias, dtype=float)
    se = np.asarray(se, dtype=float)
    lo_b = bias - z * se
    hi_b = bias + z * se
    lo = np.where(lo_b * hi_b <= 0.0, 0.0, np.minimum(lo_b**2, hi_b**2))
    hi = np.maximum(lo_b**2, hi_b**2)
    bad = ~(np.isfinite(bias) & np.isfinite(se))
    return np.where(bad, np.nan, lo), np.where(bad, np.nan, hi)


def _combine_se(se_values) -> float:
    """SE of the unweighted average of k independent group means: the
    figure code averages rel_mse over the config rows that share one x
    cell, so the matching SE is sqrt(sum se_i^2) / k (identity for k=1;
    NaN propagates so a single-seed row blanks the whole cell's band)."""
    se = np.asarray(se_values, dtype=float)
    if se.size == 0:
        return float("nan")
    return float(np.sqrt(np.sum(se**2)) / se.size)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Per (configuration x estimator): relative MSE and its bias^2 / variance
    decomposition of the normalized error (V_hat - V) / V over the M seeds."""
    df = df.copy()
    df["rel_err"] = (df["estimate"] - df["ground_truth"]) / df["ground_truth"]
    grouped = df.groupby(CONFIG_KEYS + ["estimator"], dropna=False)["rel_err"]
    agg = grouped.agg(
        rel_mse=lambda s: float(np.mean(s**2)),
        bias=lambda s: float(np.mean(s)),
        variance=lambda s: float(np.var(s)),
        n_seeds="count",
    )
    agg["bias_sq"] = agg["bias"] ** 2
    # display-only column for the figure CIs; appended last so the existing
    # aggregate csv columns keep their values and order
    agg["rel_mse_se"] = grouped.agg(
        lambda s: se_of_mean(np.asarray(s, dtype=float) ** 2)
    )
    # display-only SEs for the bias^2 / variance panels of the error-
    # decomposition figures; appended after
    # rel_mse_se so the pre-existing columns again keep values and order
    agg["bias_se"] = grouped.agg(
        lambda s: se_of_mean(np.asarray(s, dtype=float))
    )
    agg["variance_se"] = grouped.agg(
        lambda s: se_of_variance(np.asarray(s, dtype=float))
    )
    # corrupted-oracle weight-ratio error (Figure 3 x axis):
    # per-cell mean of the run_one_replicate diagnostic; appended last so
    # every pre-existing aggregate column keeps its value and order. Only
    # present when at least one run recorded it (oracle-enabled runs).
    if "oracle_ratio_error_mean" in df.columns:
        agg["oracle_ratio_error_mean"] = df.groupby(
            CONFIG_KEYS + ["estimator"], dropna=False
        )["oracle_ratio_error_mean"].mean()
    # corrupted-relevance error (fig3 condition (e) calibration check);
    # appended after oracle_ratio_error_mean, only when recorded
    if "oracle_relevance_error_mean" in df.columns:
        agg["oracle_relevance_error_mean"] = df.groupby(
            CONFIG_KEYS + ["estimator"], dropna=False
        )["oracle_relevance_error_mean"].mean()
    return agg.reset_index()


def _style_axis(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONTSIZE)


def _ordered_estimators(estimators) -> List[str]:
    known = [m for m in METHOD_ORDER if m in set(estimators)]
    unknown = sorted(set(estimators) - set(METHOD_ORDER))
    return known + unknown


def plot_relmse_vs_axis(
    agg: pd.DataFrame,
    x_col: str,
    panel_col: Optional[str],
    out_path: Path,
    logx: bool = True,
    logy: bool = True,
    title: str = "",
    xlabel: str = "",
):
    """Generic line chart: relative MSE vs a swept axis, one panel per value
    of `panel_col`, one line per estimator (fixed colors, thin marks)."""
    panels = [None] if panel_col is None else sorted(agg[panel_col].unique())
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 4.0), squeeze=False
    )
    for j, panel in enumerate(panels):
        ax = axes[0][j]
        sub = agg if panel is None else agg[agg[panel_col] == panel]
        for est in _ordered_estimators(sub["estimator"].unique()):
            style = METHOD_STYLE.get(est, dict(color=MUTED_COLOR, ls="-", marker="."))
            grp = sub[sub["estimator"] == est].groupby(x_col)
            line = grp["rel_mse"].mean().sort_index()
            if SHOW_CI:
                se = grp["rel_mse_se"].agg(_combine_se).sort_index()
                lo, hi = ci_bounds(line.values, se.values)
                ax.fill_between(
                    line.index, lo, hi, color=style["color"], alpha=0.18, linewidth=0
                )
            ax.plot(
                line.index,
                line.values,
                label=est,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                linewidth=1.6,
                markersize=5,
            )
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel or x_col, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        if j == 0:
            ax.set_ylabel("relative MSE", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        if panel is not None:
            ax.set_title(
                f"{panel_col} = {panel}", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
            )
        _style_axis(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    # anchored by its bottom edge just above the panels, so a taller legend
    # (more rows, larger font) grows upward instead of into the panel headers
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, 1.01),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )
    if title and SHOW_FIG_TITLE:
        fig.suptitle(title, y=1.22, fontsize=11, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# the three columns of every error-decomposition figure, left to right;
# rel_mse = bias_sq + variance holds per aggregate row (ddof=0), so sharing
# the y axis across the three panels makes the decomposition comparable
DECOMP_COLUMNS = (
    ("rel_mse", "Relative MSE"),
    ("bias_sq", "Bias$^2$"),
    ("variance", "Variance"),
)


def decomposition_stats(sub: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per figure cell (`by` = the x column or "estimator"): the three error
    components with their 95% CIs.

    - relative MSE: the existing rel_mse_se / ci_bounds (unchanged).
    - bias^2: the Wald interval of bias, squared (bias_sq_ci_bounds).
    - variance: log-scale delta-method interval on the influence-function SE.

    A cell can hold several config rows (e.g. the decay_K sweep before the K
    panels were dropped); values are then unweighted means and the SEs combine
    as in _combine_se. The interval of bias^2 uses the combined bias mean, so
    it is exact for the single-row cells all decomposition figures have."""

    def _cell(g: pd.DataFrame) -> pd.Series:
        rel = float(g["rel_mse"].mean())
        rel_lo, rel_hi = ci_bounds(rel, _combine_se(g["rel_mse_se"]))
        b_lo, b_hi = bias_sq_ci_bounds(
            float(g["bias"].mean()), _combine_se(g["bias_se"])
        )
        var = float(g["variance"].mean())
        var_lo, var_hi = ci_bounds(var, _combine_se(g["variance_se"]))
        return pd.Series(
            dict(
                rel_mse=rel,
                rel_mse_lo=float(rel_lo),
                rel_mse_hi=float(rel_hi),
                bias_sq=float(g["bias_sq"].mean()),
                bias_sq_lo=float(b_lo),
                bias_sq_hi=float(b_hi),
                variance=var,
                variance_lo=float(var_lo),
                variance_hi=float(var_hi),
            )
        )

    return sub.groupby(by).apply(_cell)


def plot_error_decomposition(
    agg: pd.DataFrame,
    x_col: str,
    out_path: Path,
    row_col: Optional[str] = None,
    row_order: Optional[List] = None,
    row_labels: Optional[Dict] = None,
    logx: bool = True,
    logy: bool = True,
    title: str = "",
    xlabel: str = "",
    ytick_decades: Optional[int] = None,
):
    """Error-decomposition line figure: columns are relative MSE /
    bias^2 / variance, one row per value of `row_col`, one line per estimator
    (plus its 95% CI band when SHOW_CI is on). Every panel scales its own y
    axis (a shared axis let the near-zero bias^2 panels fall out of view);
    every panel carries its error component as the y label (no column
    headers), the x label sits on the
    bottom row, and one legend serves the whole figure. A row label, when
    given, goes on the right edge of the row (the paper's Figure 1 passes
    none: its caption names the rows). Points that cannot sit on a log axis
    (<= 0 or NaN) are not drawn; with the bands on, a bias^2 band whose bias
    interval covers 0 reaches the panel bottom (lower bound 0)."""
    if row_col is None:
        rows = [None]
    else:
        present = list(agg[row_col].unique())
        rows = (
            [r for r in row_order if r in present]
            if row_order is not None
            else sorted(present)
        )
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(11.4, 3.6 * n_rows),
        squeeze=False,
        sharex=True,
    )
    estimators_seen: List[str] = []
    for i, row_val in enumerate(rows):
        sub = agg if row_val is None else agg[agg[row_col] == row_val]
        for est in _ordered_estimators(sub["estimator"].unique()):
            if est not in estimators_seen:
                estimators_seen.append(est)
            style = METHOD_STYLE.get(est, dict(color=MUTED_COLOR, ls="-", marker="."))
            cells = decomposition_stats(sub[sub["estimator"] == est], x_col).sort_index()
            xs = cells.index.values
            for j, (col, _) in enumerate(DECOMP_COLUMNS):
                ax = axes[i][j]
                y = cells[col].values.astype(float)
                lo = cells[f"{col}_lo"].values.astype(float)
                hi = cells[f"{col}_hi"].values.astype(float)
                if logy:
                    y = np.where(y > 0.0, y, np.nan)
                if SHOW_CI:
                    band = np.isfinite(lo) & np.isfinite(hi)
                    ax.fill_between(
                        xs,
                        np.where(band, lo, np.nan),
                        np.where(band, hi, np.nan),
                        color=style["color"],
                        alpha=0.18,
                        linewidth=0,
                    )
                ax.plot(
                    xs,
                    y,
                    color=style["color"],
                    linestyle=style["ls"],
                    marker=style["marker"],
                    linewidth=1.6,
                    markersize=5,
                )
        for j, (_, col_label) in enumerate(DECOMP_COLUMNS):
            ax = axes[i][j]
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_ylabel(col_label, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
            if i == n_rows - 1:
                ax.set_xlabel(
                    xlabel or x_col, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
                )
            _style_axis(ax)
        if row_labels is not None and row_val is not None:
            label = row_labels.get(row_val, str(row_val))
            ax_r = axes[i][-1]
            ax_r.yaxis.set_label_position("right")
            ax_r.set_ylabel(
                label, rotation=270, labelpad=16, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
            )
    if logy and ytick_decades:
        # wide-range figures (fig5 spans ~10 decades): major ticks every
        # `ytick_decades` decades so each axis stays readable
        for row_axes in axes:
            for ax in row_axes:
                ymin, ymax = ax.get_ylim()
                exps = [
                    e
                    for e in range(
                        int(np.ceil(np.log10(ymin))),
                        int(np.floor(np.log10(ymax))) + 1,
                    )
                    if e % ytick_decades == 0
                ]
                ticks = [10.0**e for e in exps]
                ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
                ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
                ax.yaxis.set_minor_locator(mticker.NullLocator())
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE.get(e, dict(color=MUTED_COLOR))["color"],
            linestyle=METHOD_STYLE.get(e, dict(ls="-")).get("ls", "-"),
            marker=METHOD_STYLE.get(e, dict(marker=".")).get("marker", "."),
            linewidth=1.6,
            markersize=5,
        )
        for e in estimators_seen
    ]
    # anchored by its bottom edge just above the panels (see plot_relmse_vs_axis)
    fig.legend(
        handles,
        estimators_seen,
        loc="lower center",
        ncol=min(len(estimators_seen), 6),
        bbox_to_anchor=(0.5, 1.01),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )
    if title and SHOW_FIG_TITLE:
        fig.suptitle(
            title, y=1.16 if n_rows == 1 else 1.10, fontsize=11, color=TEXT_COLOR
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_decomposition_bar_row(
    axes_row, sub: pd.DataFrame, estimators: List[str], logy: bool
) -> None:
    """One error-decomposition row as solid bars: per panel one bar per
    estimator, with a 95% CI whisker (rel MSE / squared-Wald bias^2 /
    influence-function variance) when SHOW_CI is on. All-NaN cells (fig2's
    intentionally skipped cascade-dr at epsilon=0) stay blank in all three
    panels. The caller sets titles, y limits and tick labels."""
    cells = decomposition_stats(sub, "estimator")
    xs = np.arange(len(estimators))
    colors = [
        METHOD_STYLE.get(e, dict(color=MUTED_COLOR))["color"] for e in estimators
    ]

    def _col(name: str) -> np.ndarray:
        series = cells[name] if name in cells else pd.Series(dtype=float)
        return np.array([series.get(e, np.nan) for e in estimators], dtype=float)

    for j, (col, _) in enumerate(DECOMP_COLUMNS):
        ax = axes_row[j]
        vals, lo, hi = _col(col), _col(f"{col}_lo"), _col(f"{col}_hi")
        ax.bar(xs, vals, color=colors, width=0.7)
        ok = np.isfinite(vals) & np.isfinite(lo) & np.isfinite(hi)
        if SHOW_CI and ok.any():
            ax.errorbar(
                xs[ok],
                vals[ok],
                yerr=np.vstack([vals[ok] - lo[ok], hi[ok] - vals[ok]]),
                fmt="none",
                ecolor=TEXT_COLOR,
                elinewidth=1.0,
                capsize=2.5,
            )
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(xs)
        _style_axis(ax)


def make_fig1(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Figure 1 (flagship): error decomposition vs n, top row (E1) pbm,
    bottom row (E3) ranking-dependent, log-log, each panel on its own y
    scale. The rows carry no label (the caption
    names them) and the error components sit on the y axes."""
    sub = agg[agg["setting_name"] == "fig1_data_size"]
    if sub.empty:
        return None
    out = figs_dir / "fig1_relmse_vs_n.png"
    plot_error_decomposition(
        sub,
        x_col="n_rounds",
        out_path=out,
        row_col="exposure_structure",
        row_order=["pbm", "ranking_dependent"],
        title="Fig.1: error decomposition vs data size",
        xlabel="Sample size $n$ (logarithmic scale)",
    )
    return out


def make_fig2(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Figure 2: error decomposition over policy divergence epsilon — one row
    per epsilon, solid bars on linear axes, each panel on its own y scale.
    The all-NaN cascade-dr cell at epsilon=0
    (deterministic evaluation policy) stays blank in all three panels."""
    sub = agg[agg["setting_name"] == "fig2_policy_divergence"]
    if sub.empty:
        return None
    out = figs_dir / "fig2_bias_variance_vs_epsilon.png"
    eps_values = sorted(sub["epsilon"].unique())
    estimators = _ordered_estimators(sub["estimator"].unique())
    n_rows = len(eps_values)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(11.4, 2.4 * n_rows), squeeze=False
    )
    for i, eps in enumerate(eps_values):
        draw_decomposition_bar_row(
            axes[i], sub[sub["epsilon"] == eps], estimators, logy=False
        )
        for j, ax in enumerate(axes[i]):
            ax.set_ylim(bottom=0.0)
            if i == 0:
                ax.set_title(
                    DECOMP_COLUMNS[j][1], fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
                )
            if i == n_rows - 1:
                ax.set_xticklabels(
                    estimators, rotation=90, fontsize=ESTIMATOR_TICK_FONTSIZE
                )
            else:
                ax.set_xticklabels([])
        axes[i][0].set_ylabel(
            f"$\\epsilon$ = {eps}", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
        )
    if SHOW_FIG_TITLE:
        fig.suptitle(
            "Fig.2: error decomposition vs policy divergence ($\\epsilon$)",
            fontsize=11,
            color=TEXT_COLOR,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


FIG3_ESTIMATORS = ("ed-dr (oracle)", "le-iips (oracle)")


def _parse_weight_channel(value) -> Optional[tuple]:
    """(method, strength) of a weight-side corruption cell value.

    'none' -> ("none", 0.0); 'power:<t>' -> ("R", t) (ratio corruption);
    'E:power:<t>' -> ("E", t) (exposure-model corruption). Legacy
    non-parameterized values of the retired 2x2 grid ('power' / 'shift')
    return None, so stale fig3 logs never blend into the new figure."""
    s = str(value)
    if s == "none":
        return ("none", 0.0)
    method = "R"
    if s.startswith("E:"):
        method, s = "E", s[2:]
    if s.startswith("power:"):
        try:
            strength = float(s.split(":", 1)[1])
        except ValueError:
            return None
        if np.isfinite(strength) and strength > 0.0:
            return (method, strength)
    return None


def _parse_relevance_channel(value) -> Optional[float]:
    """Strength of a relevance corruption cell value ('none' -> 0.0,
    'power:<t>' -> t); legacy kinds return None (old-format row)."""
    s = str(value)
    if s == "none":
        return 0.0
    if s.startswith("power:"):
        try:
            strength = float(s.split(":", 1)[1])
        except ValueError:
            return None
        if np.isfinite(strength) and strength > 0.0:
            return strength
    return None


def fit_loglog_slope(x, y) -> float:
    """OLS slope of log(y) on log(x) over the finite positive pairs (the
    bias-vs-ratio-error slope of Figure 3 / the prediction scorecard).
    Returns NaN below two usable points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def _fig3_frame(agg: pd.DataFrame) -> Optional[pd.DataFrame]:
    """New-format fig3 rows with the parsed corruption axes, or None when the
    logs hold no parameterized-strength runs yet (e.g. only the retired 2x2
    grid). Adds columns: w_method ('none'/'E'/'R'), w_t, rel_t."""
    sub = agg[
        (agg["setting_name"] == "fig3_dr_grid")
        & (agg["estimator"].isin(FIG3_ESTIMATORS))
    ].copy()
    if sub.empty or "oracle_ratio_error_mean" not in sub.columns:
        return None
    parsed_w = sub["corrupt_exposure_ratio"].map(_parse_weight_channel)
    parsed_rel = sub["corrupt_relevance"].map(_parse_relevance_channel)
    sub = sub[parsed_w.notna() & parsed_rel.notna()]
    if sub.empty:
        return None
    sub["w_method"] = parsed_w.loc[sub.index].str[0]
    sub["w_t"] = parsed_w.loc[sub.index].str[1].astype(float)
    sub["rel_t"] = parsed_rel.loc[sub.index].astype(float)
    # nothing but uncorrupted baselines is not a figure yet
    if not ((sub["w_t"] > 0) | (sub["rel_t"] > 0)).any():
        return None
    return sub


def _fig3_bias_ci_panel(ax, frame: pd.DataFrame, strength_col: str, title: str) -> None:
    """One (a)/(b) panel: signed bias with its 95% Wald CI per corruption
    strength (categorical axis, strength 0 = the uncorrupted run). The claim
    these panels carry is 'the CI covers 0 at every strength'."""
    strengths = sorted(frame[strength_col].unique())
    xs = np.arange(len(strengths), dtype=float)
    for offset, est in zip((-0.12, 0.12), FIG3_ESTIMATORS):
        style = METHOD_STYLE.get(est, dict(color=MUTED_COLOR, marker="."))
        rows = frame[frame["estimator"] == est]
        means = np.full(len(strengths), np.nan)
        errs = np.full(len(strengths), np.nan)
        for i, t in enumerate(strengths):
            g = rows[rows[strength_col] == t]
            if len(g) == 0:
                continue
            means[i] = float(g["bias"].mean())
            errs[i] = CI_Z * _combine_se(g["bias_se"])
        ok = np.isfinite(means) & np.isfinite(errs)
        ax.errorbar(
            xs[ok] + offset,
            means[ok],
            yerr=errs[ok],
            fmt=style.get("marker", "."),
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.2,
            capsize=3,
            markersize=6,
            linestyle="none",
        )
    ax.axhline(0.0, color=MUTED_COLOR, linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t:g}" for t in strengths], fontsize=8)
    ax.set_xlabel("corruption strength $t$", fontsize=9, color=TEXT_COLOR)
    ax.set_title(title, fontsize=9, color=TEXT_COLOR)
    _style_axis(ax)


# measured ceiling of the method-E ratio error under (E3): the power map
# saturates at t ~ 5 (0.28) and the clip pulls the ratio back at t = 20
# (measured in floor probes). Points at >= 2/3 of the ceiling are in the
# saturated regime and excluded from slope fits.
FIG3_E_SATURATION_CEIL = 0.28


def make_fig3(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Figure 3: double robustness under feasible misspecification.

    Panels (a)/(b) show the signed bias (95% CI) of the oracle estimators
    when only the relevance (a; E3, weights true) or the exposure model
    within its class (b; E2, corollary-2b regime) is corrupted — the DR
    prediction is a CI covering 0 at every strength. Panel (c) is the main
    object: |bias| against the EXACTLY KNOWN weight-ratio error
    (oracle_ratio_error_mean, method R — no MC noise floor) under a true
    (E3) ranking-dependent exposure, log-log, with three series:
      - `ed-dr (oracle)`, R x relevance (calibrated pairs): both nuisances
        carry O(t) errors, the first-order terms cancel exactly and the
        product O(t^2) remains — slope 2, the main claim;
      - `le-iips (oracle)`, R x relevance: single-robust reference, slope 1;
      - `ed-dr (oracle)`, R alone (open markers): q_hat stays true, so the
        bias is EXACTLY zero at every strength — the negative control.
    Method E (pointwise exposure-model corruption; first-order under (E3))
    moved to the appendix — see make_fig3_method_e_appendix."""
    sub = _fig3_frame(agg)
    if sub is None:
        return None
    rd = sub["exposure_structure"] == "ranking_dependent"
    e2 = sub["exposure_structure"] == "contextual_pbm"
    cond_a = sub[rd & (sub["w_method"] == "none")]
    cond_b = sub[e2 & (sub["rel_t"] == 0.0)]
    cond_c = sub[rd & (sub["w_method"] == "R")]
    cond_c_rx = cond_c[cond_c["rel_t"] > 0.0]      # (e) calibrated pairs
    cond_c_zero = cond_c[cond_c["rel_t"] == 0.0]   # (d) R alone: zero line
    baseline_c = sub[rd & (sub["w_method"] == "none") & (sub["rel_t"] == 0.0)]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    # ---- (a) relevance corrupted, exposure true (E3) ----
    if not cond_a.empty:
        _fig3_bias_ci_panel(
            axes[0],
            cond_a,
            "rel_t",
            "(a) $\\hat{r}$ corrupted, weights true (E3)\nprediction: bias CI covers 0",
        )
        axes[0].set_ylabel("bias (relative)", fontsize=9, color=TEXT_COLOR)
    else:
        axes[0].text(0.5, 0.5, "condition (a) not run", ha="center", va="center",
                     fontsize=9, color=MUTED_COLOR)
        _style_axis(axes[0])
    # ---- (b) exposure corrupted within its class (E2) ----
    if not cond_b.empty:
        _fig3_bias_ci_panel(
            axes[1],
            cond_b,
            "w_t",
            "(b) $\\hat{e}$ corrupted in-class (E2, method E)\nprediction: ratio error $\\approx$ 0, bias CI covers 0",
        )
    else:
        axes[1].text(0.5, 0.5, "condition (b) not run", ha="center", va="center",
                     fontsize=9, color=MUTED_COLOR)
        _style_axis(axes[1])
    # ---- (c) method R under (E3): |bias| vs exactly-known ratio error ----
    ax = axes[2]

    def _plot_series(frame, est, marker, filled, label):
        style = METHOD_STYLE.get(est, dict(color=MUTED_COLOR))
        g = frame[
            (frame["estimator"] == est) & (frame["w_t"] > 0.0)
        ].sort_values("oracle_ratio_error_mean")
        if g.empty:
            return None
        x = g["oracle_ratio_error_mean"].values.astype(float)
        y = np.abs(g["bias"].values.astype(float))
        lo, hi = bias_sq_ci_bounds(g["bias"].values, g["bias_se"].values)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack(
                [np.clip(y - np.sqrt(lo), 0.0, None), np.sqrt(hi) - y]
            ),
            fmt=marker,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2,
            markersize=6,
            markerfacecolor=style["color"] if filled else "none",
            linestyle="none",
            label=label,
        )
        return float(x[-1]), float(y[-1])

    for est in FIG3_ESTIMATORS:
        style = METHOD_STYLE.get(est, dict(color=MUTED_COLOR))
        base_est = baseline_c[baseline_c["estimator"] == est]
        if not base_est.empty:
            ax.axhline(
                float(base_est["bias"].abs().mean()),
                color=style["color"],
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
            )
    anchor_ed = _plot_series(
        cond_c_rx, "ed-dr (oracle)", "*", True,
        "ed-dr (oracle) / R$\\times$relevance",
    )
    anchor_le = _plot_series(
        cond_c_rx, "le-iips (oracle)", "X", True,
        "le-iips (oracle) / R$\\times$relevance",
    )
    _plot_series(
        cond_c_zero, "ed-dr (oracle)", "^", False,
        "ed-dr (oracle) / R alone (zero line)",
    )
    xs_all = cond_c[cond_c["w_t"] > 0]["oracle_ratio_error_mean"]
    if not xs_all.empty:
        span = np.array([max(xs_all.min() * 0.5, 1e-12), xs_all.max() * 2.0])
        for anchor, slope, ls in ((anchor_ed, 2.0, "-"), (anchor_le, 1.0, ":")):
            if anchor is None or anchor[0] <= 0 or anchor[1] <= 0:
                continue
            ax.plot(
                span,
                anchor[1] * (span / anchor[0]) ** slope,
                color=MUTED_COLOR,
                linewidth=1.0,
                linestyle=ls,
                alpha=0.8,
            )
            ax.text(
                span[0],
                anchor[1] * (span[0] / anchor[0]) ** slope,
                f"slope {slope:g}",
                fontsize=7,
                color=MUTED_COLOR,
                va="bottom",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        "exact exposure-ratio error (oracle_ratio_error_mean, log)",
        fontsize=9,
        color=TEXT_COLOR,
    )
    ax.set_ylabel("|bias| (log)", fontsize=9, color=TEXT_COLOR)
    ax.set_title(
        "(c) weights corrupted (method R) under (E3)\n"
        "prediction: R$\\times$rel ED-DR slope 2 / LE-IIPS slope 1 / R alone flat 0",
        fontsize=9,
        color=TEXT_COLOR,
    )
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style_axis(ax)
    if SHOW_FIG_TITLE:
        fig.suptitle(
            "Fig.3: double robustness under feasible misspecification "
            "(oracle nuisances, corrupted per channel)",
            y=1.06,
            fontsize=11,
            color=TEXT_COLOR,
        )
    fig.tight_layout()
    out = figs_dir / "fig3_dr_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_fig3_method_e_appendix(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Appendix companion of Figure 3: method E under
    (E3) — the FEASIBLE out-of-class misspecification. A pointwise map of
    the exposure model makes Delta q = (e - f(e)) r depend on the realized
    ranking, so it is not transported by the weights and a first-order term
    survives: the prediction is slope ~ 1 with `ed-dr` degraded to the
    `le-iips` level ("first-order, at the single-robust level" — the
    disclosed applicability limit of the theorem, NOT a slope-2 claim).
    The measured MC noise floor (vertical dotted line) and the saturation
    regime (ratio error >= 2/3 of the 0.28 ceiling, shaded) are annotated;
    saturated points are excluded from the scorecard slope fit."""
    sub = _fig3_frame(agg)
    if sub is None:
        return None
    rd = sub["exposure_structure"] == "ranking_dependent"
    cond_e = sub[rd & (sub["w_method"] == "E") & (sub["rel_t"] == 0.0)]
    if cond_e[cond_e["w_t"] > 0].empty:
        return None
    baseline_c = sub[rd & (sub["w_method"] == "none") & (sub["rel_t"] == 0.0)]
    floor = (
        float(baseline_c["oracle_ratio_error_mean"].mean())
        if not baseline_c.empty
        else np.nan
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    anchor = None
    for est in FIG3_ESTIMATORS:
        style = METHOD_STYLE.get(est, dict(color=MUTED_COLOR, marker="."))
        g = cond_e[
            (cond_e["estimator"] == est) & (cond_e["w_t"] > 0.0)
        ].sort_values("oracle_ratio_error_mean")
        if g.empty:
            continue
        x = g["oracle_ratio_error_mean"].values.astype(float)
        y = np.abs(g["bias"].values.astype(float))
        lo, hi = bias_sq_ci_bounds(g["bias"].values, g["bias_se"].values)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([np.clip(y - np.sqrt(lo), 0.0, None), np.sqrt(hi) - y]),
            fmt=style.get("marker", "o"),
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2,
            markersize=6,
            linestyle="none",
            label=est,
        )
        if est == "le-iips (oracle)":
            anchor = (float(x[-1]), float(y[-1]))
    if anchor is not None and anchor[0] > 0 and anchor[1] > 0:
        xs_all = cond_e[cond_e["w_t"] > 0]["oracle_ratio_error_mean"]
        span = np.array([max(xs_all.min() * 0.5, 1e-12), xs_all.max() * 2.0])
        ax.plot(
            span,
            anchor[1] * (span / anchor[0]) ** 1.0,
            color=MUTED_COLOR,
            linewidth=1.0,
            linestyle=":",
            alpha=0.8,
        )
        ax.text(span[0], anchor[1] * (span[0] / anchor[0]), "slope 1",
                fontsize=7, color=MUTED_COLOR, va="bottom")
    if np.isfinite(floor) and floor > 0:
        ax.axvline(floor, color=MUTED_COLOR, linewidth=1.0, linestyle=":")
        ax.text(floor, 0.03, " MC noise floor", fontsize=7, color=MUTED_COLOR,
                rotation=90, va="bottom", transform=ax.get_xaxis_transform())
    sat_cut = FIG3_E_SATURATION_CEIL * 2.0 / 3.0
    ax.axvspan(sat_cut, FIG3_E_SATURATION_CEIL * 2.0, color=MUTED_COLOR, alpha=0.10)
    ax.text(sat_cut, 0.97, " saturated ($t \\gtrsim 5$)", fontsize=7,
            color=MUTED_COLOR, va="top", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        "measured exposure-ratio error (oracle_ratio_error_mean, log)",
        fontsize=9,
        color=TEXT_COLOR,
    )
    ax.set_ylabel("|bias| (log)", fontsize=9, color=TEXT_COLOR)
    ax.set_title(
        "Appendix: method E under (E3) — feasible out-of-class error is "
        "FIRST-order,\ned-dr at the single-robust (le-iips) level "
        "(applicability limit of the slope-2 claim)",
        fontsize=9,
        color=TEXT_COLOR,
    )
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    out = figs_dir / "appendix_fig3_method_e.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# strength levels of the injection figure's x axis: t = 5 is dropped because
# the power map v -> v^(1+t) saturates there (the measured ratio error of
# method E stops growing and the bias falls back)
FIG3_INJECTION_MAX_T = 1.6


def make_fig3_injection(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Paper figure of the error-injection experiment: one panel, line chart
    in the style of the other paper figures, no CI whiskers.

    |bias| of the oracle estimators against the corruption strength t of
    v -> v^(1+t), one line per condition of the paper text:
      (a) (E3) truth, exposure true, relevance corrupted        -> flat at 0
          (corollary e-side);
      (b) (E2) truth, relevance true, exposure corrupted in-class (method E,
          the correction ratio degenerates to 1)                -> flat at 0
          (corollary r-side);
      (c) (E3) truth, relevance true, exposure corrupted out-of-class
          (method E, applied consistently)                      -> grows;
      (d) le-iips (oracle) under the same corrupted exposure, the
          single-robust reference the bias degrades to (labelled (d) as in
          the paper's caption).
    The strength-0 point of every line IS the uncorrupted run of its truth
    structure. t = 5 is dropped (FIG3_INJECTION_MAX_T): the power map
    saturates there and the bias falls back, which reads as noise on a t
    axis. The x axis is categorical over the strengths actually run. The
    R x relevance slope-2 series and the R-alone zero line of make_fig3 are
    not shown: the paper text does not make that claim. Returns None when
    none of the three conditions has been run."""
    sub = _fig3_frame(agg)
    if sub is None:
        return None
    rd = sub["exposure_structure"] == "ranking_dependent"
    e2 = sub["exposure_structure"] == "contextual_pbm"
    series = [
        # label, frame, strength column, estimator, style
        ("(a) ed-dr (oracle), $\\hat{r}$ corrupted (E3)",
         sub[rd & (sub["w_method"] == "none")], "rel_t", "ed-dr (oracle)",
         dict(color="#2a78d6", marker="o", ls="-")),
        ("(b) ed-dr (oracle), $\\hat{e}_k$ corrupted (E2)",
         sub[e2 & (sub["rel_t"] == 0.0)], "w_t", "ed-dr (oracle)",
         dict(color="#008300", marker="^", ls="-")),
        ("(c) ed-dr (oracle), $\\hat{e}_k$ corrupted (E3)",
         sub[rd & (sub["rel_t"] == 0.0) & (sub["w_method"].isin(["none", "E"]))], "w_t",
         "ed-dr (oracle)", dict(color=METHOD_STYLE["ed-dr"]["color"], marker="*", ls="-")),
        ("(d) le-iips (oracle), $\\hat{e}_k$ corrupted (E3)",
         sub[rd & (sub["rel_t"] == 0.0) & (sub["w_method"].isin(["none", "E"]))], "w_t",
         "le-iips (oracle)", dict(color=METHOD_STYLE["le-iips"]["color"], marker="X", ls="--")),
    ]
    lines = []
    for label, frame, col, est, style in series:
        g = frame[(frame["estimator"] == est) & (frame[col] <= FIG3_INJECTION_MAX_T)]
        pts = g.groupby(col)["bias"].mean().abs().sort_index()
        if (pts.index > 0).any():
            lines.append((label, pts, style))
    if not lines:
        return None
    strengths = sorted({float(t) for _, pts, _ in lines for t in pts.index})
    pos = {t: i for i, t in enumerate(strengths)}

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for label, pts, style in lines:
        ax.plot(
            [pos[float(t)] for t in pts.index],
            pts.values * 100.0,
            color=style["color"],
            linestyle=style["ls"],
            marker=style["marker"],
            linewidth=1.6,
            markersize=6,
            label=label,
        )
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([f"{t:g}" for t in strengths])
    ax.set_xlabel(
        "Corruption strength $t$  ($v \\mapsto v^{1+t}$)", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
    )
    ax.set_ylabel("|bias| (% of $V(\\pi)$)", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax.set_ylim(bottom=0)
    # legend above the panel, bottom-anchored, as in the other paper figures
    # (the panel has no title)
    ax.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )
    _style_axis(ax)
    fig.tight_layout()
    out = figs_dir / "fig3_injection.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_fig4_bias_bound(df_raw: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Figure 4: the Eq. (8) linear bias bound of LE-IIPS (paper: Prop. 1's
    misspecification bound) evaluated empirically, vs the realized |bias| of
    LE-IIPS. One point per experimental configuration in which the true
    exposure exists (the exposure-DGP sweeps, including the table 1
    misspecification grid) — the cascade DGP has no true exposure and is not
    included. Needs the per-seed rows (prop3_bound diagnostics), so nothing is
    drawn from an aggregate-only source."""
    scatter_src = df_raw[
        df_raw["prop3_bound"].notna() & (df_raw["estimator"] == "le-iips")
    ]
    if scatter_src.empty:
        return None
    grouped = scatter_src.groupby(CONFIG_KEYS, dropna=False)
    bounds = grouped["prop3_bound"].mean()
    bias = grouped.apply(
        lambda g: np.abs(np.mean(g["estimate"] - g["ground_truth"]))
    )
    # same 5.6:4.6 aspect as before but smaller, so that at
    # width=0.6\columnwidth in the paper the 13 pt labels land at ~9 pt like
    # the other figures; the 5.6 x 4.6 in version at \columnwidth took half a
    # page. Kept near-square on purpose: the y = x dashed line
    # is the reference, so the two axes must share one scale.
    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    ax.scatter(
        bounds.values,
        bias.values,
        s=22,
        color=METHOD_STYLE["le-iips"]["color"],
        edgecolor="white",
        linewidth=0.5,
    )
    lim = max(1e-12, np.nanmax(bounds.values), np.nanmax(bias.values))
    ax.plot([0, lim], [0, lim], color=MUTED_COLOR, linewidth=1.0, linestyle="--")
    ax.set_xlabel(
        "Eq. (8) linear bound (empirical)", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
    )
    ax.set_ylabel(
        "Realized |bias| (LE-IIPS)", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR
    )
    _style_axis(ax)
    if SHOW_FIG_TITLE:
        fig.suptitle(
            "Fig.4: LE-IIPS bias vs the Eq. (8) linear bound "
            "(exposure-DGP sweeps; dashed: $y = x$)",
            fontsize=11,
            color=TEXT_COLOR,
        )
    fig.tight_layout()
    out = figs_dir / "fig4_le_bias_bound.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_fig5_cascade(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Figure 5 (losing condition): error decomposition of all methods under
    the true-cascade DGP, one row of bars.

    Under the cascade DGP the slot reward still factors into exposure x
    relevance, so (A2)/(A3) hold; what breaks is the position-independence
    (A4), which puts the true exposure (the expected product of no-clicks
    above) outside every assumed exposure class — ED-DR runs with a
    misspecified e_hat."""
    sub = agg[agg["setting_name"] == "fig4_cascade"]
    if sub.empty:
        return None
    estimators = _ordered_estimators(sub["estimator"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    draw_decomposition_bar_row(list(axes), sub, estimators, logy=True)
    for j, ax in enumerate(axes):
        ax.set_title(DECOMP_COLUMNS[j][1], fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        ax.set_xticklabels(estimators, rotation=90, fontsize=ESTIMATOR_TICK_FONTSIZE)
    axes[0].set_ylabel(
        "cascade DGP ($\\hat{e}$ misspecified)",
        fontsize=LABEL_FONTSIZE,
        color=TEXT_COLOR,
    )
    if SHOW_FIG_TITLE:
        fig.suptitle(
            "Fig.5: limits under exposure-model misspecification (cascade DGP)",
            fontsize=11,
            color=TEXT_COLOR,
        )
    fig.tight_layout()
    out = figs_dir / "fig5_cascade.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _fig4_eta_frame(agg: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Rows of the cascade-strength sweep: the 4 new eta levels
    from setting fig4_cascade_eta, plus the eta = 1 cell REUSED from the
    existing fig4_cascade logs (200 seeds; never re-run — completion
    condition 1 holds by construction because the figure reads the very same
    aggregate cell as Figure 4).

    Raises when fig4_cascade_eta itself contains eta = 1 rows (that would
    duplicate the reused cell with a second run) and when the fig4_cascade
    baseline is missing (the eta = 1 level must come from it)."""
    eta_rows = agg[agg["setting_name"] == "fig4_cascade_eta"]
    if eta_rows.empty:
        return None
    if (eta_rows["exposure_decay_rate"] == 1.0).any():
        raise ValueError(
            "fig4_cascade_eta contains eta=1.0 rows: eta=1 must be reused "
            "from the fig4_cascade logs, not re-run — delete the "
            "eta=1.0 runs under logs/fig4_cascade_eta/"
        )
    base = agg[
        (agg["setting_name"] == "fig4_cascade")
        & (agg["exposure_decay_rate"] == 1.0)
    ]
    if base.empty:
        raise ValueError(
            "fig4_cascade_eta runs exist but the fig4_cascade (eta=1) logs "
            "they must reuse are missing from the aggregate"
        )
    return pd.concat([eta_rows, base], ignore_index=True)


def make_fig4_eta(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Cascade-strength sweep: error decomposition vs the
    cascade DGP's eta on a linear axis including 0 (eta = 0 removes the
    click discount entirely — the position-independence baseline; larger eta
    = stronger cascading). Placed right after Figure 4 in the report; the
    claim is restricted to the bias^2 crossing between ED-DR and RIPS."""
    sub = _fig4_eta_frame(agg)
    if sub is None:
        return None
    out = figs_dir / "fig4_cascade_eta.png"
    plot_error_decomposition(
        sub,
        x_col="exposure_decay_rate",
        out_path=out,
        logx=False,
        title=(
            "Fig.5': error decomposition vs cascade strength $\\eta$ "
            "($\\eta = 0$: position-independent; $\\eta = 1$: Fig.5 cell)"
        ),
        xlabel="cascade strength $\\eta$",
    )
    return out


def make_fig7_tau0(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """Figure 7: collapse under (near-)deterministic logging (tau0 -> 0.05),
    as a one-row error decomposition; the y axes span many decades, so major
    ticks sit every 2 decades. Values are not
    clipped."""
    sub = agg[agg["setting_name"] == "fig5_logging_determinism"]
    if sub.empty:
        return None
    out = figs_dir / "fig7_relmse_vs_tau0.png"
    plot_error_decomposition(
        sub,
        x_col="tau0",
        out_path=out,
        title="Fig.7: (near-)deterministic logging — everyone breaks",
        xlabel="Logging temperature $\\tau_0$ (scale)",
        ytick_decades=2,
    )
    return out


def make_table1(
    agg: pd.DataFrame, figs_dir: Path, data_dir: Optional[Path] = None
) -> Optional[Path]:
    """Table 1: 3x3 misspecification grid (true exposure structure x assumed
    exposure model class), relative MSE of ED-DR. The png/latex render into
    figs_dir; the csv goes to data_dir (defaults to figs_dir when omitted)."""
    sub = agg[
        (agg["setting_name"] == "table1_misspecification")
        & (agg["estimator"] == "ed-dr")
    ]
    if sub.empty:
        return None
    pivot = sub.pivot_table(
        index="exposure_structure",
        columns="exposure_model_class",
        values="rel_mse",
    )
    order = ["pbm", "contextual_pbm", "ranking_dependent"]
    pivot = pivot.reindex(index=order, columns=order)
    csv_dir = data_dir if data_dir is not None else figs_dir
    pivot.to_csv(csv_dir / "table1_misspecification.csv")
    with open(figs_dir / "table1_misspecification.tex", "w") as f:
        f.write(pivot.to_latex(float_format="%.3e"))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{v:.2e}" if np.isfinite(v) else "-" for v in row] for row in pivot.values],
        rowLabels=[f"true: {i}" for i in pivot.index],
        colLabels=[f"$\\hat e$: {c}" for c in pivot.columns],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.set_title(
        "Table 1: relative MSE of ED-DR, true structure x assumed class",
        fontsize=10,
        color=TEXT_COLOR,
    )
    out = figs_dir / "table1_misspecification.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# the six EM initialization schemes of the init-sensitivity appendix,
# left to right; each maps to a row filter over
# the aggregate. The historical 4 schemes live under app_em_sensitivity
# (warm_start x em_random_state); the two adversarial ones each have their
# own setting_name because `em_init` is deliberately NOT an aggregation
# column (see em_factory).
EM_INIT_SCHEMES = (
    ("warm", "app_em_sensitivity", dict(warm_start=True)),
    ("rand-0", "app_em_sensitivity", dict(warm_start=False, em_random_state=0)),
    ("rand-1", "app_em_sensitivity", dict(warm_start=False, em_random_state=1)),
    ("rand-2", "app_em_sensitivity", dict(warm_start=False, em_random_state=2)),
    ("const-0.5", "app_em_init_const", dict()),
    ("anti-mono", "app_em_init_antimono", dict()),
)
EM_INIT_ESTIMATORS = ("ed-dr", "le-iips")

# estimators whose (E1) negative-control series must stay flat in the
# a2slot scorecard check: the same family as the (E3) claim.
A2SLOT_CONTROL_ESTIMATORS = (
    "ed-dr",
    "le-iips",
    "ed-dr (oracle)",
    "le-iips (oracle)",
)


def make_em_sensitivity_fig(agg: pd.DataFrame, figs_dir: Path) -> Optional[Path]:
    """EM initialization sensitivity: the six init
    schemes on a categorical axis, ed-dr / le-iips as markers with 95% CI
    whiskers (no connecting lines — the schemes are nominal, not ordered).
    Init seeds and the per-point MC seeds are different things: every point
    aggregates its cell's MC replication seeds (count shown per tick), while
    the scheme label fixes how the EM position tower is initialized. Schemes
    absent from the logs are skipped, so the figure renders before the
    const-0.5 / anti-mono runs land."""
    sub = agg[
        agg["setting_name"].isin(
            ["app_em_sensitivity", "app_em_init_const", "app_em_init_antimono"]
        )
    ]
    if sub.empty:
        return None
    present: List[tuple] = []
    for label, setting_name, filters in EM_INIT_SCHEMES:
        rows = sub[sub["setting_name"] == setting_name]
        for col, val in filters.items():
            rows = rows[rows[col] == val]
        if not rows.empty:
            present.append((label, rows))
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    xs = np.arange(len(present), dtype=float)
    for offset, est in zip((-0.12, 0.12), EM_INIT_ESTIMATORS):
        style = METHOD_STYLE.get(est, dict(color=MUTED_COLOR, marker="."))
        means = np.full(len(present), np.nan)
        lo = np.full(len(present), np.nan)
        hi = np.full(len(present), np.nan)
        for i, (_, rows) in enumerate(present):
            g = rows[rows["estimator"] == est]
            if g.empty:
                continue
            means[i] = float(g["rel_mse"].mean())
            lo[i], hi[i] = ci_bounds(means[i], _combine_se(g["rel_mse_se"]))
        ok = np.isfinite(means)
        # whiskers only when the CIs are on (and the cell has an SE)
        band = ok & np.isfinite(lo) & np.isfinite(hi) if SHOW_CI else np.zeros_like(ok)
        if band.any():
            ax.errorbar(
                xs[band] + offset,
                means[band],
                yerr=np.vstack([means[band] - lo[band], hi[band] - means[band]]),
                fmt=style.get("marker", "."),
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.2,
                capsize=3,
                markersize=7,
                linestyle="none",
                label=est,
            )
        # cells without a whisker (CIs off, or single-seed cells with no CI)
        # still get their marker
        only_point = ok & ~band
        if only_point.any():
            ax.plot(
                xs[only_point] + offset,
                means[only_point],
                style.get("marker", "."),
                color=style["color"],
                markersize=7,
                linestyle="none",
                label=None if band.any() else est,
            )
    labels = []
    for label, rows in present:
        n_mc = int(rows[rows["estimator"].isin(EM_INIT_ESTIMATORS)]["n_seeds"].max())
        labels.append(f"{label}\n({n_mc} MC seeds)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("EM initialization scheme", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("relative MSE", fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(
        "Appendix: EM initialization sensitivity (isotonic projection on"
        + ("; whiskers: 95% CI over MC seeds)" if SHOW_CI else ")"),
        fontsize=10,
        color=TEXT_COLOR,
    )
    _style_axis(ax)
    fig.tight_layout()
    out = figs_dir / "appendix_em_sensitivity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_appendix_figs(agg: pd.DataFrame, figs_dir: Path) -> List[Path]:
    """Appendix figures: EM init sensitivity / warm start, MC
    sample size S, decay-rate lambda and slate-size K dependence, DCG alpha,
    the DBN (A2) violation and the slot-level Frechet-coupling A2' variant."""
    outs: List[Path] = []
    # EM initial-value sensitivity: six init schemes, categorical
    out = make_em_sensitivity_fig(agg, figs_dir)
    if out is not None:
        outs.append(out)
    # fig3 method E under (E3): first-order boundary demonstration
    out = make_fig3_method_e_appendix(agg, figs_dir)
    if out is not None:
        outs.append(out)
    # MC sample size S
    sub = agg[agg["setting_name"] == "app_mc_samples"]
    if not sub.empty:
        out = figs_dir / "appendix_mc_samples.png"
        plot_relmse_vs_axis(
            sub,
            x_col="n_mc_samples",
            panel_col=None,
            out_path=out,
            title="Appendix: sensitivity to the MC sample size S",
            xlabel="S (log)",
        )
        outs.append(out)
    # exposure decay rate lambda (K stays at its single production level 5;
    # the per-K panels were dropped with the decomposition redesign)
    sub = agg[agg["setting_name"] == "app_decay_K"]
    if not sub.empty:
        out = figs_dir / "appendix_decay_K.png"
        plot_error_decomposition(
            sub,
            x_col="exposure_decay_rate",
            out_path=out,
            logx=False,
            title="Appendix: exposure decay rate $\\lambda$ (K=5)",
            xlabel="Decay exponent $\\lambda$",
        )
        outs.append(out)
    # DCG position weights
    sub = agg[agg["setting_name"] == "app_position_weight"]
    if not sub.empty:
        out = figs_dir / "appendix_position_weight.png"
        plot_relmse_vs_axis(
            sub,
            x_col="n_rounds",
            panel_col="position_weight",
            out_path=out,
            title="Appendix: DCG position weights $\\alpha_k = 1/\\log_2(k+1)$",
            xlabel="n (log)",
        )
        outs.append(out)
    # (A2) robustness: DBN post-click satisfaction sweep, as a one-row error
    # decomposition vs the satisfaction probability rho on a linear axis
    # (the "(A2) violation" naming refers to the vector-level
    # independence and is kept — slot-wise (A2)/(A3) still hold, the bias
    # comes from e_hat misspecification, the same route as Figure 4).
    sub = agg[agg["setting_name"] == "app_a2_robustness"]
    if not sub.empty:
        out = figs_dir / "appendix_a2_robustness.png"
        plot_error_decomposition(
            sub,
            x_col="or_correlation",
            out_path=out,
            logx=False,
            title=(
                "Appendix: (A2) violation — DBN-generated clicks, "
                "satisfaction probability $\\rho$ ($\\rho = 0$ recovers the "
                "exposure DGP)"
            ),
            xlabel="Post-click satisfaction $\\rho$",
        )
        outs.append(out)
    # A2' — slot-level (A2)/(A3) violation via the Frechet O-R coupling
    # top row = pbm (E1, negative control: flat bias expected),
    # bottom row = ranking_dependent (E3: even the oracle cannot recover)
    sub = agg[agg["setting_name"] == "app_a2_slot_coupling"]
    if not sub.empty:
        out = figs_dir / "appendix_a2_slot_coupling.png"
        plot_error_decomposition(
            sub,
            x_col="or_coupling",
            out_path=out,
            row_col="exposure_structure",
            row_order=["pbm", "ranking_dependent"],
            row_labels={
                "pbm": "(E1) pbm — negative control",
                "ranking_dependent": "(E3) ranking-dependent",
            },
            logx=False,
            title=(
                "Appendix A2': slot-level (A2)/(A3) violation — Frechet O-R "
                "coupling $\\delta$ ($\\delta = 0$ recovers the exposure DGP; "
                "marginals preserved at every $\\delta$)"
            ),
            xlabel="within-slot coupling $\\delta$",
        )
        outs.append(out)
    return outs


def make_prediction_scorecard(
    agg: pd.DataFrame, figs_dir: Path, data_dir: Optional[Path] = None
) -> Optional[Path]:
    """The win/lose predictions of the experimental design checked against
    the measured results. One row per prediction with its verdict.
    The csv goes to data_dir (defaults to figs_dir when omitted)."""
    rows = []

    def _rel_mse(sub: pd.DataFrame, estimator: str, **filters) -> float:
        s = sub[sub["estimator"] == estimator]
        for k, v in filters.items():
            s = s[s[k] == v]
        return float(s["rel_mse"].mean()) if len(s) else np.nan

    fig1 = agg[agg["setting_name"] == "fig1_data_size"]
    if not fig1.empty:
        n_max = fig1["n_rounds"].max()
        e3 = fig1[
            (fig1["exposure_structure"] == "ranking_dependent")
            & (fig1["n_rounds"] == n_max)
        ]
        iips, eddr = _rel_mse(e3, "iips"), _rel_mse(e3, "ed-dr")
        rows.append(
            dict(
                prediction="fig1/(E3), largest n: ED-DR beats IIPS (IIPS bias floor)",
                measured=f"iips={iips:.3e}, ed-dr={eddr:.3e}",
                holds=bool(np.isfinite(iips) and np.isfinite(eddr) and eddr < iips),
            )
        )
        e1 = fig1[
            (fig1["exposure_structure"] == "pbm") & (fig1["n_rounds"] == n_max)
        ]
        iips1, eddr1 = _rel_mse(e1, "iips"), _rel_mse(e1, "ed-dr")
        rows.append(
            dict(
                prediction="fig1/(E1): ED-DR does not degrade vs IIPS (within 3x)",
                measured=f"iips={iips1:.3e}, ed-dr={eddr1:.3e}",
                holds=bool(
                    np.isfinite(iips1) and np.isfinite(eddr1) and eddr1 < 3.0 * iips1
                ),
            )
        )
    fig3 = _fig3_frame(agg)
    if fig3 is not None:
        rd = fig3["exposure_structure"] == "ranking_dependent"
        e2 = fig3["exposure_structure"] == "contextual_pbm"
        # (a)/(b): one side stays correct -> every bias CI covers 0
        ab = pd.concat(
            [
                fig3[rd & (fig3["w_method"] == "none") & (fig3["rel_t"] > 0)],
                fig3[e2 & (fig3["rel_t"] == 0.0) & (fig3["w_t"] > 0)],
            ]
        )
        if not ab.empty:
            z = (ab["bias"].abs() / ab["bias_se"]).replace([np.inf], np.nan)
            rows.append(
                dict(
                    prediction="fig3 (a)(b): bias CI covers 0 at every corruption strength (DR, corollaries 2a/2b)",
                    measured=f"max |bias|/SE={np.nanmax(z.values):.2f} over {len(ab)} cells",
                    holds=bool(np.isfinite(np.nanmax(z.values)) and np.nanmax(z.values) < CI_Z),
                )
            )
        # (c) the slope claims sit
        # on the method-R x axis, which is exactly known — no MC noise floor,
        # so no floor exclusion for the R-based series. The two-stage floor
        # protocol survives only for the method-E appendix fit, together with
        # the saturation cutoff (ratio error < 2/3 of the 0.28 ceiling).
        cond_c = fig3[rd & (fig3["w_method"] == "R")]
        cond_c_rx = cond_c[(cond_c["rel_t"] > 0.0) & (cond_c["w_t"] > 0.0)]
        cond_c_zero = cond_c[(cond_c["rel_t"] == 0.0) & (cond_c["w_t"] > 0.0)]
        cond_c_e = fig3[
            rd & (fig3["w_method"] == "E") & (fig3["rel_t"] == 0.0)
            & (fig3["w_t"] > 0.0)
        ]
        floor_rows = fig3[
            rd & (fig3["w_method"] == "none") & (fig3["rel_t"] == 0.0)
        ]
        floor = (
            float(floor_rows["oracle_ratio_error_mean"].mean())
            if not floor_rows.empty
            else 0.0
        )

        def _slope(frame: pd.DataFrame, est: str) -> float:
            g = frame[frame["estimator"] == est]
            return fit_loglog_slope(
                g["oracle_ratio_error_mean"].values, g["bias"].abs().values
            )

        def _monotone_abs_bias(frame: pd.DataFrame, est: str) -> bool:
            g = frame[frame["estimator"] == est].sort_values(
                "oracle_ratio_error_mean"
            )
            return bool(
                len(g) >= 2 and np.all(np.diff(np.abs(g["bias"].values)) > 0)
            )

        if not cond_c_rx.empty:
            s_ed = _slope(cond_c_rx, "ed-dr (oracle)")
            mono_ed = _monotone_abs_bias(cond_c_rx, "ed-dr (oracle)")
            rows.append(
                dict(
                    prediction="fig3 (c) RxRel: ED-DR (oracle) log-log slope = 2 +- 0.5 and |bias| monotone (both nuisances O(t) -> O(t^2))",
                    measured=f"slope={s_ed:.2f}, monotone={mono_ed}",
                    holds=bool(
                        np.isfinite(s_ed) and abs(s_ed - 2.0) <= 0.5 and mono_ed
                    ),
                )
            )
            s_le = _slope(cond_c_rx, "le-iips (oracle)")
            rows.append(
                dict(
                    prediction="fig3 (c) RxRel: LE-IIPS (oracle) log-log slope = 1 +- 0.3 (single-robust reference)",
                    measured=f"slope={s_le:.2f}",
                    holds=bool(np.isfinite(s_le) and abs(s_le - 1.0) <= 0.3),
                )
            )
        if not cond_c_zero.empty:
            g0 = cond_c_zero[cond_c_zero["estimator"] == "ed-dr (oracle)"]
            z0 = (g0["bias"].abs() / g0["bias_se"]).replace([np.inf], np.nan)
            zmax = float(np.nanmax(z0.values)) if len(g0) else np.nan
            rows.append(
                dict(
                    # threshold 3, not 2: 5 levels judged jointly — at 2 the
                    # false-positive rate of a truly unbiased series is ~20%
                    # (same threshold as the E1 control)
                    prediction="fig3 (c) R alone: ED-DR (oracle) exactly unbiased — all |bias|/SE < 3 (q_hat true -> DR null)",
                    measured=f"max |bias|/SE = {zmax:.2f} over {len(g0)} levels",
                    holds=bool(np.isfinite(zmax) and zmax < 3.0),
                )
            )
        if not cond_c_e.empty:
            sat_cut = FIG3_E_SATURATION_CEIL * 2.0 / 3.0
            e_fit = cond_c_e[
                (cond_c_e["oracle_ratio_error_mean"] >= 3.0 * floor)
                & (cond_c_e["oracle_ratio_error_mean"] < sat_cut)
            ]
            s_ed_e = _slope(e_fit, "ed-dr (oracle)")
            ratios = []
            for _, g in cond_c_e.groupby("w_t"):
                b_ed = g[g["estimator"] == "ed-dr (oracle)"]["bias"].abs().mean()
                b_le = g[g["estimator"] == "le-iips (oracle)"]["bias"].abs().mean()
                if np.isfinite(b_ed) and np.isfinite(b_le) and b_le > 0:
                    ratios.append(b_ed / b_le)
            ratio_ok = bool(
                ratios and all(0.5 <= v <= 2.0 for v in ratios)
            )
            rows.append(
                dict(
                    prediction="fig3 appendix method E: FIRST-order (non-saturated slope 1 +- 0.3) and ed-dr/le-iips |bias| ratio in [0.5, 2]",
                    measured=(
                        f"slope={s_ed_e:.2f} (floor={floor:.2e}, saturation cut={sat_cut:.2f}), "
                        f"ratio range=[{min(ratios):.2f}, {max(ratios):.2f}]"
                        if ratios
                        else f"slope={s_ed_e:.2f}, no ratio cells"
                    ),
                    holds=bool(
                        np.isfinite(s_ed_e)
                        and abs(s_ed_e - 1.0) <= 0.3
                        and ratio_ok
                    ),
                )
            )
    fig4 = agg[agg["setting_name"] == "fig4_cascade"]
    if not fig4.empty:
        rips, eddr4 = _rel_mse(fig4, "rips"), _rel_mse(fig4, "ed-dr")
        rows.append(
            dict(
                prediction="fig4/cascade DGP: RIPS beats ED-DR ((A4) broken, e_hat misspecified)",
                measured=f"rips={rips:.3e}, ed-dr={eddr4:.3e}",
                holds=bool(np.isfinite(rips) and np.isfinite(eddr4) and rips < eddr4),
            )
        )
    fig4eta = _fig4_eta_frame(agg)
    if fig4eta is not None:
        # the measured eta=0 side is statistically zero for both, so a
        # "crossing" cannot be claimed: at eta=0 ed-dr and rips bias^2 are both
        # indistinguishable from 0 (z < 2); at eta >= 2 ed-dr significantly
        # exceeds rips (delta-method z on bias^2 > 3). No variance-trend
        # claim (the RIPS "flat variance" prediction was refuted, 1.9 SE).
        etas = sorted(fig4eta["exposure_decay_rate"].unique())

        def _bias_and_se(est, eta):
            g = fig4eta[
                (fig4eta["estimator"] == est)
                & (fig4eta["exposure_decay_rate"] == eta)
            ]
            return float(g["bias"].mean()), _combine_se(g["bias_se"])

        b_ed0, se_ed0 = _bias_and_se("ed-dr", etas[0])
        b_r0, se_r0 = _bias_and_se("rips", etas[0])
        z_eta0 = max(abs(b_ed0) / se_ed0, abs(b_r0) / se_r0)
        z_diff_large = []
        for eta in [e for e in etas if e >= 2.0]:
            b_ed, se_ed = _bias_and_se("ed-dr", eta)
            b_r, se_r = _bias_and_se("rips", eta)
            # delta method on bias^2: var(b^2) ~ (2 b se)^2
            se_diff = np.sqrt((2 * b_ed * se_ed) ** 2 + (2 * b_r * se_r) ** 2)
            z_diff_large.append((b_ed**2 - b_r**2) / se_diff if se_diff > 0 else np.nan)
        rows.append(
            dict(
                prediction="fig4eta: bias^2 both ~0 at eta=0 (z<2); ed-dr significantly above rips at every eta>=2 (z>3)",
                measured=(
                    f"eta=0 max |bias|/SE={z_eta0:.2f}; "
                    "bias^2-diff z at eta>=2: "
                    + ", ".join(f"{z:.1f}" for z in z_diff_large)
                ),
                holds=bool(
                    np.isfinite(z_eta0)
                    and z_eta0 < 2.0
                    and len(z_diff_large) > 0
                    and np.isfinite(z_diff_large).all()
                    and all(z > 3.0 for z in z_diff_large)
                ),
            )
        )
    a2slot = agg[agg["setting_name"] == "app_a2_slot_coupling"]
    if not a2slot.empty:
        # the coupling breaks (A2)/(A3) inside the slot, so even the ORACLE
        # proposal is biased
        # under (E3) — some delta >= 0.5 with |bias|/SE > 3 AND the
        # delta=1 vs delta=0 endpoint difference z > 4 — while the (E1)
        # control stays flat (threshold 3, not 2: 4 series x 5 levels = 20
        # cells judged jointly).
        # The (E1) control is scoped to the proposal family
        # le-iips / ed-dr (+ oracle) — the same series the (E3) claim is
        # about — so both sides of the comparison cover the same
        # estimators. dm and aips are biased at delta=0 already (|bias|/SE
        # = 259 and 28) by regression misspecification and by the injected
        # noise_level, neither of which involves (A2)/(A3); including them
        # made this condition unsatisfiable regardless of the experiment.
        e3 = a2slot[a2slot["exposure_structure"] == "ranking_dependent"]
        e1 = a2slot[
            (a2slot["exposure_structure"] == "pbm")
            & (a2slot["estimator"].isin(A2SLOT_CONTROL_ESTIMATORS))
        ]
        e3_large = e3[
            (e3["or_coupling"] >= 0.5)
            & (e3["estimator"].isin(["ed-dr (oracle)", "le-iips (oracle)"]))
        ]
        z_e3 = (e3_large["bias"].abs() / e3_large["bias_se"]).max()
        z_ends = []
        for est in ("ed-dr (oracle)", "le-iips (oracle)"):
            g = e3[e3["estimator"] == est]
            lo = g[g["or_coupling"] == g["or_coupling"].min()]
            hi = g[g["or_coupling"] == g["or_coupling"].max()]
            if lo.empty or hi.empty:
                continue
            se = np.sqrt(
                _combine_se(lo["bias_se"]) ** 2 + _combine_se(hi["bias_se"]) ** 2
            )
            z_ends.append(
                abs(float(hi["bias"].mean()) - float(lo["bias"].mean())) / se
            )
        z_end = min(z_ends) if z_ends else np.nan
        z_e1 = (e1["bias"].abs() / e1["bias_se"]).max()
        rows.append(
            dict(
                prediction="a2_slot: oracle biased under (E3) — some delta>=0.5 |bias|/SE>3 and endpoint diff z>4; (E1) control flat (le-iips/ed-dr +oracle, all |bias|/SE<3)",
                measured=(
                    f"max |bias|/SE E3 oracle={z_e3:.2f}, min endpoint z={z_end:.2f}, "
                    f"E1 proposal family={z_e1:.2f}"
                ),
                holds=bool(
                    np.isfinite(z_e3)
                    and z_e3 > 3.0
                    and np.isfinite(z_end)
                    and z_end > 4.0
                    and np.isfinite(z_e1)
                    and z_e1 < 3.0
                ),
            )
        )
    tbl = agg[
        (agg["setting_name"] == "table1_misspecification")
        & (agg["estimator"] == "ed-dr")
    ]
    if not tbl.empty:
        diag = tbl[tbl["exposure_structure"] == tbl["exposure_model_class"]][
            "rel_mse"
        ].mean()
        off = tbl[tbl["exposure_structure"] != tbl["exposure_model_class"]][
            "rel_mse"
        ].mean()
        rows.append(
            dict(
                prediction="table1: matching-class diagonal <= mismatched-class off-diagonal",
                measured=f"diag={diag:.3e}, off-diag={off:.3e}",
                holds=bool(np.isfinite(diag) and np.isfinite(off) and diag <= off),
            )
        )
    if not rows:
        return None
    csv_dir = data_dir if data_dir is not None else figs_dir
    out = csv_dir / "prediction_scorecard.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def make_run_summary_figure(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Per-run quick look: relative MSE per estimator (bias^2 + variance)."""
    agg = aggregate_results(df)
    estimators = _ordered_estimators(agg["estimator"].unique())
    b = np.array(
        [agg[agg["estimator"] == e]["bias_sq"].mean() for e in estimators]
    )
    v = np.array(
        [agg[agg["estimator"] == e]["variance"].mean() for e in estimators]
    )
    colors = [METHOD_STYLE.get(e, dict(color=MUTED_COLOR))["color"] for e in estimators]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = np.arange(len(estimators))
    ax.bar(xs, b, color=colors, width=0.7)
    ax.bar(xs, v, bottom=b, color=colors, alpha=0.45, hatch="///",
           edgecolor="white", linewidth=0.5, width=0.7)
    total = np.array(
        [agg[agg["estimator"] == e]["rel_mse"].mean() for e in estimators]
    )
    if SHOW_CI:
        ses = np.array(
            [_combine_se(agg[agg["estimator"] == e]["rel_mse_se"]) for e in estimators]
        )
        lo, hi = ci_bounds(total, ses)
        ax.errorbar(
            xs,
            total,
            yerr=np.vstack([total - lo, hi - total]),
            fmt="none",
            ecolor=TEXT_COLOR,
            elinewidth=1.0,
            capsize=2.5,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(estimators, rotation=90, fontsize=8)
    ax.set_ylabel("relative MSE (bias$^2$ solid + variance hatched)", fontsize=9)
    ax.set_title(title, fontsize=10, color=TEXT_COLOR)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# entry points
# ===========================================================================
def run_experiment(cfg: DictConfig) -> pd.DataFrame:
    # flatten first: it validates the config (e.g. mutually exclusive
    # weight-side corruption channels) before hours of replications start
    flat_setting = flatten_setting(cfg.setting)
    seeds = list(range(int(cfg.start_seed), int(cfg.start_seed) + int(cfg.n_seeds)))
    n_jobs = int(cfg.n_jobs)
    if n_jobs == 1:
        all_rows = [run_one_replicate(cfg, seed) for seed in seeds]
    else:
        all_rows = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(run_one_replicate)(cfg, seed) for seed in seeds
        )
    rows = [dict(flat_setting, **row) for rows_ in all_rows for row in rows_]
    return pd.DataFrame(rows)


def summarize_seed_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per setting_name: how many config cells there are and how many seeds
    each (config x estimator) cell actually holds.

    Seeds are counted as *rows*, NaN estimates included — unlike the n_seeds
    column of `aggregate_results`, which counts non-NaN rel_err and is 0 for
    the intentionally skipped cells (fig2 epsilon=0 x cascade-dr). Rows are
    what a merged cycle adds, so a row count is the check that tells whether
    a cycle actually landed."""
    counts = (
        df.groupby(CONFIG_KEYS + ["estimator"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    per_setting = counts.groupby("setting_name", dropna=False)["n_rows"].agg(
        seeds_min="min", seeds_max="max"
    )
    n_configs = (
        counts[CONFIG_KEYS]
        .drop_duplicates()
        .groupby("setting_name", dropna=False)
        .size()
        .rename("n_configs")
    )
    return pd.concat([n_configs, per_setting], axis=1).reset_index()


def report_seed_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Print the seed-count summary and warn on uneven cells (F5 / F6).

    An uneven setting means some stage of a cycle did not finish (or only a
    subset of the sweep was re-run); the figures would then average cells with
    different numbers of replications. It is a warning, not an error — a
    partial cycle is still a valid intermediate state — and re-running the
    missing stage with the same CYCLE resolves it."""
    summary = summarize_seed_counts(df)
    if summary.empty:
        # every results.csv held headers only (a stray n_seeds=0 job leaves one
        # behind, and it then sits in logs/ forever). Say so rather than dying
        # on max() of an empty sequence in the column width below — the caller
        # still gets the empty frame and fails on the real problem downstream.
        print("seed counts per setting: no rows (every results.csv is empty)")
        return summary
    print("seed counts per setting (rows per config x estimator, NaN included):")
    width = max(len(str(s)) for s in summary["setting_name"])
    for row in summary.itertuples(index=False):
        print(
            f"  {str(row.setting_name):<{width}}  configs={row.n_configs:>3}  "
            f"seeds/cell: min={row.seeds_min} max={row.seeds_max}"
        )
    uneven = summary[summary["seeds_min"] != summary["seeds_max"]]
    if not uneven.empty:
        names = ", ".join(str(s) for s in uneven["setting_name"])
        print(
            f"WARNING: uneven seed counts per cell in setting(s) {names} — a "
            "stage of some cycle did not finish, so the figures would average "
            "cells with different numbers of replications. Re-run the missing "
            "stage with the same CYCLE.",
            file=sys.stderr,
        )
    return summary


def load_results_csvs(log_root: Path) -> Optional[pd.DataFrame]:
    """Read every results.csv under log_root, validate, and de-duplicate.

    Each csv is checked for the full CONFIG_KEYS column set *before* concat:
    a check on the concatenated frame passes as soon as one new-format csv
    is present, and rows from stale csvs would then survive de-duplication
    as NaN cells. A mix of exposure_ratio_clip
    values within one setting_name is refused outright: the figure functions
    select rows by setting_name only and average rel_mse over whatever cells
    remain, so a clipped main run and an unclipped appendix run under the
    same log_dir would silently blend into one line — appendix runs with
    exposure_ratio_clip=null need their own log_dir.

    Rows from different cycles (disjoint seed bands of the same config) all
    survive de-duplication and stack up as extra replications; the per-setting
    seed-count summary printed at the end is how a merged cycle is verified.

    Rows of RETIRED_ESTIMATORS (sips) are dropped after de-duplication: the
    earlier cycles computed them, later ones do not, and keeping them
    would both put sips back into every figure and make the seed-count check
    uneven for a reason that is not a missing stage.
    """
    csvs = sorted(log_root.glob("**/results.csv"))
    if not csvs:
        return None
    frames = []
    for p in csvs:
        frame = pd.read_csv(p)
        # the ONLY backfilled keys — both are exact, not guesses: every
        # pre-or_coupling run used the uncoupled DGPs (delta = 0) and every
        # pre-relevance_scale run used the unscaled relevance (s = 1). Any
        # other missing config column still errors below.
        if "or_coupling" not in frame.columns:
            frame["or_coupling"] = 0.0
        if "relevance_scale" not in frame.columns:
            frame["relevance_scale"] = 1.0
        missing = [k for k in CONFIG_KEYS if k not in frame.columns]
        if missing:
            raise ValueError(
                f"{p} lacks config columns {missing}: a stale run from an "
                "older code version cannot be de-duplicated safely — delete "
                "or regenerate it before plotting"
            )
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    # de-duplicate re-runs of an identical (config, seed, estimator) cell,
    # keeping the latest file (glob order is chronological by dir name)
    df = df.drop_duplicates(subset=CONFIG_KEYS + ["seed", "estimator"], keep="last")
    retired = df["estimator"].isin(RETIRED_ESTIMATORS)
    if retired.any():
        print(
            f"dropped {int(retired.sum())} rows of retired estimator(s) "
            f"{sorted(df.loc[retired, 'estimator'].unique())} (RETIRED_ESTIMATORS)"
        )
        df = df[~retired].copy()
    clip_per_setting = df.groupby("setting_name")["exposure_ratio_clip"].nunique(
        dropna=False
    )
    mixed = clip_per_setting[clip_per_setting > 1]
    if not mixed.empty:
        raise ValueError(
            f"multiple exposure_ratio_clip values under {log_root} for "
            f"setting(s) {sorted(mixed.index)}: the figures would average "
            "clipped and unclipped cells into one line — move the no-clip "
            "appendix runs to a separate log_dir"
        )
    report_seed_counts(df)
    return df


def run_plot_mode(cfg: DictConfig) -> None:
    global SHOW_CI
    if cfg.get("show_ci") is not None:
        SHOW_CI = bool(cfg.show_ci)
    figs_dir = REPO_ROOT / "figs"
    figs_dir.mkdir(exist_ok=True)
    # Aggregate CSV tables go to data/<date>/ so each plot run is archived
    # separately; data_date defaults to today and can be pinned via override
    # (e.g. `mode=plot data_date=2026-07-20`).
    date_str = str(cfg.data_date) if cfg.get("data_date") else datetime.now().strftime("%Y-%m-%d")
    data_dir = REPO_ROOT / "data" / date_str
    data_dir.mkdir(parents=True, exist_ok=True)
    log_root = REPO_ROOT / str(cfg.log_dir)
    df = load_results_csvs(log_root)
    if df is None:
        print(f"no results.csv found under {log_root}")
        return
    agg = aggregate_results(df)
    agg.to_csv(data_dir / "aggregate_all.csv", index=False)
    made = []
    for fn in (
        make_fig1, make_fig2, make_fig3, make_fig3_injection, make_fig4_eta,
        make_fig5_cascade, make_fig7_tau0,
    ):
        out = fn(agg, figs_dir)
        if out is not None:
            made.append(out)
    out_t1 = make_table1(agg, figs_dir, data_dir=data_dir)
    if out_t1 is not None:
        made.append(out_t1)
    out4 = make_fig4_bias_bound(df, figs_dir)
    if out4 is not None:
        made.append(out4)
    made += make_appendix_figs(agg, figs_dir)
    make_prediction_scorecard(agg, figs_dir, data_dir=data_dir)
    print(f"data tables (csv) written under: {data_dir}")
    for name in (
        "aggregate_all.csv",
        "table1_misspecification.csv",
        "prediction_scorecard.csv",
    ):
        print(f"  {data_dir / name}")
    print("figures written:")
    for p in made:
        print(f"  {p}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.mode == "plot":
        run_plot_mode(cfg)
        return
    df = run_experiment(cfg)
    out_dir = Path.cwd()  # hydra job dir (hydra.job.chdir=true)
    df.to_csv(out_dir / "results.csv", index=False)
    agg = aggregate_results(df)
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    print(
        agg[["estimator", "rel_mse", "bias_sq", "variance", "n_seeds"]]
        .sort_values("rel_mse")
        .to_string(index=False)
    )
    figs_dir = REPO_ROOT / "figs"
    figs_dir.mkdir(exist_ok=True)
    make_run_summary_figure(
        df,
        figs_dir / f"{cfg.setting.name}_summary.png",
        title=f"{cfg.setting.name}: n={cfg.setting.n_rounds}, "
        f"{cfg.setting.exposure_structure}, seeds={cfg.n_seeds}",
    )


if __name__ == "__main__":
    main()
