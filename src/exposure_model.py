# EM-based decomposed reward model (exposure x relevance) and the Monte-Carlo
# computation of the conditional expected exposure under the evaluation policy.
#
# The class skeleton (dataclass, sklearn `base_model`
# injected and `clone`d, fit / predict / fit_predict) follows
# `obp/ope/regression_model_slate.py::SlateRegressionModel`, but note the
# REVERSED validation: the EM needs `predict_proba`, so `relevance_model`
# must be a *classifier* (SlateRegressionModel demands a regressor — known
# copy-paste hazard).
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from obp.utils import check_array

EXPOSURE_MODEL_CLASSES = ("pbm", "contextual_pbm", "ranking_dependent")
PROB_CLIP = 1e-4


# --------------------------------------------------------------------------
# shared feature maps (also reused by the direct click regression so that the
# DM / DR baselines are compared on equal footing)
# --------------------------------------------------------------------------
def relevance_feature(
    context: np.ndarray, action: np.ndarray, n_unique_action: int, len_list: int
) -> np.ndarray:
    """Features of the relevance tower r(x, a(k)): (x, one-hot(a)); NO position.

    `action` is the flat long-format array (n * len_list,); row i * len_list + k
    corresponds to round i, slot k.
    """
    context_rep = np.repeat(context, len_list, axis=0)
    action_onehot = np.eye(n_unique_action)[action]
    return np.concatenate([context_rep, action_onehot], axis=1)


def exposure_feature(
    context: np.ndarray,
    action_2d: np.ndarray,
    exposure_model_class: str,
    len_list: int,
    n_unique_action: int,
) -> np.ndarray:
    """Features of the exposure tower e_k(x, a) in long format (n * len_list, .).

    - "pbm":              position one-hot only (K free parameters theta_k)
    - "contextual_pbm":   position one-hot + position x context interactions
    - "ranking_dependent": the above + sum over upper slots j < k of the
                          one-hot features of a(j) (which items sit above).
    """
    n = context.shape[0]
    pos_onehot = np.tile(np.eye(len_list), (n, 1))  # (n * K, K)
    blocks = [pos_onehot]
    if exposure_model_class in ("contextual_pbm", "ranking_dependent"):
        context_rep = np.repeat(context, len_list, axis=0)  # (n * K, d)
        interaction = (
            pos_onehot[:, :, None] * context_rep[:, None, :]
        ).reshape(n * len_list, -1)  # (n * K, K * d)
        blocks.append(interaction)
    if exposure_model_class == "ranking_dependent":
        onehot_slates = np.eye(n_unique_action)[action_2d]  # (n, K, m)
        cum_above = np.concatenate(
            [
                np.zeros((n, 1, n_unique_action)),
                np.cumsum(onehot_slates, axis=1)[:, :-1, :],
            ],
            axis=1,
        )  # sum_{j<k} one-hot(a(j))
        blocks.append(cum_above.reshape(n * len_list, n_unique_action))
    return np.concatenate(blocks, axis=1)


def click_feature(
    context: np.ndarray, action: np.ndarray, n_unique_action: int, len_list: int
) -> np.ndarray:
    """Features of the direct click regression q_hat_k(x, a): position one-hot
    x item features (same feature functions as the EM towers for fairness)."""
    n = context.shape[0]
    pos_onehot = np.tile(np.eye(len_list), (n, 1))
    return np.concatenate(
        [pos_onehot, relevance_feature(context, action, n_unique_action, len_list)],
        axis=1,
    )


class _ConstantProbabilityModel:
    """Fallback used when a fit would see a single class (tiny folds)."""

    def __init__(self, p: float):
        self.p = float(np.clip(p, PROB_CLIP, 1 - PROB_CLIP))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], 2))
        out[:, 1] = self.p
        out[:, 0] = 1.0 - self.p
        return out


def _fit_classifier_or_constant(
    model: BaseEstimator, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
):
    """Fit a classifier defensively; degrade to a constant when only one label
    (weighted) is present."""
    pos = float(np.dot(sample_weight, y))
    total = float(sample_weight.sum())
    if pos <= 0.0 or pos >= total:
        return _ConstantProbabilityModel(pos / max(total, 1e-12))
    fitted = clone(model)
    fitted.fit(X, y, sample_weight=sample_weight)
    return fitted


# --------------------------------------------------------------------------
# EM
# --------------------------------------------------------------------------
@dataclass
class ExposureRelevanceEM:
    """EM posterior inference of the decomposed click model Y = O * R.

    E-step (plan eq. (4.1)):
        gamma_ik = Pr(O = 1 | Y = 0) = e (1 - r) / (1 - e r);  gamma = 1 if Y = 1.
    M-step (exposure): weighted binary regression on duplicated records
        (label 1 with weight gamma, label 0 with weight 1 - gamma).
    M-step (relevance): weighted binary regression with label Y and weight
        gamma on features (x, a(k)) only (no position).

    Parameters
    ----------
    len_list, n_unique_action: slate geometry.
    relevance_model: sklearn *classifier* with `predict_proba` (validated).
    exposure_model: sklearn classifier for the exposure tower; only used for
        exposure_model_class != "pbm" ("pbm" has the closed-form weighted-mean
        solution per position). Defaults to LogisticRegression.
    exposure_model_class: which exposure feature map to use — the
        misspecification switch of the experiments.
    n_em_iter: maximum number of EM iterations.
    tol: early-stopping threshold on the mean observed-data log-likelihood.
    warm_start: initialize theta by intervention harvesting (Agarwal+ 2019).
    monotone_position_tower: project theta_k to be non-increasing in k via
        isotonic regression after each M-step (applied for the
        position-only "pbm" tower).
    init_scheme: None (default) keeps the historical initialization logic
        (warm_start decides; see below). The two explicit schemes of the
        init-sensitivity appendix require
        warm_start=False:
          "const"     — theta_k = 0.5 for every position;
          "anti_mono" — ascending sorted uniforms (monotonicity deliberately
                        reversed; adversarial start), seeded by random_state.
    random_state: used only for the random initialization when warm_start=False
        (init-sensitivity experiments). When warm_start=False and
        random_state is None, e is initialized at the constant 0.5.
    """

    len_list: int
    n_unique_action: int
    relevance_model: BaseEstimator = None
    exposure_model: Optional[BaseEstimator] = None
    exposure_model_class: str = "pbm"
    n_em_iter: int = 20
    tol: float = 1e-6
    warm_start: bool = True
    monotone_position_tower: bool = True
    init_scheme: Optional[str] = None
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        check_scalar(self.len_list, "len_list", int, min_val=1)
        check_scalar(self.n_unique_action, "n_unique_action", int, min_val=2)
        check_scalar(self.n_em_iter, "n_em_iter", int, min_val=1)
        if self.exposure_model_class not in EXPOSURE_MODEL_CLASSES:
            raise ValueError(
                f"`exposure_model_class` must be one of {EXPOSURE_MODEL_CLASSES}, "
                f"but {self.exposure_model_class} is given"
            )
        if self.relevance_model is None:
            self.relevance_model = LogisticRegression(max_iter=1000, C=100.0)
        if not isinstance(self.relevance_model, BaseEstimator):
            raise ValueError("`relevance_model` must be a sklearn BaseEstimator")
        # REVERSED w.r.t. SlateRegressionModel.__post_init__ (L56): the EM needs
        # class probabilities, so a classifier with predict_proba is REQUIRED.
        if not is_classifier(self.relevance_model):
            raise ValueError(
                "`relevance_model` must be a classifier with `predict_proba` "
                "(the EM works on probabilities), not a regressor"
            )
        if not hasattr(self.relevance_model, "predict_proba"):
            raise ValueError("`relevance_model` must implement `predict_proba`")
        if self.exposure_model is None:
            self.exposure_model = LogisticRegression(max_iter=1000, C=100.0)
        if not is_classifier(self.exposure_model):
            raise ValueError("`exposure_model` must be a classifier")
        if self.init_scheme not in (None, "const", "anti_mono"):
            raise ValueError(
                "`init_scheme` must be one of None, 'const', 'anti_mono', "
                f"but {self.init_scheme} is given"
            )
        if self.init_scheme is not None and self.warm_start:
            raise ValueError(
                "`init_scheme` overrides the initialization, so it requires "
                "warm_start=False (warm start would silently win otherwise)"
            )

    # ------------------------------------------------------------------
    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
    ) -> "ExposureRelevanceEM":
        check_array(array=context, name="context", expected_dim=2)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=position, name="position", expected_dim=1)
        n = context.shape[0]
        if action.shape[0] != n * self.len_list:
            raise ValueError(
                "Expected `action.shape[0] == context.shape[0] * len_list`"
            )
        y = reward.reshape((n, self.len_list)).astype(float)
        action_2d = action.reshape((n, self.len_list))

        # ---- initialization -------------------------------------------------
        theta_init = self._initial_theta(
            action=action, reward=reward, position=position
        )
        self.theta_ = np.clip(theta_init, PROB_CLIP, 1.0)
        self._exposure_clf_ = None  # non-pbm tower, fitted in the first M-step
        e_hat = np.tile(self.theta_, (n, 1))

        # relevance init: fit on ALL samples with label Y (the "Y=1 are the
        # positives, everything is the population" initial fit of the plan)
        relevance_features = relevance_feature(context, action, self.n_unique_action, self.len_list)
        self._relevance_clf_ = _fit_classifier_or_constant(
            self.relevance_model, relevance_features, reward, np.ones_like(reward, dtype=float)
        )
        r_hat = self._predict_relevance_fitted(relevance_features).reshape((n, self.len_list))

        exposure_features = exposure_feature(
            context, action_2d, self.exposure_model_class, self.len_list, self.n_unique_action
        )

        self.likelihood_history_: List[float] = []
        prev_loglik = -np.inf
        self.n_iter_ = 0
        for it in range(self.n_em_iter):
            # ---- E-step: eq. (4.1); Y = 1 pins the exposure to 1 ------------
            denom = np.clip(1.0 - e_hat * r_hat, PROB_CLIP, None)
            gamma = np.where(y == 1.0, 1.0, e_hat * (1.0 - r_hat) / denom)
            gamma = np.clip(gamma, 0.0, 1.0)
            gamma_flat = gamma.flatten()

            # ---- M-step (exposure tower) ------------------------------------
            if self.exposure_model_class == "pbm":
                # weighted binary regression on position one-hots == the
                # per-position weighted mean of gamma (closed form)
                self.theta_ = np.clip(gamma.mean(axis=0), PROB_CLIP, 1.0)
                if self.monotone_position_tower and self.len_list > 1:
                    iso = IsotonicRegression(
                        increasing=False, y_min=PROB_CLIP, y_max=1.0
                    )
                    self.theta_ = iso.fit_transform(
                        np.arange(self.len_list), self.theta_
                    )
                e_hat = np.tile(self.theta_, (n, 1))
            else:
                # duplicate each record: label 1 with weight gamma, label 0
                # with weight 1 - gamma
                features_dup = np.concatenate([exposure_features, exposure_features], axis=0)
                labels_dup = np.concatenate(
                    [np.ones(n * self.len_list), np.zeros(n * self.len_list)]
                )
                weights_dup = np.concatenate([gamma_flat, 1.0 - gamma_flat])
                self._exposure_clf_ = _fit_classifier_or_constant(
                    self.exposure_model, features_dup, labels_dup, weights_dup
                )
                e_hat = self._exposure_clf_.predict_proba(exposure_features)[:, 1].reshape(
                    (n, self.len_list)
                )
                e_hat = np.clip(e_hat, PROB_CLIP, 1.0)
                # keep a position-marginal theta_ for diagnostics / monotone check
                self.theta_ = np.array(
                    [e_hat[:, k].mean() for k in range(self.len_list)]
                )

            # ---- M-step (relevance tower) ------------------------------------
            # regression on the samples inferred to be exposed: label Y,
            # weight gamma (Y = 1 has gamma = 1), features (x, a(k)) only
            self._relevance_clf_ = _fit_classifier_or_constant(
                self.relevance_model, relevance_features, reward, gamma_flat
            )
            r_hat = self._predict_relevance_fitted(relevance_features).reshape((n, self.len_list))

            # ---- observed-data log-likelihood & early stopping ---------------
            q = np.clip(e_hat * r_hat, PROB_CLIP, 1 - PROB_CLIP)
            loglik = float(np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))
            self.likelihood_history_.append(loglik)
            self.n_iter_ = it + 1
            if loglik - prev_loglik < self.tol and it > 0:
                break
            prev_loglik = loglik
        return self

    # ------------------------------------------------------------------
    def _initial_theta(
        self, action: np.ndarray, reward: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        """Initial position tower theta_k. The explicit init schemes of the
        init-sensitivity appendix take precedence; otherwise the
        historical logic applies unchanged (warm start > sorted-descending
        uniforms seeded by random_state > constant 0.5)."""
        if self.init_scheme == "const":
            return np.full(self.len_list, 0.5)
        if self.init_scheme == "anti_mono":
            # ascending = the monotone-reversed adversarial start; the sorted
            # DESCENDING uniforms below are the historical random scheme
            rng = check_random_state(self.random_state)
            return np.sort(rng.uniform(0.05, 1.0, size=self.len_list))
        if self.warm_start:
            return self._warm_start_by_intervention_harvesting(
                action=action, reward=reward, position=position
            )
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            return np.sort(rng.uniform(0.05, 1.0, size=self.len_list))[::-1]
        return np.full(self.len_list, 0.5)

    # ------------------------------------------------------------------
    def _warm_start_by_intervention_harvesting(
        self, action: np.ndarray, reward: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        """theta_k / theta_1 from CTR ratios of (position, action) pairs where
        the same item appeared at multiple positions (Agarwal et al. 2019).
        Plackett-Luce logging provides sufficient overlap. theta_1 is
        normalized to 1 (only the ratio is identified; the absolute scale
        cancels in the estimators).

        WARM START ONLY, not a consistent estimator of theta_k / theta_1:
        under PL logging, WHICH contexts put item a at position k
        is itself context-dependent, so the (position, action)-conditional CTR
        ratio carries context confounding. The EM overwrites it, so this is
        harmless as an initializer — do not present it as an estimator."""
        df = pd.DataFrame(
            {"position": position, "action": action, "reward": reward}
        )
        stats = df.groupby(["position", "action"])["reward"].agg(["mean", "count"])
        ctr = stats["mean"].unstack(fill_value=np.nan)  # (K, m)
        cnt = stats["count"].unstack(fill_value=0.0)
        ctr_matrix = ctr.reindex(
            index=range(self.len_list), columns=range(self.n_unique_action)
        ).to_numpy()
        count_matrix = cnt.reindex(
            index=range(self.len_list), columns=range(self.n_unique_action)
        ).fillna(0.0).to_numpy()
        theta = np.ones(self.len_list)
        base_ctr = ctr_matrix[0]
        for k in range(1, self.len_list):
            valid = (
                (count_matrix[k] > 0)
                & (count_matrix[0] > 0)
                & np.isfinite(ctr_matrix[k])
                & np.isfinite(base_ctr)
                & (base_ctr > 0)
            )
            if valid.sum() == 0:
                theta[k] = theta[k - 1]
                continue
            weights = np.minimum(count_matrix[k][valid], count_matrix[0][valid])
            ratios = ctr_matrix[k][valid] / base_ctr[valid]
            theta[k] = float(np.average(ratios, weights=weights))
        theta = np.clip(theta, PROB_CLIP, 1.0)
        if self.monotone_position_tower and self.len_list > 1:
            iso = IsotonicRegression(increasing=False, y_min=PROB_CLIP, y_max=1.0)
            theta = iso.fit_transform(np.arange(self.len_list), theta)
        return theta

    # ------------------------------------------------------------------
    def _predict_relevance_fitted(self, relevance_features: np.ndarray) -> np.ndarray:
        return np.clip(
            self._relevance_clf_.predict_proba(relevance_features)[:, 1], PROB_CLIP, 1 - PROB_CLIP
        )

    def predict_exposure(self, context: np.ndarray, action_2d: np.ndarray) -> np.ndarray:
        """e_hat_k(x, a); shape (n, len_list)."""
        n = context.shape[0]
        if self.exposure_model_class == "pbm" or self._exposure_clf_ is None:
            return np.tile(self.theta_, (n, 1))
        X = exposure_feature(
            context, action_2d, self.exposure_model_class, self.len_list, self.n_unique_action
        )
        e = self._exposure_clf_.predict_proba(X)[:, 1].reshape((n, self.len_list))
        return np.clip(e, PROB_CLIP, 1.0)

    def predict_relevance(self, context: np.ndarray, action: np.ndarray) -> np.ndarray:
        """r_hat(x, a(k)); flat shape (n * len_list,)."""
        relevance_features = relevance_feature(context, action, self.n_unique_action, self.len_list)
        return self._predict_relevance_fitted(relevance_features)

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and return the factual nuisances on the same data:
        (exposure_factual_hat (n * len_list,), relevance_factual_hat (n * len_list,))."""
        self.fit(context=context, action=action, reward=reward, position=position)
        n = context.shape[0]
        action_2d = action.reshape((n, self.len_list))
        return (
            self.predict_exposure(context, action_2d).flatten(),
            self.predict_relevance(context, action),
        )


# --------------------------------------------------------------------------
# oracle nuisance (true e / r injected; optional controlled corruption)
# --------------------------------------------------------------------------
@dataclass
class OracleNuisance:
    """Duck-typed replacement of ExposureRelevanceEM backed by the true DGP.

    Used for milestone M3 and Figure 3: the corruption hooks deliberately
    break one side while keeping it inside its function class. NOTE (Figure 3
    redesign 2026-07-18): the corrupt_exposure hook is kept for the M3
    Corollary 2b tests only — an in-class e_hat error cancels exactly in the
    weight ratio, so Figure 3's weight-channel corruption is applied to
    e_bar^pi in src/main.py instead (see make_corruption there).
    The relevance corruption is a pointwise map of r(x, a(k)),
    so it stays a function of (x, a(k)) — the Corollary 2a regime for any true
    exposure structure. The exposure corruption is a pointwise map of e
    values; ONLY when the true exposure is (E1)/(E2) (position(/context)-only)
    does a pointwise map of theta_k(x) stay within the position x context
    class so that condition (4.6) / Corollary 2b applies. Under a true (E3)
    structure the relative error of a pointwise map depends on the rest of
    the ranking and (4.6) fails — such runs measure regime-outside bias, not
    Corollary 2b (a warning is emitted).
    """

    dataset: "ExposureSlateBanditDataset"  # noqa: F821 (imported lazily)
    exposure_model_class: str = None
    corrupt_relevance: Optional[Callable[[np.ndarray], np.ndarray]] = None
    corrupt_exposure: Optional[Callable[[np.ndarray], np.ndarray]] = None
    len_list: int = field(init=False)
    n_unique_action: int = field(init=False)

    def __post_init__(self) -> None:
        self.len_list = self.dataset.len_list
        self.n_unique_action = self.dataset.n_unique_action
        if self.exposure_model_class is None:
            # mirror the true structure so that the pbm/contextual shortcut
            # (correction ratio == 1) is applied exactly when it should be
            self.exposure_model_class = self.dataset.exposure_structure
        if (
            self.corrupt_exposure is not None
            and getattr(self.dataset, "exposure_structure", None)
            == "ranking_dependent"
        ):
            warnings.warn(
                "corrupt_exposure with a true (E3) ranking-dependent exposure: "
                "a pointwise map does not keep the relative error a function "
                "of (x, k, a(k)), so condition (4.6) / Corollary 2b does not "
                "apply to this run",
                UserWarning,
            )

    def predict_exposure(self, context: np.ndarray, action_2d: np.ndarray) -> np.ndarray:
        e = self.dataset.calc_expected_exposure(context, action_2d)
        if self.corrupt_exposure is not None:
            e = np.clip(self.corrupt_exposure(e), PROB_CLIP, 1.0)
        return e

    def predict_relevance(self, context: np.ndarray, action: np.ndarray) -> np.ndarray:
        base = self.dataset.base_expected_reward(context)  # (n, m)
        n = context.shape[0]
        action_2d = action.reshape((n, self.len_list))
        r = base[np.arange(n)[:, None], action_2d].flatten()
        if self.corrupt_relevance is not None:
            r = np.clip(self.corrupt_relevance(r), PROB_CLIP, 1 - PROB_CLIP)
        return r


# --------------------------------------------------------------------------
# direct click regression (DM / DR-IIPS baselines)
# --------------------------------------------------------------------------
@dataclass
class DirectClickRegression:
    """Plain click regression q_hat_k(x, a(k)) = Pr(Y = 1 | x, k, a(k)):
    one sklearn classifier on position one-hot x item features. Uses the same
    feature functions as the EM towers so the DM / DR baselines and the
    decomposed model are compared fairly."""

    len_list: int
    n_unique_action: int
    base_model: BaseEstimator = None

    def __post_init__(self) -> None:
        if self.base_model is None:
            self.base_model = LogisticRegression(max_iter=1000, C=100.0)
        if not is_classifier(self.base_model):
            raise ValueError("`base_model` must be a classifier with predict_proba")

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
    ) -> "DirectClickRegression":
        X = click_feature(context, action, self.n_unique_action, self.len_list)
        self._clf_ = _fit_classifier_or_constant(
            self.base_model, X, reward, np.ones_like(reward, dtype=float)
        )
        return self

    def predict_q(self, context: np.ndarray, action_2d: np.ndarray) -> np.ndarray:
        """q_hat_k(x, a(k)) for given slates; shape (n, len_list)."""
        X = click_feature(
            context, action_2d.flatten(), self.n_unique_action, self.len_list
        )
        return np.clip(
            self._clf_.predict_proba(X)[:, 1], PROB_CLIP, 1 - PROB_CLIP
        ).reshape((context.shape[0], self.len_list))


# --------------------------------------------------------------------------
# Monte-Carlo computation of e_bar^pi and the DM term
# --------------------------------------------------------------------------
def run_eval_policy_monte_carlo(
    nuisance,
    context: np.ndarray,
    action_2d: np.ndarray,
    sample_ranking_fn: Callable[[np.random.RandomState], np.ndarray],
    n_mc_samples: int = 100,
    random_state: Optional[int] = None,
    direct_q_model: Optional[DirectClickRegression] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Shared MC sweep over a ~ pi(.|x): returns
      e_bar_pi        (n * len_list,)  self-normalized MC of eq. (2.5a),
      dm_slot_values  (n * len_list,)  slot decomposition of the DM term
                                       E_{a~pi}[e_hat_k(x,a) r_hat(x,a(k))],
      dm_slot_values_direct (n * len_list,) or None — the same MC samples
                                       reused for the direct click regression.

    Self-normalization: the conditional expectation is a ratio, so dividing
    the accumulated numerator by the match count `cnt` is more stable than
    dividing by pi_k * n_mc_samples; conditional on cnt > 0 the matched-sample
    mean is unbiased for e_bar^pi (each matched e_s has conditional mean
    (2.3)). Slots never matched (cnt = 0) fall back to e_hat_k(x_i, a_i),
    i.e. correction ratio 1 — that record degenerates to IIPS, which is a
    KNOWN small bias source for LE-IIPS of order P(cnt = 0) x (e_hat - e_bar):
    with S = n_mc_samples = 100 and pi_k ~ eps/m ~ 0.02, P(cnt = 0) ~ 13% on
    non-greedy slots. ED-DR is nearly immune (the weight error
    multiplies a ~zero-mean residual), and the affected slots have small
    pi_k, so the contribution is second-order; app_mc_samples sweeps S to
    measure the sensitivity. If it ever matters, scale S with a few multiples
    of 1 / min_k pi_k or add an unnormalized variant (denominator pi_k * S).

    Consistent-construction shortcut: when the exposure model class is
    position(/context)-only ("pbm" / "contextual_pbm"), eq. (2.5a) gives
    e_bar^pi = e_hat identically, so the correction ratio is EXACTLY 1 and no
    MC is used for it (the ratio-1 degeneration behind Corollary 2b and the
    IIPS-equivalence test of milestone M3/checkpoint 1).
    """
    rng = check_random_state(random_state)
    n, len_list = action_2d.shape
    exposure_factual = nuisance.predict_exposure(context, action_2d)  # (n, K)
    use_shortcut = getattr(nuisance, "exposure_model_class", None) in (
        "pbm",
        "contextual_pbm",
    )
    num = np.zeros((n, len_list))
    cnt = np.zeros((n, len_list))
    dm = np.zeros((n, len_list))
    dm_direct_sum = np.zeros((n, len_list)) if direct_q_model is not None else None
    for _ in range(n_mc_samples):
        a_s = sample_ranking_fn(rng)  # (n, K)
        e_s = nuisance.predict_exposure(context, a_s)
        r_s = nuisance.predict_relevance(context, a_s.flatten()).reshape(
            (n, len_list)
        )
        dm += e_s * r_s
        if not use_shortcut:
            match = a_s == action_2d
            num += e_s * match
            cnt += match
        if direct_q_model is not None:
            dm_direct_sum += direct_q_model.predict_q(context, a_s)
    if use_shortcut:
        e_bar_pi = exposure_factual
    else:
        e_bar_pi = np.divide(num, np.maximum(cnt, 1.0))
        e_bar_pi = np.where(cnt > 0, e_bar_pi, exposure_factual)
    dm_slot_values = dm / n_mc_samples
    dm_slot_values_direct = (
        (dm_direct_sum / n_mc_samples).flatten() if direct_q_model is not None else None
    )
    return e_bar_pi.flatten(), dm_slot_values.flatten(), dm_slot_values_direct


def calc_expected_exposure_under_eval_policy(
    em,
    context: np.ndarray,
    action: np.ndarray,
    sample_ranking_fn: Callable[[np.random.RandomState], np.ndarray],
    n_mc_samples: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(e_bar^pi_k(x_i, a_i(k)), dm_slot_values), both flat (n * len_list,).

    Thin wrapper of `run_eval_policy_monte_carlo` (`action` is the flat
    logged action array).
    """
    n = context.shape[0]
    len_list = em.len_list
    action_2d = action.reshape((n, len_list))
    e_bar_pi, dm_slot_values, _ = run_eval_policy_monte_carlo(
        nuisance=em,
        context=context,
        action_2d=action_2d,
        sample_ranking_fn=sample_ranking_fn,
        n_mc_samples=n_mc_samples,
        random_state=random_state,
    )
    return e_bar_pi, dm_slot_values


# --------------------------------------------------------------------------
# cross-fitting (2-fold default; DR nuisances)
# --------------------------------------------------------------------------
def fit_nuisances_cross_fitted(
    em_factory: Callable[[], ExposureRelevanceEM],
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    n_folds: int = 2,
    random_state: Optional[int] = None,
    direct_q_factory: Optional[Callable[[], DirectClickRegression]] = None,
):
    """K-fold cross-fitting of the EM (and optionally the direct click model).

    Rounds (slates) are split into folds; the nuisance predicting fold f is
    trained on all other folds. Returns a dict with:
      fold_of              (n,) fold id per round
      ems                  list of fitted ExposureRelevanceEM (one per fold)
      direct_q_models      list or None
      exposure_factual_hat (n * len_list,)
      relevance_factual_hat(n * len_list,)
      q_hat_factual        (n * len_list,) or None
    With n_folds = 1 the nuisances are fit on the full data (no sample split).
    """
    rng = check_random_state(random_state)
    n = context.shape[0]
    len_list = int(action.shape[0] // n)
    fold_of = (
        np.zeros(n, dtype=int) if n_folds <= 1 else rng.permutation(n) % n_folds
    )
    record_fold = np.repeat(fold_of, len_list)
    exposure_factual_hat = np.zeros(n * len_list)
    relevance_factual_hat = np.zeros(n * len_list)
    q_hat_factual = np.zeros(n * len_list) if direct_q_factory is not None else None
    ems, fitted_direct_models = [], []
    for f in range(max(n_folds, 1)):
        train_rounds = fold_of != f if n_folds > 1 else np.ones(n, dtype=bool)
        pred_rounds = fold_of == f
        train_records = np.repeat(train_rounds, len_list)
        pred_records = record_fold == f
        em = em_factory()
        em.fit(
            context=context[train_rounds],
            action=action[train_records],
            reward=reward[train_records],
            position=position[train_records],
        )
        ems.append(em)
        action_2d_pred = action[pred_records].reshape((-1, len_list))
        exposure_factual_hat[pred_records] = em.predict_exposure(
            context[pred_rounds], action_2d_pred
        ).flatten()
        relevance_factual_hat[pred_records] = em.predict_relevance(
            context[pred_rounds], action[pred_records]
        )
        if direct_q_factory is not None:
            dq = direct_q_factory()
            dq.fit(
                context=context[train_rounds],
                action=action[train_records],
                reward=reward[train_records],
                position=position[train_records],
            )
            fitted_direct_models.append(dq)
            q_hat_factual[pred_records] = dq.predict_q(
                context[pred_rounds], action_2d_pred
            ).flatten()
    return {
        "fold_of": fold_of,
        "ems": ems,
        "direct_q_models": fitted_direct_models if direct_q_factory is not None else None,
        "exposure_factual_hat": np.clip(exposure_factual_hat, PROB_CLIP, 1.0),
        "relevance_factual_hat": np.clip(
            relevance_factual_hat, PROB_CLIP, 1 - PROB_CLIP
        ),
        "q_hat_factual": (
            np.clip(q_hat_factual, PROB_CLIP, 1 - PROB_CLIP)
            if q_hat_factual is not None
            else None
        ),
    }
