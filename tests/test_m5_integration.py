# M5: estimated-nuisance pipeline + meta_slate wiring.
# Acceptance: the default configuration reproduces the predicted ordering of
# Figure 1 (slow, reduced scale); fast tests check the full pipeline runs end
# to end, is reproducible, and routes every estimator through the extended
# meta_slate.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose
from hydra import initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parent.parent

import main as main_mod  # noqa: E402


TINY_OVERRIDES = [
    "setting.n_rounds=400",
    "setting.n_unique_action=6",
    "setting.len_list=3",
    "setting.dim_context=3",
    "setting.n_mc_samples=20",
    "setting.n_em_iter=5",
    "setting.n_folds=2",
    "setting.cascade_dr_base_model=ridge",
    "setting.ground_truth.n_mc_samples=20000",
    "n_seeds=2",
    "n_jobs=1",
]


def compose_cfg(overrides):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base="1.3"
    ):
        return compose(config_name="config", overrides=overrides)


@pytest.fixture(scope="module")
def tiny_cfg():
    return compose_cfg(TINY_OVERRIDES)


@pytest.fixture(scope="module")
def tiny_rows(tiny_cfg):
    return main_mod.run_one_replicate(tiny_cfg, seed=0)


def test_m5_pipeline_end_to_end(tiny_rows):
    df = pd.DataFrame(tiny_rows)
    # sips is retired from the default estimator list (2026-09-06,
    # RETIRED_ESTIMATORS in src/main.py)
    expected = {
        "iips",
        "rips",
        "cascade-dr",
        "aips",
        "dm",
        "dr-iips",
        "le-iips",
        "ed-dr",
        "le-iips (oracle)",
        "ed-dr (oracle)",
    }
    assert set(df["estimator"]) == expected
    assert np.isfinite(df["estimate"]).all()
    assert np.isfinite(df["ground_truth"]).all()
    assert (df["ground_truth"] > 0).all()
    # estimates should be in a sane range around the truth
    v = df["ground_truth"].iloc[0]
    assert (df["estimate"].abs() < 50 * v).all()


def test_m5_diagnostics_present(tiny_rows):
    df = pd.DataFrame(tiny_rows)
    for col in [
        "exposure_corr",
        "ratio_error_mean",
        "prop3_bound",
        "em_n_iter",
        "em_loglik_final",
        "em_loglik_monotone",
    ]:
        assert col in df.columns, col
    assert bool(df["em_loglik_monotone"].iloc[0])


def test_m5_reproducibility_same_seed(tiny_cfg, tiny_rows):
    rows2 = main_mod.run_one_replicate(tiny_cfg, seed=0)
    df1 = pd.DataFrame(tiny_rows).set_index("estimator").sort_index()
    df2 = pd.DataFrame(rows2).set_index("estimator").sort_index()
    np.testing.assert_allclose(
        df1["estimate"].values.astype(float), df2["estimate"].values.astype(float)
    )
    np.testing.assert_allclose(
        df1["ground_truth"].values.astype(float),
        df2["ground_truth"].values.astype(float),
    )


def test_m5_cascade_dgp_pipeline_runs():
    cfg = compose_cfg(TINY_OVERRIDES + ["setting=fig4_cascade"] )
    rows = main_mod.run_one_replicate(cfg, seed=0)
    df = pd.DataFrame(rows)
    # no oracle / exposure diagnostics in the cascade DGP (no true exposure)
    assert "le-iips (oracle)" not in set(df["estimator"])
    assert np.isfinite(df["estimate"]).all()


def test_m5_epsilon_zero_skips_cascade_dr():
    cfg = compose_cfg(TINY_OVERRIDES + ["setting.evaluation_policy.epsilon=0.0"])
    rows = main_mod.run_one_replicate(cfg, seed=0)
    df = pd.DataFrame(rows).set_index("estimator")
    assert np.isnan(df.loc["cascade-dr", "estimate"])


def test_m5_pl_softmax_policy_pipeline_runs():
    cfg = compose_cfg(
        TINY_OVERRIDES
        + [
            "setting.evaluation_policy.type=pl_softmax",
            "setting.evaluation_policy.tau1=1.0",
        ]
    )
    rows = main_mod.run_one_replicate(cfg, seed=0)
    df = pd.DataFrame(rows)
    assert np.isfinite(df["estimate"]).all()


def test_m5_dcg_position_weight_pipeline_runs():
    """alpha_k = 1/log2(k+1): proposed estimators take position_weight
    natively; OBP baselines receive alpha-weighted rewards."""
    cfg = compose_cfg(TINY_OVERRIDES + ["setting.position_weight=dcg"])
    rows = main_mod.run_one_replicate(cfg, seed=0)
    df = pd.DataFrame(rows)
    assert np.isfinite(df["estimate"]).all()
    # DCG value must be smaller than the uniform-alpha value of the same DGP
    cfg_u = compose_cfg(TINY_OVERRIDES)
    df_u = pd.DataFrame(main_mod.run_one_replicate(cfg_u, seed=0))
    assert df["ground_truth"].iloc[0] < df_u["ground_truth"].iloc[0]


def test_m5_mc_marginalized_pscore_pipeline_runs():
    """pscore_mc_samples > 0: behavior pscore_item_position via Monte-Carlo
    marginalization (the large-(m, K) path of the pitfalls memo)."""
    cfg = compose_cfg(TINY_OVERRIDES + ["setting.pscore_mc_samples=3000"])
    rows = main_mod.run_one_replicate(cfg, seed=0)
    df = pd.DataFrame(rows)
    assert np.isfinite(df["estimate"]).all()
    # the MC pscores should give estimates in the same ballpark as exact ones
    df_exact = pd.DataFrame(main_mod.run_one_replicate(compose_cfg(TINY_OVERRIDES), seed=0))
    v = df_exact["ground_truth"].iloc[0]
    m = df.set_index("estimator")["estimate"]
    m_exact = df_exact.set_index("estimator")["estimate"]
    assert abs(m["iips"] - m_exact["iips"]) < 0.5 * v


@pytest.mark.slow
def test_m5_default_config_ordering_reduced_scale():
    """Plan acceptance (reduced): under the (E3) default the Figure-1 ordering
    holds — ED-DR (oracle and estimated) beats IIPS in relative MSE, and IIPS
    carries systematic bias."""
    cfg = compose_cfg(
        [
            "setting.n_rounds=2000",
            "setting.n_unique_action=8",
            "setting.len_list=4",
            "setting.attention_spillover=2.0",
            "setting.n_mc_samples=100",
            "setting.cascade_dr_base_model=ridge",
            "n_seeds=40",
            "n_jobs=1",
        ]
    )
    df = main_mod.run_experiment(cfg)
    agg = main_mod.aggregate_results(df).set_index("estimator")
    assert agg.loc["ed-dr (oracle)", "rel_mse"] < agg.loc["iips", "rel_mse"]
    assert agg.loc["ed-dr", "rel_mse"] < agg.loc["iips", "rel_mse"]
    assert abs(agg.loc["iips", "bias"]) > 2 * abs(agg.loc["ed-dr (oracle)", "bias"])
