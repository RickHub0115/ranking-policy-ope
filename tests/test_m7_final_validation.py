# M7: final validation.
# Acceptance: (i) fixed-seed reproducibility of every figure input;
# (ii) the gap between oracle and estimated-nuisance versions shrinks as n
# grows (nuisance convergence consistency); (iii) the win/lose prediction
# scorecard is generated and failed predictions are visible.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose
from hydra import initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parent.parent

import main as main_mod  # noqa: E402


def compose_cfg(overrides):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base="1.3"
    ):
        return compose(config_name="config", overrides=overrides)


TINY = [
    "setting.n_rounds=300",
    "setting.n_unique_action=6",
    "setting.len_list=3",
    "setting.dim_context=3",
    "setting.n_mc_samples=20",
    "setting.n_em_iter=5",
    "setting.cascade_dr_base_model=ridge",
    "setting.ground_truth.n_mc_samples=20000",
    "n_jobs=1",
]


def test_m7_i_run_experiment_reproducible():
    cfg = compose_cfg(TINY + ["n_seeds=2"])
    df1 = main_mod.run_experiment(cfg)
    df2 = main_mod.run_experiment(cfg)
    key = ["seed", "estimator"]
    df1 = df1.sort_values(key).reset_index(drop=True)
    df2 = df2.sort_values(key).reset_index(drop=True)
    np.testing.assert_allclose(df1["estimate"].values, df2["estimate"].values)
    np.testing.assert_allclose(
        df1["ground_truth"].values, df2["ground_truth"].values
    )


def test_m7_i_different_seeds_differ():
    cfg = compose_cfg(TINY + ["n_seeds=2"])
    df = main_mod.run_experiment(cfg)
    est = df[df["estimator"] == "iips"].sort_values("seed")["estimate"].values
    assert est[0] != est[1]


def test_m7_iii_prediction_scorecard(tmp_path):
    from test_m6_experiment_suite import _synthetic_results

    agg = main_mod.aggregate_results(_synthetic_results())
    out = main_mod.make_prediction_scorecard(agg, tmp_path)
    assert out is not None and Path(out).exists()
    scorecard = pd.read_csv(out)
    assert {"prediction", "measured", "holds"} <= set(scorecard.columns)
    assert len(scorecard) >= 4


@pytest.mark.slow
def test_m7_ii_oracle_gap_shrinks_with_n():
    """The relative-MSE gap between the estimated-nuisance ED-DR and its
    oracle version shrinks as n grows (nuisance convergence)."""
    gaps = {}
    for n in (500, 4000):
        cfg = compose_cfg(
            [
                f"setting.n_rounds={n}",
                "setting.n_unique_action=8",
                "setting.len_list=4",
                "setting.attention_spillover=2.0",
                "setting.cascade_dr_base_model=ridge",
                "n_seeds=40",
                "n_jobs=1",
            ]
        )
        agg = main_mod.aggregate_results(main_mod.run_experiment(cfg)).set_index(
            "estimator"
        )
        gaps[n] = abs(
            agg.loc["ed-dr", "rel_mse"] - agg.loc["ed-dr (oracle)", "rel_mse"]
        )
    assert gaps[4000] < gaps[500], gaps
