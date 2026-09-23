# M6: experiment suite.
# Acceptance: the Hydra settings cover all experimental axes and the figure /
# table code renders Figures 1-5, Table 1 and the appendix figures into figs/.
# The rendering functions are exercised here on synthetic results so the full
# sweep is not needed to validate the plumbing.
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose
from hydra import initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parent.parent

import main as main_mod  # noqa: E402


SETTING_NAMES = [
    "default",
    "fig1_data_size",
    "fig2_policy_divergence",
    "fig3_dr_grid",
    "fig4_cascade",
    "fig4_cascade_eta",
    "fig5_logging_determinism",
    "table1_misspecification",
    "app_em_sensitivity",
    "app_em_init_const",
    "app_em_init_antimono",
    "app_mc_samples",
    "app_decay_K",
    "app_position_weight",
    "app_a2_robustness",
    "app_a2_slot_coupling",
]

REQUIRED_SETTING_KEYS = [
    # experimental axes
    "exposure_structure",
    "exposure_decay_rate",
    "attention_spillover",
    "len_list",
    "n_unique_action",
    "n_rounds",
    "tau0",
    "exposure_model_class",
    # nuisance / estimator switches
    "relevance_model",
    "n_em_iter",
    "warm_start",
    "monotone_position_tower",
    "n_folds",
    "n_mc_samples",
    "position_weight",
    "estimators",
    "include_oracle",
]


@pytest.mark.parametrize("name", SETTING_NAMES)
def test_m6_all_setting_configs_compose(name):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=[f"setting={name}"])
    assert cfg.setting.name == name
    for key in REQUIRED_SETTING_KEYS:
        assert key in cfg.setting, f"{name} misses {key}"
    assert cfg.setting.evaluation_policy.type in ("epsilon_greedy", "pl_softmax")
    assert cfg.setting.oracle_corruption.corrupt_exposure_ratio is not None


def test_m6_fig4_setting_is_cascade():
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=["setting=fig4_cascade"])
    assert cfg.setting.dgp == "cascade"
    assert not cfg.setting.include_oracle


# ---------------------------------------------------------------------------
# synthetic results covering all figure inputs
# ---------------------------------------------------------------------------
def _synthetic_results() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    estimators = [
        "sips", "iips", "rips", "cascade-dr", "dm", "dr-iips",
        "le-iips", "ed-dr", "le-iips (oracle)", "ed-dr (oracle)",
    ]
    base = main_mod.flatten_setting_defaults()
    rows = []

    def add(config_updates, n_seeds=8, oracle_ratio_error=0.02, extra_bias=0.0):
        cfg_row = dict(base)
        cfg_row.update(config_updates)
        for seed, est in itertools.product(range(n_seeds), estimators):
            noise = 0.05 * rng.normal()
            bias = 0.1 if est in ("sips", "iips") else 0.0
            rows.append(
                dict(
                    cfg_row,
                    seed=seed,
                    estimator=est,
                    ground_truth=1.0,
                    estimate=1.0 + bias + extra_bias + noise,
                    behavior_value=1.0,
                    exposure_corr=0.9,
                    ratio_error_mean=0.05,
                    oracle_ratio_error_mean=oracle_ratio_error,
                    prop3_bound=0.2 + 0.1 * rng.uniform(),
                    em_n_iter=10,
                    em_loglik_final=-0.3,
                    em_loglik_monotone=1.0,
                )
            )

    for structure, n in itertools.product(
        ["pbm", "ranking_dependent"], [500, 4000, 16000]
    ):
        add(
            dict(
                setting_name="fig1_data_size",
                exposure_structure=structure,
                n_rounds=n,
            )
        )
    for eps in [0.0, 0.5, 1.0]:
        add(dict(setting_name="fig2_policy_divergence", epsilon=eps))
    # fig3 (redesigned): parameterized strengths, three conditions
    fig3_rd = dict(
        setting_name="fig3_dr_grid",
        exposure_structure="ranking_dependent",
        exposure_model_class="ranking_dependent",
    )
    add(dict(fig3_rd), oracle_ratio_error=0.002)  # uncorrupted baseline = floor
    for t in (0.3, 0.6):  # (a) relevance corrupted
        add(dict(fig3_rd, corrupt_relevance=f"power:{t}"), oracle_ratio_error=0.002)
    # (c)-(e) of the re-composed series: method E
    # (appendix), method R alone (zero line) and R x relevance (main)
    for t, err in ((0.3, 0.05), (0.6, 0.11)):
        add(
            dict(fig3_rd, corrupt_exposure_ratio=f"E:power:{t}"),
            oracle_ratio_error=err,
            extra_bias=0.5 * err,
        )
        add(
            dict(fig3_rd, corrupt_exposure_ratio=f"power:{t}"),
            oracle_ratio_error=err,
        )
        add(
            dict(
                fig3_rd,
                corrupt_exposure_ratio=f"power:{t}",
                corrupt_relevance=f"power:{3 * t}",
            ),
            oracle_ratio_error=err,
            extra_bias=0.5 * err**2,
        )
    fig3_e2 = dict(
        setting_name="fig3_dr_grid",
        exposure_structure="contextual_pbm",
        exposure_model_class="contextual_pbm",
    )
    add(dict(fig3_e2), oracle_ratio_error=0.0)  # (b) baseline
    for t in (0.3, 0.6):  # (b) in-class exposure corruption
        add(dict(fig3_e2, corrupt_exposure_ratio=f"E:power:{t}"), oracle_ratio_error=0.0)
    add(dict(setting_name="fig4_cascade", dgp="cascade"))
    for eta in (0.0, 0.5, 2.0, 4.0):  # fig4': eta=1 reused from fig4_cascade
        add(
            dict(
                setting_name="fig4_cascade_eta",
                dgp="cascade",
                exposure_decay_rate=eta,
            )
        )
    for tau0 in [0.05, 0.5, 1.0]:
        add(dict(setting_name="fig5_logging_determinism", tau0=tau0))
    for true_s, model_s in itertools.product(
        ["pbm", "contextual_pbm", "ranking_dependent"], repeat=2
    ):
        add(
            dict(
                setting_name="table1_misspecification",
                exposure_structure=true_s,
                exposure_model_class=model_s,
            )
        )
    # Mirrors the production sweep: warm=True ignores em_random_state (single
    # job), seeds are swept only on the warm=False side.
    for warm, emrs in [(True, 0)] + [(False, e) for e in range(3)]:
        add(
            dict(
                setting_name="app_em_sensitivity",
                warm_start=warm,
                em_random_state=emrs,
            )
        )
    for s in [10, 100]:
        add(dict(setting_name="app_mc_samples", n_mc_samples=s))
    for lam, k in itertools.product([0.5, 1.0, 2.0], [3, 5]):
        add(
            dict(
                setting_name="app_decay_K",
                exposure_decay_rate=lam,
                len_list=k,
            )
        )
    for pw, n in itertools.product(["uniform", "dcg"], [1000, 4000]):
        add(
            dict(setting_name="app_position_weight", position_weight=pw, n_rounds=n)
        )
    for rho in [0.0, 0.5, 1.0]:
        add(
            dict(setting_name="app_a2_robustness", dgp="dbn", or_correlation=rho)
        )
    # extra EM-init schemes: each under its own setting_name
    add(dict(setting_name="app_em_init_const", warm_start=False, em_random_state=0))
    add(dict(setting_name="app_em_init_antimono", warm_start=False, em_random_state=0))
    # slot-level (A2) violation
    for structure, delta in itertools.product(
        ["pbm", "ranking_dependent"], [0.0, 0.5, 1.0]
    ):
        add(
            dict(
                setting_name="app_a2_slot_coupling",
                dgp="or_coupled",
                exposure_structure=structure,
                exposure_model_class=structure,
                or_coupling=delta,
            ),
            extra_bias=(0.05 * delta if structure == "ranking_dependent" else 0.0),
        )
    return pd.DataFrame(rows)


def test_m6_all_figures_and_table_render(tmp_path):
    df = _synthetic_results()
    agg = main_mod.aggregate_results(df)
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outs = []
    outs.append(main_mod.make_fig1(agg, figs_dir))
    outs.append(main_mod.make_fig2(agg, figs_dir))
    outs.append(main_mod.make_fig3(agg, figs_dir))
    outs.append(main_mod.make_fig4_bias_bound(df, figs_dir))
    outs.append(main_mod.make_fig4_eta(agg, figs_dir))
    outs.append(main_mod.make_fig5_cascade(agg, figs_dir))
    outs.append(main_mod.make_fig7_tau0(agg, figs_dir))
    outs.append(main_mod.make_table1(agg, figs_dir, data_dir=data_dir))
    outs.extend(main_mod.make_appendix_figs(agg, figs_dir))
    outs.append(main_mod.make_prediction_scorecard(agg, figs_dir, data_dir=data_dir))
    assert all(o is not None for o in outs), outs
    for o in outs:
        assert Path(o).exists()
    # CSV tables route to data_dir; the latex table and png stay in figs_dir
    assert (data_dir / "table1_misspecification.csv").exists()
    assert (data_dir / "prediction_scorecard.csv").exists()
    assert (figs_dir / "table1_misspecification.tex").exists()
    assert not (figs_dir / "table1_misspecification.csv").exists()


def test_m6_run_summary_figure(tmp_path):
    df = _synthetic_results()
    df = df[df["setting_name"] == "fig1_data_size"]
    out = tmp_path / "summary.png"
    main_mod.make_run_summary_figure(df, out, title="test")
    assert out.exists()


def test_m6_aggregate_bias_variance_decomposition():
    df = _synthetic_results()
    agg = main_mod.aggregate_results(df)
    # rel_mse == bias^2 + variance (population decomposition)
    np.testing.assert_allclose(
        agg["rel_mse"].values,
        (agg["bias_sq"] + agg["variance"]).values,
        rtol=1e-10,
    )
