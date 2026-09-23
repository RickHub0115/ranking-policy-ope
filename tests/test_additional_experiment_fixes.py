# Additional-experiment fixes: the relevance_scale knob of the
# a2slot redesign, the fig3 condition (e) R x relevance cell, and
# the bit-identity that justifies "the figure may use either the R-alone or
# the R x relevance cell for `le-iips (oracle)`" — that premise rests on the
# corruption maps being deterministic and on le-iips never touching
# r_hat / the DM term, so it is pinned explicitly here.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose
from hydra import initialize_config_dir

import main as main_mod

from conftest import make_exposure_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def compose_cfg(overrides):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base="1.3"
    ):
        return compose(config_name="config", overrides=overrides)


# ---------------------------------------------------------------------------
# relevance_scale: DGP semantics
# ---------------------------------------------------------------------------
def test_relevance_scale_scales_base_expected_reward():
    ds_1 = make_exposure_dataset(exposure_structure="ranking_dependent",
                                 attention_spillover=1.0)
    ds_s = make_exposure_dataset(exposure_structure="ranking_dependent",
                                 attention_spillover=1.0, relevance_scale=0.25)
    rng = np.random.RandomState(0)
    context = rng.normal(size=(20, ds_1.dim_context))
    np.testing.assert_allclose(
        ds_s.base_expected_reward(context),
        0.25 * ds_1.base_expected_reward(context),
        rtol=0,
        atol=0,
    )


def test_relevance_scale_lowers_attract_and_raises_e3_exposure():
    """attract_relevance_align > 0 routes the scaled relevance logit into
    attract(x, a), so s < 1 lowers attract and RAISES the (E3) exposure —
    the disclosed side effect the a2slot gate includes."""
    common = dict(exposure_structure="ranking_dependent",
                  attention_spillover=1.0, attract_relevance_align=0.5)
    ds_1 = make_exposure_dataset(**common)
    ds_s = make_exposure_dataset(relevance_scale=0.25, **common)
    rng = np.random.RandomState(1)
    context = rng.normal(size=(30, ds_1.dim_context))
    action_2d = np.array(
        [rng.permutation(ds_1.n_unique_action)[: ds_1.len_list] for _ in range(30)]
    )
    e_1 = ds_1.calc_expected_exposure(context, action_2d)
    e_s = ds_s.calc_expected_exposure(context, action_2d)
    # position 1 is exactly theta_1 in both; later positions gain exposure
    np.testing.assert_array_equal(e_s[:, 0], e_1[:, 0])
    assert (e_s[:, 1:] >= e_1[:, 1:]).all()
    assert e_s[:, 1:].mean() > e_1[:, 1:].mean()


def test_relevance_scale_one_is_bitwise_noop():
    """s = 1.0 must leave every logged array bit-identical (multiplication
    by 1.0 is exact in IEEE 754), so pre-existing runs are unaffected."""
    ds_a = make_exposure_dataset(exposure_structure="ranking_dependent",
                                 attention_spillover=1.0)
    ds_b = make_exposure_dataset(exposure_structure="ranking_dependent",
                                 attention_spillover=1.0, relevance_scale=1.0)
    ds_a.random_ = np.random.RandomState(0)
    ds_b.random_ = np.random.RandomState(0)
    bf_a = ds_a.obtain_batch_bandit_feedback(n_rounds=200)
    bf_b = ds_b.obtain_batch_bandit_feedback(n_rounds=200)
    for key in ("reward", "expected_reward_factual", "expected_exposure_factual",
                "expected_relevance_factual", "pscore_item_position"):
        np.testing.assert_array_equal(bf_a[key], bf_b[key], err_msg=key)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_relevance_scale_validation(bad):
    with pytest.raises(ValueError):
        make_exposure_dataset(relevance_scale=bad)


def test_relevance_scale_is_an_aggregation_key_everywhere():
    """Sweep-axis keys missing from an aggregation key set silently blend
    cells; the a2slot redesign sweeps E3 at s=0.25
    against pre-existing s=1 logs, so the key must separate the cells."""
    assert main_mod.CONFIG_KEYS[-1] == "relevance_scale"  # appended last
    assert main_mod.flatten_setting_defaults()["relevance_scale"] == 1.0
    cfg = compose_cfg(
        ["setting=app_a2_slot_coupling", "setting.relevance_scale=0.25"]
    )
    assert main_mod.flatten_setting(cfg.setting)["relevance_scale"] == 0.25


# ---------------------------------------------------------------------------
# fig3 condition (e): the R x relevance cell
# ---------------------------------------------------------------------------
def test_rxrel_channels_coexist_in_flatten_setting():
    """Method R and the relevance channel are NOT mutually exclusive (only
    the two weight-side channels E / R are): the calibrated-pair cell must
    flatten to both raw kinds without error."""
    cfg = compose_cfg(
        [
            "setting=fig3_dr_grid",
            "setting.oracle_corruption.corrupt_exposure_ratio=power:0.15",
            "setting.oracle_corruption.corrupt_relevance=power:0.5",
        ]
    )
    flat = main_mod.flatten_setting(cfg.setting)
    assert flat["corrupt_exposure_ratio"] == "power:0.15"
    assert flat["corrupt_relevance"] == "power:0.5"


def test_fig3_frame_classifies_rxrel_cells():
    base = main_mod.flatten_setting_defaults()
    rows = []
    for cr, rel, err in (
        ("none", "none", 0.001),
        ("power:0.3", "none", 0.05),          # (d) R alone
        ("power:0.3", "power:0.9", 0.05),     # (e) R x relevance
        ("E:power:0.3", "none", 0.05),        # (c) method E appendix
    ):
        for seed in range(3):
            rows.append(
                dict(base, setting_name="fig3_dr_grid",
                     corrupt_exposure_ratio=cr, corrupt_relevance=rel,
                     seed=seed, estimator="ed-dr (oracle)",
                     estimate=1.0 + 0.01 * seed, ground_truth=1.0,
                     oracle_ratio_error_mean=err)
            )
    sub = main_mod._fig3_frame(main_mod.aggregate_results(pd.DataFrame(rows)))
    rx = sub[(sub["w_method"] == "R") & (sub["rel_t"] > 0)]
    assert len(rx) == 1
    assert float(rx["w_t"].iloc[0]) == 0.3
    assert float(rx["rel_t"].iloc[0]) == 0.9
    r_alone = sub[(sub["w_method"] == "R") & (sub["rel_t"] == 0.0)]
    assert len(r_alone) == 1
    assert len(sub[sub["w_method"] == "E"]) == 1


FIG3_E3_TINY = [
    "setting=fig3_dr_grid",
    "setting.n_rounds=300",
    "setting.n_unique_action=6",
    "setting.len_list=3",
    "setting.dim_context=3",
    "setting.n_mc_samples=20",
    "setting.ground_truth.n_mc_samples=20000",
    "n_seeds=1",
    "n_jobs=1",
]


def _oracle_rows(extra_overrides, seed=0):
    cfg = compose_cfg(FIG3_E3_TINY + extra_overrides)
    return pd.DataFrame(main_mod.run_one_replicate(cfg, seed=seed)).set_index(
        "estimator"
    )


def test_rxrel_cell_runs_and_le_iips_is_bit_identical_to_r_alone():
    """The premise 'the figure may use either the R-alone or the R x rel
    cell for `le-iips (oracle)`': for the same seed
    and the same t_w, the two cells must be BIT-identical for le-iips
    (deterministic corruption maps; le-iips uses neither r_hat nor the DM
    term), while ed-dr must differ (its residual baseline sees the
    corrupted relevance). The exactly-known ratio error is also unaffected
    by the relevance channel."""
    r_alone = _oracle_rows(
        ["setting.oracle_corruption.corrupt_exposure_ratio=power:0.5"]
    )
    rxrel = _oracle_rows(
        [
            "setting.oracle_corruption.corrupt_exposure_ratio=power:0.5",
            "setting.oracle_corruption.corrupt_relevance=power:1.5",
        ]
    )
    # oracle-only runs: estimators=[] leaves just the two oracle rows
    assert set(r_alone.index) == {"le-iips (oracle)", "ed-dr (oracle)"}
    assert set(rxrel.index) == set(r_alone.index)
    assert np.isfinite(rxrel["estimate"].astype(float)).all()
    assert (
        rxrel.loc["le-iips (oracle)", "estimate"]
        == r_alone.loc["le-iips (oracle)", "estimate"]
    )
    assert (
        rxrel.loc["ed-dr (oracle)", "estimate"]
        != r_alone.loc["ed-dr (oracle)", "estimate"]
    )
    assert (
        rxrel.loc["ed-dr (oracle)", "oracle_ratio_error_mean"]
        == r_alone.loc["ed-dr (oracle)", "oracle_ratio_error_mean"]
    )
    # the relevance-error diagnostic separates the two cells
    assert r_alone.loc["ed-dr (oracle)", "oracle_relevance_error_mean"] == 0.0
    assert rxrel.loc["ed-dr (oracle)", "oracle_relevance_error_mean"] > 0.0


def test_oracle_relevance_error_aggregates_after_ratio_error():
    base = main_mod.flatten_setting_defaults()
    rows = [
        dict(base, seed=s, estimator="ed-dr", estimate=1.0 + 0.01 * s,
             ground_truth=1.0, oracle_ratio_error_mean=0.03,
             oracle_relevance_error_mean=0.015 + 0.001 * s)
        for s in range(3)
    ]
    agg = main_mod.aggregate_results(pd.DataFrame(rows))
    assert list(agg.columns[-2:]) == [
        "oracle_ratio_error_mean", "oracle_relevance_error_mean",
    ]
    np.testing.assert_allclose(agg["oracle_relevance_error_mean"].iloc[0], 0.016)
    # without the diagnostic the column is absent (old logs stay old)
    agg_old = main_mod.aggregate_results(
        pd.DataFrame(rows).drop(columns=["oracle_relevance_error_mean"])
    )
    assert "oracle_relevance_error_mean" not in agg_old.columns


# ---------------------------------------------------------------------------
# figures: three-series panel (c) + the method-E appendix
# ---------------------------------------------------------------------------
def _fig3_synthetic_agg(with_rxrel=True, with_e=True):
    base = main_mod.flatten_setting_defaults()
    fig3 = dict(base, setting_name="fig3_dr_grid")
    rows = []

    def add(cr, rel, err, bias):
        for seed in range(4):
            for est in ("ed-dr (oracle)", "le-iips (oracle)"):
                rows.append(
                    dict(fig3, corrupt_exposure_ratio=cr, corrupt_relevance=rel,
                         seed=seed, estimator=est,
                         estimate=1.0 + bias + 0.005 * seed, ground_truth=1.0,
                         oracle_ratio_error_mean=err)
                )

    add("none", "none", 0.002, 0.0)
    for t, err in ((0.3, 0.05), (0.6, 0.11)):
        add(f"power:{t}", "none", err, 0.0005)                    # (d) zero line
        if with_rxrel:
            add(f"power:{t}", f"power:{3 * t}", err, 0.3 * err**2)  # (e)
        if with_e:
            add(f"E:power:{t}", "none", err, 0.4 * err)           # appendix E
    return main_mod.aggregate_results(pd.DataFrame(rows))


def test_make_fig3_renders_three_series(tmp_path):
    out = main_mod.make_fig3(_fig3_synthetic_agg(), tmp_path)
    assert out is not None and out.exists()
    # still renders while the R x relevance jobs have not landed yet
    out_partial = main_mod.make_fig3(
        _fig3_synthetic_agg(with_rxrel=False), tmp_path
    )
    assert out_partial is not None and out_partial.exists()


def test_make_fig3_method_e_appendix_renders_and_skips(tmp_path):
    out = main_mod.make_fig3_method_e_appendix(_fig3_synthetic_agg(), tmp_path)
    assert out is not None and out.exists()
    assert out.name == "appendix_fig3_method_e.png"
    # no method-E rows -> no appendix figure
    assert (
        main_mod.make_fig3_method_e_appendix(
            _fig3_synthetic_agg(with_e=False), tmp_path
        )
        is None
    )


def test_scorecard_reports_the_three_fig3_series(tmp_path):
    out = main_mod.make_prediction_scorecard(
        _fig3_synthetic_agg(), tmp_path, data_dir=tmp_path
    )
    assert out is not None
    preds = pd.read_csv(out)["prediction"].str.cat(sep="\n")
    assert "RxRel: ED-DR (oracle) log-log slope = 2" in preds
    assert "RxRel: LE-IIPS (oracle) log-log slope = 1" in preds
    assert "R alone: ED-DR (oracle) exactly unbiased" in preds
    assert "appendix method E: FIRST-order" in preds


def _fig3_injection_synthetic_agg(with_a=True, with_b=True, with_e=True):
    """Cells of the paper's three injection conditions: (a) relevance-only
    under (E3),
    (b) method E in-class under (E2), (c) method E under (E3)."""
    base = main_mod.flatten_setting_defaults()
    fig3 = dict(base, setting_name="fig3_dr_grid")
    rows = []

    def add(structure, cr, rel, err, bias):
        for seed in range(4):
            for est in ("ed-dr (oracle)", "le-iips (oracle)"):
                rows.append(
                    dict(fig3, exposure_structure=structure, exposure_model_class=structure,
                         corrupt_exposure_ratio=cr, corrupt_relevance=rel,
                         seed=seed, estimator=est,
                         estimate=1.0 + bias + 0.005 * seed, ground_truth=1.0,
                         oracle_ratio_error_mean=err)
                )

    add("ranking_dependent", "none", "none", 0.002, 0.0)
    for t, err in ((0.3, 0.05), (0.6, 0.11)):
        if with_a:
            add("ranking_dependent", "none", f"power:{t}", 0.002, 0.0)
        if with_e:
            add("ranking_dependent", f"E:power:{t}", "none", err, 0.4 * err)
    if with_b:
        add("contextual_pbm", "none", "none", 0.0, -0.004)
        for t in (0.3, 0.6):
            add("contextual_pbm", f"E:power:{t}", "none", 0.0, -0.004)
    return main_mod.aggregate_results(pd.DataFrame(rows))


def test_make_fig3_injection_renders_paper_figure(tmp_path):
    out = main_mod.make_fig3_injection(_fig3_injection_synthetic_agg(), tmp_path)
    assert out is not None and out.exists()
    assert out.name == "fig3_injection.png"
    # each condition alone still renders (the others are marked "not run")
    for kw in (dict(with_b=False, with_e=False), dict(with_a=False, with_e=False),
               dict(with_a=False, with_b=False)):
        out = main_mod.make_fig3_injection(_fig3_injection_synthetic_agg(**kw), tmp_path)
        assert out is not None and out.exists()


def test_make_fig3_injection_skips_without_any_condition(tmp_path):
    agg = _fig3_injection_synthetic_agg(with_a=False, with_b=False, with_e=False)
    assert main_mod.make_fig3_injection(agg, tmp_path) is None
    # no fig3 rows at all
    other = agg[agg["setting_name"] != "fig3_dr_grid"]
    assert main_mod.make_fig3_injection(other, tmp_path) is None
