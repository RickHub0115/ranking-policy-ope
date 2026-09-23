# Additional-experiment plumbing:
# fig3 redesign (parameterized corruption strengths, method E/R encoding,
# the corrupted-oracle ratio-error diagnostic), the fig4' eta sweep's reuse
# of the fig4_cascade cell, the explicit EM init schemes, and the
# or_coupling-only backfill of mode=plot.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose
from hydra import initialize_config_dir
from sklearn.utils import check_random_state

import main as main_mod
from exposure_model import ExposureRelevanceEM

REPO_ROOT = Path(__file__).resolve().parent.parent


def compose_cfg(overrides):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base="1.3"
    ):
        return compose(config_name="config", overrides=overrides)


# ---------------------------------------------------------------------------
# make_corruption: parameterized strengths
# ---------------------------------------------------------------------------
def test_make_corruption_parameterized_power():
    v = np.array([0.2, 0.5, 0.9])
    f = main_mod.make_corruption("power:0.3")
    np.testing.assert_allclose(f(v), v ** 1.3, rtol=1e-12)
    # legacy fixed maps unchanged (M3 tests / old logs)
    np.testing.assert_allclose(main_mod.make_corruption("power")(v), v ** 1.5)
    assert main_mod.make_corruption("none") is None


@pytest.mark.parametrize("kind", ["power:0", "power:-1", "power:nan", "power:x", "cube"])
def test_make_corruption_rejects_malformed_kinds(kind):
    # strength 0 must run as "none" (the uncorrupted config itself), so
    # power:0 is rejected rather than silently acting as identity-with-clip
    with pytest.raises(ValueError):
        main_mod.make_corruption(kind)


def test_parse_weight_and_relevance_channels():
    assert main_mod._parse_weight_channel("none") == ("none", 0.0)
    assert main_mod._parse_weight_channel("power:0.25") == ("R", 0.25)
    assert main_mod._parse_weight_channel("E:power:0.25") == ("E", 0.25)
    # legacy 2x2-grid values must NOT parse (stale fig3 logs stay out)
    assert main_mod._parse_weight_channel("power") is None
    assert main_mod._parse_weight_channel("shift") is None
    assert main_mod._parse_relevance_channel("none") == 0.0
    assert main_mod._parse_relevance_channel("power:0.5") == 0.5
    assert main_mod._parse_relevance_channel("power") is None


def test_fit_loglog_slope_recovers_power_law():
    x = np.geomspace(1e-3, 1e-1, 6)
    assert abs(main_mod.fit_loglog_slope(x, 3.0 * x ** 2) - 2.0) < 1e-9
    assert abs(main_mod.fit_loglog_slope(x, 0.5 * x) - 1.0) < 1e-9
    assert np.isnan(main_mod.fit_loglog_slope(x[:1], x[:1]))


# ---------------------------------------------------------------------------
# weight-channel encoding (method E merged into corrupt_exposure_ratio)
# ---------------------------------------------------------------------------
def test_weight_channel_column_encodes_method_e():
    cfg = compose_cfg(
        ["setting=fig3_dr_grid", "setting.oracle_corruption.corrupt_exposure=power:0.1"]
    )
    flat = main_mod.flatten_setting(cfg.setting)
    assert flat["corrupt_exposure_ratio"] == "E:power:0.1"
    cfg_r = compose_cfg(
        [
            "setting=fig3_dr_grid",
            "setting.oracle_corruption.corrupt_exposure_ratio=power:0.1",
        ]
    )
    assert main_mod.flatten_setting(cfg_r.setting)["corrupt_exposure_ratio"] == "power:0.1"


def test_weight_channels_are_mutually_exclusive():
    cfg = compose_cfg(
        [
            "setting=fig3_dr_grid",
            "setting.oracle_corruption.corrupt_exposure=power:0.1",
            "setting.oracle_corruption.corrupt_exposure_ratio=power:0.1",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        main_mod.flatten_setting(cfg.setting)


# ---------------------------------------------------------------------------
# corrupted-oracle ratio error (the fig3 x axis)
# ---------------------------------------------------------------------------
FIG3_TINY = [
    "setting=fig3_dr_grid",
    "setting.exposure_structure=contextual_pbm",
    "setting.exposure_model_class=contextual_pbm",
    "setting.n_rounds=300",
    "setting.n_unique_action=6",
    "setting.len_list=3",
    "setting.dim_context=3",
    "setting.n_mc_samples=20",
    "setting.ground_truth.n_mc_samples=20000",
    "n_seeds=1",
    "n_jobs=1",
]


def _logged_exposures(cfg, seed):
    """Rebuild the replicate's logged (E2) exposures outside run_one_replicate
    (same dataset seed and sampling call, so bit-identical)."""
    dataset = main_mod.build_dataset(cfg.setting)
    dataset.random_ = check_random_state(seed)
    bf = dataset.obtain_batch_bandit_feedback(
        n_rounds=int(cfg.setting.n_rounds), return_pscore_item_position=True,
        clip_logit_value=None,
    )
    n = bf["n_rounds"]
    action_2d = bf["action"].reshape((n, dataset.len_list))
    return dataset.calc_expected_exposure(bf["context"], action_2d).flatten()


def test_oracle_ratio_error_is_exact_for_method_r():
    """Method R under (E2): the pbm/contextual shortcut makes e_bar^pi equal
    the factual exposure exactly, so the recorded ratio error must equal
    mean(|e^(1+t) - e| / e) to machine precision — no MC noise floor."""
    cfg = compose_cfg(
        FIG3_TINY + ["setting.oracle_corruption.corrupt_exposure_ratio=power:0.5"]
    )
    rows = pd.DataFrame(main_mod.run_one_replicate(cfg, seed=0))
    e = _logged_exposures(cfg, seed=0)
    expected = np.mean(np.abs(np.clip(e ** 1.5, 1e-6, 1.0) - e) / e)
    np.testing.assert_allclose(
        rows["oracle_ratio_error_mean"].iloc[0], expected, rtol=1e-12
    )


def test_oracle_ratio_error_is_zero_for_in_class_method_e():
    """Method E under (E2) (fig3 condition (b)): the SAME corrupted exposure
    model feeds numerator and denominator, and the shortcut keeps the ratio
    exactly 1 — the corollary-2b cancellation — so the measured ratio error
    must be exactly 0."""
    cfg = compose_cfg(
        FIG3_TINY + ["setting.oracle_corruption.corrupt_exposure=power:0.5"]
    )
    rows = pd.DataFrame(main_mod.run_one_replicate(cfg, seed=0))
    assert (rows["oracle_ratio_error_mean"] == 0.0).all()
    # oracle-only run: estimators=[] leaves just the two oracle rows
    assert set(rows["estimator"]) == {"le-iips (oracle)", "ed-dr (oracle)"}


def test_run_one_replicate_rejects_both_weight_channels():
    cfg = compose_cfg(
        FIG3_TINY
        + [
            "setting.oracle_corruption.corrupt_exposure=power:0.1",
            "setting.oracle_corruption.corrupt_exposure_ratio=power:0.1",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        main_mod.run_one_replicate(cfg, seed=0)


def test_aggregate_appends_oracle_ratio_error_after_existing_columns():
    base = main_mod.flatten_setting_defaults()
    rows = [
        dict(base, seed=s, estimator="ed-dr", estimate=1.0 + 0.01 * s,
             ground_truth=1.0, oracle_ratio_error_mean=0.03 + 0.01 * s)
        for s in range(3)
    ]
    agg = main_mod.aggregate_results(pd.DataFrame(rows))
    # appended LAST: every pre-existing aggregate column keeps value & order
    assert list(agg.columns) == main_mod.CONFIG_KEYS + [
        "estimator", "rel_mse", "bias", "variance", "n_seeds", "bias_sq",
        "rel_mse_se", "bias_se", "variance_se", "oracle_ratio_error_mean",
    ]
    np.testing.assert_allclose(agg["oracle_ratio_error_mean"].iloc[0], 0.04)
    # without the diagnostic the column is absent (old logs stay old)
    agg_old = main_mod.aggregate_results(
        pd.DataFrame(rows).drop(columns=["oracle_ratio_error_mean"])
    )
    assert "oracle_ratio_error_mean" not in agg_old.columns
    assert list(agg_old.columns) == list(agg.columns[:-1])


def test_make_fig3_ignores_legacy_grid_rows():
    base = main_mod.flatten_setting_defaults()
    rows = []
    for ce, cr in (("none", "none"), ("power", "none"), ("power", "power")):
        for seed in range(3):
            rows.append(
                dict(base, setting_name="fig3_dr_grid",
                     corrupt_exposure_ratio=ce, corrupt_relevance=cr,
                     seed=seed, estimator="ed-dr (oracle)",
                     estimate=1.0 + 0.01 * seed, ground_truth=1.0,
                     oracle_ratio_error_mean=0.01)
            )
    agg = main_mod.aggregate_results(pd.DataFrame(rows))
    assert main_mod.make_fig3(agg, Path("/nonexistent")) is None


# ---------------------------------------------------------------------------
# fig4' eta sweep: eta = 1 is REUSED from fig4_cascade
# ---------------------------------------------------------------------------
def _eta_agg(etas_new, include_base=True, base_rel_mse=0.011):
    base = main_mod.flatten_setting_defaults()
    rows = []
    for name, etas in (("fig4_cascade_eta", etas_new),
                       (("fig4_cascade"), [1.0] if include_base else [])):
        for eta in etas:
            for seed in range(3):
                for est in ("ed-dr", "rips"):
                    rows.append(
                        dict(base, setting_name=name, dgp="cascade",
                             exposure_decay_rate=eta, seed=seed, estimator=est,
                             estimate=1.0 + (base_rel_mse if est == "ed-dr" else 0.0)
                             + 0.01 * seed,
                             ground_truth=1.0)
                    )
    return main_mod.aggregate_results(pd.DataFrame(rows))


def test_fig4_eta_frame_reuses_the_fig4_cascade_cell():
    agg = _eta_agg([0.0, 0.5, 2.0, 4.0])
    sub = main_mod._fig4_eta_frame(agg)
    assert sorted(sub["exposure_decay_rate"].unique()) == [0.0, 0.5, 1.0, 2.0, 4.0]
    eta1 = sub[sub["exposure_decay_rate"] == 1.0]
    assert set(eta1["setting_name"]) == {"fig4_cascade"}
    # the reused cell is the very same aggregate row
    base_row = agg[(agg["setting_name"] == "fig4_cascade")
                   & (agg["estimator"] == "ed-dr")]
    np.testing.assert_array_equal(
        eta1[eta1["estimator"] == "ed-dr"]["rel_mse"].values,
        base_row["rel_mse"].values,
    )


def test_fig4_eta_frame_rejects_a_rerun_eta_one():
    agg = _eta_agg([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="eta=1"):
        main_mod._fig4_eta_frame(agg)


def test_fig4_eta_frame_requires_the_fig4_cascade_baseline():
    agg = _eta_agg([0.0, 0.5], include_base=False)
    with pytest.raises(ValueError, match="fig4_cascade"):
        main_mod._fig4_eta_frame(agg)


def test_fig4_eta_frame_returns_none_without_eta_runs():
    agg = _eta_agg([], include_base=True)
    assert main_mod._fig4_eta_frame(agg) is None
    assert main_mod.make_fig4_eta(agg, Path("/nonexistent")) is None


def test_make_fig4_eta_renders(tmp_path):
    out = main_mod.make_fig4_eta(_eta_agg([0.0, 0.5, 2.0, 4.0]), tmp_path)
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# EM init schemes
# ---------------------------------------------------------------------------
def _em(**kwargs):
    return ExposureRelevanceEM(len_list=4, n_unique_action=6, **kwargs)


def test_em_initial_theta_schemes():
    dummy = dict(
        action=np.zeros(8, dtype=int),
        reward=np.zeros(8),
        position=np.tile(np.arange(4), 2),
    )
    em_const = _em(warm_start=False, init_scheme="const", random_state=0)
    np.testing.assert_array_equal(em_const._initial_theta(**dummy), np.full(4, 0.5))
    em_anti = _em(warm_start=False, init_scheme="anti_mono", random_state=0)
    theta = em_anti._initial_theta(**dummy)
    assert np.all(np.diff(theta) >= 0)  # ascending = monotone-reversed
    # the historical random scheme is the SAME draws sorted descending
    em_rand = _em(warm_start=False, init_scheme=None, random_state=0)
    np.testing.assert_array_equal(em_rand._initial_theta(**dummy), theta[::-1])
    # warm_start=False + random_state=None keeps the historical constant init
    em_default = _em(warm_start=False)
    np.testing.assert_array_equal(em_default._initial_theta(**dummy), np.full(4, 0.5))


def test_em_init_scheme_validation():
    with pytest.raises(ValueError, match="warm_start=False"):
        _em(warm_start=True, init_scheme="const")
    with pytest.raises(ValueError, match="init_scheme"):
        _em(warm_start=False, init_scheme="bogus")


def test_em_init_settings_compose_and_route_to_the_em():
    for name, scheme in (
        ("app_em_init_const", "const"),
        ("app_em_init_antimono", "anti_mono"),
    ):
        cfg = compose_cfg([f"setting={name}"])
        assert cfg.setting.em_init == scheme
        assert cfg.setting.warm_start is False
        assert list(cfg.setting.estimators) == ["le-iips", "ed-dr"]
        # em_init is deliberately NOT an aggregation column (own setting_name
        # per scheme instead — see default.yaml); the guard here is that it
        # never silently becomes one without a migration plan for old logs
        assert "em_init" not in main_mod.CONFIG_KEYS
        assert "em_init" not in main_mod.flatten_setting(cfg.setting)


def test_make_em_sensitivity_fig_renders_partial_and_full(tmp_path):
    base = main_mod.flatten_setting_defaults()
    rows = []

    def add(setting_name, warm, emrs, est, offset=0.0):
        for seed in range(3):
            rows.append(
                dict(base, setting_name=setting_name, warm_start=warm,
                     em_random_state=emrs, seed=seed, estimator=est,
                     estimate=1.05 + offset + 0.01 * seed, ground_truth=1.0)
            )

    # partial: only the historical schemes -> figure still renders
    for est in ("ed-dr", "le-iips"):
        add("app_em_sensitivity", True, 0, est)
        for emrs in range(3):
            add("app_em_sensitivity", False, emrs, est)
    agg = main_mod.aggregate_results(pd.DataFrame(rows))
    out = main_mod.make_em_sensitivity_fig(agg, tmp_path)
    assert out is not None and out.exists()
    # full: the two extra schemes join the categorical axis
    for est in ("ed-dr", "le-iips"):
        add("app_em_init_const", False, 0, est, offset=0.01)
        add("app_em_init_antimono", False, 0, est, offset=0.02)
    agg_full = main_mod.aggregate_results(pd.DataFrame(rows))
    out2 = main_mod.make_em_sensitivity_fig(agg_full, tmp_path)
    assert out2 is not None and out2.exists()


# ---------------------------------------------------------------------------
# mode=plot: or_coupling / relevance_scale are the ONLY backfilled config
# columns — both backfills are exact, not
# guesses (pre-existing runs used delta = 0 and s = 1)
# ---------------------------------------------------------------------------
def _write_run(log_root, run_name, drop_cols=()):
    row = dict(main_mod.flatten_setting_defaults())
    row.update(seed=0, estimator="ed-dr", estimate=1.0, ground_truth=2.0)
    df = pd.DataFrame([row])
    if drop_cols:
        df = df.drop(columns=list(drop_cols))
    run_dir = log_root / "default" / run_name
    run_dir.mkdir(parents=True)
    df.to_csv(run_dir / "results.csv", index=False)


def test_missing_or_coupling_is_backfilled_with_zero(tmp_path):
    _write_run(tmp_path, "run_1", drop_cols=["or_coupling"])
    df = main_mod.load_results_csvs(tmp_path)
    assert (df["or_coupling"] == 0.0).all()


def test_missing_relevance_scale_is_backfilled_with_one(tmp_path):
    _write_run(tmp_path, "run_1", drop_cols=["or_coupling", "relevance_scale"])
    df = main_mod.load_results_csvs(tmp_path)
    assert (df["relevance_scale"] == 1.0).all()
    assert (df["or_coupling"] == 0.0).all()


def test_other_missing_config_columns_still_error(tmp_path):
    _write_run(
        tmp_path,
        "run_1",
        drop_cols=["or_coupling", "relevance_scale", "corrupt_relevance"],
    )
    with pytest.raises(ValueError, match="corrupt_relevance"):
        main_mod.load_results_csvs(tmp_path)
