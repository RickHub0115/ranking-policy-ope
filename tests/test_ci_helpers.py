# CI helpers for the figures (heavy-tailed IPS variants are handled by CI
# bands in the plots, not by more seeds). The SE is the
# closed-form standard error of rel_mse = mean of per-seed s^2 — fully
# deterministic, display-only, so results.csv and the existing aggregate
# columns are untouched.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main as main_mod


def test_se_of_mean_matches_closed_form():
    values = np.array([0.4, 1.1, 0.7, 2.3, 0.2])
    expected = np.std(values, ddof=1) / np.sqrt(len(values))
    assert main_mod.se_of_mean(values) == expected


def test_se_of_mean_single_value_is_nan():
    assert np.isnan(main_mod.se_of_mean([0.5]))
    assert np.isnan(main_mod.se_of_mean([]))


def test_ci_bounds_is_multiplicative_on_the_log_scale():
    z = main_mod.CI_Z
    # symmetric on the log scale: the same factor up and down
    lo, hi = main_mod.ci_bounds(np.array([1.0, 5.0]), np.array([0.2, 1.0]))
    np.testing.assert_allclose(np.log(hi / [1.0, 5.0]), np.log([1.0, 5.0] / lo))
    np.testing.assert_allclose(hi / [1.0, 5.0], np.exp(z * np.array([0.2, 0.2])))

    # se == mean (the tau0=0.05 cells): the lower bound stays strictly positive
    lo, hi = main_mod.ci_bounds(np.array([296.0]), np.array([296.0]))
    assert lo[0] > 0.0
    np.testing.assert_allclose(lo, 296.0 * np.exp(-z))
    np.testing.assert_allclose(hi, 296.0 * np.exp(z))

    # mean <= 0 cannot be logged: fall back to the 0-clipped linear interval
    lo, hi = main_mod.ci_bounds(np.array([0.0, -1.0]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(lo, [0.0, 0.0])
    np.testing.assert_allclose(hi, [0.5 * z, -1.0 + 0.5 * z])

    # a single-seed cell has a NaN SE: both bounds stay NaN, so no band
    lo, hi = main_mod.ci_bounds(np.array([2.0]), np.array([np.nan]))
    assert np.isnan(lo[0]) and np.isnan(hi[0])

    # se/mean = 1%: the ends move less than 1% from the linear interval
    mean, se = np.array([100.0]), np.array([1.0])
    lo, hi = main_mod.ci_bounds(mean, se)
    np.testing.assert_allclose(lo, mean - z * se, rtol=0.01)
    np.testing.assert_allclose(hi, mean + z * se, rtol=0.01)


def test_combine_se_identity_and_average():
    # k=1 leaves the SE unchanged; k=2 equal SEs give se/sqrt(2)
    assert main_mod._combine_se([0.3]) == 0.3
    np.testing.assert_allclose(
        main_mod._combine_se([0.3, 0.3]), 0.3 / np.sqrt(2)
    )
    assert np.isnan(main_mod._combine_se([0.3, np.nan]))


def test_aggregate_results_rel_mse_se_column():
    """rel_mse_se equals std(s^2, ddof=1)/sqrt(n) over the seeds of one
    (config x estimator) cell, and the existing columns keep their values."""
    rel_err = np.array([0.1, -0.2, 0.3, -0.1])
    setting = _flat_setting()
    rows = [
        dict(setting, seed=i, estimator="ed-dr", ground_truth=1.0, estimate=1.0 + e)
        for i, e in enumerate(rel_err)
    ]
    agg = main_mod.aggregate_results(pd.DataFrame(rows))
    assert len(agg) == 1
    row = agg.iloc[0]
    sq = rel_err**2
    np.testing.assert_allclose(row["rel_mse"], np.mean(sq))
    np.testing.assert_allclose(row["bias"], np.mean(rel_err))
    np.testing.assert_allclose(row["variance"], np.var(rel_err))
    np.testing.assert_allclose(
        row["rel_mse_se"], np.std(sq, ddof=1) / np.sqrt(len(sq))
    )
    # single seed: SE is NaN (no spread information), rel_mse still defined
    agg1 = main_mod.aggregate_results(pd.DataFrame(rows[:1]))
    assert np.isnan(agg1.iloc[0]["rel_mse_se"])
    np.testing.assert_allclose(agg1.iloc[0]["rel_mse"], sq[0])


def test_se_of_variance_influence_function_form():
    """SE of the ddof=0 sample variance is sqrt((m4 - var^2)/n) with m4 the
    fourth central moment."""
    values = np.array([0.4, 1.1, 0.7, 2.3, 0.2])
    var = np.var(values)
    m4 = np.mean((values - values.mean()) ** 4)
    np.testing.assert_allclose(
        main_mod.se_of_variance(values), np.sqrt((m4 - var**2) / len(values))
    )
    assert np.isnan(main_mod.se_of_variance([0.5]))
    assert np.isnan(main_mod.se_of_variance([]))


def test_bias_sq_ci_bounds_squares_the_wald_interval():
    z = main_mod.CI_Z
    # interval away from 0: both ends are the squared Wald ends
    lo, hi = main_mod.bias_sq_ci_bounds(np.array([1.0]), np.array([0.1]))
    np.testing.assert_allclose(lo, (1.0 - z * 0.1) ** 2)
    np.testing.assert_allclose(hi, (1.0 + z * 0.1) ** 2)
    # sign does not matter (bias^2 is symmetric)
    lo_n, hi_n = main_mod.bias_sq_ci_bounds(np.array([-1.0]), np.array([0.1]))
    np.testing.assert_allclose([lo_n, hi_n], [lo, hi])
    # bias interval covers 0: the lower bound is 0 (band to the panel bottom)
    lo, hi = main_mod.bias_sq_ci_bounds(np.array([0.1]), np.array([0.1]))
    assert lo[0] == 0.0
    np.testing.assert_allclose(hi, (0.1 + z * 0.1) ** 2)
    # NaN SE (single-seed cell): no band
    lo, hi = main_mod.bias_sq_ci_bounds(np.array([0.1]), np.array([np.nan]))
    assert np.isnan(lo[0]) and np.isnan(hi[0])


def test_aggregate_results_appends_bias_and_variance_se_columns():
    """bias_se / variance_se are display-only columns appended after
    rel_mse_se, so the pre-existing aggregate columns keep their order."""
    rel_err = np.array([0.1, -0.2, 0.3, -0.1])
    setting = _flat_setting()
    rows = [
        dict(setting, seed=i, estimator="ed-dr", ground_truth=1.0, estimate=1.0 + e)
        for i, e in enumerate(rel_err)
    ]
    agg = main_mod.aggregate_results(pd.DataFrame(rows))
    assert list(agg.columns[-3:]) == ["rel_mse_se", "bias_se", "variance_se"]
    row = agg.iloc[0]
    np.testing.assert_allclose(
        row["bias_se"], np.std(rel_err, ddof=1) / np.sqrt(len(rel_err))
    )
    np.testing.assert_allclose(
        row["variance_se"], main_mod.se_of_variance(rel_err)
    )
    # single seed: both SEs NaN, mirroring rel_mse_se
    agg1 = main_mod.aggregate_results(pd.DataFrame(rows[:1]))
    assert np.isnan(agg1.iloc[0]["bias_se"])
    assert np.isnan(agg1.iloc[0]["variance_se"])


def _flat_setting():
    return {k: "x" for k in main_mod.CONFIG_KEYS}


# SHOW_CI (2026-09-06): the CI bands / whiskers are off by default; the SE
# columns are still computed, and flipping the module switch (mode=plot
# show_ci=true) draws them again from the same aggregate.
def _fig1_like_agg():
    rows = []
    base = main_mod.flatten_setting_defaults()
    for structure in ("pbm", "ranking_dependent"):
        for n in (400, 800, 1600):
            for seed in range(3):
                for est, off in (("ed-dr", 0.0), ("iips", 0.1)):
                    rows.append(
                        dict(
                            base,
                            setting_name="fig1_data_size",
                            exposure_structure=structure,
                            n_rounds=n,
                            seed=seed,
                            estimator=est,
                            estimate=1.0 + off + 0.01 * seed,
                            ground_truth=1.0,
                        )
                    )
    return main_mod.aggregate_results(pd.DataFrame(rows))


def test_show_ci_off_by_default_and_toggles_the_bands(tmp_path, monkeypatch):
    import matplotlib.axes

    assert main_mod.SHOW_CI is False
    agg = _fig1_like_agg()
    bands = []
    whiskers = []
    orig_fill = matplotlib.axes.Axes.fill_between
    orig_err = matplotlib.axes.Axes.errorbar

    def fill(self, *a, **k):
        bands.append(1)
        return orig_fill(self, *a, **k)

    def err(self, *a, **k):
        whiskers.append(1)
        return orig_err(self, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", fill)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", err)

    monkeypatch.setattr(main_mod, "SHOW_CI", False)
    assert main_mod.make_fig1(agg, tmp_path) is not None
    main_mod.draw_decomposition_bar_row(
        plt.subplots(1, 3)[1], agg[agg["n_rounds"] == 400], ["ed-dr", "iips"], logy=False
    )
    plt.close("all")
    assert bands == [] and whiskers == []

    monkeypatch.setattr(main_mod, "SHOW_CI", True)
    assert main_mod.make_fig1(agg, tmp_path) is not None
    main_mod.draw_decomposition_bar_row(
        plt.subplots(1, 3)[1], agg[agg["n_rounds"] == 400], ["ed-dr", "iips"], logy=False
    )
    plt.close("all")
    assert bands and whiskers
