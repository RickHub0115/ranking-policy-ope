# scripts/merge_archive_aggregate.py: pooling an archived aggregate (earlier
# seed band, per-cell means only) with the aggregate of a later band must give
# what aggregate_results computes on the union of the per-seed rows.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as main_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_archive_aggregate as merge_mod  # noqa: E402


def _rows(seeds, estimators=("ed-dr", "iips"), nan_for=(), rng=None,
          setting="fig1_data_size", axis="n_rounds", levels=(400, 800)):
    """Per-seed rows of `setting` sweeping `axis` over `levels` (fig1's n_rounds
    by default; app_decay_K sweeps exposure_decay_rate)."""
    rng = rng or np.random.RandomState(0)
    base = main_mod.flatten_setting_defaults()
    rows = []
    for level in levels:
        for seed in seeds:
            for est in estimators:
                est_val = np.nan if est in nan_for else 1.0 + 0.1 * (est == "iips") + 0.05 * rng.normal()
                rows.append(dict(base, setting_name=setting, seed=seed, estimator=est,
                                 estimate=est_val, ground_truth=1.0, **{axis: level}))
    return pd.DataFrame(rows)


def _decay_rows(seeds, **kw):
    return _rows(seeds, setting="app_decay_K", axis="exposure_decay_rate",
                 levels=(0.5, 1.0, 2.0), **kw)


def _assert_pooled_matches_union(band1, band2):
    agg1 = main_mod.aggregate_results(band1)
    agg2 = main_mod.aggregate_results(band2)
    ref = main_mod.aggregate_results(pd.concat([band1, band2], ignore_index=True))
    pooled = merge_mod.pool_cells(agg1, agg2)
    keys = merge_mod.KEYS
    pooled = pooled.sort_values(keys).reset_index(drop=True)
    ref = ref.sort_values(keys).reset_index(drop=True)
    assert len(pooled) == len(ref)
    for col in ("rel_mse", "bias", "variance", "bias_sq"):
        np.testing.assert_allclose(
            pooled[col].to_numpy(float), ref[col].to_numpy(float), rtol=1e-10, atol=1e-12,
            err_msg=col,
        )
    # se_of_mean does not skip NaN seeds (a cell with NaN and non-NaN seeds
    # never occurs in the real runs), so compare the SEs where the reference
    # has one; the NaN-cell test checks the pooled SE separately
    for col in ("rel_mse_se", "bias_se"):
        have = np.isfinite(ref[col].to_numpy(float))
        np.testing.assert_allclose(
            pooled[col].to_numpy(float)[have], ref[col].to_numpy(float)[have],
            rtol=1e-10, atol=1e-12, err_msg=col,
        )
    assert (pooled["n_seeds"].to_numpy() == ref["n_seeds"].to_numpy()).all()
    assert pooled["variance_se"].isna().all()
    return pooled


def test_pooling_is_exact_for_disjoint_bands():
    rng = np.random.RandomState(1)
    _assert_pooled_matches_union(_rows(range(0, 5), rng=rng), _rows(range(5, 12), rng=rng))


def test_pooling_handles_an_all_nan_cell_in_one_band():
    # e.g. fig2's cascade-dr at epsilon=0 (n_seeds=0 in the archive) later
    # joined by a band where the cell has values: the pooled cell must equal the
    # band that has data, and its SE must stay finite
    rng = np.random.RandomState(2)
    band1 = _rows(range(0, 4), nan_for=("iips",), rng=rng)
    band2 = _rows(range(4, 9), rng=rng)
    pooled = _assert_pooled_matches_union(band1, band2)
    iips = pooled[pooled["estimator"] == "iips"].sort_values("n_rounds")
    assert (iips["n_seeds"] == 5).all()
    # the band with data carries the whole cell, SE included
    agg2 = main_mod.aggregate_results(band2)
    iips2 = agg2[agg2["estimator"] == "iips"].sort_values("n_rounds")
    for col in ("rel_mse", "bias", "variance", "rel_mse_se", "bias_se"):
        np.testing.assert_allclose(iips[col].to_numpy(float), iips2[col].to_numpy(float),
                                   rtol=1e-10, atol=1e-12, err_msg=col)


def test_pooling_refuses_mismatched_cell_sets():
    agg1 = main_mod.aggregate_results(_rows(range(0, 3)))
    agg2 = main_mod.aggregate_results(_rows(range(3, 6), estimators=("ed-dr",)))
    with pytest.raises(ValueError, match="cell sets differ"):
        merge_mod.pool_cells(agg1, agg2)


def test_pooling_is_exact_for_the_decay_setting():
    # Figure 3 (app_decay_K): archived appendix band 0-99 (cycles 1-2) + the
    # seed 100-499 run; here 0-3 + 4-9 with the same cell layout (3 lambdas)
    rng = np.random.RandomState(3)
    pooled = _assert_pooled_matches_union(_decay_rows(range(0, 4), rng=rng),
                                          _decay_rows(range(4, 10), rng=rng))
    assert sorted(pooled["exposure_decay_rate"].unique()) == [0.5, 1.0, 2.0]
    assert (pooled["n_seeds"] == 10).all()
    assert merge_mod.sweep_columns(pooled) == ["exposure_decay_rate"]


def test_select_new_band_refuses_overlap_without_seed_from():
    raw = _decay_rows(range(0, 6))
    with pytest.raises(SystemExit, match="seed bands overlap"):
        merge_mod.select_new_band(raw, archive_band_end=4)


def test_select_new_band_drops_the_archived_cycles_with_seed_from():
    # the server's logs/ still holds seeds 0-3 (already in the archive): with
    # --seed-from 4 only the new band survives, and pooling stays exact
    raw = _decay_rows(range(0, 10))
    kept = merge_mod.select_new_band(raw, archive_band_end=4, seed_from=4)
    assert int(kept["seed"].min()) == 4 and int(kept["seed"].max()) == 9
    assert len(kept) == len(raw[raw["seed"] >= 4])
    # seed_from above the band end is allowed too (a gap is just fewer seeds)
    kept = merge_mod.select_new_band(raw, archive_band_end=4, seed_from=6)
    assert int(kept["seed"].min()) == 6


def test_select_new_band_refuses_seed_from_inside_the_archive():
    raw = _decay_rows(range(0, 10))
    with pytest.raises(SystemExit, match="inside the archive band"):
        merge_mod.select_new_band(raw, archive_band_end=4, seed_from=2)
    with pytest.raises(SystemExit, match="no rows with seed >= 50"):
        merge_mod.select_new_band(raw, archive_band_end=4, seed_from=50)


def test_latest_archive_picks_the_newest_date_dir(tmp_path):
    for d in ("2026-07-29", "2026-09-07", "2026-07-20"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "aggregate_all.csv").write_text("x\n")
    (tmp_path / "notes").mkdir()  # a non-date dir without an aggregate is ignored
    assert merge_mod.latest_archive(tmp_path) == tmp_path / "2026-09-07" / "aggregate_all.csv"
    with pytest.raises(SystemExit, match="no data/<date>/aggregate_all.csv"):
        merge_mod.latest_archive(tmp_path / "notes")


# ---- several settings in one run (Figure 2 + Table 1 at 500 seeds, 2026-09-09)

def _table1_rows(seeds, **kw):
    """table1_misspecification sweeps two axes (true structure x assumed class);
    build the 3x3 grid by sweeping exposure_model_class for each structure."""
    frames = []
    for structure in ("pbm", "contextual_pbm", "ranking_dependent"):
        f = _rows(seeds, setting="table1_misspecification", axis="exposure_model_class",
                  levels=("pbm", "contextual_pbm", "ranking_dependent"), **kw)
        f["exposure_structure"] = structure
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _fig2_rows(seeds, **kw):
    return _rows(seeds, setting="fig2_policy_divergence", axis="epsilon",
                 levels=(0.0, 0.2, 0.5), **kw)


def test_parse_settings():
    assert merge_mod.parse_settings("fig1_data_size") == ["fig1_data_size"]
    assert merge_mod.parse_settings(" fig2_policy_divergence, table1_misspecification ,") == [
        "fig2_policy_divergence", "table1_misspecification"
    ]
    with pytest.raises(SystemExit, match="twice"):
        merge_mod.parse_settings("app_decay_K,app_decay_K")
    with pytest.raises(SystemExit, match="at least one"):
        merge_mod.parse_settings(" , ")


def test_pooling_is_exact_for_the_table1_grid():
    rng = np.random.RandomState(4)
    pooled = _assert_pooled_matches_union(_table1_rows(range(0, 4), rng=rng),
                                          _table1_rows(range(4, 10), rng=rng))
    assert len(pooled) == 9 * 2  # 3x3 grid x 2 estimators
    assert (pooled["n_seeds"] == 10).all()
    assert merge_mod.sweep_columns(pooled) == ["exposure_structure", "exposure_model_class"]


def test_merge_setting_pools_each_setting_from_its_own_logs(tmp_path):
    # archive = mode=plot aggregate over seeds 0-3 of both settings (plus an
    # untouched third setting); the later band 4-9 sits in logs/<setting>/
    rng = np.random.RandomState(5)
    fig2_old, fig2_new = _fig2_rows(range(0, 4), rng=rng), _fig2_rows(range(4, 10), rng=rng)
    t1_old, t1_new = _table1_rows(range(0, 4), rng=rng), _table1_rows(range(4, 10), rng=rng)
    other = _decay_rows(range(0, 4), rng=rng)
    archive = main_mod.aggregate_results(pd.concat([fig2_old, t1_old, other], ignore_index=True))
    logs = tmp_path / "logs"
    for setting, new in (("fig2_policy_divergence", fig2_new), ("table1_misspecification", t1_new)):
        # the machine still holds the archived band too: --seed-from drops it
        run_dir = logs / setting / "multirun_x" / "0"
        run_dir.mkdir(parents=True)
        band_old = fig2_old if setting.startswith("fig2") else t1_old
        pd.concat([band_old, new], ignore_index=True).to_csv(run_dir / "results.csv", index=False)

    merged = [
        merge_mod.merge_setting(archive, s, logs / s, seed_from=4)
        for s in ("fig2_policy_divergence", "table1_misspecification")
    ]
    assert [m.setting for m in merged] == ["fig2_policy_divergence", "table1_misspecification"]
    for m, old_rows, new_rows in zip(merged, (fig2_old, t1_old), (fig2_new, t1_new)):
        assert (m.archive_band_end, m.new_lo, m.new_hi, m.new_n_seeds, m.pooled_n_seeds) == (4, 4, 9, 6, 10)
        ref = main_mod.aggregate_results(pd.concat([old_rows, new_rows], ignore_index=True))
        ref = ref.sort_values(merge_mod.KEYS).reset_index(drop=True)
        got = m.pooled.sort_values(merge_mod.KEYS).reset_index(drop=True)
        assert len(got) == len(ref)
        np.testing.assert_allclose(got["rel_mse"].to_numpy(float), ref["rel_mse"].to_numpy(float),
                                   rtol=1e-10, atol=1e-12)
    # a setting absent from the logs is refused, not silently skipped
    with pytest.raises(SystemExit, match="no results.csv under"):
        merge_mod.merge_setting(archive, "app_decay_K", logs / "app_decay_K", seed_from=4)


def test_parse_seed_from():
    settings = ["fig4_cascade", "fig5_logging_determinism", "app_a2_robustness"]
    assert merge_mod.parse_seed_from(None, settings) == [None, None, None]
    assert merge_mod.parse_seed_from("200", settings) == [200, 200, 200]
    assert merge_mod.parse_seed_from(" 200, 200 ,100", settings) == [200, 200, 100]
    assert merge_mod.parse_seed_from(100, ["app_decay_K"]) == [100]
    with pytest.raises(SystemExit, match="3 values for 2 settings"):
        merge_mod.parse_seed_from("200,200,100", settings[:2])
    with pytest.raises(SystemExit, match="empty entry"):
        merge_mod.parse_seed_from("200,,100", settings)
    with pytest.raises(SystemExit, match="integer"):
        merge_mod.parse_seed_from("200,x,100", settings)


def _a2_rows(seeds, **kw):
    return _rows(seeds, setting="app_a2_robustness", axis="or_correlation",
                 levels=(0.0, 0.25, 0.5, 0.75, 1.0), **kw)


def test_pooling_is_exact_for_fig4_single_config_with_skipped_oracle():
    # fig4_cascade has one config; the oracle estimators are skipped (all-NaN
    # cells) under the cascade DGP, so those cells must pool to n_seeds=0/NaN
    rng = np.random.RandomState(6)
    ests = ("ed-dr", "iips", "le-iips (oracle)")
    band1 = _rows(range(0, 4), estimators=ests, nan_for=("le-iips (oracle)",), rng=rng,
                  setting="fig4_cascade", axis="exposure_decay_rate", levels=(1.0,))
    band2 = _rows(range(4, 10), estimators=ests, nan_for=("le-iips (oracle)",), rng=rng,
                  setting="fig4_cascade", axis="exposure_decay_rate", levels=(1.0,))
    pooled = _assert_pooled_matches_union(band1, band2)
    assert len(pooled) == 3
    assert merge_mod.sweep_columns(pooled) == []
    oracle = pooled[pooled["estimator"] == "le-iips (oracle)"].iloc[0]
    assert oracle["n_seeds"] == 0 and np.isnan(oracle["rel_mse"])


def test_pooling_is_exact_for_the_a2_sweep_with_appendix_band():
    # appendix band: archive 0-99 would meet new 100-499; here 0-3 + 4-9
    rng = np.random.RandomState(7)
    pooled = _assert_pooled_matches_union(_a2_rows(range(0, 4), rng=rng), _a2_rows(range(4, 10), rng=rng))
    assert len(pooled) == 5 * 2
    assert merge_mod.sweep_columns(pooled) == ["or_correlation"]


def _fig3_rows(seeds, rng=None):
    """Oracle-corruption cells of fig3_dr_grid: the per-seed rows carry
    oracle_ratio_error_mean, which aggregate_results averages per cell and
    the log-log panels use as the x axis."""
    rng = rng or np.random.RandomState(0)
    base = main_mod.flatten_setting_defaults()
    rows = []
    for cr, err in (("none", 0.01), ("E:power:0.5", 0.06), ("power:0.5", 0.3)):
        for seed in seeds:
            for est in ("ed-dr (oracle)", "le-iips (oracle)"):
                rows.append(dict(base, setting_name="fig3_dr_grid", seed=seed, estimator=est,
                                 corrupt_exposure_ratio=cr, corrupt_relevance="none",
                                 estimate=1.0 + 0.02 * rng.normal(), ground_truth=1.0,
                                 oracle_ratio_error_mean=err + 0.01 * rng.normal()))
    return pd.DataFrame(rows)


def test_pooling_is_exact_for_fig3_oracle_error_columns():
    rng = np.random.RandomState(7)
    band1, band2 = _fig3_rows(range(0, 100), rng=rng), _fig3_rows(range(100, 500), rng=rng)
    pooled = _assert_pooled_matches_union(band1, band2)
    ref = main_mod.aggregate_results(pd.concat([band1, band2], ignore_index=True))
    ref = ref.sort_values(merge_mod.KEYS).reset_index(drop=True)
    np.testing.assert_allclose(
        pooled["oracle_ratio_error_mean"].to_numpy(float),
        ref["oracle_ratio_error_mean"].to_numpy(float), rtol=1e-10, atol=1e-12,
    )
    # the column absent from both bands stays NaN, not 0
    assert pooled["oracle_relevance_error_mean"].isna().all()
    assert (pooled["n_seeds"] == 500).all()


def test_pooling_keeps_oracle_columns_nan_for_settings_without_them():
    rng = np.random.RandomState(8)
    pooled = merge_mod.pool_cells(
        main_mod.aggregate_results(_rows(range(0, 5), rng=rng)),
        main_mod.aggregate_results(_rows(range(5, 12), rng=rng)),
    )
    assert pooled["oracle_ratio_error_mean"].isna().all()


def test_setting_archive_replaces_the_archives_rows_of_the_setting(tmp_path):
    rng = np.random.RandomState(9)
    # the main archive holds fig1 and a retired design of fig3 (as the 2026-09
    # archives hold the 2x2-grid rows); the setting archive holds the real cells
    legacy = _fig3_rows(range(0, 8), rng=rng).assign(n_mc_samples=100)
    archive = main_mod.aggregate_results(
        pd.concat([_rows(range(0, 4), rng=rng), legacy], ignore_index=True)
    )
    real = _fig3_rows(range(0, 4), rng=rng).assign(n_mc_samples=10000)
    other = main_mod.aggregate_results(
        pd.concat([real, _decay_rows(range(0, 4), rng=rng)], ignore_index=True)
    )
    path = tmp_path / "aggregate_all.csv"
    other.to_csv(path, index=False)
    new_archive, replaced = merge_mod.take_setting_rows_from(archive, path, ["fig3_dr_grid"])
    assert replaced == {"fig3_dr_grid": int((archive["setting_name"] == "fig3_dr_grid").sum())}
    fig3 = new_archive[new_archive["setting_name"] == "fig3_dr_grid"]
    assert (fig3["n_seeds"] == 4).all() and (fig3["n_mc_samples"] == 10000).all()
    # only the requested setting is taken; the archive's other rows are untouched
    assert set(new_archive["setting_name"]) == {"fig1_data_size", "fig3_dr_grid"}
    # fig1's 2 levels x 2 estimators = 4 cells
    assert len(new_archive[new_archive["setting_name"] == "fig1_data_size"]) == 2 * 2
    # a setting the setting archive lacks is refused
    with pytest.raises(SystemExit, match="has no rows for setting"):
        merge_mod.take_setting_rows_from(archive, path, ["fig2_policy_divergence"])


def test_render_fig3_draws_the_paper_and_report_figures(tmp_path):
    rng = np.random.RandomState(10)
    agg = main_mod.aggregate_results(_fig3_rows(range(0, 6), rng=rng))
    made = merge_mod.render(agg, "fig3_dr_grid", tmp_path / "figs", tmp_path / "data")
    names = {p.name for p in made}
    assert {"fig3_dr_grid.png", "fig3_injection.png", "appendix_fig3_method_e.png"} <= names
