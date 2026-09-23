# Regression guards for mode=plot csv loading:
# (N2) the CONFIG_KEYS column check must run per csv, before concat — a
#      check on the concatenated frame passes as soon as one new-format csv
#      is present, and stale rows then survive as NaN cells;
# (N1) a clipped main run and an unclipped appendix run of the same setting
#      under one log_dir must be refused — the figure functions select rows
#      by setting_name only and average rel_mse over the remaining cells,
#      so the proposal lines would silently become the auto/null average.
import pandas as pd
import pytest

import main as main_mod


def _rows(seed=0, estimator="ed-dr", estimate=1.0, **overrides):
    row = dict(main_mod.flatten_setting_defaults())
    row.update(
        seed=seed,
        estimator=estimator,
        estimate=estimate,
        ground_truth=2.0,
    )
    row.update(overrides)
    return row


def _write(log_root, run_name, rows, drop_cols=()):
    df = pd.DataFrame(rows)
    if drop_cols:
        df = df.drop(columns=list(drop_cols))
    run_dir = log_root / "default" / run_name
    run_dir.mkdir(parents=True)
    df.to_csv(run_dir / "results.csv", index=False)


def test_stale_csv_alone_is_rejected(tmp_path):
    _write(tmp_path, "run_1", [_rows()], drop_cols=["exposure_ratio_clip"])
    with pytest.raises(ValueError, match="exposure_ratio_clip"):
        main_mod.load_results_csvs(tmp_path)


def test_stale_csv_mixed_with_new_csv_is_still_rejected(tmp_path):
    # N2: one new-format csv must not mask the stale one
    _write(tmp_path, "run_1", [_rows()], drop_cols=["exposure_ratio_clip"])
    _write(tmp_path, "run_2", [_rows(seed=1)])
    with pytest.raises(ValueError, match="lacks config columns"):
        main_mod.load_results_csvs(tmp_path)


def test_mixed_clip_values_within_a_setting_are_rejected(tmp_path):
    # N1: auto main run + null appendix run under the same log_dir
    _write(tmp_path, "run_1", [_rows(exposure_ratio_clip="auto")])
    _write(tmp_path, "run_2", [_rows(exposure_ratio_clip="none")])
    with pytest.raises(ValueError, match="separate log_dir"):
        main_mod.load_results_csvs(tmp_path)


def test_valid_csvs_load_and_dedup_keeps_latest(tmp_path):
    _write(tmp_path, "run_1", [_rows(estimate=1.0)])
    _write(tmp_path, "run_2", [_rows(estimate=3.0)])
    df = main_mod.load_results_csvs(tmp_path)
    assert len(df) == 1
    assert df["estimate"].iloc[0] == 3.0


def test_empty_log_dir_returns_none(tmp_path):
    assert main_mod.load_results_csvs(tmp_path) is None


# Seed-accumulation cycles: merging an extra cycle must add replications,
# not replace them.
def test_disjoint_seed_bands_accumulate(tmp_path):
    # cycle 1 (seeds 0,1) and cycle 2 (seeds 2,3) of the same config
    _write(tmp_path, "run_1", [_rows(seed=s, estimate=1.0) for s in (0, 1)])
    _write(tmp_path, "run_2", [_rows(seed=s, estimate=1.5) for s in (2, 3)])
    df = main_mod.load_results_csvs(tmp_path)
    assert len(df) == 4
    agg = main_mod.aggregate_results(df)
    assert len(agg) == 1
    assert agg["n_seeds"].iloc[0] == 4


def test_overlapping_seed_bands_are_deduplicated(tmp_path):
    # same seed band re-run: start_seed is not part of CONFIG_KEYS, so the
    # rows collapse onto each other instead of doubling the replications
    _write(tmp_path, "run_1", [_rows(seed=s, estimate=1.0) for s in (0, 1)])
    _write(tmp_path, "run_2", [_rows(seed=s, estimate=1.5) for s in (0, 1)])
    df = main_mod.load_results_csvs(tmp_path)
    assert len(df) == 2
    assert set(df["estimate"]) == {1.5}  # keep="last"
    assert main_mod.aggregate_results(df)["n_seeds"].iloc[0] == 2


def test_uneven_seed_counts_warn(tmp_path, capsys):
    # n_rounds=4000 got both cycles, n_rounds=8000 only the first
    _write(
        tmp_path,
        "run_1",
        [_rows(seed=s, n_rounds=4000) for s in (0, 1)]
        + [_rows(seed=0, n_rounds=8000)],
    )
    df = main_mod.load_results_csvs(tmp_path)
    assert len(df) == 3
    err = capsys.readouterr().err
    assert "uneven seed counts" in err
    assert "default" in err

    summary = main_mod.summarize_seed_counts(df)
    assert summary.loc[0, "n_configs"] == 2
    assert summary.loc[0, "seeds_min"] == 1
    assert summary.loc[0, "seeds_max"] == 2


def test_even_seed_counts_do_not_warn(tmp_path, capsys):
    # NaN estimates are counted as rows, so the intentionally skipped cells
    # (fig2 epsilon=0 x cascade-dr) must not trip the F6 warning
    _write(
        tmp_path,
        "run_1",
        [_rows(seed=s, estimator="ed-dr") for s in (0, 1)]
        + [_rows(seed=s, estimator="cascade-dr", estimate=float("nan"))
           for s in (0, 1)],
    )
    df = main_mod.load_results_csvs(tmp_path)
    out = capsys.readouterr()
    assert "uneven seed counts" not in out.err
    assert "seeds/cell: min=2 max=2" in out.out
    # the aggregate n_seeds column disagrees — that is why F5 counts rows
    agg = main_mod.aggregate_results(df)
    assert set(agg["n_seeds"]) == {0, 2}


# Retired estimators (main.RETIRED_ESTIMATORS, 2026-09-06): the cycles up to
# 2026-07 wrote sips rows, later cycles do not compute sips. The loader drops
# the old rows so sips neither reaches the figures nor makes the seed-count
# check uneven for a reason that is not a missing stage.
def test_retired_estimator_rows_are_dropped_on_load(tmp_path, capsys):
    _write(
        tmp_path,
        "run_1",
        [_rows(seed=s, estimator="ed-dr") for s in (0, 1)]
        + [_rows(seed=s, estimator="sips") for s in (0, 1)],
    )
    _write(tmp_path, "run_2", [_rows(seed=s, estimator="ed-dr") for s in (2, 3)])
    df = main_mod.load_results_csvs(tmp_path)
    assert set(df["estimator"]) == {"ed-dr"}
    assert len(df) == 4
    out = capsys.readouterr()
    assert "dropped 2 rows of retired estimator(s) ['sips']" in out.out
    assert "WARNING: uneven seed counts" not in out.err


def test_header_only_csv_does_not_break_the_seed_summary(tmp_path, capsys):
    # a stray n_seeds=0 job leaves a 0-row results.csv in logs/ for good; the
    # F5 summary must not turn that into an opaque max()-of-empty crash
    run_dir = tmp_path / "default" / "run_1"
    run_dir.mkdir(parents=True)
    # full column set, zero rows — _write() cannot express this
    pd.DataFrame([_rows()]).iloc[:0].to_csv(run_dir / "results.csv", index=False)
    df = main_mod.load_results_csvs(tmp_path)
    assert df.empty
    assert "no rows" in capsys.readouterr().out
