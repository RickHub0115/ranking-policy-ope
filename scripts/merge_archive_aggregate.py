#!/usr/bin/env python3
"""Merge an archived aggregate (a mode=plot output: per-cell means over an
earlier seed band) with the raw results.csv of a later seed band, for when the
earlier band's per-seed rows are not at hand.

Typical use: seeds 0-199 of a setting were run earlier and archived as a
mode=plot aggregate, seeds 200-499 are run later with `START_SEED=200
N_SEEDS=300 ./run_experiments.sh <stage>`, and the two bands are pooled here
(--seed-from keeps the merge exact even when logs/ still holds the archived
seeds).

Why this is exact for the point estimates
-----------------------------------------
Per cell (CONFIG_KEYS x estimator) `aggregate_results` stores, over the
normalised errors s = (estimate - V)/V of the n_seeds non-NaN seeds,

    rel_mse = mean(s^2),   bias = mean(s),   variance = mean(s^2) - mean(s)^2   (ddof=0).

Two disjoint seed bands therefore pool by seed-weighted means:

    m = (n1 m1 + n2 m2)/(n1 + n2),   b = (n1 b1 + n2 b2)/(n1 + n2),   var = m - b^2,

which is what `aggregate_results` would compute on the concatenated rows (the
DGP is fixed by DGP_RANDOM_STATE, so V is the same in both bands). rel_mse_se
and bias_se are SEs of a mean: the ddof=1 sample variance of s^2 (of s) over
both bands follows exactly from the per-band sample variances and means, so
those two columns are exact as well. variance_se (influence-function SE, needs
the 4th central moment) cannot be recovered from the archive and is written as
NaN -- harmless while the figures run with SHOW_CI off (src/main.py).

Usage (from the repository root):
    .venv/bin/python scripts/merge_archive_aggregate.py
        # = --setting fig1_data_size --archive <newest data/*/aggregate_all.csv>
        #   --logs logs/fig1_data_size --data-date <today> --render fig1_data_size
        #   --figs-dir figs,figs/no_sips
    .venv/bin/python scripts/merge_archive_aggregate.py --setting app_decay_K --seed-from 100
        # pool the archived seeds 0-99 with the seed 100-499 rows found under
        # logs/app_decay_K (rows with seed < 100 -- the old cycles, if still
        # present -- are dropped before pooling), write
        # data/<today>/aggregate_all.csv and re-render appendix_decay_K.png into
        # figs/ and figs/no_sips/
    .venv/bin/python scripts/merge_archive_aggregate.py \
        --setting fig2_policy_divergence,table1_misspecification --seed-from 200
        # several settings in ONE run, each pooled from its own logs/<setting>/
        # (archived seeds 0-199 +
        # the seed 200-499 rows). One run is required when the settings are
        # merged on the same day: a second run would find today's output as the
        # newest archive and refuse to overwrite it.
    .venv/bin/python scripts/merge_archive_aggregate.py \
        --setting fig4_cascade,fig5_logging_determinism,app_a2_robustness \
        --seed-from 200,200,100
        # settings from DIFFERENT seed bands in one run -- fig4/fig5 are main-band
        # settings (archived seeds 0-199, new 200-499), app_a2_robustness is an
        # appendix-band setting (archived 0-99, new 100-499) -- so --seed-from is
        # given per setting, in the order of --setting. A single value applies
        # to every setting. fig4_cascade renders the cascade bar row
        # (fig5_cascade.png) from the pooled
        # cells; the Eq. (9) bound scatter (fig4_le_bias_bound.png) is not a
        # merge product -- refresh it from the per-seed rows under logs/ with
        # `src/visualization.py --scatter-only`.
    .venv/bin/python scripts/merge_archive_aggregate.py --setting fig3_dr_grid \
        --seed-from 100 --setting-archive data/<older date>/aggregate_all.csv
        # The error-injection figures (fig3_injection.png / fig3_dr_grid.png /
        # appendix method-E figure): when the archived cells of a setting live
        # in an OLDER archive than the newest one (which may still hold rows of
        # a retired design of the same setting), --setting-archive supplies the
        # archived cells (and drops the retired rows) while --archive stays the
        # newest chain link. The per-cell oracle_ratio_error_mean (the x axis
        # of the log-log panels) is pooled too, seed-weighted, which is exact
        # for these cells (no NaN estimates). Needs the SAME strength levels as
        # the archived run (FIG3_* env vars): pool_cells refuses a cell set
        # that differs.
    .venv/bin/python scripts/merge_archive_aggregate.py \
        --render fig1_data_size,fig2_policy_divergence,app_decay_K
        # also re-render the other paper figures from the (unchanged) archive rows,
        # e.g. after a change of the plotting code

The archive defaults to the newest data/<date>/aggregate_all.csv, so successive
merges chain (each one carries the earlier merged settings forward). Running the
same merge twice is refused by the seed-band check, because the newest archive
then already holds the new band.

Outputs:
    data/<date>/aggregate_all.csv  the archive (retired estimators dropped) with the
                                   merged settings' rows replaced by the pooled cells
    data/<date>/table1_misspecification.csv  when table1_misspecification is rendered
                                   (make_table1 writes its csv next to the aggregate)
    data/<date>/merge_note.txt     provenance: sources, seed bands, what is exact
    <figs-dir>/...                 the figures of the settings named in --render, in
                                   every directory of --figs-dir (comma-separated)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import main as eddr_main  # noqa: E402

KEYS: List[str] = eddr_main.CONFIG_KEYS + ["estimator"]
METRICS: List[str] = [
    "rel_mse", "bias", "variance", "n_seeds", "bias_sq",
    "rel_mse_se", "bias_se", "variance_se",
]
# per-cell means aggregate_results appends only when the logs record them
# (the oracle-corruption runs of fig3_dr_grid): pooled as seed-weighted means,
# which is exact because those cells have no NaN estimates (n_seeds = rows).
# Absent from an archive -> NaN in the output for that archive's settings.
OPTIONAL_METRICS: List[str] = ["oracle_ratio_error_mean", "oracle_relevance_error_mean"]
ALL_METRICS: List[str] = METRICS + OPTIONAL_METRICS


def latest_archive(data_root: Path) -> Path:
    """Newest data/<date>/aggregate_all.csv by directory name (the dirs are
    ISO dates, so lexical order is chronological)."""
    candidates = sorted(
        p for p in data_root.glob("*/aggregate_all.csv") if p.parent.name[:4].isdigit()
    )
    if not candidates:
        raise SystemExit(f"no data/<date>/aggregate_all.csv under {data_root}")
    return candidates[-1]


def select_new_band(raw: pd.DataFrame, archive_band_end: int, seed_from=None) -> pd.DataFrame:
    """Rows of the later band to pool with an archive covering seeds
    [0, archive_band_end).

    Without `seed_from` the logs must lie entirely above the archive band
    (SystemExit otherwise: a re-run of the merge, or a logs/ that still holds
    the archived cycles). With `seed_from` (>= archive_band_end) rows below it
    are dropped first: they are the archived cycles themselves, already in the
    archive's cell means, so dropping them keeps the pooling exact while the
    machine keeps its full logs/ for mode=plot."""
    if seed_from is not None:
        if seed_from < archive_band_end:
            raise SystemExit(
                f"--seed-from {seed_from} lies inside the archive band 0-{archive_band_end - 1}: "
                "rows below the archive's n_seeds are already pooled in the archive"
            )
        below = raw["seed"] < seed_from
        if below.any():
            print(
                f"dropping {int(below.sum())} rows with seed < {seed_from} "
                f"(seeds {int(raw.loc[below, 'seed'].min())}-{int(raw.loc[below, 'seed'].max())}: "
                "the archived band, already pooled)"
            )
            raw = raw[~below]
        if raw.empty:
            raise SystemExit(f"no rows with seed >= {seed_from} in the logs")
    new_lo = int(raw["seed"].min())
    if new_lo < archive_band_end:
        raise SystemExit(
            f"seed bands overlap: archive covers seeds 0-{archive_band_end - 1} "
            f"(n_seeds={archive_band_end}), new logs start at seed {new_lo} "
            "(pass --seed-from <first new seed> if logs/ still holds the archived cycles)"
        )
    return raw


def sweep_columns(cells: pd.DataFrame) -> List[str]:
    """CONFIG_KEYS that take more than one value over the cells: the axes the
    setting sweeps (for the quick look printed at the end)."""
    return [k for k in eddr_main.CONFIG_KEYS if k != "setting_name" and cells[k].nunique(dropna=False) > 1]


def backfill_config_columns(agg: pd.DataFrame) -> pd.DataFrame:
    """The two exact backfills mode=plot applies to raw logs (load_results_csvs),
    for archived aggregates written before those columns existed."""
    agg = agg.copy()
    if "or_coupling" not in agg.columns:
        agg["or_coupling"] = 0.0
    if "relevance_scale" not in agg.columns:
        agg["relevance_scale"] = 1.0
    return agg


def pooled_se_of_mean(n1, m1, se1, n2, m2, se2, n, m):
    """SE of the pooled mean from per-band (n_i, mean_i, SE_i of the mean).

    v_i = se_i^2 n_i is the ddof=1 sample variance of band i; the ddof=1
    variance of the union is
        [(n1-1) v1 + (n2-1) v2 + n1 (m1-m)^2 + n2 (m2-m)^2] / (n-1).
    Bands with n_i = 0 contribute nothing (their NaN stats are masked); n <= 1
    gives NaN, as se_of_mean does for a single seed."""
    n1 = np.asarray(n1, float); n2 = np.asarray(n2, float); n = np.asarray(n, float)
    m1 = np.where(n1 > 0, m1, 0.0); m2 = np.where(n2 > 0, m2, 0.0)
    v1 = np.where(n1 > 1, np.asarray(se1, float) ** 2 * n1, 0.0)
    v2 = np.where(n2 > 1, np.asarray(se2, float) ** 2 * n2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_var = (
            np.maximum(n1 - 1, 0) * v1 + np.maximum(n2 - 1, 0) * v2
            + n1 * (m1 - m) ** 2 + n2 * (m2 - m) ** 2
        ) / (n - 1)
        se = np.sqrt(pooled_var / n)
    return np.where(n > 1, se, np.nan)


def pool_cells(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Pool two aggregates of the SAME cells (KEYS) from disjoint seed bands.

    Raises if the two frames do not cover exactly the same cells: a partial
    merge would silently mix 200- and 500-seed cells in one figure."""
    merged = old.reindex(columns=KEYS + ALL_METRICS).merge(
        new.reindex(columns=KEYS + ALL_METRICS),
        on=KEYS, how="outer", suffixes=("_1", "_2"), indicator=True,
    )
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            "cell sets differ between the archive and the new logs "
            f"({int((unmatched['_merge'] == 'left_only').sum())} archive-only, "
            f"{int((unmatched['_merge'] == 'right_only').sum())} logs-only cells); "
            "both sides must hold every (config x estimator) cell of the setting"
        )
    n1 = merged["n_seeds_1"].to_numpy(float)
    n2 = merged["n_seeds_2"].to_numpy(float)
    n = n1 + n2
    m1 = np.where(n1 > 0, merged["rel_mse_1"].to_numpy(float), 0.0)
    m2 = np.where(n2 > 0, merged["rel_mse_2"].to_numpy(float), 0.0)
    b1 = np.where(n1 > 0, merged["bias_1"].to_numpy(float), 0.0)
    b2 = np.where(n2 > 0, merged["bias_2"].to_numpy(float), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(n > 0, (n1 * m1 + n2 * m2) / n, np.nan)
        b = np.where(n > 0, (n1 * b1 + n2 * b2) / n, np.nan)
    out = merged[KEYS].copy()
    out["rel_mse"] = m
    out["bias"] = b
    out["variance"] = m - b**2
    out["n_seeds"] = n.astype(int)
    out["bias_sq"] = b**2
    out["rel_mse_se"] = pooled_se_of_mean(
        n1, m1, merged["rel_mse_se_1"], n2, m2, merged["rel_mse_se_2"], n, m
    )
    out["bias_se"] = pooled_se_of_mean(
        n1, b1, merged["bias_se_1"], n2, b2, merged["bias_se_2"], n, b
    )
    out["variance_se"] = np.nan  # needs the 4th central moment; see module docstring
    for col in OPTIONAL_METRICS:
        v1 = merged[f"{col}_1"].to_numpy(float)
        v2 = merged[f"{col}_2"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pooled = (n1 * np.where(n1 > 0, v1, 0.0) + n2 * np.where(n2 > 0, v2, 0.0)) / n
        # a cell that records the mean in neither band stays NaN (most settings)
        out[col] = np.where(np.isfinite(v1) | np.isfinite(v2), pooled, np.nan)
    return out


def parse_settings(spec: str) -> List[str]:
    """`--setting a,b` -> ['a', 'b'] (order kept, blanks dropped, duplicates
    refused: a setting pooled twice would count its new band twice)."""
    settings = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not settings:
        raise SystemExit("--setting needs at least one setting_name")
    dups = sorted({s for s in settings if settings.count(s) > 1})
    if dups:
        raise SystemExit(f"--setting lists a setting twice: {', '.join(dups)}")
    return settings


def parse_seed_from(spec, settings: List[str]) -> List[Optional[int]]:
    """`--seed-from` per setting: None (not given) -> [None, ...]; a single
    integer applies to every setting; a comma-separated list must have one
    entry per --setting, in the same order (settings run in different seed
    bands -- main band 200-499 vs appendix band 100-499 -- need different
    first seeds, and one shared value would either be refused for the
    main-band setting or silently drop new seeds of the appendix one)."""
    if spec is None:
        return [None] * len(settings)
    parts = [x.strip() for x in str(spec).split(",")]
    if any(p == "" for p in parts):
        raise SystemExit(f"--seed-from has an empty entry: {spec!r}")
    try:
        values = [int(x) for x in parts]
    except ValueError:
        raise SystemExit(f"--seed-from must be an integer or a comma-separated list of integers: {spec!r}")
    if len(values) == 1:
        return values * len(settings)
    if len(values) != len(settings):
        raise SystemExit(
            f"--seed-from lists {len(values)} values for {len(settings)} settings; give one "
            "value (applied to every setting) or one per --setting in the same order"
        )
    return values


def take_setting_rows_from(archive: pd.DataFrame, setting_archive_path: Path,
                           settings: List[str]):
    """Replace, in `archive`, the rows of every setting in `settings` by the
    rows the setting archive holds for it (--setting-archive). For settings
    the main archive lacks, or holds only in a retired design (e.g. the
    retired 2x2-grid rows of fig3_dr_grid with the unparameterized 'power'
    corruption). Every listed setting must be in the setting
    archive (SystemExit otherwise); the rest of the setting archive is not
    carried forward. Returns the new archive and, per setting, how many rows
    of the main archive were replaced."""
    other = backfill_config_columns(pd.read_csv(setting_archive_path))
    other = other[~other["estimator"].isin(eddr_main.RETIRED_ESTIMATORS)]
    replaced: Dict[str, int] = {}
    parts = [archive[~archive["setting_name"].isin(settings)]]
    for setting in settings:
        rows = other[other["setting_name"] == setting]
        if rows.empty:
            raise SystemExit(
                f"the setting archive {setting_archive_path} has no rows for setting {setting!r}"
            )
        replaced[setting] = int((archive["setting_name"] == setting).sum())
        parts.append(rows)
    return pd.concat(parts, ignore_index=True), replaced


class MergedSetting:
    """One setting's pooled cells plus the provenance printed to merge_note.txt."""

    def __init__(self, setting: str, logs: Path, pooled: pd.DataFrame,
                 archive_band_end: int, new_lo: int, new_hi: int, new_n_seeds: int):
        self.setting = setting
        self.logs = logs
        self.pooled = pooled
        self.archive_band_end = archive_band_end
        self.new_lo = new_lo
        self.new_hi = new_hi
        self.new_n_seeds = new_n_seeds
        self.pooled_n_seeds = int(pooled["n_seeds"].max())


def merge_setting(archive: pd.DataFrame, setting: str, logs: Path, seed_from=None) -> MergedSetting:
    """Pool the archive's cells of `setting` with the later band found under
    `logs` (recursively). Nothing is written; main() assembles the output."""
    old = archive[archive["setting_name"] == setting]
    if old.empty:
        raise SystemExit(f"the archive has no rows for setting {setting!r}")

    raw = eddr_main.load_results_csvs(logs)  # validates, de-duplicates, drops retired
    if raw is None:
        raise SystemExit(f"no results.csv under {logs}")
    raw = raw[raw["setting_name"] == setting]
    if raw.empty:
        raise SystemExit(f"no rows of setting {setting!r} under {logs}")
    # the archive's cycles are contiguous from seed 0, so its band is
    # [0, max n_seeds); the new rows must lie entirely above it
    archive_band_end = int(old["n_seeds"].max())
    raw = select_new_band(raw, archive_band_end, seed_from)
    new_lo, new_hi = int(raw["seed"].min()), int(raw["seed"].max())
    new = eddr_main.aggregate_results(raw)
    pooled = pool_cells(old, new)
    return MergedSetting(setting, logs, pooled, archive_band_end, new_lo, new_hi,
                         int(new["n_seeds"].max()))


# figure functions per setting_name that need only the aggregate. The
# cascade bar row (fig5_cascade.png) is aggregate-only too; the Eq. (9)
# bound scatter (fig4_le_bias_bound.png) needs
# the per-seed rows of the exposure-DGP sweeps and is refreshed separately with
# `src/visualization.py --scatter-only` (it is not a seed-count claim of any
# single setting, so it is not merged here).
def render_fig3(agg: pd.DataFrame, figs_dir: Path) -> List[Path]:
    """The three figures drawn from the fig3_dr_grid cells: the report's
    fig3_dr_grid.png (R x relevance slope-2 series), the paper's
    fig3_injection.png (conditions (a)(b)(c)) and the
    method-E appendix figure. All aggregate-only (bias, bias_se and the
    pooled oracle_ratio_error_mean)."""
    outs = []
    for fn in (eddr_main.make_fig3, eddr_main.make_fig3_injection,
               eddr_main.make_fig3_method_e_appendix):
        out = fn(agg, figs_dir)
        if out is not None:
            outs.append(out)
    return outs


RENDERERS: Dict[str, Callable[[pd.DataFrame, Path], object]] = {
    "fig1_data_size": eddr_main.make_fig1,
    "fig2_policy_divergence": eddr_main.make_fig2,
    "fig3_dr_grid": render_fig3,
    "fig5_logging_determinism": eddr_main.make_fig7_tau0,
    "fig4_cascade": eddr_main.make_fig5_cascade,
    "fig4_cascade_eta": eddr_main.make_fig4_eta,
}


def render(agg: pd.DataFrame, setting: str, figs_dir: Path, data_dir: Path,
           log_root: Optional[Path] = None) -> List[Path]:
    figs_dir.mkdir(parents=True, exist_ok=True)
    if setting in RENDERERS:
        out = RENDERERS[setting](agg, figs_dir)
        if isinstance(out, (list, tuple)):
            return [p for p in out if p is not None]
        return [out] if out is not None else []
    if setting == "table1_misspecification":
        out = eddr_main.make_table1(agg, figs_dir, data_dir=data_dir)
        return [out] if out is not None else []
    if setting.startswith("app_"):
        # make_appendix_figs renders only the settings present in the frame
        return eddr_main.make_appendix_figs(agg[agg["setting_name"] == setting], figs_dir)
    raise SystemExit(f"no aggregate-only renderer for setting {setting!r}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--setting", default="fig1_data_size",
                    help="setting_name whose cells are pooled (default fig1_data_size); "
                         "comma-separated for several settings merged in one run, each "
                         "from its own logs/<setting>/ (or all from --logs)")
    ap.add_argument("--archive", type=Path, default=None,
                    help="archived aggregate_all.csv of the earlier seed band "
                         "(default: the newest data/<date>/aggregate_all.csv)")
    ap.add_argument("--setting-archive", type=Path, default=None,
                    help="aggregate_all.csv whose rows of the merged setting(s) are used as "
                         "the archived cells INSTEAD of --archive's rows for them (which "
                         "are dropped), e.g. --setting fig3_dr_grid "
                         "--setting-archive data/<older date>/aggregate_all.csv")
    ap.add_argument("--logs", type=Path, default=None,
                    help="directory scanned recursively for the later band's results.csv "
                         "(default logs/<setting>, per setting)")
    ap.add_argument("--seed-from", default=None,
                    help="first seed of the new band; rows below it are dropped before "
                         "pooling (for a logs/ that still holds the archived cycles). "
                         "Must be >= the archive's n_seeds of the setting. One integer "
                         "for every setting, or a comma-separated list with one entry "
                         "per --setting in the same order (settings from different "
                         "seed bands)")
    ap.add_argument("--data-date", default=None,
                    help="output dir data/<data-date>/ (default today)")
    ap.add_argument("--figs-dir", default="figs,figs/no_sips",
                    help="comma-separated output dirs for the figures, relative to the "
                         "repository root unless absolute (default figs,figs/no_sips)")
    ap.add_argument("--render", default=None,
                    help="comma-separated setting_names to (re)render from the output "
                         "aggregate (default: the merged settings only; '' for none)")
    args = ap.parse_args()

    settings = parse_settings(args.setting)
    seed_froms = parse_seed_from(args.seed_from, settings)
    date_str = args.data_date or datetime.now().strftime("%Y-%m-%d")
    data_dir = REPO_ROOT / "data" / date_str
    render_list = (
        list(settings) if args.render is None
        else [s.strip() for s in args.render.split(",") if s.strip()]
    )
    figs_dirs = [
        (Path(d) if Path(d).is_absolute() else REPO_ROOT / d)
        for d in (x.strip() for x in str(args.figs_dir).split(",")) if d
    ]
    archive_path = args.archive if args.archive is not None else latest_archive(REPO_ROOT / "data")
    if archive_path.resolve() == (data_dir / "aggregate_all.csv").resolve():
        raise SystemExit(
            f"the archive {archive_path} is also the output file; pass a different "
            "--data-date (or --archive) so the earlier aggregate is not overwritten"
        )
    print(f"archive: {archive_path}")

    archive = backfill_config_columns(pd.read_csv(archive_path))
    n_retired = int(archive["estimator"].isin(eddr_main.RETIRED_ESTIMATORS).sum())
    archive = archive[~archive["estimator"].isin(eddr_main.RETIRED_ESTIMATORS)]
    replaced: Dict[str, int] = {}
    if args.setting_archive is not None:
        archive, replaced = take_setting_rows_from(archive, args.setting_archive, settings)
        for s_, n_ in replaced.items():
            print(f"setting {s_}: archived cells taken from {args.setting_archive}"
                  f" ({n_} rows of the archive for this setting dropped)")

    # every setting is pooled before anything is written: a failure on the
    # second setting must not leave a half-merged data/<date>/
    merged: List[MergedSetting] = []
    for setting, seed_from in zip(settings, seed_froms):
        logs = args.logs if args.logs is not None else REPO_ROOT / "logs" / setting
        merged.append(merge_setting(archive, setting, logs, seed_from))

    rest = archive[~archive["setting_name"].isin(settings)].reindex(columns=KEYS + ALL_METRICS)
    out = (
        pd.concat([rest] + [m.pooled for m in merged], ignore_index=True)
        .sort_values(KEYS, kind="stable")
        .reset_index(drop=True)
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    out_csv = data_dir / "aggregate_all.csv"
    out.to_csv(out_csv, index=False)

    note = data_dir / "merge_note.txt"
    lines = [
        f"aggregate_all.csv written by scripts/merge_archive_aggregate.py on {date_str}",
        f"  archive : {archive_path}  (retired estimator rows dropped: {n_retired})",
    ]
    for s_, n_ in replaced.items():
        lines.append(f"  archived cells of {s_}: taken from {args.setting_archive} "
                     f"({n_} rows the archive above held for this setting were dropped)")
    for m, seed_from in zip(merged, seed_froms):
        lines += [
            f"  new logs: {m.logs}"
            + (f"  (rows with seed < {seed_from} dropped)" if seed_from is not None else ""),
            f"  setting : {m.setting}  ({len(m.pooled)} cells)",
            f"  seed bands: archive 0-{m.archive_band_end - 1} (n_seeds={m.archive_band_end}) "
            f"+ new {m.new_lo}-{m.new_hi} (n_seeds={m.new_n_seeds}) "
            f"-> pooled n_seeds={m.pooled_n_seeds}",
        ]
    lines += [
        "  exact: rel_mse, bias, variance, bias_sq, n_seeds, rel_mse_se, bias_se "
        "(seed-weighted pooling; see the script docstring)",
        "  exact: oracle_ratio_error_mean / oracle_relevance_error_mean where recorded "
        "(seed-weighted means; cells without NaN estimates)",
        "  NaN  : variance_se of the merged setting(s) (4th moment not in the archive)",
        "  other settings: copied from the archive unchanged",
    ]
    note.write_text("\n".join(lines) + "\n")
    for m in merged:
        print(f"merged {len(m.pooled)} cells of {m.setting}: seeds 0-{m.archive_band_end - 1} + "
              f"{m.new_lo}-{m.new_hi} -> n_seeds={m.pooled_n_seeds}")
    print(f"written: {out_csv}\n         {note}")

    made: List[Path] = []
    for figs_dir in figs_dirs:
        for s in render_list:
            made += render(out, s, figs_dir, data_dir)
    for p in made:
        print(f"figure: {p}")

    # quick look at the merged cells for the two headline estimators, along the
    # axes each setting sweeps (fig1: exposure_structure x n_rounds; app_decay_K:
    # exposure_decay_rate; table1: exposure_structure x exposure_model_class)
    for m in merged:
        show = m.pooled[m.pooled["estimator"].isin(["ed-dr", "iips"])]
        if show.empty:  # oracle-only settings (fig3_dr_grid)
            show = m.pooled
        axes = sweep_columns(m.pooled)
        cols = axes + ["estimator", "rel_mse", "bias_sq", "variance", "n_seeds"]
        print(f"--- {m.setting}")
        print(show.sort_values(axes + ["estimator"])[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
