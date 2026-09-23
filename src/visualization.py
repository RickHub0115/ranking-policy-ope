"""Paper figures with selected estimators dropped (default: the SIPS family).

SIPS carries an enormous variance in most of our settings, so its line/bar
dominates the log axes and squeezes every other method into the bottom of the
panel. This script re-renders the whole figure set from the same results with
those rows removed, so the remaining methods get the full dynamic range.

Nothing in src/main.py changes: the figure functions there stay the single
source of truth and are imported as-is; the estimators are filtered out of the
aggregated (and raw) frames *before* make_fig* / make_table1 /
make_appendix_figs / make_run_summary_figure are called, and the output goes to
its own directory so the canonical figs/ and data/<date>/ artifacts stay
untouched.

Usage (from the repository root):
    # from the raw logs (default; needed for fig4's Eq. (8) bound scatter and the
    # per-setting *_summary.png quick looks). NOTE: this redraws EVERY figure in
    # --out-dir from the per-seed rows this machine's logs/ holds; where those
    # are a shorter seed band than the merged data/<date>/ cells (the settings
    # run elsewhere), the committed 500-seed figures regress. To refresh only
    # the scatter, use --scatter-only.
    .venv/bin/python src/visualization.py

    # only the Eq. (8) bound scatter (fig4_le_bias_bound.png), leaving every
    # other figure in --out-dir untouched; run once per figure directory
    .venv/bin/python src/visualization.py --scatter-only --out-dir figs
    .venv/bin/python src/visualization.py --scatter-only

    # from an archived aggregate table instead (no raw per-seed rows)
    .venv/bin/python src/visualization.py --agg data/<date>/aggregate_all.csv

    # keep SIPS but drop something else, and write elsewhere
    .venv/bin/python src/visualization.py --exclude cascade-dr --out-dir figs/no_cascade

Importable too, e.g. from a notebook:
    from visualization import drop_estimators, render_all

Outputs (images only, all under --out-dir, default figs/no_sips/):
    fig1_relmse_vs_n.png, fig2_bias_variance_vs_epsilon.png, fig3_dr_grid.png,
    fig3_injection.png (the paper's three-condition injection figure),
    fig4_le_bias_bound.png, fig5_cascade.png, fig7_relmse_vs_tau0.png,
    table1_misspecification.png,
    appendix_*.png, <setting>_summary.png

The csv/tex tables of mode=plot are deliberately not written: under the default
SIPS exclusion they duplicate data/<date>/ (see render_all). Pass --csv-dir if
an exclusion actually changes them.

SIPS is retired upstream (RETIRED_ESTIMATORS in src/main.py:
no longer computed, and dropped when mode=plot merges the logs), so with the
default exclusion this script reproduces figs/ itself; it stays useful for
excluding other estimators (--exclude) and for archived aggregates (--agg)
that still carry sips rows.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import main as eddr_main  # noqa: E402  (src/main.py: the single source of truth)

# "SIPS" in the paper's tables means both the plain and the self-normalized
# variant; snsips is absent from the current runs but listing it keeps the
# default honest if the self-normalized baselines are re-enabled.
DEFAULT_EXCLUDE = ("sips", "snsips")

# columns make_fig4_bias_bound touches; an empty frame with these lets
# the existing guard fall through to its "no data" placeholder instead of
# raising KeyError when we only have an aggregate table.
RAW_COLUMNS = eddr_main.CONFIG_KEYS + [
    "seed",
    "estimator",
    "estimate",
    "ground_truth",
    "prop3_bound",
]


def drop_estimators(df: pd.DataFrame, exclude: Sequence[str]) -> pd.DataFrame:
    """Rows whose `estimator` is not in `exclude`.

    Matching is on the plain estimator label, so the oracle-injected variants
    ("ed-dr (oracle)") are unaffected unless named explicitly."""
    if df.empty or not exclude:
        return df
    return df[~df["estimator"].isin(set(exclude))].copy()


def make_summary_figures(
    df_raw: pd.DataFrame, out_dir: Path, exclude: Sequence[str]
) -> List[Path]:
    """Per-setting quick looks, the filtered counterpart of the `*_summary.png`
    that a training run writes for its own single configuration. Here one
    figure covers every configuration of a setting_name (the sweeps are Hydra
    multiruns), so the bars are means over that setting's cells."""
    outs: List[Path] = []
    for setting, sub in df_raw.groupby("setting_name", dropna=False):
        out = out_dir / f"{setting}_summary.png"
        n_seeds = int(sub["seed"].nunique())
        n_configs = int(sub.groupby(eddr_main.CONFIG_KEYS, dropna=False).ngroups)
        eddr_main.make_run_summary_figure(
            sub,
            out,
            title=(
                f"{setting}: mean over {n_configs} config(s), seeds={n_seeds}"
                f" (excl. {', '.join(exclude)})"
            ),
        )
        outs.append(out)
    return outs


def load_frames(
    logs: Optional[Path] = None,
    agg_csv: Optional[Path] = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """(agg, df_raw) with `exclude` dropped from both, or (None, _) when the
    logs hold no results.csv.

    With `agg_csv` the per-seed rows do not exist, so df_raw comes back as an
    empty frame carrying just the columns make_fig4_bias_bound inspects."""
    if agg_csv is not None:
        return drop_estimators(pd.read_csv(agg_csv), exclude), pd.DataFrame(
            columns=RAW_COLUMNS
        )
    # reuse the validation / de-duplication of mode=plot verbatim: stale csvs
    # and mixed exposure_ratio_clip cells raise here, rather than silently
    # blending into a line
    df_raw = eddr_main.load_results_csvs(logs if logs is not None else REPO_ROOT / "logs")
    if df_raw is None:
        return None, pd.DataFrame(columns=RAW_COLUMNS)
    df_raw = drop_estimators(df_raw, exclude)
    return drop_estimators(eddr_main.aggregate_results(df_raw), exclude), df_raw


def render_all(
    agg: pd.DataFrame,
    df_raw: pd.DataFrame,
    out_dir: Path,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    with_summaries: bool = True,
    csv_dir: Optional[Path] = None,
) -> List[Path]:
    """Render the figures of mode=plot into `out_dir`, from frames the caller
    has already filtered. Returns the paths written.

    Only images land in `out_dir`. The csv/tex tables mode=plot also emits are
    thrown away unless `csv_dir` is given: with the default SIPS exclusion they
    are pure duplicates of data/<date>/ (table1 and the scorecard read ed-dr /
    iips / rips only, so they come out byte-identical; aggregate_all.csv is the
    canonical one minus the dropped rows), and a second copy under figs/ would
    compete with data/<date>/ as the source of the paper's numbers. `csv_dir`
    is for the case where the exclusion *does* move them, e.g.
    `--exclude iips`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        # make_table1 / make_prediction_scorecard write their csv
        # unconditionally; point them at a scratch dir when the caller wants
        # figures only
        tables_dir = csv_dir if csv_dir is not None else Path(tmp)
        tables_dir.mkdir(parents=True, exist_ok=True)
        if csv_dir is not None:
            agg.to_csv(csv_dir / "aggregate_all.csv", index=False)
        made: List[Path] = []
        for fn in (
            eddr_main.make_fig1,
            eddr_main.make_fig2,
            eddr_main.make_fig3,
            eddr_main.make_fig3_injection,
            eddr_main.make_fig4_eta,
            eddr_main.make_fig5_cascade,
            eddr_main.make_fig7_tau0,
        ):
            out = fn(agg, out_dir)
            if out is not None:
                made.append(out)
        out_t1 = eddr_main.make_table1(agg, out_dir, data_dir=tables_dir)
        if out_t1 is not None:
            made.append(out_t1)
            # the .tex goes to figs_dir with no override; keep out_dir images-only
            tex = out_dir / "table1_misspecification.tex"
            if tex.exists():
                if csv_dir is not None:
                    tex.replace(csv_dir / tex.name)
                else:
                    tex.unlink()
        out4 = eddr_main.make_fig4_bias_bound(df_raw, out_dir)
        if out4 is not None:
            made.append(out4)
        made += eddr_main.make_appendix_figs(agg, out_dir)
        if with_summaries and not df_raw.empty:
            made += make_summary_figures(df_raw, out_dir, exclude)
        eddr_main.make_prediction_scorecard(agg, out_dir, data_dir=tables_dir)
    if csv_dir is not None:
        made += [
            csv_dir / n
            for n in (
                "aggregate_all.csv",
                "table1_misspecification.tex",
                "table1_misspecification.csv",
                "prediction_scorecard.csv",
            )
            if (csv_dir / n).exists()
        ]
    return made


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=REPO_ROOT / "logs",
        help="directory scanned recursively for results.csv (default: logs/)",
    )
    parser.add_argument(
        "--agg",
        type=Path,
        default=None,
        help="use an existing aggregate_all.csv instead of --logs; fig4's "
        "Prop.3 scatter and the *_summary.png figures are then skipped "
        "(they need per-seed rows)",
    )
    parser.add_argument(
        "--scatter-only",
        action="store_true",
        help="render only fig4_le_bias_bound.png (the Eq. (8) bound scatter, "
        "from the per-seed rows of --logs) into --out-dir and leave every "
        "other figure there untouched",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "figs" / "no_sips",
        help="output directory for figures and csv tables (default: figs/no_sips/)",
    )
    parser.add_argument(
        "--exclude",
        default=",".join(DEFAULT_EXCLUDE),
        help=f"comma-separated estimator labels to drop (default: {','.join(DEFAULT_EXCLUDE)}); "
        "pass an empty string to keep everything",
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="skip the per-setting *_summary.png figures",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="also write the csv/tex tables (aggregate_all, table1, "
        "prediction_scorecard) here; off by default because with the default "
        "SIPS exclusion they duplicate data/<date>/",
    )
    args = parser.parse_args()

    exclude = [e.strip() for e in str(args.exclude).split(",") if e.strip()]
    out_dir: Path = args.out_dir

    agg, df_raw = load_frames(logs=args.logs, agg_csv=args.agg, exclude=exclude)
    if agg is None:
        print(f"no results.csv found under {args.logs}")
        return
    if agg.empty:
        print(f"nothing left to plot after excluding {exclude}")
        return
    print(f"source: {args.agg if args.agg is not None else args.logs}")
    print(f"excluded: {exclude or '(none)'}")
    if args.scatter_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = eddr_main.make_fig4_bias_bound(df_raw, out_dir)
        if out is None:
            print("no per-seed rows with prop3_bound: fig4_le_bias_bound.png not written")
            return
        src = df_raw[df_raw["prop3_bound"].notna() & (df_raw["estimator"] == "le-iips")]
        n_cfg = src.groupby(eddr_main.CONFIG_KEYS, dropna=False).ngroups
        print(f"written:\n  {out}  ({n_cfg} configurations)")
        return
    print(
        "estimators plotted: "
        + ", ".join(eddr_main._ordered_estimators(agg["estimator"].unique()))
    )

    made = render_all(
        agg,
        df_raw,
        out_dir,
        exclude=exclude,
        with_summaries=not args.no_summaries,
        csv_dir=args.csv_dir,
    )
    print("written:")
    for p in made:
        print(f"  {p}")
    if args.csv_dir is None:
        print("csv/tex tables: not written (duplicates of data/<date>/; --csv-dir to keep)")


if __name__ == "__main__":
    main()
