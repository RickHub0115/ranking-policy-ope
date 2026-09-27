# ranking-policy-ope

## Setup

Python 3.9 is required (`obp==0.5.5` does not support 3.10+).

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS arm64, install PyYAML first (see the top of `requirements.txt`).

## Tests

```bash
pytest  # fast tests
pytest -m slow
```

## Experiments

`run_experiments.sh` runs the experiment stages (Hydra multiruns of `src/main.py`) and renders the figures.

```bash
./run_experiments.sh list          # stages and options
./run_experiments.sh fig1          # one stage
./run_experiments.sh all           # every stage, then plot
./run_experiments.sh plot          # figures and tables from logs/
```

Per-seed results go to `logs/<setting>/`, figures to `figs/`, and aggregated tables to `data/<date>/`.
`N_JOBS` sets the number of worker processes (default 8).

The `fig3` stage requires the corruption strengths as environment variables:

```bash
export FIG3_E_STRENGTHS='power:0.05,power:0.16,power:0.5,power:1.6,power:5'
export FIG3_R_STRENGTHS='power:0.05,power:0.16,power:0.5,power:1.6,power:5'
export FIG3_REL_STRENGTHS='power:0.5,power:1.6,power:5'
export FIG3_RXREL_PAIRS='0.15/0.5,0.22/0.75,0.33/1.1,0.5/1.55,0.75/2.1'
```

## Layout

```text
conf/            Hydra configs (conf/setting/*.yaml: one file per experiment)
src/
  dataset.py         synthetic data generation (exposure, cascade, DBN)
  exposure_model.py  EM for exposure and relevance, Monte Carlo marginalization, cross-fitting
  estimators_slate.py  ranking OPE estimators
  aips.py            AIPS
  meta_slate.py      nuisance routing to the estimators
  utils.py           policies, ground truth, input validation
  main.py            experiment, aggregation, and plotting entry point
  visualization.py   re-renders the figures with selected estimators excluded
scripts/         pooling of archived aggregates with new logs
tests/           unit and integration tests
```

## License

Apache License 2.0. See `NOTICE` for the code derived from zr-obp and kdd2023-aips.
