#!/usr/bin/env bash
# Runs the experiment stages (Hydra multiruns of src/main.py) and the plotting.
#
# Usage:
#   source .venv/bin/activate
#   ./run_experiments.sh list                        # available stages
#   ./run_experiments.sh fig1                        # one stage
#   ./run_experiments.sh fig1 fig2 table1            # several stages, sequentially
#   ./run_experiments.sh all                         # every stage, then plot
#   ./run_experiments.sh parallel fig1 fig2 fig3     # stages side by side (cores split by weight)
#   ./run_experiments.sh plot                        # render figures / tables from logs/
#   ./run_experiments.sh detach fig1                 # launch under nohup, detached from the terminal
#   CYCLE=2 ./run_experiments.sh all                 # second seed band (see "Seed bands" below)
#   START_SEED=200 N_SEEDS=300 ./run_experiments.sh fig1   # explicit seed band 200-499
#   DRY_RUN=1 ./run_experiments.sh fig1              # print the python commands, run nothing
set -euo pipefail

STAGES=(fig1 fig2 fig3 fig4 fig4eta fig5 table1 appendix decay a2 a2slot eminit plot)
ALL_STAGES=(fig1 fig2 fig3 fig4 fig4eta fig5 table1 appendix decay a2 a2slot eminit plot)

# parallel mode splits the cores in proportion to these weights (rough run time)
weight_of() {
  case "$1" in
    fig1|table1|appendix|a2slot) echo 2 ;;
    *) echo 1 ;;
  esac
}

MAIN_N_SEEDS=100
APPENDIX_N_SEEDS=50

N_JOBS="${N_JOBS:-8}"
if [[ ! "${N_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_JOBS must be a positive integer: '${N_JOBS}'" >&2
  exit 1
fi

# DRY_RUN: shadow python with a function that prints the command line
if [[ -n "${DRY_RUN:-}" ]]; then
  python() { echo "[dry-run] python $*"; }
fi

# Seed bands. One cycle runs every stage on one band of seeds: main-text stages
# use start_seed = 100 * (CYCLE - 1) with 100 seeds, appendix stages
# 50 * (CYCLE - 1) with 50 seeds. START_SEED overrides CYCLE for every stage,
# and N_SEEDS requires START_SEED. mode=plot pools every results.csv under
# logs/, so additional cycles add replications per cell.
# Seeds must stay below 100,000 - n_folds: main.py derives the per-estimator
# RNG streams as seed + k * 100,000 (+ fold), so larger seeds would collide
# with the streams of other seeds.
SEED_MAX=99998

if [[ -n "${N_SEEDS:-}" ]]; then
  if [[ ! "${N_SEEDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_SEEDS must be a positive integer: '${N_SEEDS}'" >&2
    exit 1
  fi
  if [[ -z "${START_SEED:-}" ]]; then
    echo "N_SEEDS requires START_SEED (the CYCLE seed bands assume 100 main / 50 appendix seeds per cycle)" >&2
    echo "  e.g. START_SEED=200 N_SEEDS=300 ./run_experiments.sh fig1   # seeds 200-499" >&2
    exit 1
  fi
  MAIN_N_SEEDS="${N_SEEDS}"
  APPENDIX_N_SEEDS="${N_SEEDS}"
fi

cycle_given=0
if [[ -n "${CYCLE:-}" ]]; then cycle_given=1; fi
CYCLE="${CYCLE:-1}"

if [[ -n "${START_SEED:-}" ]]; then
  if [[ ! "${START_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "START_SEED must be a non-negative integer: '${START_SEED}'" >&2
    exit 1
  fi
  if [[ ${cycle_given} -eq 1 ]]; then
    echo "START_SEED=${START_SEED} takes precedence; CYCLE=${CYCLE} is ignored"
  fi
  MAIN_START_SEED="${START_SEED}"
  APPENDIX_START_SEED="${START_SEED}"
else
  if [[ ! "${CYCLE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CYCLE must be a positive integer: '${CYCLE}'" >&2
    exit 1
  fi
  MAIN_START_SEED=$(( (CYCLE - 1) * MAIN_N_SEEDS ))
  APPENDIX_START_SEED=$(( (CYCLE - 1) * APPENDIX_N_SEEDS ))
fi

if [[ $(( MAIN_START_SEED + MAIN_N_SEEDS - 1 )) -gt ${SEED_MAX} ]] ||
   [[ $(( APPENDIX_START_SEED + APPENDIX_N_SEEDS - 1 )) -gt ${SEED_MAX} ]]; then
  echo "seed band exceeds the maximum seed ${SEED_MAX}" \
       "(main start_seed=${MAIN_START_SEED} / appendix start_seed=${APPENDIX_START_SEED})" >&2
  exit 1
fi

start_seed_of() {
  case "$1" in
    appendix|decay|a2|eminit) echo "${APPENDIX_START_SEED}" ;;
    *) echo "${MAIN_START_SEED}" ;;
  esac
}

n_seeds_of() {
  case "$1" in
    appendix|decay|a2|eminit) echo "${APPENDIX_N_SEEDS}" ;;
    *) echo "${MAIN_N_SEEDS}" ;;
  esac
}

seed_label_of() {
  if [[ "$1" == "plot" ]]; then
    echo ""
    return
  fi
  local start n
  start=$(start_seed_of "$1")
  n=$(n_seeds_of "$1")
  echo "start_seed=${start}, n_seeds=${n}, "
}

# one BLAS thread per joblib worker, so the process count equals n_jobs
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The run_* functions are called from an `if` context, where errexit is
# disabled, so multi-command stages must chain their commands with &&.

run_fig1() {
  local njobs="${1:-8}"
  # true exposure structure and assumed model class are paired (two multiruns)
  python src/main.py -m setting=fig1_data_size \
    setting.exposure_structure=pbm setting.exposure_model_class=pbm \
    setting.n_rounds=400,800,1600,3200,6400 \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}" &&
  python src/main.py -m setting=fig1_data_size \
    setting.exposure_structure=ranking_dependent \
    setting.exposure_model_class=ranking_dependent \
    setting.n_rounds=400,800,1600,3200,6400 \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}"
}

run_fig2() {
  local njobs="${1:-8}"
  python src/main.py -m setting=fig2_policy_divergence \
    setting.evaluation_policy.epsilon=0.0,0.2,0.5,0.8,1.0 \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}"
}

run_fig3() {
  local njobs="${1:-8}"
  # Corruption strength levels come from the environment; strength 0 is run
  # as "none" inside the (a)/(b) commands.
  if [[ -z "${FIG3_E_STRENGTHS:-}" || -z "${FIG3_R_STRENGTHS:-}" || -z "${FIG3_REL_STRENGTHS:-}" || -z "${FIG3_RXREL_PAIRS:-}" ]]; then
    echo "fig3: FIG3_E_STRENGTHS / FIG3_R_STRENGTHS / FIG3_REL_STRENGTHS / FIG3_RXREL_PAIRS are not set." >&2
    echo "  Values used for the reported experiments:" >&2
    echo "    FIG3_E_STRENGTHS='power:0.05,power:0.16,power:0.5,power:1.6,power:5'" >&2
    echo "    FIG3_R_STRENGTHS='power:0.05,power:0.16,power:0.5,power:1.6,power:5'" >&2
    echo "    FIG3_REL_STRENGTHS='power:0.5,power:1.6,power:5'" >&2
    echo "    FIG3_RXREL_PAIRS='0.15/0.5,0.22/0.75,0.33/1.1,0.5/1.55,0.75/2.1'" >&2
    return 1
  fi
  # (a) (E3) truth, exposure true, relevance corrupted
  python src/main.py -m setting=fig3_dr_grid \
    "setting.oracle_corruption.corrupt_relevance=none,${FIG3_REL_STRENGTHS}" \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}" &&
  # (b) (E2) truth, relevance true, exposure corrupted in class (method E)
  python src/main.py -m setting=fig3_dr_grid \
    setting.exposure_structure=contextual_pbm \
    setting.exposure_model_class=contextual_pbm \
    "setting.oracle_corruption.corrupt_exposure=none,${FIG3_E_STRENGTHS}" \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}" &&
  # (c) (E3) truth, exposure corrupted with method E (out of class)
  python src/main.py -m setting=fig3_dr_grid \
    "setting.oracle_corruption.corrupt_exposure=${FIG3_E_STRENGTHS}" \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}" &&
  # (d) (E3) truth, method R alone (weights corrupted, q_hat true)
  python src/main.py -m setting=fig3_dr_grid \
    "setting.oracle_corruption.corrupt_exposure_ratio=${FIG3_R_STRENGTHS}" \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}" || return 1
  # (e) (E3) truth, method R x relevance, calibrated pairs tw/tr (one run per pair)
  local pair tw tr
  for pair in ${FIG3_RXREL_PAIRS//,/ }; do
    tw="${pair%%/*}"
    tr="${pair##*/}"
    if [[ -z "${tw}" || -z "${tr}" || "${tw}" == "${pair}" ]]; then
      echo "fig3: malformed FIG3_RXREL_PAIRS entry: '${pair}' (expected e.g. '0.15/0.5')" >&2
      return 1
    fi
    python src/main.py setting=fig3_dr_grid \
      "setting.oracle_corruption.corrupt_exposure_ratio=power:${tw}" \
      "setting.oracle_corruption.corrupt_relevance=power:${tr}" \
      start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}" || return 1
  done
}

run_fig4() {
  local njobs="${1:-8}"
  python src/main.py setting=fig4_cascade \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}"
}

run_fig4eta() {
  local njobs="${1:-8}"
  # eta = 1.0 is not run here: the figure reuses the fig4_cascade cell
  python src/main.py -m setting=fig4_cascade_eta \
    setting.exposure_decay_rate=0.0,0.5,2.0,4.0 \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}"
}

run_fig5() {
  local njobs="${1:-8}"
  python src/main.py -m setting=fig5_logging_determinism \
    setting.tau0=0.05,0.2,0.5,1.0,2.0 \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}"
}

run_table1() {
  local njobs="${1:-8}"
  python src/main.py -m setting=table1_misspecification \
    setting.exposure_structure=pbm,contextual_pbm,ranking_dependent \
    setting.exposure_model_class=pbm,contextual_pbm,ranking_dependent \
    start_seed="${MAIN_START_SEED}" n_seeds="${MAIN_N_SEEDS}" n_jobs="${njobs}"
}

run_appendix() {
  local njobs="${1:-8}"
  # intentionally unquoted: expands to two words
  local seeds="start_seed=${APPENDIX_START_SEED} n_seeds=${APPENDIX_N_SEEDS}"
  python src/main.py -m setting=app_em_sensitivity setting.warm_start=false \
    setting.em_random_state=0,1,2 ${seeds} n_jobs="${njobs}" &&
  python src/main.py -m setting=app_em_sensitivity setting.warm_start=true \
    setting.em_random_state=0 ${seeds} n_jobs="${njobs}" &&
  python src/main.py -m setting=app_mc_samples setting.n_mc_samples=10,30,100,300 ${seeds} n_jobs="${njobs}" &&
  python src/main.py -m setting=app_position_weight setting.position_weight=uniform,dcg \
    setting.n_rounds=1000,4000 ${seeds} n_jobs="${njobs}"
}

run_decay() {
  local njobs="${1:-8}"
  python src/main.py -m setting=app_decay_K \
    setting.exposure_decay_rate=0.5,1.0,2.0 \
    start_seed="${APPENDIX_START_SEED}" n_seeds="${APPENDIX_N_SEEDS}" n_jobs="${njobs}"
}

run_a2() {
  local njobs="${1:-8}"
  python src/main.py -m setting=app_a2_robustness \
    setting.or_correlation=0.0,0.25,0.5,0.75,1.0 \
    start_seed="${APPENDIX_START_SEED}" n_seeds="${APPENDIX_N_SEEDS}" n_jobs="${njobs}"
}

run_a2slot() {
  local njobs="${1:-8}"
  # true structure and assumed class paired as in fig1; the (E3) command
  # sets relevance_scale=0.25 (see conf/setting/app_a2_slot_coupling.yaml)
  local seeds="start_seed=${MAIN_START_SEED} n_seeds=${MAIN_N_SEEDS}"
  python src/main.py -m setting=app_a2_slot_coupling \
    setting.exposure_structure=pbm setting.exposure_model_class=pbm \
    setting.or_coupling=0.0,0.25,0.5,0.75,1.0 ${seeds} n_jobs="${njobs}" &&
  python src/main.py -m setting=app_a2_slot_coupling \
    setting.exposure_structure=ranking_dependent \
    setting.exposure_model_class=ranking_dependent \
    setting.relevance_scale=0.25 \
    setting.or_coupling=0.0,0.25,0.5,0.75,1.0 ${seeds} n_jobs="${njobs}"
}

run_eminit() {
  local njobs="${1:-8}"
  local seeds="start_seed=${APPENDIX_START_SEED} n_seeds=${APPENDIX_N_SEEDS}"
  python src/main.py -m setting=app_em_init_const ${seeds} n_jobs="${njobs}" &&
  python src/main.py -m setting=app_em_init_antimono ${seeds} n_jobs="${njobs}"
}

run_plot() {
  python src/main.py mode=plot
}

detect_cores() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    echo 8
  fi
}

usage() {
  echo "usage: $0 <stage> [<stage> ...]"
  echo "       $0 parallel <stage> [<stage> ...]"
  echo "       $0 detach <stage> [<stage> ...]      (detach parallel ... also works)"
  echo "stages: ${STAGES[*]}"
  echo "        list     - print this help"
  echo "        all      - run every stage sequentially, then plot (plot is skipped"
  echo "                   when a stage failed)"
  echo "        decay    - decay-rate sweep (setting app_decay_K) only, appendix seed band;"
  echo "                   not part of appendix"
  echo "        a2       - (A2) robustness sweep (setting app_a2_robustness) only, appendix"
  echo "                   seed band; not part of appendix"
  echo "        parallel - run the given stages side by side, cores split by weight"
  echo "                   ('parallel all' runs every stage except plot)"
  echo "        detach   - run the remaining arguments under nohup (+ setsid where"
  echo "                   available), detached from the terminal; output goes to"
  echo "                   logs_console/detach_<stage>_<timestamp>.log, the PID to the"
  echo "                   .pid file of the same name"
  echo "environment variables:"
  echo "  CYCLE=C      - seed band of cycle C (default 1): main-text stages start at"
  echo "                 100*(C-1), appendix stages at 50*(C-1). Keeping earlier"
  echo "                 cycles under logs/ pools their replications in mode=plot"
  echo "  START_SEED=S - start_seed=S for every stage (takes precedence over CYCLE)"
  echo "  N_SEEDS=N    - n_seeds=N for every stage (requires START_SEED), e.g."
  echo "                 START_SEED=200 N_SEEDS=300 ./run_experiments.sh fig1"
  echo "  N_JOBS=J     - joblib processes in sequential mode (default 8); parallel"
  echo "                 mode ignores it"
  echo "  DRY_RUN=1    - print the python commands instead of running them"
  echo "  FIG3_E_STRENGTHS / FIG3_R_STRENGTHS / FIG3_REL_STRENGTHS"
  echo "               - corruption strength levels of fig3 (e.g. 'power:0.05,power:0.16');"
  echo "                 fig3 exits with an error when they are unset"
  echo "  FIG3_RXREL_PAIRS"
  echo "               - calibrated tw/tr pairs of fig3 condition (e) (e.g. '0.15/0.5,0.22/0.75')"
  echo "current seed bands: main start_seed=${MAIN_START_SEED} (n_seeds=${MAIN_N_SEEDS}) /" \
       "appendix start_seed=${APPENDIX_START_SEED} (n_seeds=${APPENDIX_N_SEEDS}) /" \
       "sequential n_jobs=${N_JOBS}"
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "list" ]]; then
  usage
  exit 0
fi

# ---- detach mode ----------------------------------------------------------
# Re-launches this script with the remaining arguments under nohup (and setsid
# where available) and returns immediately. Environment variables are inherited.
if [[ "$1" == "detach" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    echo "detach needs at least one stage (or parallel <stage>...)" >&2
    exit 1
  fi
  for arg in "$@"; do
    if [[ "${arg}" == "detach" || "${arg}" == "list" ]]; then
      echo "'${arg}' cannot follow detach" >&2
      exit 1
    fi
    if [[ "${arg}" != "parallel" && "${arg}" != "all" && ! " ${STAGES[*]} " =~ " ${arg} " ]]; then
      echo "unknown stage: ${arg}" >&2
      usage
      exit 1
    fi
  done
  mkdir -p logs_console
  stamp=$(date '+%Y%m%d_%H%M%S')
  tag=$(printf '%s_' "$@"); tag="${tag%_}"
  logfile="logs_console/detach_${tag}_${stamp}.log"
  pidfile="logs_console/detach_${tag}_${stamp}.pid"
  launcher=(nohup)
  if command -v setsid >/dev/null 2>&1; then
    launcher+=(setsid)
  fi
  # the detached side writes its own PID and appends its exit status to the log
  "${launcher[@]}" bash -c \
    'echo $$ >"$1"; shift; bash "$0" "$@"; rc=$?; echo "=== exit status: ${rc} ($(date "+%Y-%m-%d %H:%M:%S")) ==="; exit ${rc}' \
    "$0" "${pidfile}" "$@" </dev/null >"${logfile}" 2>&1 &
  disown 2>/dev/null || true
  child=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -s "${pidfile}" ]]; then child=$(cat "${pidfile}"); break; fi
    sleep 0.5
  done
  if [[ -z "${child}" ]]; then
    echo "detach: no PID file was written; see the log: ${logfile}" >&2
    tail -n 20 "${logfile}" >&2 || true
    exit 1
  fi
  sleep 1
  if ! kill -0 "${child}" 2>/dev/null; then
    if grep -q '^=== exit status: 0 ' "${logfile}" 2>/dev/null; then
      echo "=== detached job already finished (exit 0; DRY_RUN?) — log: ${logfile} ==="
      exit 0
    fi
    echo "detach: the job exited right after launch; see the log: ${logfile}" >&2
    tail -n 20 "${logfile}" >&2 || true
    exit 1
  fi
  echo "=== detached: $* (pid ${child}, $(date '+%Y-%m-%d %H:%M:%S')) ==="
  echo "  seed bands: main start_seed=${MAIN_START_SEED} n_seeds=${MAIN_N_SEEDS} /" \
       "appendix start_seed=${APPENDIX_START_SEED} n_seeds=${APPENDIX_N_SEEDS} / sequential n_jobs=${N_JOBS}"
  echo "  log       : ${logfile}"
  echo "  progress  : tail -f ${logfile}        (stage start/end lines: grep '^===' ${logfile})"
  echo "  alive?    : kill -0 \$(cat ${pidfile}) && echo running || echo finished"
  echo "  finished? : grep '^=== exit status' ${logfile}   (0 = success)"
  if [[ ${#launcher[@]} -eq 2 ]]; then
    echo "  stop      : kill -- -\$(cat ${pidfile})   (whole setsid process group, incl. python / joblib workers)"
  else
    echo "  stop      : pkill -TERM -P \$(cat ${pidfile}); kill \$(cat ${pidfile})   (no setsid: children first)"
  fi
  echo "  The job keeps running after the terminal or SSH session is closed."
  exit 0
fi

# ---- parallel mode --------------------------------------------------------
if [[ "$1" == "parallel" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    echo "parallel needs at least one stage" >&2
    exit 1
  fi
  if [[ "$1" == "all" ]]; then
    set -- "${ALL_STAGES[@]:0:$((${#ALL_STAGES[@]} - 1))}"  # everything except plot
  fi

  for stage in "$@"; do
    if [[ ! " ${STAGES[*]} " =~ " ${stage} " ]] || [[ "${stage}" == "plot" ]]; then
      echo "stage cannot run in parallel mode: ${stage} (run plot on its own)" >&2
      exit 1
    fi
  done

  cores=$(detect_cores)
  total_weight=0
  for stage in "$@"; do
    total_weight=$((total_weight + $(weight_of "${stage}")))
  done

  echo "detected cores: ${cores} / stages: $* (total weight: ${total_weight})"

  PIDS=()
  STAGE_OF_PID=()
  for stage in "$@"; do
    njobs=$(( cores * $(weight_of "${stage}") / total_weight ))
    if [[ ${njobs} -lt 1 ]]; then njobs=1; fi
    mkdir -p logs_console
    logfile="logs_console/parallel_${stage}.log"
    echo "=== launching: ${stage} ($(seed_label_of "${stage}")n_jobs=${njobs}, log: ${logfile}, $(date '+%Y-%m-%d %H:%M:%S')) ==="
    ( "run_${stage}" "${njobs}" ) >"${logfile}" 2>&1 &
    PIDS+=("$!")
    STAGE_OF_PID+=("${stage}")
  done

  FAILED=()
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "=== done: ${STAGE_OF_PID[$i]} ($(date '+%Y-%m-%d %H:%M:%S')) ==="
    else
      echo "=== FAILED: ${STAGE_OF_PID[$i]} (see logs_console/parallel_${STAGE_OF_PID[$i]}.log) ==="
      FAILED+=("${STAGE_OF_PID[$i]}")
    fi
  done

  if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "some stages failed: ${FAILED[*]}" >&2
    exit 1
  fi
  exit 0
fi

# ---- sequential mode ------------------------------------------------------
if [[ "$1" == "all" ]]; then
  set -- "${ALL_STAGES[@]}"
fi

FAILED=()

for stage in "$@"; do
  if [[ ! " ${STAGES[*]} " =~ " ${stage} " ]]; then
    echo "unknown stage: ${stage}" >&2
    usage
    exit 1
  fi
  # figures from partial logs are indistinguishable from complete ones
  if [[ "${stage}" == "plot" && ${#FAILED[@]} -gt 0 ]]; then
    echo "=== skipping plot: failed stages exist (${FAILED[*]}) ===" >&2
    continue
  fi
  echo "=== running: ${stage} ($(seed_label_of "${stage}")n_jobs=${N_JOBS}, $(date '+%Y-%m-%d %H:%M:%S')) ==="
  if "run_${stage}" "${N_JOBS}"; then
    echo "=== done: ${stage} ($(date '+%Y-%m-%d %H:%M:%S')) ==="
  else
    echo "=== FAILED: ${stage} ($(date '+%Y-%m-%d %H:%M:%S')) — continuing to next stage ==="
    FAILED+=("${stage}")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "some stages failed: ${FAILED[*]}" >&2
  exit 1
fi

# stage -> setting_name for stages that run a single setting (used by the
# merge hint below; scripts/merge_archive_aggregate.py --setting)
setting_of_stage() {
  case "$1" in
    fig1) echo fig1_data_size ;;
    fig2) echo fig2_policy_divergence ;;
    fig4) echo fig4_cascade ;;
    fig5) echo fig5_logging_determinism ;;
    table1) echo table1_misspecification ;;
    decay) echo app_decay_K ;;
    a2) echo app_a2_robustness ;;
    *) echo "" ;;
  esac
}

# after an explicit additional seed band, print how to pool it with an
# archived aggregate (one merge run for all settings of this invocation)
if [[ -n "${START_SEED:-}" ]]; then
  merge_settings=""
  for stage in "$@"; do
    s=$(setting_of_stage "${stage}")
    if [[ -n "${s}" ]]; then
      merge_settings="${merge_settings:+${merge_settings},}${s}"
    fi
  done
  if [[ -n "${merge_settings}" ]]; then
    echo "=== additional seed band finished. To pool it with an archived aggregate and re-render the figures: ==="
    echo "  .venv/bin/python scripts/merge_archive_aggregate.py --setting ${merge_settings} --seed-from ${START_SEED}"
    echo "  (settings launched with different START_SEEDs can be merged in one run by listing them in"
    echo "   --setting and giving --seed-from per setting, comma-separated, in the same order)"
  fi
fi
echo "=== all requested stages finished ($(date '+%Y-%m-%d %H:%M:%S')) ==="
