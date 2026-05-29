#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-}"
shift || true

slugify() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g'
}

ensure_date_prefixed_run_name() {
    local raw_name="$1"
    if [[ "${raw_name}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}(-|$) ]]; then
        echo "${raw_name}"
    else
        echo "${RUN_DATE}-${raw_name}"
    fi
}

show_help() {
    cat <<'EOF'
Usage:
  bash script/run_train_tmux.sh <stage1|stage2> [HYDRA_OVERRIDES...]

Environment variables:
  CONFIG_NAME
    Hydra config name. Defaults to the stage argument.
    Examples: stage1, stage2, stage1_dinov3_large, stage2_dinov3_large,
              stage1_dinov3, stage2_dinov3.
  SESSION_NAME
    tmux session name. Defaults to csvit2-<stage>.
  GPU_IDS
    Comma-separated GPU ids passed to accelerate --gpu_ids. Default: 0,1,2,3.
  NUM_PROCESSES
    Number of accelerate worker processes. Default: 4.
  MAIN_PROCESS_PORT
    Distributed port passed to accelerate. Default: 0.
  STAGE1_WEIGHT
    Required when stage is stage2. Directory containing model.safetensors or a
    direct .safetensors path.
  RUN_NAME
    Optional run name. A YYYY-MM-DD prefix is added automatically unless present.
  DRY_RUN
    Set to 1 to print the resolved tmux command without launching training.

Examples:
  bash script/run_train_tmux.sh stage1
  CONFIG_NAME=stage1_dinov3_large bash script/run_train_tmux.sh stage1 LOSS.heatmap_sigma=4.0
  CONFIG_NAME=stage2_dinov3_large STAGE1_WEIGHT=/path/to/best_model bash script/run_train_tmux.sh stage2
EOF
}

if [[ "${STAGE}" == "-h" || "${STAGE}" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ "${STAGE}" != "stage1" && "${STAGE}" != "stage2" ]]; then
    echo "Usage: bash script/run_train_tmux.sh <stage1|stage2> [HYDRA_OVERRIDES...]" >&2
    echo "Run 'bash script/run_train_tmux.sh --help' for details." >&2
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but was not found in PATH." >&2
    exit 1
fi

if [[ ! -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
    echo "Missing ${ROOT_DIR}/.venv/bin/activate. Run 'uv venv .venv --python=3.12 && uv sync' first." >&2
    exit 1
fi

SESSION_NAME="${SESSION_NAME:-csvit2-${STAGE}}"
CONFIG_NAME="${CONFIG_NAME:-${STAGE}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
STAGE1_WEIGHT="${STAGE1_WEIGHT:-}"
RUN_NAME="${RUN_NAME:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_DATE="$(date +%F)"

if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(ensure_date_prefixed_run_name "$(date +%H-%M-%S)-$(slugify "${SESSION_NAME}")")"
else
    RUN_NAME="$(ensure_date_prefixed_run_name "$(slugify "${RUN_NAME}")")"
fi

RUN_DIR="${ROOT_DIR}/checkpoint/${RUN_DATE}/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/tmux.log"

if [[ ! -f "${ROOT_DIR}/config/${CONFIG_NAME}.yaml" ]]; then
    echo "Missing Hydra config: ${ROOT_DIR}/config/${CONFIG_NAME}.yaml" >&2
    exit 1
fi

if [[ "${STAGE}" == "stage2" && -z "${STAGE1_WEIGHT}" ]]; then
    echo "STAGE1_WEIGHT is required for stage2. Example:" >&2
    echo "  make train-stage2 STAGE1_WEIGHT=/path/to/stage1/best_model" >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session '${SESSION_NAME}' already exists. Attach or stop it first." >&2
    exit 1
fi

train_cmd=(
    accelerate
    launch
    --main_process_port "${MAIN_PROCESS_PORT}"
    --gpu_ids "${GPU_IDS}"
    --num_processes "${NUM_PROCESSES}"
    -m script.train
    --config-name="${CONFIG_NAME}"
)

if [[ "${STAGE}" == "stage2" ]]; then
    train_cmd+=("MODEL.stage1_weight=${STAGE1_WEIGHT}")
fi

if [[ "$#" -gt 0 ]]; then
    train_cmd+=("$@")
fi

printf -v quoted_train_cmd "%q " "${train_cmd[@]}"
printf -v quoted_root_dir "%q" "${ROOT_DIR}"
printf -v quoted_run_dir "%q" "${RUN_DIR}"
printf -v quoted_run_name "%q" "${RUN_NAME}"
printf -v quoted_log_file "%q" "${LOG_FILE}"
tmux_cmd="set -euo pipefail && cd ${quoted_root_dir} && source .venv/bin/activate && CSVIT2_RUN_DIR=${quoted_run_dir} CSVIT2_RUN_NAME=${quoted_run_name} ${quoted_train_cmd} 2>&1 | tee -a ${quoted_log_file}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "session: ${SESSION_NAME}"
    echo "config_name: ${CONFIG_NAME}"
    echo "run_name: ${RUN_NAME}"
    echo "run_dir: ${RUN_DIR}"
    echo "log: ${LOG_FILE}"
    echo "command: ${tmux_cmd}"
    exit 0
fi

mkdir -p "${RUN_DIR}"
: > "${LOG_FILE}"

printf -v quoted_tmux_cmd "%q" "${tmux_cmd}"
tmux new-session -d -s "${SESSION_NAME}" "bash -lc ${quoted_tmux_cmd}"
tmux set-environment -t "${SESSION_NAME}" CSVIT2_RUN_DIR "${RUN_DIR}"
tmux set-environment -t "${SESSION_NAME}" CSVIT2_RUN_NAME "${RUN_NAME}"
tmux set-environment -t "${SESSION_NAME}" CSVIT2_LOG_FILE "${LOG_FILE}"

echo "Started tmux session '${SESSION_NAME}'."
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Run name: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Log: ${LOG_FILE}"
