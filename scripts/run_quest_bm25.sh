#!/bin/bash
# QUEST-30B on BrowseComp-Plus with offline BM25 retrieval.
#
# Applied decisions:
#   Q1  context management = discard_all (no external API; QUEST's own finding)
#   Q2  visit tool ENABLED on bm25://<docid>; disallowed tools -> clean error + log
#   Q3  max turns = 100
#   Q4  BM25 top-k = 15, snippet max tokens = 512 (repository defaults)
#   Q5  full 830-question set
#   R2  YaRN rope-scaling to 131072; MEMORY_CONTEXT_THRESHOLD = 80000
set -euo pipefail

# Derived from this script's own location so the harness is portable.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
INFERENCE_DIR="${REPO_ROOT}/systems/QUEST/inference"

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"

# This host routes all HTTP through a proxy. The agent talks to the local vLLM
# server, so localhost must bypass it or every model call 503s.
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# --- Offline BM25 retrieval (Decision Q4) ---
export BCP_OFFLINE=true
export BM25_INDEX_PATH="${REPO_ROOT}/data/browsecomp_plus/indexes/bm25"
export BM25_TOP_K="${BM25_TOP_K:-15}"
export BM25_SNIPPET_MAX_TOKENS="${BM25_SNIPPET_MAX_TOKENS:-512}"
export FAISS_INDEX_PATH=""          # must stay empty so the BM25 path is selected
export BCP_VISIT_MAX_TOKENS="${BCP_VISIT_MAX_TOKENS:-8192}"

# --- No live web anywhere (Decision Q2: visit stays ON, but offline) ---
export SERPER_KEY_ID=""
export JINA_API_KEYS=""
export DISABLE_VISIT_TOOL=false
export ENABLE_PYTHON_TOOL=false
export ENABLE_SCHOLAR_TOOL=false

# --- Model endpoint ---
export SERVER_ENDPOINTS_FILE="${SERVER_ENDPOINTS_FILE:-${INFERENCE_DIR}/server_endpoints.conf}"
export HOSTNAME_LIST="${HOSTNAME_LIST:-localhost}"
export PORTS="${PORTS:-6000}"
export MODEL_NAME="${MODEL_NAME:-deepresearch}"
# Which checkpoint this run is (stamped into every event-log record).
export BCP_CHECKPOINT="${BCP_CHECKPOINT:-base}"
# Base Qwen3 only: run non-thinking. SFT/RL use their shipped template
# (which forces thinking) and must leave this unset.
export QUEST_ENABLE_THINKING="${QUEST_ENABLE_THINKING:-false}"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/Qwen3-30B-A3B-Base}"
export MEMORY_TOKENIZER_PATH="${MEMORY_TOKENIZER_PATH:-${MODEL_PATH}}"

# --- Dataset / output ---
export DATASET="${DATASET:-${REPO_ROOT}/data/browsecomp_plus/browsecomp_plus_830.jsonl}"
export OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/trajectories/quest30b_base_bm25}"
export TASK_LOG_DIR="${TASK_LOG_DIR:-${OUTPUT_PATH}/logs}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PATH}/cache}"
export BCP_EVENT_LOG="${BCP_EVENT_LOG:-${OUTPUT_PATH}/events.jsonl}"
export BCP_MULTI_BLOCK_LOG="${BCP_MULTI_BLOCK_LOG:-${OUTPUT_PATH}/multi_block_events.jsonl}"
# Condition A (uncapped) = 0. Condition B (capped) = 24576 total tokens per visit call.
export BCP_VISIT_TOTAL_TOKEN_CAP="${BCP_VISIT_TOTAL_TOKEN_CAP:-0}"
export BCP_VISIT_CAP_LOG="${BCP_VISIT_CAP_LOG:-${OUTPUT_PATH}/visit_cap_events_${BCP_CHECKPOINT}.jsonl}"

# --- Inference hyperparameters (Decision Q3, mentor settings) ---
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-1}"
export TEMPERATURE="${TEMPERATURE:-1}"        # QUEST's own BrowseComp-Plus default
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.1}"
export MAX_TURN="${MAX_TURN:-60}"
export MAX_LLM_CALL_PER_RUN="${MAX_LLM_CALL_PER_RUN:-${MAX_TURN}}"
export MAX_WORKERS="${MAX_WORKERS:-8}"
export MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-1440}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-8192}"

# --- Context management (Decision Q1 + Ruling 2) ---
export MEMORY_ENABLED=true
export MEMORY_STRATEGY=discard_all
# No rope-scaling anywhere. Base is served at its native 40,960 window, so its
# discard-all threshold must sit well under that; SFT/RL are served at 131,072
# (capped from native 262,144) and use QUEST's intended 80,000.
if [ "${BCP_CHECKPOINT}" = "base" ]; then
    # Qwen3-30B-A3B-Base has a 32,768 native window (smaller than the instruct
    # release's 40,960), so discard-all must fire well below it.
    export MEMORY_CONTEXT_THRESHOLD="${MEMORY_CONTEXT_THRESHOLD:-25000}"
else
    export MEMORY_CONTEXT_THRESHOLD="${MEMORY_CONTEXT_THRESHOLD:-80000}"
fi
export MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-${MEMORY_CONTEXT_THRESHOLD}}"

# --- Caching off: every question must retrieve independently ---
export SEARCH_CACHE_ENABLED=false
export VISIT_CACHE_ENABLED=false

# --- Resume / distribution ---
export RESUME_FROM_MESSAGES="${RESUME_FROM_MESSAGES:-false}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"

mkdir -p "${OUTPUT_PATH}" "${TASK_LOG_DIR}" "${CACHE_DIR}"

cat <<EOF
==========================================
 QUEST-30B | BrowseComp-Plus | BM25 offline
 model      : ${MODEL_PATH}
 dataset    : ${DATASET}
 output     : ${OUTPUT_PATH}
 bm25 top-k : ${BM25_TOP_K} (snippet ${BM25_SNIPPET_MAX_TOKENS} tok)
 max turns  : ${MAX_TURN}   max out tok: ${LLM_MAX_TOKENS}
 context mgmt: ${MEMORY_STRATEGY} @ ${MEMORY_CONTEXT_THRESHOLD}
 visit tool : ENABLED (bm25://docid)
 workers    : ${MAX_WORKERS}   rollouts: ${ROLLOUT_COUNT}
==========================================
EOF

cd "${INFERENCE_DIR}"
exec "${REPO_ROOT}/venv/bin/python" -u run_multi_react.py \
    --dataset "${DATASET}" \
    --output "${OUTPUT_PATH}" \
    --max_workers "${MAX_WORKERS}" \
    --model "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --temperature "${TEMPERATURE}" \
    --presence_penalty "${PRESENCE_PENALTY}" \
    --roll_out_count "${ROLLOUT_COUNT}" \
    --total_splits "${WORLD_SIZE}" \
    --worker_split "$((RANK + 1))"
