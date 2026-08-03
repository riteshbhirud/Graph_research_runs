#!/usr/bin/env bash
# Serve the three checkpoints and launch the three Table 1 runs.
#
# Auto-detects GPU VRAM and picks the layout:
#   >= 70 GB/GPU (H200, H100 NVL, B200, B300, A100-80) -> 1 GPU per checkpoint
#   <  70 GB/GPU (A100-40, L40S ...)                   -> tensor-parallel 2
# A 30B-A3B checkpoint is ~57 GB in bf16, so smaller cards cannot hold one alone.
#
#   ./run_all.sh            launch everything (resumes automatically)
#   ./run_all.sh --status   show progress and exit
#   ./run_all.sh --judge    judge completed runs and build Table 1
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f env.sh ]] && source env.sh
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
DATA=$ROOT/data/browsecomp_plus/browsecomp_plus_830.jsonl
mkdir -p logs trajectories checkpoints results

# checkpoint | model dir | output dir | thinking | discard threshold
RUNS=(
  "base|models/Qwen3-30B-A3B|quest30b_base_bm25_top5_uncapped|false|30000"
  "sft|models/QUEST-30B-MT-Plus-SFT|quest30b_sft_bm25_top5_uncapped|true|80000"
  "rl|models/QUEST-30B-RL|quest30b_rl_bm25_top5_uncapped|true|80000"
)

count() { local f; f=$(ls "$1"/deepresearch/*/iter1.jsonl 2>/dev/null | head -1); [[ -n "$f" ]] && wc -l < "$f" || echo 0; }

if [[ "${1:-}" == "--status" ]]; then
    printf '%-42s %s\n' "RUN" "COMPLETED"
    for r in "${RUNS[@]}"; do IFS='|' read -r ck md od th mt <<< "$r"
        printf '%-42s %s/830\n' "$od" "$(count "trajectories/$od")"; done
    echo; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
    exit 0
fi

if [[ "${1:-}" == "--judge" ]]; then
    GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)
    echo "==> serving judge on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup ./venv/bin/vllm serve models/Qwen3-32B \
        --served-model-name judge --port 6100 --max-model-len 16384 \
        --gpu-memory-utilization 0.90 > logs/vllm_judge.log 2>&1 < /dev/null &
    until curl -s -m 3 --noproxy '*' http://localhost:6100/v1/models 2>/dev/null | grep -q judge; do sleep 15; done
    for r in "${RUNS[@]}"; do IFS='|' read -r ck md od th mt <<< "$r"
        [[ $(count "trajectories/$od") -gt 0 ]] || continue
        echo "==> analyzing + judging $od"
        ./venv/bin/python analyze_run.py --run_dir "trajectories/$od" --system QUEST-30B \
            --checkpoint "$ck" --out_dir "results/analyzed_$od" >/dev/null
        ./venv/bin/python judge_run.py --dir "results/analyzed_$od" --base_url http://localhost:6100/v1
    done
    ./venv/bin/python scripts/build_table1.py
    exit 0
fi

# ------------------------------------------------------------ GPU layout
mapfile -t VRAM < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader | awk '{print int($1/1024)}')
NGPU=${#VRAM[@]}
PER=${VRAM[0]}
if (( PER >= 70 )); then TP=1; else TP=2; fi
NEED=$(( 3 * TP ))
echo "==> ${NGPU} GPU(s) @ ${PER} GB  ->  tensor-parallel=${TP}, need ${NEED} GPU(s) for 3 concurrent runs"
if (( NGPU < NEED )); then
    echo "    only ${NGPU} available: runs will be launched on what fits; re-run this script"
    echo "    after one finishes to start the rest (all runs resume automatically)."
fi

start_server() {  # gpus port model maxlen
    local gpus=$1 port=$2 model=$3 maxlen=$4
    curl -s -m 3 --noproxy '*' "http://localhost:$port/v1/models" 2>/dev/null | grep -q deepresearch && { echo "    :$port already up"; return 0; }
    echo "    serving $model on GPU(s) $gpus -> :$port (TP=$TP, max_len=$maxlen)"
    CUDA_VISIBLE_DEVICES=$gpus VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 \
        setsid nohup ./venv/bin/vllm serve "$model" --served-model-name deepresearch \
        --port "$port" --tensor-parallel-size "$TP" --max-model-len "$maxlen" \
        --gpu-memory-utilization 0.92 > "logs/vllm_${port}.log" 2>&1 < /dev/null &
    until curl -s -m 3 --noproxy '*' "http://localhost:$port/v1/models" 2>/dev/null | grep -q deepresearch; do
        grep -q "Engine core initialization failed" "logs/vllm_${port}.log" 2>/dev/null && { echo "    FAILED - see logs/vllm_${port}.log"; return 1; }
        sleep 15
    done
    grep -o "GPU KV cache size: [0-9,]*" "logs/vllm_${port}.log" | tail -1 | sed 's/^/    /'
}

g=0; port=6000
for r in "${RUNS[@]}"; do
    IFS='|' read -r CK MODEL OUT THINK THRESH <<< "$r"
    [[ $(count "trajectories/$OUT") -ge 830 ]] && { echo "==> $OUT already complete"; port=$((port+2)); continue; }
    (( g + TP > NGPU )) && { echo "==> no GPUs left for $OUT - re-run after one finishes"; break; }

    gpus=$(seq -s, $g $((g+TP-1)))
    # Base (Qwen3-30B-A3B) has a 40,960 native window; SFT/RL are 262,144 natively.
    if [[ "$CK" == "base" ]]; then MAXLEN=40960; else MAXLEN=98304; fi
    echo "==> $CK"
    start_server "$gpus" "$port" "$MODEL" "$MAXLEN" || { g=$((g+TP)); port=$((port+2)); continue; }

    CONF=$ROOT/systems/QUEST/inference/endpoints_${CK}.conf
    printf 'HOSTNAME_LIST=localhost\nPORTS=%s\n' "$port" > "$CONF"

    # workers scale with KV headroom: KV is the binding constraint
    if   (( PER >= 170 )); then W=48
    elif (( PER >= 130 )); then W=32
    elif (( PER >= 70  )); then W=16
    else                        W=12; fi

    setsid nohup env DATASET="$DATA" OUTPUT_PATH="$ROOT/trajectories/$OUT" \
        BCP_CHECKPOINT="$CK" BM25_TOP_K=5 BCP_VISIT_TOTAL_TOKEN_CAP=0 MAX_WORKERS=$W \
        MODEL_PATH="$ROOT/$MODEL" QUEST_ENABLE_THINKING="$THINK" \
        MEMORY_CONTEXT_THRESHOLD="$THRESH" SERVER_ENDPOINTS_FILE="$CONF" \
        ./run_quest_bm25.sh >> "logs/run_${OUT}.log" 2>&1 < /dev/null &
    echo "    launched ($W workers, resumes from $(count "trajectories/$OUT")/830)"
    g=$((g+TP)); port=$((port+2))
done

pgrep -f "disk_watc[h]" >/dev/null || { setsid nohup ./disk_watch.sh >/dev/null 2>&1 < /dev/null & echo "==> disk watchdog started"; }
echo; echo "==> ./run_all.sh --status   to check progress"
echo "==> ./run_all.sh --judge    once runs finish"
