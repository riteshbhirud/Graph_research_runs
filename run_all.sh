#!/usr/bin/env bash
# Run the three Table 1 rows to completion, supervising until done.
#
#   ./run_all.sh            start + supervise (safe to re-run at any time)
#   ./run_all.sh --status   progress, then exit
#   ./run_all.sh --stop     stop all runs and servers cleanly
#   ./run_all.sh --judge    judge finished runs -> results/table1.md
#
# Designed for interruption. Re-running after a crash, preemption, reboot or
# SSH drop is always the correct recovery action: it detects what is complete,
# what is stalled, and what never started, then repairs only what is broken.
#
# GPU allocation: one checkpoint per GPU when VRAM >= 70 GB (a 30B-A3B
# checkpoint is ~57 GB in bf16), otherwise tensor-parallel 2. Long runs (SFT,
# RL) are started first so that on a 2-GPU box they occupy both slots and the
# much cheaper Base run takes whichever frees first.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f env.sh ]] && source env.sh
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
DATA=$ROOT/data/browsecomp_plus/browsecomp_plus_830.jsonl
TOTAL=830
SUPLOG=$ROOT/logs/supervisor.log
mkdir -p logs trajectories checkpoints results

# order matters: SFT and RL are ~6x more expensive per question than Base
#   checkpoint | model dir | output dir | thinking | discard threshold
RUNS=(
  "sft|models/QUEST-30B-MT-Plus-SFT|quest30b_sft_bm25_top5_uncapped|true|80000"
  "rl|models/QUEST-30B-RL|quest30b_rl_bm25_top5_uncapped|true|80000"
  "base|models/Qwen3-30B-A3B|quest30b_base_bm25_top5_uncapped|false|30000"
)

log() { local m; m="$(date '+%F %T') $*"; echo "$m";
        [[ "${IN_SUPERVISOR:-0}" == 1 ]] || echo "$m" >> "$SUPLOG"; }
count()        { local f; f=$(ls "trajectories/$1"/deepresearch/*/iter1.jsonl 2>/dev/null | head -1); [[ -n "$f" ]] && wc -l < "$f" || echo 0; }
agent_alive()  { pgrep -f "run_multi_react.py.*$ROOT/trajectories/$1" >/dev/null; }
server_alive() { curl -s -m 4 --noproxy '*' "http://localhost:$1/v1/models" 2>/dev/null | grep -q deepresearch; }
port_for()     { case "$1" in sft) echo 6000;; rl) echo 6002;; base) echo 6004;; esac; }

# ----------------------------------------------------------------- status
if [[ "${1:-}" == "--status" ]]; then
    printf '%-40s %-10s %-9s %s\n' RUN DONE AGENT SERVER
    for r in "${RUNS[@]}"; do IFS='|' read -r ck md od th mt <<< "$r"; p=$(port_for "$ck")
        a=$(agent_alive "$od" && echo running || echo "-")
        s=$(server_alive "$p" && echo "up:$p" || echo "-")
        printf '%-40s %-10s %-9s %s\n' "$od" "$(count "$od")/$TOTAL" "$a" "$s"
    done
    echo; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo; tail -5 "$SUPLOG" 2>/dev/null
    exit 0
fi

# ------------------------------------------------------------------- stop
if [[ "${1:-}" == "--stop" ]]; then
    log "stopping supervisor, agents and servers"
    [[ -f logs/supervisor.pid ]] && kill "$(cat logs/supervisor.pid)" 2>/dev/null
    rm -f logs/supervisor.pid
    for r in "${RUNS[@]}"; do IFS='|' read -r ck md od th mt <<< "$r"
        for p in $(pgrep -f "run_multi_react.py.*$ROOT/trajectories/$od"); do kill -9 "$p" 2>/dev/null; done; done
    for p in $(pgrep -f "vllm.*serve.*served-model-name deepresearch"); do kill "$p" 2>/dev/null; done
    sleep 5; log "stopped"
    exit 0
fi

# ----------------------------------------------------------------- bundle
# Package everything needed for the paper into one archive. Deliberately
# excludes trajectories/ (raw QUEST logs, ~140 GB): results/analyzed_*/ already
# contains every trajectory in the agreed schema - turns, reasoning, tool calls,
# observations and all metrics - at ~3 MB/question instead of ~57 MB.
if [[ "${1:-}" == "--bundle" ]]; then
    STAMP=$(date +%Y%m%d)
    OUTDIR="results_bundle_${STAMP}"
    rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
    cp -r results/analyzed_* "$OUTDIR"/ 2>/dev/null
    cp results/table1.md "$OUTDIR"/ 2>/dev/null
    cp results/oracle_bm25_ceiling.json results/oracle_k_sweep.json "$OUTDIR"/ 2>/dev/null
    mkdir -p "$OUTDIR/event_logs"
    for r in "${RUNS[@]}"; do IFS='|' read -r ck md od th mt <<< "$r"
        for f in events.jsonl multi_block_events.jsonl visit_cap_events_*.jsonl; do
            [[ -f "trajectories/$od/$f" ]] && cp "trajectories/$od/$f" "$OUTDIR/event_logs/${od}__${f}" 2>/dev/null
        done; done
    cp -r docs "$OUTDIR"/ 2>/dev/null
    {
        echo "# Results bundle — QUEST-30B on BrowseComp-Plus (BM25 top-5, 830q)"
        echo
        echo "Generated: $(date -u '+%F %T UTC')"
        echo "Host: $(hostname)   GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr '\n' ' ')"
        echo "Harness commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
        echo
        echo "## Completion"
        for r in "${RUNS[@]}"; do IFS='|' read -r ck md od th mt <<< "$r"
            echo "- $od: $(count "$od")/$TOTAL"; done
        echo
        echo "## Contents"
        echo "- analyzed_<run>/<qid>.json  full trajectory per question, agreed schema"
        echo "- analyzed_<run>/_summary.json  SR, Turns, all behavioural metrics"
        echo "- table1.md                  Table 1 + behavioural progression"
        echo "- event_logs/                per-tool-call event streams"
        echo "- oracle_*.json              BM25 single-query ceiling"
        echo "- docs/                      what may/may not be claimed vs published numbers"
        echo
        echo "NOT included: trajectories/ (raw QUEST logs, ~140 GB). The analyzed"
        echo "JSONs contain the same trajectories in structured form. Ask for the raw"
        echo "logs only if byte-level reproduction is needed."
    } > "$OUTDIR/MANIFEST.md"
    tar czf "${OUTDIR}.tar.gz" "$OUTDIR"
    echo "bundle: ${OUTDIR}.tar.gz  ($(du -sh "${OUTDIR}.tar.gz" | cut -f1))"
    echo "share it, or: rsync -avP ${OUTDIR}.tar.gz <dest>"
    exit 0
fi

# ------------------------------------------------------------------ judge
if [[ "${1:-}" == "--judge" ]]; then
    GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)
    VR=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | awk '{print int($1/1024)}')
    JTP=1; (( VR < 70 )) && JTP=2
    if ! curl -s -m 4 --noproxy '*' http://localhost:6100/v1/models 2>/dev/null | grep -q judge; then
        log "serving judge (Qwen3-32B, TP=$JTP)"
        CUDA_VISIBLE_DEVICES=$( [[ $JTP == 2 ]] && echo 0,1 || echo "$GPU" ) \
        setsid nohup ./venv/bin/vllm serve models/Qwen3-32B --served-model-name judge \
            --port 6100 --tensor-parallel-size $JTP --max-model-len 16384 \
            --gpu-memory-utilization 0.90 > logs/vllm_judge.log 2>&1 < /dev/null &
        until curl -s -m 4 --noproxy '*' http://localhost:6100/v1/models 2>/dev/null | grep -q judge; do sleep 15; done
    fi
    for r in "${RUNS[@]}"; do IFS='|' read -r ck md od th mt <<< "$r"
        n=$(count "$od"); [[ $n -gt 0 ]] || continue
        log "analyzing + judging $od ($n questions)"
        ./venv/bin/python analyze_run.py --run_dir "trajectories/$od" --system QUEST-30B \
            --checkpoint "$ck" --out_dir "results/analyzed_$od" >/dev/null
        ./venv/bin/python judge_run.py --dir "results/analyzed_$od" --base_url http://localhost:6100/v1
    done
    ./venv/bin/python scripts/build_table1.py
    exit 0
fi

# ------------------------------------------------------- hardware layout
mapfile -t VRAM < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader | awk '{print int($1/1024)}')
NGPU=${#VRAM[@]}; PER=${VRAM[0]}
if (( PER >= 70 )); then TP=1; else TP=2; fi
SLOTS=$(( NGPU / TP ))
if   (( PER >= 170 )); then WORKERS=48
elif (( PER >= 130 )); then WORKERS=32     # H200 141 GB
elif (( PER >= 70  )); then WORKERS=16     # H100 NVL / A100 80
else                        WORKERS=12; fi
log "hardware: ${NGPU}x ${PER}GB -> TP=$TP, $SLOTS concurrent slot(s), $WORKERS workers/run"
(( SLOTS < ${#RUNS[@]} )) && log "note: ${#RUNS[@]} runs but $SLOTS slot(s) - remaining runs start automatically as slots free"

start_server() {  # slot_index port model checkpoint
    local slot=$1 port=$2 model=$3 ck=$4 maxlen=98304 gpus
    [[ "$ck" == "base" ]] && maxlen=40960          # Qwen3-30B-A3B native window
    gpus=$(seq -s, $(( slot * TP )) $(( slot * TP + TP - 1 )))
    log "  serving $model on GPU(s) $gpus -> :$port (TP=$TP, max_len=$maxlen)"
    CUDA_VISIBLE_DEVICES=$gpus VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 \
        setsid nohup ./venv/bin/vllm serve "$model" --served-model-name deepresearch \
        --port "$port" --tensor-parallel-size "$TP" --max-model-len "$maxlen" \
        --gpu-memory-utilization 0.92 > "logs/vllm_${port}.log" 2>&1 < /dev/null &
    local waited=0
    until server_alive "$port"; do
        grep -q "Engine core initialization failed\|CUDA out of memory" "logs/vllm_${port}.log" 2>/dev/null && {
            log "  server :$port FAILED - see logs/vllm_${port}.log"; return 1; }
        sleep 15; waited=$((waited+15)); (( waited > 1800 )) && { log "  server :$port timed out"; return 1; }
    done
    grep -o "GPU KV cache size: [0-9,]*" "logs/vllm_${port}.log" | tail -1 | sed 's/^/    /'
    return 0
}

start_agent() {  # ck model out thinking threshold port
    local ck=$1 model=$2 out=$3 think=$4 thresh=$5 port=$6
    local conf=$ROOT/systems/QUEST/inference/endpoints_${ck}.conf
    printf 'HOSTNAME_LIST=localhost\nPORTS=%s\n' "$port" > "$conf"
    log "  launching $out from $(count "$out")/$TOTAL ($WORKERS workers)"
    setsid nohup env DATASET="$DATA" OUTPUT_PATH="$ROOT/trajectories/$out" \
        BCP_CHECKPOINT="$ck" BM25_TOP_K=5 BCP_VISIT_TOTAL_TOKEN_CAP=0 MAX_WORKERS=$WORKERS \
        MODEL_PATH="$ROOT/$model" QUEST_ENABLE_THINKING="$think" \
        MEMORY_CONTEXT_THRESHOLD="$thresh" SERVER_ENDPOINTS_FILE="$conf" \
        ./run_quest_bm25.sh >> "logs/run_${out}.log" 2>&1 < /dev/null &
}

# ------------------------------------------------------------- supervisor
supervise() {
    log "supervisor started (pid $$)"
    echo $$ > logs/supervisor.pid
    while true; do
        local busy=0 pending=0
        # pass 1: reap and repair anything already assigned
        for r in "${RUNS[@]}"; do
            IFS='|' read -r CK MODEL OUT THINK THRESH <<< "$r"
            local port; port=$(port_for "$CK"); local n; n=$(count "$OUT")

            if (( n >= TOTAL )); then
                if agent_alive "$OUT"; then log "$OUT complete ($n) - stopping agent"; pkill -9 -f "run_multi_react.py.*$ROOT/trajectories/$OUT"; fi
                if server_alive "$port"; then log "$OUT complete - freeing :$port"; pkill -f "vllm.*--port $port"; sleep 10; fi
                # analyse immediately (CPU only) so structured results exist as
                # soon as a run finishes, without waiting for the whole job
                if [[ ! -f "results/analyzed_$OUT/_summary.json" ]]; then
                    log "$OUT complete - analysing (no GPU needed)"
                    ./venv/bin/python analyze_run.py --run_dir "trajectories/$OUT" \
                        --system QUEST-30B --checkpoint "$CK" \
                        --out_dir "results/analyzed_$OUT" >> "$SUPLOG" 2>&1 \
                        && log "$OUT analysed -> results/analyzed_$OUT"
                fi
                continue
            fi

            if agent_alive "$OUT"; then
                # STALLED: agent alive but its server is gone (observed failure mode).
                # The agent blocks on retries and makes no progress - restart both.
                if ! server_alive "$port"; then
                    log "$OUT STALLED (agent alive, server :$port dead) - restarting pair"
                    pkill -9 -f "run_multi_react.py.*$ROOT/trajectories/$OUT"; sleep 5
                else
                    busy=$((busy+1)); continue
                fi
            fi
            pending=$((pending+1))
        done

        # pass 2: fill free slots with pending runs
        local slot=0
        for r in "${RUNS[@]}"; do
            IFS='|' read -r CK MODEL OUT THINK THRESH <<< "$r"
            local port; port=$(port_for "$CK"); local n; n=$(count "$OUT")
            (( n >= TOTAL )) && continue
            agent_alive "$OUT" && { slot=$((slot+1)); continue; }
            if (( slot >= SLOTS )); then continue; fi
            log "$OUT needs a slot -> slot $slot"
            if ! server_alive "$port"; then
                pkill -f "vllm.*--port $port" 2>/dev/null; sleep 5
                start_server "$slot" "$port" "$MODEL" "$CK" || { log "  slot $slot unavailable, will retry"; slot=$((slot+1)); continue; }
            fi
            start_agent "$CK" "$MODEL" "$OUT" "$THINK" "$THRESH" "$port"
            sleep 30
            slot=$((slot+1))
        done

        # done?
        local alldone=1
        for r in "${RUNS[@]}"; do IFS='|' read -r CK MODEL OUT THINK THRESH <<< "$r"
            (( $(count "$OUT") >= TOTAL )) || alldone=0; done
        if (( alldone )); then
            log "ALL RUNS COMPLETE - judging automatically"
            "$ROOT/run_all.sh" --judge >> "$SUPLOG" 2>&1 && log "judging done"
            "$ROOT/run_all.sh" --bundle >> "$SUPLOG" 2>&1 && log "results bundle ready"
            rm -f logs/supervisor.pid
            return 0
        fi
        sleep 300
    done
}

if [[ "${1:-}" == "--supervise-loop" ]]; then
    export IN_SUPERVISOR=1
    supervise
    exit 0
fi

# a second supervisor would fight the first over GPUs
if [[ -f logs/supervisor.pid ]] && kill -0 "$(cat logs/supervisor.pid)" 2>/dev/null; then
    log "supervisor already running (pid $(cat logs/supervisor.pid)) - nothing to do"
    log "use --status to watch, --stop to halt"
    exit 0
fi

pgrep -f "disk_watc[h]" >/dev/null || { setsid nohup ./disk_watch.sh >/dev/null 2>&1 < /dev/null & log "disk watchdog started"; }
IN_SUPERVISOR=1 setsid nohup "$ROOT/run_all.sh" --supervise-loop >> "$SUPLOG" 2>&1 < /dev/null &
sleep 3
log "supervisor detached - survives logout. ./run_all.sh --status to watch"
