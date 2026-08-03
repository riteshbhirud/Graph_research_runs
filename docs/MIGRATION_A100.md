# Migrating the remaining QUEST work to the 6x A100-40GB node

## What is left

| Run | Done | Remaining |
| --- | ---: | ---: |
| SFT top-5 uncapped | 69/830 | 761 |
| RL top-5 uncapped | 71/830 | 759 |
| Base top-5 uncapped | 0/830 | 830 |
| Judge all 3 rows + Table 1 | — | ~3h |

Already complete and needing nothing further: instruct-Base top-15 uncapped and
capped (830/830 each), the pretrained-base supplementary run, the oracle
ceiling, and all diagnostics.

## Step 0 — the one question that decides everything

**Is `/home/x5o` the same filesystem on both nodes?** hudson mounts it over NFS
as `fs01:/pool/home/x5o`.

    df -h /home/x5o        # on the A100 node

- **Same NFS mount** -> nothing to copy. Weights, BM25 index, all trajectories
  and resume state are already there. Skip to Step 2.
- **Different filesystem** -> transfer (~180 GB):

      rsync -av --info=progress2 \
        hudson:/home/x5o/ChenWork/{models,data,trajectories,checkpoints,results,systems,paper_notes} \
        /path/on/a100/ChenWork/

  Trajectories are the irreplaceable part - `models/` can be re-downloaded from
  the Hub in ~1h if bandwidth is better than the copy.

## Step 1 — environment (only if not shared)

    python3 -m venv venv
    ./venv/bin/pip install vllm pyserini json5 openai transformers datasets \
        huggingface_hub litellm qwen-agent tiktoken pandas
    export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64   # pyserini needs a JVM

Check for a proxy, as on hudson: if `http_proxy` is set, every run needs
`no_proxy=localhost,127.0.0.1` or vLLM calls will 503. `run_quest_bm25.sh`
already exports this.

## Step 2 — serve (TP=2 is mandatory)

A 57 GB checkpoint does not fit on a 40 GB card. TP must divide
`num_attention_heads=32` and `num_key_value_heads=4`, so **TP=2 or TP=4 only**
(TP=3 is invalid). Use 3 instances x TP=2 across all 6 GPUs:

    ./serve_a100.sh 0,1 6000 models/QUEST-30B-MT-Plus-SFT
    ./serve_a100.sh 2,3 6002 models/QUEST-30B-RL
    ./serve_a100.sh 4,5 6004 models/Qwen3-30B-A3B

The script prints the KV cache size on startup - **record it**. On hudson it was
274,544 tokens; expect roughly 150k-190k here. That number is the single best
predictor of throughput, because these runs are KV-bound.

## Step 3 — endpoint configs

    printf 'HOSTNAME_LIST=localhost\nPORTS=6000\n' > systems/QUEST/inference/server_endpoints.conf
    printf 'HOSTNAME_LIST=localhost\nPORTS=6002\n' > systems/QUEST/inference/server_endpoints_gpu1.conf
    printf 'HOSTNAME_LIST=localhost\nPORTS=6004\n' > systems/QUEST/inference/server_endpoints_base_top5.conf

## Step 4 — relaunch (resumes automatically)

Same `OUTPUT_PATH` = completed question ids are skipped. Expect the launch
banner to report `already successfully processed: 69` / `71`.

    # SFT
    setsid nohup env DATASET=$PWD/data/browsecomp_plus/browsecomp_plus_830.jsonl \
      OUTPUT_PATH=$PWD/trajectories/quest30b_sft_bm25_top5_uncapped \
      BCP_CHECKPOINT=sft BM25_TOP_K=5 BCP_VISIT_TOTAL_TOKEN_CAP=0 MAX_WORKERS=12 \
      MODEL_PATH=$PWD/models/QUEST-30B-MT-Plus-SFT \
      QUEST_ENABLE_THINKING=true MEMORY_CONTEXT_THRESHOLD=80000 \
      ./run_quest_bm25.sh >> logs/run_sft_top5_uncapped.log 2>&1 < /dev/null &

    # RL
    setsid nohup env DATASET=$PWD/data/browsecomp_plus/browsecomp_plus_830.jsonl \
      OUTPUT_PATH=$PWD/trajectories/quest30b_rl_bm25_top5_uncapped \
      BCP_CHECKPOINT=rl BM25_TOP_K=5 BCP_VISIT_TOTAL_TOKEN_CAP=0 MAX_WORKERS=12 \
      MODEL_PATH=$PWD/models/QUEST-30B-RL \
      QUEST_ENABLE_THINKING=true MEMORY_CONTEXT_THRESHOLD=80000 \
      SERVER_ENDPOINTS_FILE=$PWD/systems/QUEST/inference/server_endpoints_gpu1.conf \
      ./run_quest_bm25.sh >> logs/run_rl_top5_uncapped.log 2>&1 < /dev/null &

    # Base (top-5) - no longer needs queueing, there is a third instance
    setsid nohup env DATASET=$PWD/data/browsecomp_plus/browsecomp_plus_830.jsonl \
      OUTPUT_PATH=$PWD/trajectories/quest30b_base_bm25_top5_uncapped \
      BCP_CHECKPOINT=base BM25_TOP_K=5 BCP_VISIT_TOTAL_TOKEN_CAP=0 MAX_WORKERS=12 \
      MODEL_PATH=$PWD/models/Qwen3-30B-A3B \
      QUEST_ENABLE_THINKING=false MEMORY_CONTEXT_THRESHOLD=30000 \
      SERVER_ENDPOINTS_FILE=$PWD/systems/QUEST/inference/server_endpoints_base_top5.conf \
      ./run_quest_bm25.sh > logs/run_base_top5_uncapped.log 2>&1 < /dev/null &

`MAX_WORKERS` drops 16 -> 12 because KV per instance is smaller; raise it only
if the KV cache figure from Step 2 comes in above ~200k tokens.

Base-specific settings are not optional: `QUEST_ENABLE_THINKING=false` (Qwen3
thinking mode emits unparseable tool calls - 0/5 vs 5/5) and
`MEMORY_CONTEXT_THRESHOLD=30000` (native window is 40,960).

## Step 5 — timing probe before committing ~3.5 days

Let each run reach ~20 completions, then:

    ./venv/bin/python write_checkpoint.py --run_dir trajectories/quest30b_sft_bm25_top5_uncapped \
        --name sft --out checkpoints/checkpoint_sft.json

Measure q/min over a >=60 min window with the pipeline full. hudson steady state
was **0.18-0.20 q/min** per run. If A100 lands near 0.11, expect ~90h; if NVLink
helps and it lands near 0.15, expect ~65h.

## Step 6 — judging (unchanged)

    ./serve_a100.sh 0,1 6001 models/Qwen3-32B     # 62 GB, also needs TP=2
    ./venv/bin/python judge_run.py --dir <analyzed_dir> --base_url http://localhost:6001/v1

## Also start the watchdog

    setsid nohup ./disk_watch.sh > /dev/null 2>&1 < /dev/null &

Trajectory growth measured on hudson: ~57 MB/question for uncapped runs, so the
three remaining runs need roughly 130 GB.

## Not portable to this node

**O-Researcher-72B.** 145 GB needs TP=4 (160 GB across 4x40 GB), leaving ~10 GB
for KV - effectively zero concurrency. It was already infeasible on hudson at
~200h per checkpoint; on 40 GB cards it is worse. Needs different hardware or an
FP8 checkpoint.
