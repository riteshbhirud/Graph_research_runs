# QUEST-30B on BrowseComp-Plus with offline BM25

Evaluation harness for QUEST-30B (Base / SFT / RL) on the full 830-question
BrowseComp-Plus set using offline BM25 retrieval. Produces Table 1 (SR, Turns)
plus a behavioural progression across training stages.

## Quick start

```bash
git clone <this-repo> && cd bcp-quest-eval
./setup.sh          # env, code, data, ~230 GB of weights (skips anything present)
./run_all.sh        # serve models + launch all three runs
./run_all.sh --status
./run_all.sh --judge   # when runs finish -> results/table1.md
```

`setup.sh` is **idempotent** — every asset is checked before download, and
weight completeness is verified against each repo's safetensors index, so an
interrupted download resumes rather than being silently accepted.
Use `./setup.sh --no-models` for a code/data-only install (~3 GB).

## Hardware

The checkpoints are ~57 GB in bf16, so a GPU with **≥70 GB** holds one alone.
`run_all.sh` detects VRAM and configures itself:

| GPU | Layout | Notes |
| --- | --- | --- |
| H200 141 GB, B200, B300, H100 NVL 94 GB, A100 80 GB | 1 GPU per checkpoint | 3 GPUs runs all three concurrently |
| A100 40 GB, L40S 48 GB | tensor-parallel 2 | 6 GPUs needed for three concurrent runs |

Tensor parallelism must divide `num_attention_heads=32` and
`num_key_value_heads=4` → **TP ∈ {1, 2, 4}**. TP=3 is invalid.

**Throughput is KV-cache-bound, not compute-bound.** After 57 GB of weights,
whatever VRAM remains becomes KV cache and determines concurrency. Measured on
H100 NVL 94 GB: 30 GB KV = 274,544 tokens = **0.19 questions/min per GPU**.
An H200 has ~2.7× that headroom, so expect roughly 0.3 q/min. `run_all.sh`
scales `MAX_WORKERS` with VRAM for this reason.

Whole job is ~2,350 remaining questions across three runs.

## What it runs

| Row | Model | Notes |
| --- | --- | --- |
| Base | `Qwen/Qwen3-30B-A3B` (instruction-tuned) | non-thinking mode, 40,960 window, discard-all @ 30k |
| SFT | `osunlp/QUEST-30B-MT-Plus-SFT` | shipped chat template (forces thinking), discard-all @ 80k |
| RL | `osunlp/QUEST-30B-RL` | same as SFT |

Common: BM25 top-5, 512-token snippets, visit enabled on `bm25://<docid>`,
max 100 turns, 8,192 max output tokens, single rollout, temperature 1.0,
presence penalty 1.1 (QUEST's own BrowseComp-Plus settings).

Two Base-specific settings are **not optional**: Qwen3's thinking mode emits
structurally malformed tool calls (0/5 parseable vs 5/5 with thinking off), and
its native window is 40,960 — not the 131,072 of the QUEST checkpoints.

## Scaffold modifications

`patches/react_agent.patch` applies to QUEST at pinned commit `225686f`:

- **offline `visit`** — resolves `bm25://<docid>` against the Lucene index and
  returns document text, instead of fetching a live page through Jina
- **disallowed-tool handling** — anything outside `{search, visit}` returns a
  uniform error and is logged (measured rate: 0/1106 for SFT)
- **minimal tool-call repair** — trims the fewest trailing `}` that yield valid
  JSON, never altering names or arguments; every repair logged
- **non-thinking switch** — `QUEST_ENABLE_THINKING=false` for the Base model

`patches/bcp_offline.py` implements the BM25 backend, an optional visit token
cap, and structured event logging.

## Outputs

```
trajectories/<run>/                 raw QUEST output + per-question logs
trajectories/<run>/events.jsonl     one record per executed tool call
results/analyzed_<run>/<qid>.json   full trajectory in the agreed schema
results/analyzed_<run>/_summary.json
results/table1.md                   Table 1 + behavioural progression
```

Trajectories are reconstructed from `trajectories_no_memory.jsonl`, **not** the
run's final message list — discard-all context management truncates the latter
and would silently drop most of the tool history.

Per-trajectory fields include the query-diversity block (exact/near-duplicate
rates), gold retrieval, `non_tool_turn_fraction`, `termination_type`, and an
`IMPORTANT_FOR_PAPER` block recording what may be claimed about the numbers.

## Metric definitions

- **SR** = judged-correct / 830 × 100, using the **official BrowseComp-Plus
  grader prompt** (`judge_run.py`) with Qwen3-32B, thinking enabled, T=0.
- **Turns** = search **invocations** / 830. `total_search_queries` is recorded
  alongside because QUEST batches ~3.5 queries per invocation, so the two differ
  by ~3.5×.

## Repro utilities

```bash
./venv/bin/python oracle_bm25_ceiling.py   # single-query BM25 ceiling (76.4% @ k=15, 70.7% @ k=5)
./venv/bin/python oracle_k_sweep.py        # ceiling vs top-k
./venv/bin/python diag_search_quality.py results/analyzed_<run>
./venv/bin/python write_checkpoint.py --run_dir trajectories/<run> --name x --out checkpoints/x.json
```

## Resume

Every run is resumable: rerun with the same `OUTPUT_PATH` and completed question
ids are skipped (the launch banner reports `already successfully processed: N`).
Runs are launched detached (`setsid nohup`) so they survive an SSH disconnect.

## Not included

`O-Researcher-72B` — ~145 GB per checkpoint, needs live-web tools replaced with
a BM25 shim, and estimated ~200 h per checkpoint. See
`docs/paper_notes/baseline_comparison.md`.
