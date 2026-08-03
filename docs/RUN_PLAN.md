# QUEST-30B on BrowseComp-Plus (BM25) — FINAL SCOPE

Scope reduced to **Table 1 primary conditions only**. All ablations cancelled.

Primary condition = **BM25 top-5, uncapped visit, 830 questions** — the official
BrowseComp-Plus protocol (their README: "a local retriever tool that returns the
top 5 relevant documents, with a maximum context length of 512 tokens across all
methods for fair comparison").

## Table 1 rows

| Row | Model | Status |
| --- | --- | --- |
| Base | `Qwen/Qwen3-30B-A3B` (instruction-tuned) | ✅ **830/830 complete** (top-15 uncapped + capped) |
| SFT | `osunlp/QUEST-30B-MT-Plus-SFT` | 🔄 running — top-5 uncapped, GPU0 |
| RL | `osunlp/QUEST-30B-RL` | 🔄 running — top-5 uncapped, GPU1 |

Then: judge all completed runs (Qwen3-32B, official BC-Plus grader) → Table 1.

## Cancelled

- SFT/RL top-15 (retrieval-depth ablation) — partials kept as
  `quest30b_sft_bm25_top15_{uncapped,capped}_partial/`
- SFT/RL top-5 capped (visit-cap ablation)
- **O-Researcher-72B** — not feasible here: ~145 GB per checkpoint (435 GB for
  three vs 283 GB free), needs both GPUs at `tensor_parallel_size=2`, and dense
  72B is ~5-10x the compute per token of 30B-A3B (3B active). Estimated
  ~200h per checkpoint, ~3-4 weeks total. **Awaiting mentor guidance.**

## Known caveat on the Base row

The Base row was run at **top-15**, not top-5, because the scope change came
after it completed. It is internally consistent (both its conditions are
top-15) but sits at a different retrieval depth from SFT/RL. Options: note it in
the table, or re-run Base at top-5 (~8h — far cheaper than SFT/RL, since Base
averages 12.6 search calls/question vs 62.7).

## Supplementary (not Table 1 rows)

- `quest30b_pretrained_base_supplementary/` — `Qwen3-30B-A3B-Base` pretrained,
  stopped at 500/830, 2 tool events total. Evidence that the pretrained base
  never invokes tools (see `results/base_parser_confirmation/`).
- `_archive_base_redundant_conditions/` — proof the retrieval config is inert
  when no tool calls occur.

## Weights on disk

Only `QUEST-30B-MT-Plus-SFT` and `QUEST-30B-RL` (in use) plus `Qwen3-32B`
(judge, re-downloading). `Qwen3-30B-A3B` and `Qwen3-30B-A3B-Base` were deleted
after their runs completed — re-fetch from the Hub if ever needed.

## Reporting cadence

Every 100 completed questions, for SFT and RL together: SR, turn-cap rate, gold
retrieval, duplicate query rate, tool calls/question, non-tool turn fraction.

## Resume

Rerun the same command with the same `OUTPUT_PATH`; completed question ids are
skipped. All runs launched detached (`setsid nohup ... < /dev/null &`).
