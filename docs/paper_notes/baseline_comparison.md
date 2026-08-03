# Baseline comparison notes — BrowseComp-Plus (hudson, BM25)

What may and may not be claimed about prior results for the systems we run on
this machine. Every fact below was checked against released code or model
cards; provenance is given inline.

---

## QUEST-30B — a published BrowseComp-Plus number exists

**Published: 48.2 (avg@3), QUEST-30B-RL.**

Provenance: the results table in the `osunlp/QUEST-30B-RL` **model card**. The
GitHub README (`OSU-NLP-Group/QUEST`, commit `225686f`) contains no results
table — only a directory index — so the model card is the source.

The number applies to the **RL checkpoint only**. No published BrowseComp-Plus
figure was found for the Base or MT+SFT checkpoints.

### Their evaluation configuration (verified from `inference/scripts/run_react_infer_bcp.sh`)

| Setting | Their published run | Our run |
| --- | --- | --- |
| Questions | **130-question subset** (`browsecomp_plus_quest_130.jsonl`) | **full 830** |
| Retriever | **Qwen3-Embedding-8B FAISS dense, top-k=5** | **BM25 (Lucene), top-k=15** |
| Snippet length | 512 tokens | 512 tokens |
| `visit` tool | **disabled** (search-only) | **enabled** on `bm25://<docid>` |
| Context management | `discard_all` @ 80,000 | `discard_all` @ 30,000 (Base) / 80,000 (SFT, RL) |
| Rollouts | **3** (metric is avg@3) | **1** |
| Rope scaling | n/a | none (Base at native 40,960, non-thinking) |

### Three corrections to earlier assumptions

These were assumed in an earlier draft and are **wrong**:

1. **"full set"** — their released BrowseComp-Plus input is a **130-question
   subset**, not the full 830. Both `run_react_infer_bcp.sh` and
   `run_eval_quest.sh` default to `browsecomp_plus_quest_130`.
2. **"unknown retriever config"** — the retriever is fully specified in their
   launch script: `FAISS_MODEL_NAME=Qwen/Qwen3-Embedding-8B`, `FAISS_TOP_K=5`,
   `FAISS_SNIPPET_MAX_TOKENS=512`. It is documented, not unknown.
3. **"with context condenser"** — they set `MEMORY_STRATEGY=discard_all` for
   BrowseComp-Plus, i.e. **the same context-management strategy we use**. The
   condenser is their default elsewhere, but not on this benchmark.

### What may be claimed

- ✅ Cross-checkpoint comparison **Base vs SFT vs RL is valid** — all three run
  under identical conditions in our setup.
- ✅ Retriever comparison across systems is valid — BM25 is held constant.
- ❌ **Do not claim reproduction of 48.2.** Different retriever, different
  question set, visit enabled, and single-rollout vs avg@3.
- ❌ Do not describe our QUEST numbers as "matching" or "failing to match"
  published results; the configurations are not comparable in either direction.

Suggested wording:

> Our QUEST-30B numbers are not directly comparable to the published 48.2
> (avg@3) for QUEST-30B-RL, which was measured on a 130-question subset with
> Qwen3-Embedding-8B dense retrieval (top-5), the visit tool disabled, and
> averaged over three rollouts. We evaluate the full 830-question set with BM25
> (top-15) and the visit tool enabled, single rollout. Cross-checkpoint
> comparisons remain valid as all checkpoints share our configuration.

---

## O-Researcher-72B — no published BrowseComp-Plus number

**Published BrowseComp-Plus result: none.**

Checked the `PersonalAILab/O-Researcher-72B-{sft,rl}` model cards and the
`OPPO-PersonalAI/O-Researcher` GitHub README (commit `adbddc3`). Neither
reports BrowseComp-Plus. Their repo has no BrowseComp-Plus evaluation
directory.

### What may be claimed

- ✅ **First evaluation of O-Researcher-72B on BrowseComp-Plus under offline
  BM25.** All numbers are new measurements.
- ✅ Cross-checkpoint Base/SFT/RL comparison is valid (identical conditions).
- ❌ No reproduction claim is possible — there is nothing to reproduce.

### Adaptation that must be disclosed

O-Researcher ships **live-web tools only**: a Serper-backed `web_search`
service and a `crawl_page` service using Jina plus an LLM summarizer. There is
no offline code path. We replaced both with local BM25-backed HTTP services
that reproduce the observation formats found in their released SFT training
data (43,082 `# Search Summary` SERP-style observations and 35,150
`# Synthesis Report` crawl observations). This adaptation must be stated in the
methods section, since it is our construction and not theirs.

Base checkpoint (`Qwen2.5-72B-Instruct`) is run with the **same** system
prompt, tag schema and tool definitions as SFT/RL — the paper does not state
how they configured a base model, so we choose the most conservative fair
option and document it.

---

## Machine-readable form

The same content is stamped as an `IMPORTANT_FOR_PAPER` block into every
trajectory JSON and every `_summary.json` produced by `analyze_run.py`, via
`paper_notes_blocks.py`. Trajectories are therefore self-describing: anyone
reading a single trajectory file can see what may be claimed about it.

---

# Base Model Query Behavior (167-question sample, Base uncapped)

**Key finding: the base model exhibits severe query looping, not genuine
exploration.**

| Metric | Base (uncapped) |
| --- | ---: |
| Search invocations per question | **12.60** |
| Search queries per question | **45.42** |
| Queries per invocation (batching) | 3.61 |
| **Duplicate query rate** | **82.4%** (exact 49.8% + near-dup 32.6%) |
| Unique documents retrieved per question | 73.3 |
| **Gold retrieval rate** | **15.0%** |

Duplicate = exact match, or Jaccard token overlap >= 0.8, against an earlier
query in the *same* trajectory. Computed identically for Base, SFT and RL.

## BM25 ceiling (all 830 questions, `results/oracle_bm25_ceiling.json`)

Querying BM25 with the gold **answer** string — an oracle query the agent
cannot know in advance — gives the maximum gold retrieval achievable at k=15:

| | |
| --- | ---: |
| Gold retrievable at k=15 | **634 / 830 = 76.4%** |
| Mean rank when found | 2.02 |
| Median rank when found | **1** |
| Found at rank 1 | 471 (56.7% of all questions) |

**The retriever is not the bottleneck.** BM25 surfaces a gold document for
76.4% of questions, usually at rank 1. The base model reaches only 15.0%. The
~61-point gap is attributable to query formulation, not retrieval
infrastructure.

Caveat for the write-up: 76.4% is an *upper bound*, not an achievable target —
it uses the answer as the query. It bounds the search half of the task; it does
not bound end-to-end SR, since the agent must also reason correctly over what it
retrieves.

## Expected Base -> SFT -> RL trajectory

If training teaches search behaviour rather than just protocol conformance,
SFT/RL should show: lower duplicate rate, higher unique-document count, higher
gold retrieval, and fewer search invocations (more efficient). **To be
confirmed** once those runs complete.

## Framing rules

Do **not** describe 12.6 invocations as "thorough search".

Use instead:

> Base models exhibit repetitive query behavior, with 82% of issued queries
> duplicating an earlier query in the same trajectory, achieving only 15% gold
> retrieval despite high search volume — against a BM25 ceiling of 76.4% at the
> same top-k. SFT and RL training substantially reduce this redundancy
> [to be confirmed].

---

# ⚠️ Name collision: "OpenResearcher" vs "O-Researcher"

Two distinct projects with near-identical names. They must never be conflated
in the paper, in Table 1 row labels, or in citations.

| | **O-Researcher** | **OpenResearcher** |
| --- | --- | --- |
| Org | OPPO Personal AI Lab | TIGER-AI-Lab |
| HF | `PersonalAILab/O-Researcher-72B-{sft,rl}` | `OpenResearcher/OpenResearcher-30B-A3B` |
| GitHub | `OPPO-PersonalAI/O-Researcher` (`adbddc3`) | `TIGER-AI-Lab/OpenResearcher` (`785fd6b`) |
| arXiv | 2601.03743 | 2603.20278 |
| Size | 72B (Qwen2.5-72B-Instruct) | 30B-A3B |
| Tools | `web_search`, `crawl_page` (XML tags) | `browser.search`, `browser.open`, `browser.find` (native tool calling) |
| Offline retrieval | **None** — Serper + Jina only; we built a BM25 shim | **Native BM25** (pyserini `LuceneSearcher`, `SEARCHER_TYPE=bm25`) |
| Published BrowseComp-Plus | **none** | **54.8%** |
| In our scope | ✅ scheduled (72B, both checkpoints) | ❓ candidate unified scaffold — not yet scheduled |

## Rules

1. Always write the hyphen exactly: **O-Researcher** (OPPO) vs
   **OpenResearcher** (TIGER-AI-Lab). Never "Open-Researcher" or
   "OResearcher".
2. On first mention of each, give the org in parentheses.
3. The "first BrowseComp-Plus evaluation" claim applies **only to
   O-Researcher (OPPO)**. OpenResearcher (TIGER-AI-Lab) already reports 54.8%
   on BrowseComp-Plus, so no first-evaluation claim may be made for it.
4. Table 1 row labels must carry the org until the distinction is established
   in text.
5. When citing a BrowseComp-Plus baseline number, state which project it came
   from — 54.8% belongs to OpenResearcher (TIGER-AI-Lab), not to O-Researcher.

---

# Pretrained base cannot conform to the ReAct protocol (confirmed)

**Qwen3-30B-A3B-Base never emitted `<tool_call>` tags across 20 generations,
4 serving configurations, and 2 independent scaffolds.** Raw outputs saved to
`results/base_parser_confirmation/` for reviewer verification. This finding
demonstrates that QUEST's mid-training stage is prerequisite for ReAct protocol
compliance.

## Configurations tested (5 questions each, greedy decoding)

| Config | QUEST parser | OpenResearcher parser |
| --- | ---: | ---: |
| T1 thinking mode ON | 0/5 | 0/5 |
| T2 raw text completion (no chat template) | 0/5 | 0/5 |
| T3 max_tokens=16,384, thinking OFF | 0/5 | 0/5 |
| T4 OpenResearcher scaffold (`browser.search/open/find`) | 0/5 | 0/5 |
| **TOTAL** | **0/20** | **0/20** |

Both parsers are the production implementations, applied verbatim: QUEST's
`matches[-1]` + strict `json.loads` + our trailing-brace repair; OpenResearcher's
`<tool_call>` / bare `</tool_call>` + `json5`.

## What the model emitted instead (all 20 generations)

| Wrapper | Count |
| --- | ---: |
| `<tools>` tag (wrong; should be `<tool_call>`) | 7 |
| bare JSON, no wrapper | 4 |
| no JSON call at all | 9 |
| **`<tool_call>` tags** | **0** |

The model can form syntactically correct tool-call JSON but never wraps it in
the required tags. It also fails to sustain a trajectory: under T1 it emits one
valid call then loops `<tools>` / `工具调用` / `tools` indefinitely; under T3 it
runs away to 42,258 characters of ever-growing query lists.

## Decision

A parser extension accepting `<tools>` and bare JSON was considered and
**rejected**: it would invent a tool-call convention that neither QUEST nor
OpenResearcher uses, and the degenerate looping means parsed first turns would
not survive into usable multi-turn trajectories. The result is recorded as
genuine model behaviour, not a harness artifact.

This is distinct from the trailing-brace repair, which corrects a
one-character JSON defect **inside an already-correct wrapper** and is applied
symmetrically to all checkpoints with the repair rate logged.

---

# CORRECTION — the BM25 ceiling is a *single-query* ceiling

**76.4% is a single-query ceiling at top-15. SFT surpasses this through
multi-query exploration across 100 turns, achieving 80% gold retrieval. The
ceiling describes single-query retrieval capacity, not multi-turn agentic
retrieval capacity.**

## Why the earlier framing was wrong

`results/oracle_bm25_ceiling.json` was computed by issuing **one** oracle query
(the gold answer string) per question and checking the top-15. That bounds what
a single perfect query can retrieve. It does **not** bound what an agent can
retrieve by issuing many queries across many turns.

Measured on QUEST-30B-SFT (first 15 completed questions, top-15 uncapped):

| | Base (instruct) | SFT |
| --- | ---: | ---: |
| Search queries per question | 45.4 | **195.7** |
| Unique documents surfaced per question | 74.8 | **389.7** |
| Gold retrieval rate | 15.7% | **80.0%** |
| Mean gold recall | 0.075 | **0.723** |

SFT exceeds the 76.4% single-query ceiling because breadth (196 queries x 15
documents, 390 unique docs) beats one perfectly-worded query.

## How to use each number

- **76.4% / 70.7%** (top-15 / top-5): the ceiling for **one** ideal query.
  Use it to argue that BM25 *can* surface the evidence, and to attribute the
  Base model's 15.7% to query formulation rather than retrieval capacity.
- **Do NOT** present it as an upper bound on agent gold retrieval, or as an
  upper bound on SR. Multi-turn agents legitimately exceed it.

## Related finding: the bottleneck moves

SFT retrieves a gold document on **80%** of questions yet **60%** of questions
terminate as `answer not found` after exhausting all 100 turns. Retrieval is no
longer the limiting factor for SFT; synthesis and termination are. Whether RL
closes this gap is a primary question for the RL runs.

---

# FRAMING RULE: report the Base -> SFT -> RL progression, not any single failure

Termination behaviour is **one component** of the progression story, not a
standalone finding. Do not write a section titled "SFT fails to terminate".
Write the trajectory of what each training stage changes, with termination as
one of several dimensions that move together.

## The progression (dimensions tracked identically for all three checkpoints)

| Dimension | Base (instruct) | SFT | RL |
| --- | ---: | ---: | ---: |
| Protocol conformance (valid tool calls) | partial | 1106/1106 | TBD |
| Search invocations / question | 12.6 | 62.7 | TBD |
| Duplicate query rate | 82.4% | 61.0% | TBD |
| — exact duplicates | 49.8% | 12.0% | TBD |
| Unique docs / question | 74.8 | 389.7 | TBD |
| Gold retrieval rate | 15.7% | 80.0% | TBD |
| Hit 100-turn cap | ~5% | 62.5% | TBD |
| Non-tool turn fraction (cap-hitters) | n/a | 0.000 | TBD |
| Non-tool turn fraction (natural) | n/a | 0.054 | TBD |
| SR | TBD | TBD | TBD |

Base figures are the instruction-tuned `Qwen3-30B-A3B` (the Table 1 Base row);
SFT figures are from 16 completed top-15 uncapped questions and are provisional.

## The claim this supports

Each stage moves a different bottleneck rather than uniformly improving:
mid-training/SFT converts a looping, low-yield searcher into a high-yield one
(gold retrieval 15.7% -> 80.0%, exact duplicates 49.8% -> 12.0%), and in doing
so shifts the limiting factor from *finding* evidence to *deciding when enough
has been found*. The measurable prediction for RL is a **higher non-tool turn
fraction on cap-hitting questions** — i.e. the agent learns to stop searching
and answer.

Report the whole trajectory. Termination is the dimension SFT moves *backwards*
(or exposes), which is what makes the progression interesting; it is not a
verdict on SFT.
