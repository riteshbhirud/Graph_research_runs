"""Emit Table 1 and the behavioural-progression table from judged runs."""
import glob
import json
import os

ROWS = [("base", "Base", "quest30b_base_bm25_top5_uncapped"),
        ("sft", "SFT", "quest30b_sft_bm25_top5_uncapped"),
        ("rl", "RL", "quest30b_rl_bm25_top5_uncapped")]

out = []
out.append("# Table 1 — QUEST-30B on BrowseComp-Plus (BM25 top-5, 830 questions)\n")
out.append("Primary condition: BM25 top-5, 512-token snippets, uncapped visit, "
           "max 100 turns, discard-all context management, single rollout.\n")
out.append("| System | Stage | SR↑ | Turns↓ |")
out.append("| --- | --- | ---: | ---: |")

summaries = {}
for ck, label, d in ROWS:
    sp = f"results/analyzed_{d}/_summary.json"
    if not os.path.exists(sp):
        out.append(f"| QUEST-30B | {label} | – | – |")
        continue
    s = json.load(open(sp))
    summaries[ck] = s
    sr = s.get("TABLE1_SR_percent")
    turns = s.get("TABLE1_Turns_search_invocations")
    n = s.get("n_questions", 0)
    star = "" if n >= 830 else f" *(n={n})*"
    out.append(f"| QUEST-30B | {label} | {sr}%{star} | {turns} |")

out.append("\n`Turns` = search invocations / 830. `total_search_queries` is also "
           "recorded per trajectory, since QUEST batches ~3.5 queries per invocation.\n")

out.append("\n# Behavioural progression\n")
hdr = "| Metric | " + " | ".join(l for _, l, _ in ROWS) + " |"
out.append(hdr)
out.append("| --- | " + " | ".join("---:" for _ in ROWS) + " |")
METRICS = [
    ("SR (%)", "TABLE1_SR_percent"),
    ("Search invocations / q", "TABLE1_Turns_search_invocations"),
    ("Search queries / q", "TABLE1_Turns_search_queries"),
    ("Duplicate query rate", "TABLE1_query_duplicate_rate"),
    ("Unique docs / q", "TABLE1_unique_docs_per_question"),
    ("Gold retrieval rate (%)", "TABLE1_gold_retrieval_rate"),
    ("Hit 100-turn cap (%)", "pct_hit_turn_cap"),
    ("Non-tool turn fraction", "TABLE1_non_tool_turn_fraction"),
    ("  — cap-hitters", "non_tool_turn_fraction_cap_hitters"),
    ("  — natural finishers", "non_tool_turn_fraction_natural"),
    ("Tool-call repair rate", "overall_repair_rate"),
    ("Disallowed tool attempts", "total_disallowed_attempts"),
    ("Docid hallucinations", "total_docid_hallucinations"),
]
for label, key in METRICS:
    vals = [str(summaries.get(ck, {}).get(key, "–")) for ck, _, _ in ROWS]
    out.append(f"| {label} | " + " | ".join(vals) + " |")

out.append("\nSee docs/paper_notes/baseline_comparison.md for what may and may not "
           "be claimed against published QUEST numbers.\n")

os.makedirs("results", exist_ok=True)
text = "\n".join(out)
open("results/table1.md", "w", encoding="utf-8").write(text)
print(text)
print("\n-> results/table1.md")
