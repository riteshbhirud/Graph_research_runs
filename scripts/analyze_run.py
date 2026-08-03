"""Build the full trajectory records + summary metrics for one QUEST run.

Reads the raw QUEST output (iter*.jsonl), the per-task message logs, and the
bcp_offline event log, then emits one JSON per question in the agreed schema
plus a summary. Judging is done separately by judge_run.py.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_notes_blocks import block_for


def load_events(path):
    ev = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                ev.append(json.loads(line))
            except Exception:
                pass
    return ev


def count_tokens(text, tok):
    return len(tok.encode(text, add_special_tokens=False))


def _norm_q(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def query_diversity(search_queries, n_unique_docs):
    """Duplicate-query analysis within a single trajectory.

    A query is a duplicate if it exactly matches an earlier query in the SAME
    trajectory, or has Jaccard token overlap >= 0.8 with one. Computed
    identically for every checkpoint so Base/SFT/RL are comparable.
    """
    seen = []
    exact = near = 0
    for q in search_queries:
        nq = _norm_q(q)
        tq = set(nq.split())
        for ps, pt in seen:
            if nq == ps:
                exact += 1
                break
            if tq and pt and len(tq & pt) / len(tq | pt) >= 0.8:
                near += 1
                break
        seen.append((nq, tq))
    total = len(search_queries)
    uniq = len({_norm_q(q) for q in search_queries})
    return {
        "total_queries": total,
        "exact_duplicates": exact,
        "near_duplicates_jaccard_08": near,
        "combined_duplicate_rate": round((exact + near) / total, 4) if total else 0.0,
        "unique_queries": uniq,
        "unique_docs_retrieved": n_unique_docs,
        "docs_per_unique_query": round(n_unique_docs / uniq, 2) if uniq else 0.0,
    }


def parse_trajectory(messages):
    """Turn the raw message list into the turn-structured trajectory."""
    turns = []
    counts = collections.Counter()
    disallowed = []
    turn_id = 0
    pending = None
    for m in messages:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "assistant":
            turn_id += 1
            reasoning = ""
            tm = re.search(r"<think>(.*?)</think>", content, re.S)
            if tm:
                reasoning = tm.group(1).strip()
            else:
                reasoning = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip()
            calls = []
            for b in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.S):
                s = b.strip()
                obj = None
                try:
                    obj = json.loads(s)
                except Exception:
                    # mirror the repair so the saved trajectory matches what ran
                    n = len(s) - len(s.rstrip("}"))
                    for k in range(1, n + 1):
                        try:
                            cand = json.loads(s[: len(s) - k])
                        except Exception:
                            continue
                        if isinstance(cand, dict) and isinstance(cand.get("name"), str):
                            obj = cand
                            break
                if isinstance(obj, dict):
                    calls.append({"tool": obj.get("name"), "args": obj.get("arguments", {}), "observation": ""})
            pending = {"turn_id": turn_id, "reasoning": reasoning, "tool_calls": calls}
            turns.append(pending)
        elif role == "user" and pending is not None and "<tool_response>" in content:
            obs = re.sub(r"</?tool_response>", "", content).strip()
            if pending["tool_calls"]:
                pending["tool_calls"][0]["observation"] = obs
            else:
                pending["tool_calls"].append({"tool": None, "args": {}, "observation": obs})
            for c in pending["tool_calls"]:
                t = c.get("tool")
                if t:
                    counts[t] += 1
                if "is not available in this offline evaluation setting" in obs and t:
                    disallowed.append(t)
            pending = None
    return turns, counts, disallowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer", default="models/Qwen3-30B-A3B")
    ap.add_argument("--max_turns", type=int, default=100,
                    help="turn cap; used to classify termination_type")
    ap.add_argument("--denominator", type=int, default=830,
                    help="Table 1 denominator (full dataset size), not just completed count")
    args = ap.parse_args()

    MAX_TURNS = args.max_turns

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # gold documents per query, for retrieval recall
    GOLD = {}
    gp = "data/browsecomp_plus/browsecomp_plus_decrypted.jsonl"
    if os.path.exists(gp):
        for line in open(gp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            GOLD[str(r["query_id"])] = [str(d["docid"]) for d in (r.get("gold_docs") or [])]

    events = load_events(os.path.join(args.run_dir, "events.jsonl"))
    ev_by_q = collections.defaultdict(list)
    for e in events:
        ev_by_q[str(e.get("question_id", ""))].append(e)

    iters = sorted(glob.glob(os.path.join(args.run_dir, "*", "*", "iter*.jsonl")))
    iters = [p for p in iters if os.sep + "logs" + os.sep not in p]
    if not iters:
        raise SystemExit(f"no iter*.jsonl under {args.run_dir}")

    def load_full_history(qid):
        """Full, untruncated trajectory from QUEST's no-memory log.

        `messages` in iter*.jsonl is reset by discard_all and loses the tool
        history, so trajectories must be reconstructed from this file instead.
        Returns (messages, ctx_at_termination, max_ctx, n_discard_triggers).
        """
        pat = os.path.join(args.run_dir, "logs", str(qid), "iter*", "trajectories_no_memory.jsonl")
        files = sorted(glob.glob(pat))
        if not files:
            return None, 0, 0, 0
        recs = []
        for fp in files:
            for line in open(fp, encoding="utf-8"):
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
        if not recs:
            return None, 0, 0, 0
        finals = [r for r in recs if r.get("is_final")] or recs
        final = finals[-1]
        n_discard = sum(1 for r in recs if r.get("trigger_reason") == "CONTEXT_THRESHOLD_DISCARD_ALL")

        # token_count in the no-memory log is the CUMULATIVE transcript, not the
        # live context. The live context (what the model actually receives) is
        # recorded in the memory-side trajectories.jsonl.
        mem_pat = os.path.join(args.run_dir, "logs", str(qid), "iter*", "trajectories.jsonl")
        live = []
        for fp in sorted(glob.glob(mem_pat)):
            for line in open(fp, encoding="utf-8"):
                try:
                    live.append(json.loads(line))
                except Exception:
                    pass
        ctx_term = 0
        ctx_max = 0
        if live:
            lf = [r for r in live if r.get("is_final")] or live
            ctx_term = lf[-1].get("token_count", 0) or 0
            ctx_max = max((r.get("token_count", 0) or 0) for r in live)
        return final.get("messages", []), ctx_term, ctx_max, n_discard

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    for line in open(iters[0], encoding="utf-8"):
        item = json.loads(line)
        qid = str(item.get("filename"))
        full_msgs, ctx_term, ctx_max, n_discard = load_full_history(qid)
        # Prefer the untruncated history; fall back only if the log is missing.
        msgs = full_msgs if full_msgs else item.get("messages", [])
        turns, counts, disallowed = parse_trajectory(msgs)
        # Query batching: QUEST packs several queries into one search call.
        # Both are reported so Table 1 can use either without a rerun.
        n_search_queries = 0
        n_batched = 0
        search_queries = []
        for t in turns:
            for c in t["tool_calls"]:
                if c.get("tool") == "search":
                    q = c.get("args", {}).get("query", [])
                    q = q if isinstance(q, list) else [q]
                    n_search_queries += len(q)
                    if len(q) > 1:
                        n_batched += 1
                    search_queries.extend(str(x) for x in q)

        # docids surfaced by search, and whether the gold docs were among them
        retrieved = set()
        n_err_obs = 0
        n_obs = 0
        for t in turns:
            for c in t["tool_calls"]:
                obs = c.get("observation") or ""
                if c.get("tool") == "search" and obs:
                    n_obs += 1
                    if obs.startswith("[Tool Error]") or "No results found" in obs:
                        n_err_obs += 1
                    retrieved.update(re.findall(r"bm25://(\d+)", obs))
        gold_docids = GOLD.get(qid, [])
        qev = ev_by_q.get(qid, [])
        attempts = [e for e in qev if e["event"] == "tool_call_attempt"]
        executed = [e for e in attempts if e.get("parsed_ok")]
        n_search_exec = sum(1 for e in executed if e.get("tool") == "search")
        n_visit_exec = sum(1 for e in executed if e.get("tool") == "visit")

        # Turns that produced an executed tool call vs turns that did not.
        # A non-tool turn is the agent reasoning/answering instead of searching;
        # cap-hitting SFT questions show 0 of them (it never stops to answer).
        tool_turn_ids = {e.get("turn_id") for e in executed if e.get("turn_id") is not None}
        n_rounds = item.get("num_rounds", len(turns)) or 0
        n_tool_turns = len(tool_turn_ids)
        n_non_tool_turns = max(0, n_rounds - n_tool_turns)
        term_raw = (item.get("termination", "") or "").strip().lower()
        if n_rounds >= MAX_TURNS:
            termination_type = "cap"
        elif term_raw.startswith("answer not found"):
            termination_type = "answer_not_found"
        elif term_raw.startswith("answer"):
            termination_type = "answer"
        else:
            termination_type = term_raw or "unknown"
        repairs = [e for e in qev if e["event"] == "tool_call_repair"]
        halluc = [e for e in qev if e["event"] == "docid_hallucination"]
        dis_ev = [e for e in qev if e["event"] == "disallowed_tool"]

        ctx = ctx_term or count_tokens("\n".join(str(m.get("content", "")) for m in msgs), tok)
        pred = item.get("prediction", "") or ""
        am = re.search(r"<answer>(.*?)</answer>", pred, re.S)
        short = am.group(1).strip() if am else pred.strip()

        rec = {
            "system": args.system,
            "checkpoint": args.checkpoint,
            "question_id": qid,
            "question": item.get("question", ""),
            "gold_answer": item.get("answer", ""),
            "retriever": "bm25",
            "dataset": "browsecomp-plus",
            "trajectory": turns,
            "final_answer": short,
            "raw_final_output": pred,
            "extracted_short_answer": short,
            "is_correct": None,
            "judge_response": None,
            "judge_model": None,
            "total_turns": item.get("num_rounds", len(turns)),
            # Authoritative: one event is logged per EXECUTED tool call at
            # execution time. The transcript parser can miss calls whose
            # tool_response never materialised, so it is kept only as a
            # cross-check.
            "total_search_calls": n_search_exec,
            "total_open_calls": n_visit_exec,
            "total_search_calls_from_transcript": counts.get("search", 0),
            "total_open_calls_from_transcript": counts.get("visit", 0),
            # --- Table 1 candidate metrics: invocations vs queries ---
            "total_search_invocations": n_search_exec,
            "total_search_queries": n_search_queries,
            "avg_queries_per_invocation": round(n_search_queries / n_search_exec, 2) if n_search_exec else 0.0,
            "batched_calls_count": n_batched,
            # --- retrieval quality ---
            "retrieved_docids": sorted(retrieved),
            "gold_docids": sorted(str(d) for d in gold_docids),
            "gold_retrieved": bool(gold_docids and (set(map(str, gold_docids)) & retrieved)),
            "gold_recall": (len(set(map(str, gold_docids)) & retrieved) / len(gold_docids)) if gold_docids else None,
            "query_diversity": query_diversity(search_queries, len(retrieved)),
            "search_obs_error_or_empty": n_err_obs,
            "search_obs_total": n_obs,
            "total_find_calls": 0,
            "total_tool_calls": sum(counts.values()),
            # QUEST batches several queries into one search invocation. Table 1's
            # "Turns" counts INVOCATIONS (total_search_calls); the query count is
            # kept alongside because the two differ substantially.
            "total_search_queries": n_search_queries,
            "tool_turns": n_tool_turns,
            "non_tool_turns": n_non_tool_turns,
            "non_tool_turn_fraction": round(n_non_tool_turns / n_rounds, 4) if n_rounds else 0.0,
            "termination_type": termination_type,
            "context_management": "discard_all",
            "disallowed_tool_attempts": len(dis_ev),
            "disallowed_tools_called": sorted({e["tool"] for e in dis_ev}),
            "docid_hallucinations": len(halluc),
            "context_length_at_termination": ctx,
            "max_context_length": ctx_max or ctx,
            "discard_all_triggers": n_discard,
            "trajectory_source": "trajectories_no_memory.jsonl" if full_msgs else "iter.jsonl (truncated fallback)",
            "terminated_reason": item.get("termination", ""),
            "total_tool_call_attempts": len(attempts),
            "total_valid_tool_calls": sum(1 for a in attempts if a.get("parsed_ok")),
            "tool_call_repair_count": len(repairs),
            "tool_call_repair_rate": (len(repairs) / len(attempts)) if attempts else 0.0,
            "IMPORTANT_FOR_PAPER": block_for(args.system),
            "serp_num_values_requested": [],
            "search_query_batches": counts.get("search", 0),
        }
        records.append(rec)
        with open(os.path.join(args.out_dir, f"{qid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    n = len(records)
    # Table 1 denominator is the full dataset, not just what has finished.
    denom = args.denominator if args.denominator else n
    summary = {
        "IMPORTANT_FOR_PAPER": block_for(args.system),
        "system": args.system,
        "checkpoint": args.checkpoint,
        "n_questions": n,
        "denominator_for_table1": denom,
        # === Table 1 metrics ===
        # Turns = sum(total_search_calls) / denominator  (SEARCH invocations only)
        "TABLE1_Turns": round(sum(r["total_search_calls"] for r in records) / denom, 2),
        # SR is filled in by judge_run.py as correct_count / denominator * 100
        "TABLE1_SR_percent": None,
        "TABLE1_Turns_search_invocations": round(sum(r["total_search_invocations"] for r in records) / denom, 2),
        "TABLE1_Turns_search_queries": round(sum(r["total_search_queries"] for r in records) / denom, 2),
        "avg_queries_per_invocation": round(
            sum(r["total_search_queries"] for r in records) / max(1, sum(r["total_search_invocations"] for r in records)), 2),
        "batched_calls_total": sum(r["batched_calls_count"] for r in records),
        "pct_invocations_batched": round(
            100.0 * sum(r["batched_calls_count"] for r in records) / max(1, sum(r["total_search_invocations"] for r in records)), 1),
        "TABLE1_non_tool_turn_fraction": round(
            sum(r["non_tool_turns"] for r in records) / max(1, sum(r["total_turns"] for r in records)), 4),
        "termination_type_counts": dict(collections.Counter(r["termination_type"] for r in records)),
        "pct_hit_turn_cap": round(100.0 * sum(1 for r in records if r["termination_type"] == "cap") / max(1, len(records)), 1),
        "non_tool_turn_fraction_cap_hitters": round(
            (sum(r["non_tool_turns"] for r in records if r["termination_type"] == "cap")
             / max(1, sum(r["total_turns"] for r in records if r["termination_type"] == "cap"))), 4),
        "non_tool_turn_fraction_natural": round(
            (sum(r["non_tool_turns"] for r in records if r["termination_type"] != "cap")
             / max(1, sum(r["total_turns"] for r in records if r["termination_type"] != "cap"))), 4),
        "TABLE1_query_duplicate_rate": round(
            sum(r["query_diversity"]["exact_duplicates"] + r["query_diversity"]["near_duplicates_jaccard_08"] for r in records)
            / max(1, sum(r["query_diversity"]["total_queries"] for r in records)), 4),
        "TABLE1_unique_docs_per_question": round(
            sum(r["query_diversity"]["unique_docs_retrieved"] for r in records) / max(1, len(records)), 1),
        "TABLE1_gold_retrieval_rate": round(
            100.0 * sum(1 for r in records if r["gold_retrieved"]) / max(1, len(records)), 1),
        "gold_retrieved_pct": round(100.0 * sum(1 for r in records if r["gold_retrieved"]) / max(1, len(records)), 1),
        "avg_gold_recall": round(sum(r["gold_recall"] or 0 for r in records) / max(1, len(records)), 3),
        "search_obs_error_or_empty_total": sum(r["search_obs_error_or_empty"] for r in records),
        "total_search_calls": sum(r["total_search_calls"] for r in records),
        "total_search_queries": sum(r["total_search_queries"] for r in records),
        "avg_search_queries_per_question": round(sum(r["total_search_queries"] for r in records) / denom, 2),
        # kept for analysis, NOT the Table 1 "Turns" figure
        "avg_agent_turns_all_tools": sum(r["total_turns"] for r in records) / n,
        "avg_context_at_termination": sum(r["context_length_at_termination"] for r in records) / n,
        "max_context_seen": max(r["max_context_length"] for r in records),
        "total_discard_all_triggers": sum(r["discard_all_triggers"] for r in records),
        "trajectory_sources": dict(collections.Counter(r["trajectory_source"] for r in records)),
        "total_tool_call_attempts": sum(r["total_tool_call_attempts"] for r in records),
        "total_repairs": sum(r["tool_call_repair_count"] for r in records),
        "overall_repair_rate": (
            sum(r["tool_call_repair_count"] for r in records)
            / max(1, sum(r["total_tool_call_attempts"] for r in records))
        ),
        "pct_questions_with_repair": 100.0 * sum(1 for r in records if r["tool_call_repair_count"]) / n,
        "total_disallowed_attempts": sum(r["disallowed_tool_attempts"] for r in records),
        "pct_questions_with_disallowed": 100.0 * sum(1 for r in records if r["disallowed_tool_attempts"]) / n,
        "total_docid_hallucinations": sum(r["docid_hallucinations"] for r in records),
        "avg_search_calls": sum(r["total_search_calls"] for r in records) / n,
        "avg_visit_calls": sum(r["total_open_calls"] for r in records) / n,
        "termination_reasons": dict(collections.Counter(r["terminated_reason"] for r in records)),
    }
    with open(os.path.join(args.out_dir, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
