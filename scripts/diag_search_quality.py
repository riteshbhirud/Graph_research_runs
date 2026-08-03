"""Confirmation-2 diagnostics: is 13.4 search calls/question a real signal?

Checks for inflation (errors/retries), gold retrieval rate, termination mode,
and query duplication. Reads saved data only — safe while runs continue.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/chk3"
recs = [json.load(open(f, encoding="utf-8"))
        for f in sorted(glob.glob(os.path.join(d, "*.json")))
        if not os.path.basename(f).startswith("_")]
print(f"analysed {len(recs)} completed questions\n")

# --- 1. inflation check -------------------------------------------------
inv = sum(r["total_search_invocations"] for r in recs)
errs = sum(r["search_obs_error_or_empty"] for r in recs)
obs = sum(r["search_obs_total"] for r in recs)
print("1. INFLATION CHECK")
print(f"   search invocations (executed) : {inv}")
print(f"   per question                  : {inv/len(recs):.2f}")
print(f"   search observations recorded  : {obs}")
print(f"   of which error/no-result      : {errs} ({100*errs/max(1,obs):.1f}%)")
print("   note: API retries are NOT counted - events are logged per executed tool call\n")

# --- 2. gold retrieval --------------------------------------------------
withgold = [r for r in recs if r["gold_docids"]]
hit = [r for r in withgold if r["gold_retrieved"]]
print("2. GOLD RETRIEVAL")
print(f"   questions with gold docs listed: {len(withgold)}")
print(f"   gold doc surfaced >=1 time     : {len(hit)} ({100*len(hit)/max(1,len(withgold)):.1f}%)")
if withgold:
    mr = sum(r["gold_recall"] or 0 for r in withgold) / len(withgold)
    print(f"   mean gold recall (frac of gold docs seen): {mr:.3f}")
    uniq = sum(len(r["retrieved_docids"]) for r in recs) / len(recs)
    print(f"   unique docids surfaced per question      : {uniq:.1f}")
print()

# --- 3. termination mode ------------------------------------------------
print("3. TERMINATION MODE")
disc = sum(1 for r in recs if r["discard_all_triggers"] > 0)
print(f"   questions where discard-all fired >=1x: {disc} ({100*disc/len(recs):.1f}%)")
print(f"   mean discard-all triggers per question : {sum(r['discard_all_triggers'] for r in recs)/len(recs):.2f}")
print(f"   termination reasons: {dict(Counter(r['terminated_reason'] for r in recs))}")
print(f"   hit 100-turn cap    : {sum(1 for r in recs if r['total_turns'] >= 100)}")
print()

# --- 4. batching --------------------------------------------------------
q = sum(r["total_search_queries"] for r in recs)
b = sum(r["batched_calls_count"] for r in recs)
print("4. BATCHING")
print(f"   total_search_invocations : {inv}   (per question {inv/len(recs):.2f})")
print(f"   total_search_queries     : {q}   (per question {q/len(recs):.2f})")
print(f"   avg queries per invocation: {q/max(1,inv):.2f}")
print(f"   invocations that were batched (>1 query): {b} ({100*b/max(1,inv):.1f}%)")
print()

# --- 5. query diversity -------------------------------------------------
def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())


def toks(s):
    return set(norm(s).split())


exact_dup = near_dup = total_q = 0
per_q_dup = []
for r in recs:
    seen = []
    dq = 0
    for t in r["trajectory"]:
        for c in t["tool_calls"]:
            if c.get("tool") != "search":
                continue
            qs = c.get("args", {}).get("query", [])
            qs = qs if isinstance(qs, list) else [qs]
            for x in qs:
                total_q += 1
                nx, tx = norm(x).strip(), toks(x)
                is_dup = False
                for ps, pt in seen:
                    if nx == ps:
                        exact_dup += 1
                        is_dup = True
                        break
                    if tx and pt:
                        j = len(tx & pt) / len(tx | pt)
                        if j >= 0.8:
                            near_dup += 1
                            is_dup = True
                            break
                if is_dup:
                    dq += 1
                seen.append((nx, tx))
    per_q_dup.append(100 * dq / max(1, sum(1 for t in r["trajectory"] for c in t["tool_calls"] if c.get("tool") == "search")))
print("5. QUERY DIVERSITY (duplicate = exact match or Jaccard>=0.8 vs earlier query in SAME trajectory)")
print(f"   total queries examined : {total_q}")
print(f"   exact duplicates       : {exact_dup} ({100*exact_dup/max(1,total_q):.1f}%)")
print(f"   near duplicates        : {near_dup} ({100*near_dup/max(1,total_q):.1f}%)")
print(f"   COMBINED duplicate rate: {100*(exact_dup+near_dup)/max(1,total_q):.1f}%")
