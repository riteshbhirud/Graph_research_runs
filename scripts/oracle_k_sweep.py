"""BM25 oracle ceiling as a function of top-k.

Tells us how much retrieval headroom is lost moving from top-15 to top-5,
which determines whether a top-5 primary run is comparable to the top-15 runs.
Read-only; safe beside live runs.
"""
import json
import os

from pyserini.search.lucene import LuceneSearcher

INDEX = "data/browsecomp_plus/indexes/bm25"
SRC = "data/browsecomp_plus/browsecomp_plus_decrypted.jsonl"
OUT = "results/oracle_k_sweep.json"
KS = [1, 3, 5, 10, 15]

searcher = LuceneSearcher(INDEX)
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
maxk = max(KS)

hits_at = {k: 0 for k in KS}
recall_at = {k: 0.0 for k in KS}
for i, r in enumerate(rows, 1):
    gold = {str(d["docid"]) for d in (r.get("gold_docs") or [])}
    if not gold:
        continue
    hits = searcher.search(str(r["answer"]), maxk)
    ids = [h.docid for h in hits]
    for k in KS:
        top = set(ids[:k])
        if gold & top:
            hits_at[k] += 1
        recall_at[k] += len(gold & top) / len(gold)
    if i % 200 == 0:
        print(f"  {i}/{len(rows)}", flush=True)

n = len(rows)
agg = {
    "total_questions": n,
    "gold_hit_rate_by_k": {str(k): round(100.0 * hits_at[k] / n, 1) for k in KS},
    "mean_gold_recall_by_k": {str(k): round(recall_at[k] / n, 3) for k in KS},
    "headroom_lost_15_to_5_pp": round(100.0 * (hits_at[15] - hits_at[5]) / n, 1),
    "note": "Oracle query = gold answer string. Upper bound on the search component only.",
}
os.makedirs("results", exist_ok=True)
json.dump(agg, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(agg, indent=2))
