"""BM25 retrieval ceiling for BrowseComp-Plus.

For each question, query BM25 with the gold ANSWER string (an oracle query the
agent could never know in advance) and record whether a gold document appears
in the top-15. This is the maximum gold-retrieval rate achievable with perfect
query formulation at k=15, and therefore the ceiling that agent behaviour is
measured against.

Read-only against the Lucene index; safe to run beside live agent runs.
"""
import json
import os

from pyserini.search.lucene import LuceneSearcher

INDEX = "data/browsecomp_plus/indexes/bm25"
SRC = "data/browsecomp_plus/browsecomp_plus_decrypted.jsonl"
OUT = "results/oracle_bm25_ceiling.json"
K = 15

searcher = LuceneSearcher(INDEX)
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
print(f"oracle pass over {len(rows)} questions, k={K}")

per_q = []
for i, r in enumerate(rows, 1):
    qid = str(r["query_id"])
    gold = {str(d["docid"]) for d in (r.get("gold_docs") or [])}
    oracle_query = str(r["answer"])
    hits = searcher.search(oracle_query, K)
    rank = None
    score = None
    for pos, h in enumerate(hits, 1):
        if h.docid in gold:
            rank = pos
            score = float(h.score)
            break
    per_q.append({
        "question_id": qid,
        "gold_answer": r["answer"],
        "oracle_query": oracle_query,
        "gold_in_top15": rank is not None,
        "gold_rank": rank,
        "bm25_score": score,
        "n_gold_docs": len(gold),
    })
    if i % 100 == 0:
        print(f"  {i}/{len(rows)}", flush=True)

found = [p for p in per_q if p["gold_in_top15"]]
agg = {
    "total_questions": len(per_q),
    "k": K,
    "gold_retrievable_at_15": len(found),
    "gold_retrievable_rate": round(100.0 * len(found) / len(per_q), 1),
    "mean_gold_rank_when_found": round(sum(p["gold_rank"] for p in found) / len(found), 2) if found else None,
    "median_gold_rank_when_found": (sorted(p["gold_rank"] for p in found)[len(found) // 2] if found else None),
    "note": (
        "Oracle query is the GOLD ANSWER string, which the agent cannot know in "
        "advance. This is an upper bound on gold retrieval at k=15, not an "
        "achievable target."
    ),
}
os.makedirs("results", exist_ok=True)
json.dump({"aggregate": agg, "per_question": per_q},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(agg, indent=2))
print(f"-> {OUT}")
