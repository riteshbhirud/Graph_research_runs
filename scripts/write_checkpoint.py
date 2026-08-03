"""Write a per-condition progress checkpoint.

QUEST's own durable resume state is the appended iter*.jsonl under each
OUTPUT_PATH, so conditions with different OUTPUT_PATHs already resume
independently. This just snapshots that state into a readable checkpoint file.
"""
import argparse
import glob
import json
import os


def snapshot(run_dir, name, total=830):
    iters = [p for p in glob.glob(os.path.join(run_dir, "*", "*", "iter*.jsonl"))
             if os.sep + "logs" + os.sep not in p]
    done_ids, terms = [], {}
    for p in iters:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done_ids.append(str(r.get("filename")))
            t = r.get("termination", "")
            terms[t] = terms.get(t, 0) + 1
    uniq = sorted(set(done_ids))

    def count(fn):
        p = os.path.join(run_dir, fn)
        return sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0

    return {
        "condition": name,
        "run_dir": run_dir,
        "completed": len(uniq),
        "total": total,
        "pct": round(100.0 * len(uniq) / total, 1),
        "remaining": total - len(uniq),
        "completed_question_ids": uniq,
        "termination_reasons": terms,
        "tool_events": count("events.jsonl"),
        "multi_block_events": count("multi_block_events.jsonl"),
        "visit_cap_events": sum(count(os.path.basename(f))
                                for f in glob.glob(os.path.join(run_dir, "visit_cap_events_*.jsonl"))),
        "resume": "rerun the same script with the same OUTPUT_PATH; completed question_ids are skipped",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=830)
    a = ap.parse_args()
    s = snapshot(a.run_dir, a.name, a.total)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(s, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{s['condition']}: {s['completed']}/{s['total']} ({s['pct']}%)  "
          f"tool_events={s['tool_events']} cap_events={s['visit_cap_events']}")
