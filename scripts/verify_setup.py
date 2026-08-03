"""Verify every prerequisite before a multi-day run is launched.

Checks the things that actually broke during development: missing JVM, BM25
index unreadable, wrong question count, missing scaffold patch, proxy without
no_proxy, and incomplete weights.
"""
import glob
import json
import os
import sys

FAIL = []
WARN = []


def ok(msg):
    print(f"    \033[32mok\033[0m {msg}")


def bad(msg):
    print(f"    \033[31mFAIL\033[0m {msg}")
    FAIL.append(msg)


def warn(msg):
    print(f"    \033[33m!!\033[0m {msg}")
    WARN.append(msg)


# --- dataset ---------------------------------------------------------------
p = "data/browsecomp_plus/browsecomp_plus_830.jsonl"
if os.path.exists(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    if len(rows) == 830 and {"question", "answer", "filename"} == set(rows[0]):
        ok(f"dataset: 830 questions, correct schema")
    else:
        bad(f"dataset: {len(rows)} rows / keys {sorted(rows[0])} (expected 830 and question/answer/filename)")
else:
    bad("dataset missing - run setup.sh")

# --- BM25 index ------------------------------------------------------------
os.environ.setdefault("JAVA_HOME", os.environ.get("JAVA_HOME", ""))
try:
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher("data/browsecomp_plus/indexes/bm25")
    n = s.num_docs
    hits = s.search("Queen Arwa University Yemen", 5)
    if n == 100195 and hits:
        ok(f"BM25 index: {n:,} docs, test query returned {len(hits)} hits")
    else:
        warn(f"BM25 index: {n:,} docs (expected 100,195), {len(hits)} hits")
except Exception as e:
    bad(f"BM25/pyserini unusable: {type(e).__name__}: {str(e)[:120]} "
        f"(JAVA_HOME={os.environ.get('JAVA_HOME') or 'UNSET'})")

# --- scaffold patch --------------------------------------------------------
ra = "systems/QUEST/inference/react_agent.py"
if os.path.exists(ra):
    src = open(ra, encoding="utf-8").read()
    checks = {
        "BCP_OFFLINE flag": "BCP_OFFLINE" in src,
        "disallowed-tool handling": "disallowed_tool_message" in src,
        "bm25 visit routing": "visit_bm25" in src,
        "tool-call repair": "repair_tool_call" in src,
        "non-thinking switch": "QUEST_ENABLE_THINKING" in src,
    }
    for k, v in checks.items():
        ok(f"patch: {k}") if v else bad(f"patch missing: {k}")
else:
    bad("systems/QUEST not present - run setup.sh")

if os.path.exists("systems/QUEST/inference/bcp_offline.py"):
    ok("bcp_offline.py present")
else:
    bad("bcp_offline.py missing")

# --- weights ---------------------------------------------------------------
for name, d in [("Base", "models/Qwen3-30B-A3B"), ("SFT", "models/QUEST-30B-MT-Plus-SFT"),
                ("RL", "models/QUEST-30B-RL"), ("judge", "models/Qwen3-32B")]:
    idx = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx):
        warn(f"weights {name}: not downloaded ({d})")
        continue
    need = set(json.load(open(idx))["weight_map"].values())
    have = {os.path.basename(f) for f in glob.glob(os.path.join(d, "*.safetensors"))}
    if need <= have:
        ok(f"weights {name}: {len(have)} shards complete")
    else:
        bad(f"weights {name}: missing {len(need - have)} shards")

# --- proxy -----------------------------------------------------------------
if os.environ.get("http_proxy") or os.environ.get("https_proxy"):
    npx = (os.environ.get("no_proxy", "") + os.environ.get("NO_PROXY", "")).lower()
    if "localhost" in npx or "127.0.0.1" in npx:
        ok("proxy set, localhost excluded")
    else:
        warn("http_proxy set WITHOUT no_proxy=localhost - vLLM calls will fail "
             "(run_quest_bm25.sh exports this itself, so runs are fine)")
else:
    ok("no proxy configured")

# --- gpus ------------------------------------------------------------------
try:
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    for i, line in enumerate(out.splitlines()):
        name, mem = [x.strip() for x in line.split(",")]
        gb = int(mem.split()[0]) / 1024
        note = "fits 30B on one GPU" if gb >= 70 else "NEEDS tensor-parallel (30B is ~57 GB)"
        print(f"    gpu{i}: {name}  {gb:.0f} GB  - {note}")
except Exception as e:
    warn(f"nvidia-smi unavailable: {e}")

print()
if FAIL:
    print(f"\033[31m{len(FAIL)} blocking problem(s) - fix before running.\033[0m")
    sys.exit(1)
print(f"\033[32mAll critical checks passed.\033[0m" + (f" ({len(WARN)} warning(s))" if WARN else ""))
