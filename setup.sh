#!/usr/bin/env bash
# One-shot setup for the QUEST BrowseComp-Plus / BM25 evaluation.
#
# Idempotent: every step checks for existing local assets and skips work that
# is already done. Safe to re-run after an interruption.
#
#   ./setup.sh              # env + code + data + all model weights (~230 GB)
#   ./setup.sh --no-models  # env + code + data only (~3 GB)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SKIP_MODELS=0
[[ "${1:-}" == "--no-models" ]] && SKIP_MODELS=1

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()  { printf '    \033[32mok\033[0m %s\n' "$*"; }
warn(){ printf '    \033[33m!!\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- 1. python
say "Python environment"
if [[ -x venv/bin/python ]]; then
    ok "venv exists ($(venv/bin/python --version 2>&1))"
else
    python3 -m venv venv
    ok "created venv"
fi
if venv/bin/python -c "import vllm, pyserini, json5" 2>/dev/null; then
    ok "dependencies already installed"
else
    say "Installing dependencies (several minutes)"
    venv/bin/pip install -q --upgrade pip
    venv/bin/pip install -q vllm pyserini json5 openai transformers datasets \
        "huggingface_hub" litellm "qwen-agent==0.0.26" tiktoken pandas numpy tqdm
    ok "installed"
fi

# ------------------------------------------------------------------- 2. java
say "Java (pyserini needs a JVM)"
if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    ok "JAVA_HOME=$JAVA_HOME"
else
    JH=$(dirname "$(dirname "$(readlink -f "$(command -v java 2>/dev/null)" 2>/dev/null)")" 2>/dev/null || true)
    if [[ -n "$JH" && -x "$JH/bin/java" ]]; then
        echo "export JAVA_HOME=$JH" >> "$ROOT/env.sh"
        ok "found java at $JH (written to env.sh)"
    else
        warn "no java found - install a JRE 21 and set JAVA_HOME, else BM25 will fail"
    fi
fi

# ------------------------------------------------------------- 3. QUEST code
say "QUEST scaffold (pinned commit + our offline-BM25 patch)"
QUEST_COMMIT=225686f6967ac46915df797a6d2ef006e0ec2839
if [[ -d systems/QUEST/.git ]]; then
    ok "already cloned"
else
    mkdir -p systems
    git clone https://github.com/OSU-NLP-Group/QUEST.git systems/QUEST
    git -C systems/QUEST checkout -q "$QUEST_COMMIT"
    ok "cloned at $QUEST_COMMIT"
fi
if grep -q "BCP_OFFLINE" systems/QUEST/inference/react_agent.py 2>/dev/null; then
    ok "patch already applied"
else
    git -C systems/QUEST apply "$ROOT/patches/react_agent.patch"
    ok "applied react_agent.patch"
fi
cp -n patches/bcp_offline.py systems/QUEST/inference/ 2>/dev/null || true
ok "bcp_offline.py in place"
cp -n scripts/run_quest_bm25.sh scripts/*.py scripts/disk_watch.sh . 2>/dev/null || true
chmod +x run_quest_bm25.sh disk_watch.sh 2>/dev/null || true

# ------------------------------------------------------------------ 4. data
say "BrowseComp-Plus data"
mkdir -p data/browsecomp_plus
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"

if [[ -s data/browsecomp_plus/browsecomp_plus_830.jsonl ]]; then
    ok "830-question input exists ($(wc -l < data/browsecomp_plus/browsecomp_plus_830.jsonl) lines)"
else
    [[ -d BrowseComp-Plus ]] || git clone -q https://github.com/texttron/BrowseComp-Plus.git
    venv/bin/python BrowseComp-Plus/scripts_build_index/decrypt_dataset.py \
        --output data/browsecomp_plus/browsecomp_plus_decrypted.jsonl \
        --generate-tsv data/browsecomp_plus/queries.tsv
    venv/bin/python - <<'PY'
import json
inp="data/browsecomp_plus/browsecomp_plus_decrypted.jsonl"
out="data/browsecomp_plus/browsecomp_plus_830.jsonl"
n=0
with open(inp) as f, open(out,"w") as w:
    for line in f:
        r=json.loads(line)
        json.dump({"question":r["query"],"answer":r["answer"],"filename":str(r["query_id"])},w,ensure_ascii=False)
        w.write("\n"); n+=1
print(f"    wrote {n} questions")
PY
    ok "decrypted + built 830-question input"
fi

if [[ -f data/browsecomp_plus/indexes/bm25/segments_3 ]]; then
    ok "BM25 Lucene index exists ($(du -sh data/browsecomp_plus/indexes/bm25 | cut -f1))"
else
    venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Tevatron/browsecomp-plus-indexes", repo_type="dataset",
                  allow_patterns=["bm25/*"], local_dir="data/browsecomp_plus/indexes")
print("    BM25 index downloaded")
PY
    ok "BM25 index ready"
fi

# ---------------------------------------------------------------- 5. models
if [[ $SKIP_MODELS -eq 1 ]]; then
    warn "skipping model download (--no-models)"
else
    say "Model weights (~230 GB total; each checked before download)"
    venv/bin/python scripts/download_models.py
fi

# ------------------------------------------------------------------ 6. check
say "Verification"
venv/bin/python scripts/verify_setup.py

cat <<'EOF'

==> Setup complete. Next:

    ./run_all.sh            # serve models + launch all remaining runs
    ./run_all.sh --status   # progress of any in-flight runs

EOF
