"""Download model weights, skipping any already complete locally.

Completeness is checked against each repo's safetensors index, so a partial or
interrupted download is detected and resumed rather than silently accepted.
"""
import json
import os
import sys

from huggingface_hub import snapshot_download

MODELS = [
    ("Qwen/Qwen3-30B-A3B",            "models/Qwen3-30B-A3B",            "Base row (instruction-tuned)"),
    ("osunlp/QUEST-30B-MT-Plus-SFT",  "models/QUEST-30B-MT-Plus-SFT",    "SFT row"),
    ("osunlp/QUEST-30B-RL",           "models/QUEST-30B-RL",             "RL row"),
    ("Qwen/Qwen3-32B",                "models/Qwen3-32B",                "LLM judge"),
]


def is_complete(local_dir):
    """True only if every shard named in the index is present on disk."""
    idx = os.path.join(local_dir, "model.safetensors.index.json")
    cfg = os.path.join(local_dir, "config.json")
    if not (os.path.exists(idx) and os.path.exists(cfg)):
        return False
    try:
        need = set(json.load(open(idx))["weight_map"].values())
    except Exception:
        return False
    have = {f for f in os.listdir(local_dir) if f.endswith(".safetensors")}
    return need.issubset(have)


def main():
    for repo, local, role in MODELS:
        if is_complete(local):
            size = sum(os.path.getsize(os.path.join(local, f)) for f in os.listdir(local)
                       if f.endswith(".safetensors")) / 1e9
            print(f"    ok   {role:<32} {repo}  ({size:.0f} GB, already complete)")
            continue
        print(f"    ..   {role:<32} {repo}  downloading -> {local}")
        snapshot_download(repo, local_dir=local, max_workers=8)
        if is_complete(local):
            print(f"    ok   {role:<32} done")
        else:
            print(f"    FAIL {role:<32} incomplete after download", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
