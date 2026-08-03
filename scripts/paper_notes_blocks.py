"""IMPORTANT_FOR_PAPER blocks stamped into every trajectory and results file.

Facts here were verified against the released code/model cards, not assumed.
See paper_notes/baseline_comparison.md for provenance.
"""

QUEST_BLOCK = {
    "bcp_baseline_exists": True,
    "published_number": "48.2 (avg@3) — QUEST-30B-RL, BrowseComp-Plus",
    "published_number_source": "osunlp/QUEST-30B-RL model card results table (not the GitHub README, which has no results table)",
    "published_number_applies_to": "RL checkpoint only; no published BrowseComp-Plus number was found for the Base or MT+SFT checkpoints",
    "published_eval_config_VERIFIED": {
        "dataset": "130-question subset (browsecomp_plus_quest_130.jsonl), NOT the full 830",
        "retriever": "Qwen3-Embedding-8B FAISS dense, top-k=5, 512-token snippets",
        "visit_tool": "DISABLED (search-only fair comparison)",
        "context_management": "discard_all @ 80,000 tokens",
        "rollouts": "3 (metric is avg@3)",
    },
    "our_eval_config": {
        "dataset": "full 830 questions",
        "retriever": "BM25 (Lucene), top-k=15, 512-token snippets",
        "visit_tool": "ENABLED on bm25://<docid>",
        "context_management": "discard_all @ 30,000 (Base) / 80,000 (SFT, RL)",
        "rollouts": "1",
        "base_serving": "non-thinking mode, native 40,960 window, no rope-scaling",
    },
    "why_not_comparable": (
        "Different retriever (BM25 top-15 vs Qwen3-Embedding-8B dense top-5), different "
        "question set (830 vs a 130-question subset), visit enabled vs disabled, and 1 "
        "rollout vs avg@3. Context management is the SAME (discard-all) in both. "
        "Cross-checkpoint comparison Base/SFT/RL within our setup remains valid because "
        "all three are run under identical conditions."
    ),
    "corrections_to_earlier_assumptions": [
        "The published number is NOT on the full set — their released BrowseComp-Plus input is a 130-question subset.",
        "The retriever config is NOT unknown — it is specified in inference/scripts/run_react_infer_bcp.sh as Qwen3-Embedding-8B FAISS, k=5.",
        "They did NOT use the context condenser for BrowseComp-Plus — their script sets MEMORY_STRATEGY=discard_all, the same strategy we use.",
    ],
    "do_not_claim_reproduction": True,
}

ORESEARCHER_BLOCK = {
    "bcp_baseline_exists": False,
    "statement": (
        "O-Researcher-72B has no published BrowseComp-Plus results. "
        "All numbers here are new measurements."
    ),
    "oresearcher_published_bcp": "None",
    "paper_framing": (
        "We present the first evaluation of O-Researcher-72B on "
        "BrowseComp-Plus under offline BM25."
    ),
    "adaptation_note": (
        "O-Researcher ships live-web tools only (Serper web_search + crawl_page). "
        "We replaced both with local BM25-backed HTTP services matching their "
        "training-data observation formats."
    ),
    "do_not_claim_reproduction": True,
}


def block_for(system: str):
    s = (system or "").lower()
    if "quest" in s:
        return QUEST_BLOCK
    if "researcher" in s:
        return ORESEARCHER_BLOCK
    return None
