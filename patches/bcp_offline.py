"""
Offline BrowseComp-Plus adaptation for the QUEST ReAct scaffold.

This module supplies the two behaviours the upstream scaffold does not have
when running fully offline against a Lucene BM25 index:

1. ``visit`` on ``bm25://<docid>`` URLs. Upstream ``visit`` only knows how to
   fetch live web pages through Jina. Here it resolves a docid against the
   BrowseComp-Plus Lucene index and returns the stored document text, so
   search stays discovery (snippets) and visit stays deep reading (full text).

2. Disallowed-tool handling. QUEST SFT/RL checkpoints were trained with tools
   that are not part of this evaluation (google_scholar, PythonInterpreter).
   Calls to them get a clean, uniform error instead of a scaffold traceback,
   and every attempt is logged so the rate can be reported per checkpoint.

Every event is appended to ``BCP_EVENT_LOG`` as JSONL for later analysis.
"""

import json
import os
import threading
import time

BM25_INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "")
# Full document text is returned for visit/deep-reading. This cap only guards
# against pathological outliers blowing up the context window; it is far above
# the corpus median so ordinary documents are returned whole.
VISIT_MAX_TOKENS = int(os.environ.get("BCP_VISIT_MAX_TOKENS", "8192"))
EVENT_LOG = os.environ.get("BCP_EVENT_LOG", "")

# Tools present in the checkpoints' training distribution but not part of this
# offline evaluation. Anything outside search/visit lands here.
ALLOWED_TOOLS = {"search", "visit"}

_searcher = None
_searcher_lock = threading.Lock()
_tokenizer = None
_tokenizer_lock = threading.Lock()
_log_lock = threading.Lock()


def _get_searcher():
    global _searcher
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                from pyserini.search.lucene import LuceneSearcher

                _searcher = LuceneSearcher(BM25_INDEX_PATH)
                print(f"[bcp_offline] Lucene searcher initialized: {BM25_INDEX_PATH}")
    return _searcher


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer

                _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _tokenizer


def log_event(event: dict):
    """Append one structured event to the run's event log."""
    if not EVENT_LOG:
        return
    event = dict(event)
    event.setdefault("ts", time.time())
    event.setdefault("thread", threading.current_thread().name)
    try:
        with _log_lock:
            os.makedirs(os.path.dirname(EVENT_LOG), exist_ok=True)
            with open(EVENT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:  # logging must never break a run
        print(f"[bcp_offline] event log write failed: {e}")


def normalize_docid(url: str) -> str:
    """Accept bm25://<docid>, corpus URLs, or a bare docid."""
    u = str(url).strip()
    for prefix in ("bm25://", "http://corpus.local/", "https://corpus.local/"):
        if u.startswith(prefix):
            return u[len(prefix):].strip().strip("/")
    return u


def get_doc_text(docid: str):
    """Return stored document text for a docid, or None if absent."""
    try:
        doc = _get_searcher().doc(str(docid))
    except Exception as e:
        print(f"[bcp_offline] lucene lookup error for {docid!r}: {e}")
        return None
    if doc is None:
        return None
    raw = doc.raw()
    if not raw:
        return None
    try:
        return json.loads(raw).get("contents", "")
    except Exception:
        return raw


def _truncate(text: str) -> str:
    tok = _get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= VISIT_MAX_TOKENS:
        return text
    return tok.decode(ids[:VISIT_MAX_TOKENS], skip_special_tokens=True) + "\n\n[... document truncated ...]"


def visit_bm25(url, goal: str, filename: str = "", turn_id: int = -1) -> str:
    """Offline replacement for the visit tool over bm25://<docid> URLs.

    Condition B (capped) limits the TOTAL tokens returned by one visit call to
    VISIT_TOTAL_TOKEN_CAP. Documents are processed in the order requested and
    the call stops before the first document that would exceed the cap.
    Condition A (uncapped, cap=0) returns every requested document.
    """
    urls = url if isinstance(url, list) else [url]
    blocks = []
    tokens_returned = 0
    docids_processed = 0
    capped = False
    for u in urls:
        docid = normalize_docid(u)
        text = get_doc_text(docid)
        if text is not None and VISIT_TOTAL_TOKEN_CAP > 0:
            n_tok = len(_get_tokenizer().encode(_truncate(text), add_special_tokens=False))
            if tokens_returned and tokens_returned + n_tok > VISIT_TOTAL_TOKEN_CAP:
                capped = True
                break
            tokens_returned += n_tok
        if text is None:
            log_event({
                "event": "docid_hallucination",
                "filename": filename,
                "requested_url": str(u),
                "docid": docid,
            })
            blocks.append(
                f"The useful information in {u} for user goal {goal} as follows:\n\n"
                f"Evidence in page:\nDocument not found in offline corpus.\n\n"
                f"Summary:\nDocument not found in offline corpus. "
                f"Only bm25://<docid> URLs returned by the search tool can be visited.\n"
            )
            continue
        log_event({
            "event": "visit",
            "filename": filename,
            "docid": docid,
        })
        docids_processed += 1
        blocks.append(
            f"The useful information in bm25://{docid} for user goal {goal} as follows:\n\n"
            f"Evidence in page:\n{_truncate(text)}\n\n"
            f"Summary:\nFull text of document {docid} is provided above.\n"
        )

    out = "\n=======\n".join(blocks).strip()
    if capped:
        out += (
            f"\n\nNote: visit response capped at {VISIT_TOTAL_TOKEN_CAP} tokens. "
            f"{docids_processed} of {len(urls)} requested documents returned."
        )
    if VISIT_TOTAL_TOKEN_CAP > 0:
        log_visit_cap({
            "question_id": filename,
            "turn_id": turn_id,
            "checkpoint": CHECKPOINT,
            "docids_requested": len(urls),
            "docids_processed": docids_processed,
            "tokens_returned": tokens_returned,
            "capped": capped,
        })
    return out


CHECKPOINT = os.environ.get("BCP_CHECKPOINT", "")
MULTI_BLOCK_LOG = os.environ.get("BCP_MULTI_BLOCK_LOG", "")
# Condition B: total tokens returned by a single visit call. 0 = uncapped (A).
VISIT_TOTAL_TOKEN_CAP = int(os.environ.get("BCP_VISIT_TOTAL_TOKEN_CAP", "0"))
VISIT_CAP_LOG = os.environ.get("BCP_VISIT_CAP_LOG", "")
_mb_lock = threading.Lock()
_vc_lock = threading.Lock()


def log_visit_cap(record: dict):
    """One record per visit call in the capped condition."""
    if not VISIT_CAP_LOG:
        return
    record = dict(record)
    record.setdefault("ts", time.time())
    try:
        with _vc_lock:
            os.makedirs(os.path.dirname(VISIT_CAP_LOG), exist_ok=True)
            with open(VISIT_CAP_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[bcp_offline] visit-cap log write failed: {e}")


def log_multi_block(record: dict):
    """Record turns where the model emitted >1 <tool_call> block.

    QUEST's scaffold executes only matches[-1]; we keep that behaviour and log
    what was discarded so the cost of multi-block emission is measurable.
    """
    if not MULTI_BLOCK_LOG:
        return
    record = dict(record)
    record.setdefault("ts", time.time())
    try:
        with _mb_lock:
            os.makedirs(os.path.dirname(MULTI_BLOCK_LOG), exist_ok=True)
            with open(MULTI_BLOCK_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[bcp_offline] multi-block log write failed: {e}")


def _valid_shape(obj) -> bool:
    """The tool-call schema the scaffold requires: name:str, arguments:dict."""
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and isinstance(obj.get("arguments"), dict)
    )


def repair_tool_call(raw: str, filename: str = "", turn_id: int = -1):
    """Minimal, symmetric trailing-brace repair for malformed tool calls.

    Removes the FEWEST trailing '}' characters that yield valid JSON with the
    expected schema. Never modifies names, arguments, query content, or any
    other field. Returns (parsed_obj, braces_removed) or (None, 0).

    Applied identically to every checkpoint so the repair rate is comparable.
    """
    s = raw.strip()
    n_trailing = len(s) - len(s.rstrip("}"))
    for k in range(1, n_trailing + 1):          # ascending => minimum removal
        candidate = s[: len(s) - k]
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if _valid_shape(obj):
            log_event({
                "event": "tool_call_repair",
                "question_id": filename,
                "turn_id": turn_id,
                "checkpoint": CHECKPOINT,
                "original_raw": s,
                "repaired_raw": candidate,
                "repair_type": "trailing_brace_trim",
                "braces_removed": k,
            })
            return obj, k
    return None, 0


def disallowed_tool_message(tool_name: str, filename: str = "") -> str:
    """Uniform error for tools outside the offline evaluation's tool set."""
    log_event({
        "event": "disallowed_tool",
        "filename": filename,
        "tool": tool_name,
    })
    return f"Tool {tool_name} is not available in this offline evaluation setting."
