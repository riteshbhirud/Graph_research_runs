"""LLM-as-judge using the official BrowseComp-Plus grader prompt (Ruling 1).

Judge model: Qwen3-32B, thinking ENABLED (judge only), max_tokens=1024, T=0.
The same component also performs answer extraction from long-form responses
(Decision O4), so extraction is uniform across all five systems.
"""
import argparse
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# Verbatim from BrowseComp-Plus/scripts_evaluation (official grader).
GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""


def parse_judge(text: str) -> dict:
    """Official parsing: regex the four fields; correctness from correct: yes|no."""
    out = {
        "extracted_final_answer": None,
        "judge_reasoning": None,
        "judge_correct": None,
        "judge_confidence": None,
        "parse_error": False,
    }
    if not text:
        out["parse_error"] = True
        return out
    # strip judge's own thinking block before field extraction
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    m = re.search(r"extracted_final_answer:\s*(.*?)(?=\n|$)", body, re.I | re.S)
    if m:
        out["extracted_final_answer"] = m.group(1).strip()
    m = re.search(r"reasoning:\s*(.*?)(?=\ncorrect:|$)", body, re.I | re.S)
    if m:
        out["judge_reasoning"] = m.group(1).strip()
    m = re.search(r"correct:\s*(yes|no)", body, re.I)
    if m:
        out["judge_correct"] = m.group(1).lower() == "yes"
    m = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", body, re.I)
    if m:
        out["judge_confidence"] = m.group(1)
    if out["judge_correct"] is None:
        out["parse_error"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base_url", default="http://localhost:6001/v1")
    ap.add_argument("--judge_model", default="judge")
    ap.add_argument("--judge_name", default="Qwen3-32B")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--denominator", type=int, default=830,
                    help="Table 1 denominator: SR = correct/denominator*100")
    args = ap.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=1200.0)
    files = [f for f in sorted(glob.glob(os.path.join(args.dir, "*.json")))
             if not os.path.basename(f).startswith("_")]

    def judge_one(path):
        rec = json.load(open(path, encoding="utf-8"))
        # The grader sees the FULL response and does its own extraction,
        # so long-form answers are handled identically to short ones.
        response = rec.get("raw_final_output") or rec.get("final_answer") or ""
        prompt = GRADER_TEMPLATE.format(
            question=rec["question"],
            response=response[:30000],
            correct_answer=rec["gold_answer"],
        )
        r = client.chat.completions.create(
            model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        text = r.choices[0].message.content or ""
        parsed = parse_judge(text)
        rec.update({
            "extracted_final_answer": parsed["extracted_final_answer"],
            "extracted_short_answer": parsed["extracted_final_answer"],
            "judge_reasoning": parsed["judge_reasoning"],
            "judge_correct": bool(parsed["judge_correct"]),
            "judge_confidence": parsed["judge_confidence"],
            "judge_response": text,
            "judge_model": args.judge_name,
            "judge_thinking_enabled": True,
            "judge_parse_error": parsed["parse_error"],
            "is_correct": bool(parsed["judge_correct"]),
        })
        json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return bool(parsed["judge_correct"]), parsed["parse_error"]

    n_correct = n_parse_err = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(judge_one, f) for f in files]
        for fu in as_completed(futs):
            ok, perr = fu.result()
            n_correct += ok
            n_parse_err += perr

    n = len(files)
    denom = args.denominator or n
    sr = 100.0 * n_correct / denom if denom else 0.0
    sp = os.path.join(args.dir, "_summary.json")
    summary = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    summary.update({
        "n_correct": n_correct,
        "n_judged": n,
        "denominator_for_table1": denom,
        "TABLE1_SR_percent": round(sr, 1),
        "SR_percent": round(sr, 1),
        "judge_model": args.judge_name,
        "judge_prompt": "official BrowseComp-Plus GRADER_TEMPLATE",
        "judge_thinking_enabled": True,
        "judge_parse_errors": n_parse_err,
    })
    json.dump(summary, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"SR = {n_correct}/{denom} = {sr:.1f}%   (judged {n} files, judge={args.judge_name}, parse_errors={n_parse_err})")


if __name__ == "__main__":
    main()
