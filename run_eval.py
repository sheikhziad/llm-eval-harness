"""Run a fixed set of test prompts against an LLM and report pass rates.

Usage:
    python run_eval.py --provider mock                       # dry run, no API key needed
    python run_eval.py --provider gemini --model <model>     # real run, needs GEMINI_API_KEY

The same cases are run once per prompt version, so you can see whether a prompt
change improved or hurt accuracy (a simple regression test for prompts).
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from scoring import check_response

HERE = Path(__file__).parent

# Prompt versions to compare. Add your own to test prompt changes.
PROMPT_VERSIONS = {
    "v1-plain": "{prompt}",
    "v2-strict": (
        "Follow the instructions exactly. Be concise. "
        "Do not add commentary unless it is asked for.\n\n{prompt}"
    ),
}


def load_cases(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mock_model(prompt: str) -> str:
    """Fake model for a dry run. It answers some cases wrongly on purpose.

    Its results mean nothing about any real model. It only proves the harness works.
    """
    p = prompt.lower()
    if "capital city of australia" in p:
        return "Sydney"  # deliberately wrong
    if "chemical symbol for gold" in p:
        return "Au"
    if "17 multiplied by 23" in p:
        return "391"
    if "exactly one word" in p:
        return "Yes, of course!"  # deliberately breaks the instruction
    if "keys name and age" in p:
        return '{"name": "Sam", "age": 30}'
    if "pick my neighbour" in p:
        return "I can't help with that."
    return "I'm not sure."


def make_gemini_model(model_name: str, delay: float = 0.0):
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit("Install the SDK first: pip install google-genai") from exc
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set the GEMINI_API_KEY environment variable first.")
    client = genai.Client(api_key=api_key)

    def call(prompt: str) -> str:
        time.sleep(delay)  # pacing: stay under the free-tier requests-per-minute limit
        for attempt in range(6):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text or ""
            except Exception as err:
                msg = str(err)
                # A 429 is either an RPM window (recovers in seconds) or the
                # daily request quota (recovers only at the midnight-PT reset;
                # its body carries a "retry in Nh" hint). One backoff retries
                # the window; a second 429 returns [QUOTA] so the caller can
                # stop and resume later instead of scoring error text.
                if "429" in msg and "RESOURCE_EXHAUSTED" in msg:
                    if attempt >= 1:
                        return f"[QUOTA] {err}"
                    time.sleep(5)
                    continue
                # 503 is temporary server capacity; keep backing off longer.
                if "503" in msg:
                    if attempt == 9:
                        return f"[CAPACITY] {err}"
                    time.sleep(min(60, 5 * 2 ** min(attempt, 4)))
                    continue
                if attempt == 5:
                    return f"[ERROR] {err}"
                time.sleep(min(60, 5 * 2 ** attempt))  # 5s, 10s, 20s, 40s, 60s
        return ""

    return call


QUOTA_MARKERS = ("[QUOTA]", "[CAPACITY]")


def load_checkpoint(path: Path):
    """Previously collected rows, keyed by (model, prompt_version, case_id)."""
    done = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[(r.get("model", ""), r["prompt_version"], r["case_id"])] = r
    return done


def run(model_fn, cases, checkpoint_path: Path | None = None, model_label: str = ""):
    """Score every (version, case). Returns (rows, complete).

    With a checkpoint path, each scored row is appended there immediately and
    already-finished (version, case) pairs are skipped. If the model returns a
    [QUOTA]/[CAPACITY] marker the loop stops early (the marker row is not
    checkpointed, so it will be retried on the next run) and reports incomplete.
    """
    done = load_checkpoint(checkpoint_path) if checkpoint_path else {}
    rows = [r for r in done.values() if r.get("model", "") == model_label]
    fh = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path else None
    complete = True
    try:
        for version, template in PROMPT_VERSIONS.items():
            for case in cases:
                key = (model_label, version, case["id"])
                if key in done:
                    continue
                response = model_fn(template.format(prompt=case["prompt"]))
                if response.startswith(QUOTA_MARKERS):
                    print(f"stopped at {key}: {response[:160]}", flush=True)
                    complete = False
                    return rows, complete
                passed, reason = check_response(case["check"], response)
                row = {
                    "case_id": case["id"],
                    "category": case["category"],
                    "prompt_version": version,
                    "model": model_label,
                    "passed": passed,
                    "reason": reason,
                    "response": response.replace("\n", " ")[:300],
                }
                rows.append(row)
                if fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
    finally:
        if fh:
            fh.close()
    return rows, complete


def write_report(rows, out_dir: Path, label: str):
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_version = defaultdict(list)
    by_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_version[r["prompt_version"]].append(r["passed"])
        by_cat[r["category"]][r["prompt_version"]].append(r["passed"])

    versions = list(PROMPT_VERSIONS)
    lines = [f"# Evaluation report ({label})", "", "## Pass rate by prompt version", ""]
    lines += ["| Prompt version | Passed | Total | Pass rate |", "| --- | --- | --- | --- |"]
    for v in versions:
        res = by_version.get(v, [])
        if not res:
            lines.append(f"| {v} | 0 | 0 | n/a |")
            continue
        lines.append(f"| {v} | {sum(res)} | {len(res)} | {sum(res) / len(res):.0%} |")

    lines += ["", "## Pass rate by category", ""]
    lines += ["| Category | " + " | ".join(versions) + " |", "| --- | " + " | ".join("---" for _ in versions) + " |"]
    for cat, per_v in by_cat.items():
        cells = [("n/a" if not per_v.get(v) else f"{sum(per_v[v]) / len(per_v[v]):.0%}") for v in versions]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")

    failures = [r for r in rows if not r["passed"]]
    lines += ["", f"## Failures ({len(failures)})", ""]
    for r in failures:
        lines.append(f"- `{r['case_id']}` ({r['prompt_version']}): {r['reason']}. Got: {r['response'][:120]!r}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["mock", "gemini"], default="mock")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", ""),
                        help="Gemini model name. Check the current model list in Google AI Studio.")
    parser.add_argument("--cases", default=str(HERE / "cases.jsonl"))
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to wait before each API call (stay under free-tier RPM limits).")
    parser.add_argument("--checkpoint", default="",
                        help="JSONL file for incremental results. With it, finished (case, version) "
                             "pairs are skipped on rerun, so a run can resume after the daily quota resets.")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if args.provider == "gemini":
        if not args.model:
            raise SystemExit("Pass --model <name> or set GEMINI_MODEL.")
        model_fn, label = make_gemini_model(args.model, args.delay), f"gemini: {args.model}"
    else:
        model_fn, label = mock_model, "MOCK model, dry run only"

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    rows, complete = run(model_fn, cases, ckpt, model_label=label)
    if not rows:
        raise SystemExit("no results collected (quota exhausted before the first call?)")
    print(write_report(rows, Path(args.out), label))
    if not complete:
        print(f"\nINCOMPLETE: daily quota/capacity hit; rerun later with the same --checkpoint "
              f"to finish. {len(rows)}/{len(cases) * len(PROMPT_VERSIONS)} scored so far.")


if __name__ == "__main__":
    main()
