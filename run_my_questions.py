"""Bailey's own-questions scorecard: test the assistant with questions YOU wrote.

1. Copy eval/bailey_questions.template.json to eval/bailey_questions.json
   (RUN-MY-QUESTIONS.cmd does this for you) and fill in 7 real questions + 3 traps.
2. Run:  python run_my_questions.py
3. Open the eval/my_scorecard_<date-time>.md it writes and mark each row right / half / wrong.

Each question runs ONCE through rag.ask, the same path as the chat page: the real chat
model at temperature 0, seed 1 (rag.CHAT_OPTIONS), so the same question gives the same
answer next time. MIN_SCORE is always rag.MIN_SCORE: this script has no option to change it,
so your questions measure the system as tuned; they never tune it.

Options (for testing this script itself):
  --questions <file>   another questions file (default eval/bailey_questions.json)
  --out-dir <folder>   where to write the scorecard (default eval/)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import rag

HERE = Path(__file__).parent
QUESTIONS = HERE / "eval" / "bailey_questions.json"
TEMPLATE = HERE / "eval" / "bailey_questions.template.json"
OUT_DIR = HERE / "eval"
KINDS = ("real", "trap")
ANSWER_CHARS = 200                    # answers are trimmed to this length in the table


class QuestionsProblem(Exception):
    """The questions file is missing, broken or empty. The message says how to fix it."""


def load_my_questions(path: Path) -> list[dict]:
    """The filled-in slots as [{"q", "kind"}], real first. Empty strings are skipped."""
    if not path.exists():
        raise QuestionsProblem(f"{path} does not exist yet. Copy {TEMPLATE.name} to {path.name} "
                               "(in the eval folder), type your questions between the quotes, save, run again.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise QuestionsProblem(f"{path.name} is not valid JSON ({e}). Usually a missing quote or comma: "
                               "compare it with the template, fix that line, save, run again.") from e
    questions = [{"q": q.strip(), "kind": kind} for kind in KINDS for q in data.get(kind, [])
                 if isinstance(q, str) and q.strip()]
    if not questions:
        raise QuestionsProblem(f"{path.name} has no questions yet: every slot is empty. "
                               "Type at least one question between the quotes, save, run again.")
    return questions


def cell(text: str) -> str:
    """Make text safe for one Markdown table cell: one line, no '|'."""
    return " ".join(str(text).split()).replace("|", "/")


def run(questions: list[dict], index: dict) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['kind']}: {q['q']}", flush=True)
        hits = rag.retrieve(q["q"], index)                          # closest notes, shown even when refused
        r = rag.ask(q["q"], index, min_score=rag.MIN_SCORE)         # the real path: gate 1, model, citations
        refused = r["answer"] == rag.REFUSAL
        answer = cell(r["answer"])
        rows.append({"q": q["q"], "kind": q["kind"], "score": r["score"],
                     "top": rag.label(hits[0][1]) if hits else "-",
                     "answer": answer if len(answer) <= ANSWER_CHARS else answer[:ANSWER_CHARS - 3] + "...",
                     "refused": refused, "found_source": bool(r["cited"]), "model_called": r["model_called"]})
        print(f"      top score {r['score']:.3f} -> {'REFUSED' if refused else 'answered, cites ' + str(r['cited'])}")
    return rows


def summary_lines(rows: list[dict]) -> list[str]:
    real = [r for r in rows if r["kind"] == "real"]
    traps = [r for r in rows if r["kind"] == "trap"]
    return [
        f"- Real questions that found a source (answered with at least one [n] citation): "
        f"**{sum(r['found_source'] for r in real)}/{len(real)}**",
        f"- Traps refused (\"{rag.REFUSAL}\"): **{sum(r['refused'] for r in traps)}/{len(traps)}**"
        f" - of those, refused before the model was called (score under MIN_SCORE): "
        f"{sum(r['refused'] and not r['model_called'] for r in traps)}",
    ]


def scorecard(rows: list[dict], when: datetime, index: dict) -> str:
    lines = [
        f"# My scorecard - {when:%Y-%m-%d %H:%M}",
        "",
        f"Settings (never changed by this script): MIN_SCORE {rag.MIN_SCORE:.2f}, top {rag.TOP_K} pieces, "
        f"chat model {rag.CHAT_MODEL} with {rag.CHAT_OPTIONS}, embeddings {rag.EMBED_MODEL}. "
        f"Notes: {index['files']} files, {len(index['chunks'])} pieces.",
        "",
        "**Your job:** read each answer against your notes and put an x in ONE of right / half / wrong.",
        "- real question: right = correct and from the notes; half = partly right or missing a key point; "
        "wrong = wrong, made up, or refused although the notes have it.",
        "- trap: right = refused (\"Not in my notes.\"); wrong = it answered anyway.",
        "",
        "| # | question | kind | top score | top source note | answer (trimmed) | refused? | right | half | wrong |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {cell(r['q'])} | {r['kind']} | {r['score']:.3f} | {cell(r['top'])} | "
                     f"{r['answer']} | {'yes' if r['refused'] else 'no'} |  |  |  |")
    lines += ["", "## Summary (counted by the script)", *summary_lines(rows),
              "", "## Your count (fill in)", "- right: __  half: __  wrong: __  (out of " f"{len(rows)})", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score the assistant on YOUR questions (MIN_SCORE never changes).")
    p.add_argument("--questions", type=Path, default=QUESTIONS)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    try:
        questions = load_my_questions(args.questions)
    except QuestionsProblem as e:
        print(f"PROBLEM: {e}")
        return 1
    try:
        index = rag.load_index(rag.DEFAULT_FOLDER)
        if not index["chunks"]:
            print(f"0 files found in {rag.DEFAULT_FOLDER} - nothing to ask.")
            return 1
        rows = run(questions, index)
    except rag.ModelError as e:                 # Ollama off, model missing or out of memory: the fix, no traceback
        print(f"\n{e}")
        return 1
    when = datetime.now()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"my_scorecard_{when:%Y%m%d-%H%M%S}.md"
    out.write_text(scorecard(rows, when, index), encoding="utf-8")
    print("\n" + "\n".join(summary_lines(rows)))
    print(f"\nScorecard written: {out}\nOpen it and mark each row right / half / wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
