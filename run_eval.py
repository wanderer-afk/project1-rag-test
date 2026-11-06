"""Score the assistant on eval/eval_questions.json - and print the evidence for MIN_SCORE.

Numbers it prints:
  hit-rate      real questions whose BEST piece came from the expected note
  hit@3         real questions with the expected note anywhere in the top 3
  MRR           mean reciprocal rank: 1/position of the first right piece (1.0 = always first)
  trap pass     trap questions refused (best score under MIN_SCORE)
It also prints every question's top score, the lowest real score, the highest trap
score and the gap between them - MIN_SCORE belongs inside that gap.
Each run is appended to eval/EVAL_LOG.md.

Run:  python run_eval.py                    retrieval only (fast, no chat model)
      python run_eval.py --answers          also asks the chat model (slow)
      python run_eval.py --min-score 0.55   try another threshold without editing rag.py
      python run_eval.py --no-log           do not append to EVAL_LOG.md
      python run_eval.py --answers --model gemma3:12b
                                            answer with another chat model (only generate() reads it)
      python run_eval.py --held-out --answers
                                            the HELD-OUT set (eval/held_out_questions.json, written
                                            blind). It is the real test: it is never used to pick
                                            MIN_SCORE, so it is logged under its own heading.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import rag

HERE = Path(__file__).parent
QUESTIONS = HERE / "eval" / "eval_questions.json"
HELD_OUT = HERE / "eval" / "held_out_questions.json"           # blind-written questions, exactly as given
HELD_OUT_KEY = HERE / "eval" / "held_out_expected.json"        # where each SHOULD be answered (+ reclassifications)
HELD_OUT_TITLE = "HELD-OUT run, not used for tuning"
GROUP_KIND = {"real": "real", "off_topic_traps": "trap", "near_topic_traps": "near-topic trap"}
LOG = HERE / "eval" / "EVAL_LOG.md"
RANK_DEPTH = 10                       # how far down MRR looks for the right note


def load_questions(held_out: bool = False) -> list[dict]:
    """The tuning set, or the held-out set joined with its answer key. Each question:
    {"q", "kind", "expect_source": str | list | None, "expect_section" (held-out only)}."""
    if not held_out:
        return json.loads(QUESTIONS.read_text(encoding="utf-8"))
    raw = json.loads(HELD_OUT.read_text(encoding="utf-8"))
    key = json.loads(HELD_OUT_KEY.read_text(encoding="utf-8"))["questions"]
    questions = []
    for group, kind in GROUP_KIND.items():
        for q in raw[group]:
            k = key[q]                                   # KeyError = a question with no answer key: fix the key file
            questions.append({"q": q, "kind": k["kind"], "expect_source": k["expect_source"],
                              "expect_section": k["expect_section"], "asked_as": kind})
    return questions


def is_expected(source: str, expect: str | list[str]) -> bool:
    """A piece counts as right if its file name starts with the expected note (or any of them)."""
    return any(source.startswith(e) for e in ([expect] if isinstance(expect, str) else expect))


def run(min_score: float, with_answers: bool, held_out: bool = False) -> dict:
    questions = load_questions(held_out)
    index = rag.load_index(rag.DEFAULT_FOLDER)
    if not index["chunks"]:
        raise SystemExit(f"0 files found in {rag.DEFAULT_FOLDER} - nothing to score.")
    rows = []
    for q in questions:
        hits = rag.retrieve(q["q"], index, k=RANK_DEPTH)
        score, top = hits[0]
        row = {"q": q["q"], "kind": q["kind"], "score": score, "top": rag.label(top),
               "expect_section": q.get("expect_section"), "asked_as": q.get("asked_as")}
        if q["kind"] == "real":
            ranks = [i for i, (_, c) in enumerate(hits, 1) if is_expected(c["source"], q["expect_source"])]
            row["rank"] = ranks[0] if ranks else None
            row["ok"] = row["rank"] == 1
            row["mark"] = "HIT" if row["ok"] else ("HIT@3" if row["rank"] and row["rank"] <= 3 else "MISS")
        else:
            row["ok"] = score < min_score
            row["mark"] = "REFUSED" if row["ok"] else "GUESSED"
        if with_answers:
            r = rag.ask(q["q"], index, min_score=min_score)
            row["answer"], row["model_called"], row["why"] = r["answer"], r["model_called"], r["why"]
        rows.append(row)

    res = summarise(rows, min_score, files=index["files"], pieces=len(index["chunks"]))
    res["held_out"] = held_out
    if with_answers:
        real, traps, near = split_kinds(rows)
        res["answered"] = sum(1 for r in real if r["answer"] != rag.REFUSAL)
        res["trap_no_model"] = sum(1 for r in traps if not r["model_called"])
        res["near_end2end"] = sum(1 for r in near if r["answer"] == rag.REFUSAL)
        res["empty"] = sum(1 for r in rows if "empty reply" in r.get("why", ""))   # a blank is NOT a real refusal
    return res


def split_kinds(rows: list[dict]) -> tuple[list, list, list]:
    real = [r for r in rows if r["kind"] == "real"]
    traps = [r for r in rows if r["kind"] == "trap"]               # off-topic: gate 1 (MIN_SCORE) must catch these
    near = [r for r in rows if r["kind"] == "near-topic trap"]     # same topic, answer absent: gate 2 (the model)
    return real, traps, near


def summarise(rows: list[dict], min_score: float, files: int = 0, pieces: int = 0) -> dict:
    """The numbers. Works with any mix of questions - even no real ones or no traps -
    a missing kind just gives None instead of crashing."""
    real, traps, near = split_kinds(rows)
    low_real = min((r["score"] for r in real), default=None)
    high_trap = max((r["score"] for r in traps), default=None)
    both = low_real is not None and high_trap is not None
    return {
        "rows": rows, "min_score": min_score, "pieces": pieces, "files": files,
        "hit": sum(r["ok"] for r in real), "hit3": sum(1 for r in real if r["rank"] and r["rank"] <= 3),
        "mrr": sum(1 / r["rank"] for r in real if r["rank"]) / len(real) if real else 0.0,
        "trap": sum(r["ok"] for r in traps), "n_real": len(real), "n_trap": len(traps),
        "near_gate1": sum(r["ok"] for r in near), "n_near": len(near),
        "low_real": low_real, "high_trap": high_trap,
        "high_near": max((r["score"] for r in near), default=0.0),
        "suggest": round((low_real + high_trap) / 2, 2) if both else None,
    }


def report(res: dict) -> str:
    lines = [f"| # | result | top score | kind | question | best piece |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(res["rows"], 1):
        lines.append(f"| {i} | {r['mark']} | {r['score']:.3f} | {r['kind']} | {r['q']} | {r['top']} |")
    if res["suggest"] is None:
        evidence = "- MIN_SCORE evidence needs at least one real AND one trap question (add them to eval_questions.json)"
    else:
        gap = res["low_real"] - res["high_trap"]
        evidence = (f"- lowest real score {res['low_real']:.3f}, highest trap score {res['high_trap']:.3f}, gap {gap:+.3f}"
                    + (f" -> midpoint {res['suggest']:.2f}" if gap > 0 else " -> OVERLAP: no threshold separates them"))
        if res.get("held_out"):
            evidence += " (for information only: the held-out set never moves MIN_SCORE)"
    lines += [
        "",
        f"- files {res['files']}, pieces {res['pieces']}, MIN_SCORE {res['min_score']:.2f}",
        f"- **hit-rate {res['hit']}/{res['n_real']}**, hit@3 {res['hit3']}/{res['n_real']}, MRR {res['mrr']:.3f}, "
        f"**trap pass-rate {res['trap']}/{res['n_trap']}** (gate 1, MIN_SCORE)",
        evidence,
        f"- near-topic traps refused by gate 1: {res['near_gate1']}/{res['n_near']} "
        f"(highest {res['high_near']:.3f} - same topic as the notes, so the score alone cannot catch them)",
    ]
    if "answered" in res:
        lines += [
            f"- with answers: real questions answered with citations {res['answered']}/{res['n_real']}, "
            f"traps refused WITHOUT calling the model {res['trap_no_model']}/{res['n_trap']}, "
            f"**near-topic traps refused end to end (gate 1 or gate 2) {res['near_end2end']}/{res['n_near']}**",
            f"- empty model replies (counted as refusals above, but NOT real refusals): {res['empty']}",
        ]
    if any(r.get("expect_section") for r in res["rows"]):
        lines += ["", "Expected (answer key written before the run):"]
        for i, r in enumerate(res["rows"], 1):
            moved = r.get("asked_as") and r["asked_as"] != r["kind"]
            lines.append(f"{i}. {r['expect_section']}"
                         + (f"  [RECLASSIFIED: asked as {r['asked_as']}, scored as {r['kind']}]" if moved else ""))
    if any("answer" in r for r in res["rows"]):
        lines += ["", "Answers:"]
        for i, r in enumerate(res["rows"], 1):
            called = "model called" if r["model_called"] else "model NOT called"
            if "empty reply" in r.get("why", ""):
                called += ", EMPTY reply"
            lines.append(f"{i}. ({called}) {r['answer'][:300].replace(chr(10), ' ')}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--answers", action="store_true")
    p.add_argument("--min-score", type=float, default=rag.MIN_SCORE)
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--held-out", action="store_true", help="score the blind held-out set (never used for tuning)")
    p.add_argument("--model", default=rag.CHAT_MODEL, help="chat model for --answers (default: rag.CHAT_MODEL)")
    args = p.parse_args()
    rag.CHAT_MODEL = args.model                     # only generate() reads it
    try:
        res = run(args.min_score, args.answers, args.held_out)
    except rag.ModelError as e:
        raise SystemExit(str(e))
    text = report(res)
    print(text)
    if not args.no_log:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        if not LOG.exists():
            LOG.write_text("# Eval log\n\nEvery run of `run_eval.py`, newest at the bottom.\n", encoding="utf-8")
        with LOG.open("a", encoding="utf-8") as f:
            title = f"{HELD_OUT_TITLE} - " if args.held_out else ""
            f.write(f"\n## {title}{stamp} - MIN_SCORE {res['min_score']:.2f}"
                    f"{f' (with answers, {args.model})' if args.answers else ''}\n\n{text}\n")


if __name__ == "__main__":
    main()
