"""Ask each near-topic trap N times and count how often the assistant refuses.

Near-topic traps (machine-learning questions the notes do NOT answer) score above
MIN_SCORE, so gate 1 lets them through and only gate 2 (the chat model saying
"Not in my notes.") stops them. This measures how reliable gate 2 is.

Run:  python eval/near_topic_repeat.py                     5 runs each, gemma4:26b
      python eval/near_topic_repeat.py --model gemma3:4b    the Colab model
      python eval/near_topic_repeat.py --runs 3
The chat runs at temperature 0 / seed 1 (rag.CHAT_OPTIONS), so repeats should match;
"answers seen" shows how many different replies the N runs gave.
An EMPTY model reply is counted on its own, never as a refusal.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import rag  # noqa: E402

LOG = HERE / "EVAL_LOG.md"


def traps() -> list[tuple[str, str]]:
    """(question, what the right behaviour is). The held-out k-means question was reclassified
    to real (the notes cover it), so for that one an ANSWER is right and a refusal is wrong."""
    old = [q["q"] for q in json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))
           if q["kind"] == "near-topic trap"]
    new = json.loads((HERE / "held_out_questions.json").read_text(encoding="utf-8"))["near_topic_traps"]
    key = json.loads((HERE / "held_out_expected.json").read_text(encoding="utf-8"))["questions"]
    return ([(q, "refuse") for q in old]
            + [(q, "answer (reclassified real)" if key[q]["kind"] == "real" else "refuse") for q in new])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--model", default=rag.CHAT_MODEL)
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    rag.CHAT_MODEL = args.model                     # only generate() reads it
    index = rag.load_index(rag.DEFAULT_FOLDER)
    lines = ["| question | right behaviour | top score | refused | empty replies | answers seen |",
             "|---|---|---|---|---|---|"]
    for q, right in traps():
        results = [rag.ask(q, index) for _ in range(args.runs)]
        empty = sum("empty reply" in r["why"] for r in results)       # a blank is not a real refusal
        refused = sum(r["answer"] == rag.REFUSAL for r in results) - empty
        distinct = len({r["answer"] for r in results})
        lines.append(f"| {q} | {right} | {results[0]['score']:.3f} | {refused}/{args.runs} | {empty} | "
                     f"{distinct} distinct |")
        print(lines[-1], flush=True)
        answered = [r["answer"] for r in results if r["answer"] != rag.REFUSAL]
        if answered:
            lines.append(f"|  - answered: {answered[0][:220].replace(chr(10), ' ').replace('|', '/')} | | | | | |")
            print(lines[-1], flush=True)
    text = "\n".join(lines)
    print(f"\nmodel {args.model}, options {rag.CHAT_OPTIONS}, think {rag.THINK}, MIN_SCORE {rag.MIN_SCORE}\n{text}")
    if not args.no_log:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"\n## Near-topic refusal rate - {stamp} - {args.model}, {args.runs} runs each, "
                    f"temperature {rag.CHAT_OPTIONS['temperature']} seed {rag.CHAT_OPTIONS['seed']}, think {rag.THINK}, "
                    f"MIN_SCORE {rag.MIN_SCORE:.2f}\n\n{text}\n")


if __name__ == "__main__":
    main()
