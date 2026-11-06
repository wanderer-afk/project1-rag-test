"""Project 1 - Agentic Knowledge Base Assistant (RAG).

A question-answering assistant for ANY folder of .md / .txt notes. It knows
nothing about the subject: it reads the folder, answers ONLY from it, cites the
pieces it used, and says "Not in my notes." when nothing is close enough.

The pipeline, one function per box (M1-M8):
  build once :  load_notes -> chunk -> embed -> build_index / load_index
  every ask  :  embed_question -> retrieve -> decide -> answer (with sources)

Only TWO functions talk to the model stack: embed() and generate().
To move to Colab / n8n / another API, rewrite those two; nothing else changes.

Run:
  python rag.py                       chat with the notes in corpus/
  python rag.py --folder <path>       chat with any other folder
  python rag.py --ask "a question"    answer one question and exit
  python rag.py --check               self-test of both models, no files needed
  python rag.py --rebuild             re-embed even if nothing changed
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import ollama

HERE = Path(__file__).parent
DEFAULT_FOLDER = HERE / "corpus"
INDEX_DIR = HERE / ".index"
EMBED_MODEL = "nomic-embed-text"     # turns text into 768 numbers
CHAT_MODEL = "gemma4:26b"            # writes the answer
TOP_K = 3                            # how many pieces the model gets to read
MAX_CHARS = 1200                     # longest piece, in characters
MIN_SCORE = 0.57                     # best score under this -> refuse. Set between the lowest real-question score and the highest trap score on a tuning set (see README)
REFUSAL = "Not in my notes."
CHAT_OPTIONS = {"temperature": 0, "seed": 1}   # same question -> same answer, so the eval can be repeated
# think=False: a "thinking" model (gemma4) otherwise writes a long hidden reasoning first, can run out of
# room (done_reason 'length') and return an EMPTY answer - seen 2026-10-09 on 2 of the 7 real questions.
THINK = False


# ---------- M1 load ----------
def note_files(folder: Path) -> list[Path]:
    """The .md / .txt FILES in the folder (a sub-folder called 'x.md' is skipped)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".md", ".txt"))


def load_notes(folder: Path) -> list[dict]:
    """Every .md / .txt file in the folder -> {"source": file name, "text": whole file}.
    errors='replace' keeps a badly encoded file readable (odd bytes become a ? mark)."""
    notes = [{"source": p.name, "text": p.read_text(encoding="utf-8", errors="replace")} for p in note_files(folder)]
    return [n for n in notes if n["text"].strip()]


def folder_stamp(folder: Path) -> tuple[int, float]:
    """(how many note files, sum of their change times). Adding, editing or deleting a
    note changes it, so the web page knows to re-read the folder."""
    files = note_files(folder)
    return len(files), sum(p.stat().st_mtime for p in files)


# ---------- M2 chunk ----------
def split_section(section: str, max_chars: int) -> list[str]:
    """Pack whole paragraphs into pieces of at most max_chars.
    Only a single paragraph longer than max_chars gets cut mid-text."""
    pieces, current = [], ""
    for para in section.split("\n\n"):
        while len(para) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(para[:max_chars])
            para = para[max_chars:]
        if current and len(current) + 2 + len(para) > max_chars:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def chunk(note: dict, max_chars: int = MAX_CHARS) -> list[dict]:
    """Cut a note at its '## ' headings, then into pieces of at most max_chars.
    Each piece remembers its file and heading, so an answer can cite it."""
    chunks = []
    for section in re.split(r"\n(?=## )", note["text"]):
        for text in split_section(section.strip(), max_chars):
            chunks.append({"source": note["source"], "heading": heading_of(section), "text": text})
    return chunks


def heading_of(section: str) -> str:
    """The section's '#' heading line (cut to 60 characters). Text before the first
    heading, or a note with no heading at all, is called '(start)'."""
    first_line = section.strip().split("\n", 1)[0]
    if not first_line.startswith("#"):
        return "(start)"
    return first_line.lstrip("#").strip()[:60] or "(start)"


def label(c: dict) -> str:
    """'file name > heading' - shown in citations and glued in front of the text before embedding."""
    return f"{Path(c['source']).stem} > {c['heading']}"


class ModelError(RuntimeError):
    """The model stack failed: Ollama off, a model not downloaded, or a model that cannot load.
    The message is complete (cause + fix + Ollama's own words), so callers just print it."""


def is_missing_model(e: Exception) -> bool:
    """True when Ollama answered 'this model is not downloaded' (HTTP 404 / 'not found')."""
    return getattr(e, "status_code", None) == 404 or "not found" in str(e).lower()


def how_to_fix(model: str, e: Exception) -> str:
    """Plain-words cause and fix for one model failure, with Ollama's own message at the end."""
    if isinstance(e, ConnectionError):                             # Ollama program closed
        return f"Cannot reach the model. Start Ollama, then run: ollama pull {model}   (Ollama said: {e})"
    if is_missing_model(e):                                        # Ollama runs, but this model was never pulled
        return (f"The model {model} is not downloaded yet. Fix: open a Command Prompt and run: "
                f"ollama pull {model}   (Ollama said: {e})")
    return (f"The model {model} is downloaded but could not run. Likely cause: this PC may not have "
            f"enough free memory for {model}. Fix: close other programs (browser tabs, games) and try "
            f"again, or use the Colab notebook as a retrieval demo.   (Ollama said: {e})")


# ---------- M3 embed (swappable function 1 of 2) ----------
def embed(texts: list[str], kind: str) -> np.ndarray:
    """Text -> unit-length rows of numbers. kind = 'document' (storing) or 'query' (asking).
    nomic-embed-text wants that prefix; without it retrieval quietly gets worse."""
    rows = []
    try:
        for start in range(0, len(texts), 32):                 # small batches keep each call short
            batch = [f"search_{kind}: {t}" for t in texts[start:start + 32]]
            rows += ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"]
    except (ollama.ResponseError, ConnectionError) as e:       # Ollama off, model missing or cannot load
        raise ModelError(how_to_fix(EMBED_MODEL, e)) from e
    vectors = np.array(rows, dtype=np.float32).reshape(len(texts), -1)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # unit length -> dot product = cosine


# ---------- swappable function 2 of 2 ----------
def generate(prompt: str) -> str:
    """Prompt in, answer text out. Local models sometimes return a blank; ask once more."""
    for _ in range(2):
        try:
            reply = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
                                options=CHAT_OPTIONS, think=THINK)
        except (ollama.ResponseError, ConnectionError) as e:   # Ollama off, model missing or cannot load
            raise ModelError(how_to_fix(CHAT_MODEL, e)) from e
        text = reply["message"]["content"].strip()
        if text:
            return text
    return ""


# ---------- M4 store ----------
def index_file(folder: Path) -> Path:
    """One saved index per folder, named after the folder's full path."""
    folder = Path(folder).resolve()
    return INDEX_DIR / f"{folder.name}-{hashlib.sha1(str(folder).encode()).hexdigest()[:8]}.json"


def signature(chunks: list[dict]) -> str:
    """Fingerprint of every piece (file, heading, text) + the embedding model. Change a note,
    or change how notes are cut, and the index rebuilds itself."""
    h = hashlib.sha1(EMBED_MODEL.encode())
    for c in chunks:
        h.update(f"{label(c)}\0{c['text']}\0".encode())
    return h.hexdigest()


def build_index(folder: Path, notes: list[dict] | None = None) -> dict:
    notes = load_notes(folder) if notes is None else notes
    chunks = [c for note in notes for c in chunk(note)]
    if not chunks:
        return {"chunks": [], "matrix": np.zeros((0, 0), dtype=np.float32), "files": 0}
    print(f"embedding {len(chunks)} pieces from {len(notes)} files in {Path(folder).name}/ ...", file=sys.stderr)
    matrix = embed([f"{label(c)}\n{c['text']}" for c in chunks], "document")
    INDEX_DIR.mkdir(exist_ok=True)
    saved = {"signature": signature(chunks), "chunks": chunks, "vectors": matrix.round(6).tolist()}
    index_file(folder).write_text(json.dumps(saved), encoding="utf-8")
    return {"chunks": chunks, "matrix": matrix, "files": len(notes)}


def load_index(folder: Path = DEFAULT_FOLDER, rebuild: bool = False) -> dict:
    """Load the saved index if the notes have not changed; otherwise build it.
    A damaged index file (half-written, edited by hand) is simply rebuilt."""
    notes = load_notes(folder)
    path = index_file(folder)
    if notes and not rebuild and path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["signature"] == signature([c for note in notes for c in chunk(note)]):
                return {"chunks": saved["chunks"], "matrix": np.array(saved["vectors"], dtype=np.float32),
                        "files": len(notes)}
        except (ValueError, KeyError, TypeError):
            print(f"index file {path.name} is damaged - rebuilding it", file=sys.stderr)
    return build_index(folder, notes)


# ---------- M5 embed the question ----------
def embed_question(question: str) -> np.ndarray:
    return embed([question], "query")[0]


# ---------- M6 retrieve ----------
def retrieve(question: str, index: dict, k: int = TOP_K) -> list[tuple[float, dict]]:
    """The k pieces whose numbers point most the same way as the question's (cosine similarity)."""
    if not index["chunks"]:
        return []
    scores = index["matrix"] @ embed_question(question)       # one score per piece, all at once
    best = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), index["chunks"][i]) for i in best]


# ---------- M7 decide (the agentic step) ----------
def decide(hits: list[tuple[float, dict]], min_score: float = MIN_SCORE) -> bool:
    """True = the best piece is close enough to answer from. False = refuse without calling the model."""
    return bool(hits) and hits[0][0] >= min_score


# ---------- M8 answer with sources ----------
def build_prompt(question: str, hits: list[tuple[float, dict]]) -> str:
    notes = "\n\n".join(f"[{n}] ({label(c)})\n{c['text']}" for n, (_, c) in enumerate(hits, 1))
    return ("Answer the question using ONLY the numbered notes below, in 2-4 sentences.\n"
            "Citation or nothing: every sentence must end with the number of the note it came from, like [1] or [2][3].\n"
            f"If the notes do not contain the answer, reply exactly: {REFUSAL}\n\n"
            f"NOTES:\n{notes}\n\nQUESTION: {question}\nANSWER:")


def citations(text: str, k: int) -> list[int]:
    """The note numbers [n] the answer actually cites (only 1..k count)."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= k})


def is_refusal(text: str) -> bool:
    """True only if the WHOLE reply is 'Not in my notes' (any case, with or without the
    final period, quotes or bold). A real answer that merely contains the phrase is kept."""
    return text.strip().strip("*\"'` ").rstrip(".").strip().lower() == REFUSAL.rstrip(".").lower()


def ask(question: str, index: dict, min_score: float = MIN_SCORE, k: int = TOP_K) -> dict:
    """The whole every-question path. Returns the answer, its sources and why."""
    hits = retrieve(question, index, k)                           # M5 + M6
    top = hits[0][0] if hits else 0.0
    sources = [{"n": n, "source": c["source"], "heading": c["heading"], "score": s, "text": c["text"]}
               for n, (s, c) in enumerate(hits, 1)]
    result = {"question": question, "score": top, "min_score": min_score, "sources": sources,
              "cited": [], "model_called": False}

    def refuse(why: str) -> dict:
        return {**result, "answer": REFUSAL, "sources": [], "why": why}

    if not decide(hits, min_score):                               # M7 gate 1: too far away, model not called
        return refuse(f"best score {top:.3f} is under MIN_SCORE {min_score:.2f} - model not called")
    prompt = build_prompt(question, hits)                         # M8
    text = generate(prompt)
    result["model_called"] = True
    if not is_refusal(text) and not citations(text, len(hits)):  # forgot the [n]s: ask once more
        text = generate(prompt + "\n(Your last answer had no [n] citations. Cite every sentence.)")
    if not text:                                                  # blank twice: refuse, but say why honestly
        return refuse(f"best score {top:.3f} passed, but the model returned an empty reply")
    if is_refusal(text):                                          # gate 2: the model says the pieces lack it
        return refuse(f"best score {top:.3f} passed, but the model found no answer in the pieces")
    cited = citations(text, len(hits))
    if not cited:
        return refuse("the model answered without citing any note - citation or nothing")
    return {**result, "answer": text, "cited": cited,
            "why": f"best score {top:.3f} >= MIN_SCORE {min_score:.2f}; cites {cited}"}


def print_result(r: dict) -> None:
    print(f"\nbot> {r['answer']}")
    for s in r["sources"]:
        mark = "*" if s["n"] in r["cited"] else " "
        print(f"   {mark}[{s['n']}] {Path(s['source']).stem} > {s['heading']}  (score {s['score']:.2f})")
    print(f"     ({r['why']})\n")


# ---------- self-test, no files needed ----------
def self_check() -> bool:
    """Proves both models work and that scores mean something, before any note exists."""
    a, b, c = ("The sun heats water and it rises as vapor.", "Warm water evaporates into the air.",
               "The invoice is due next Friday.")
    f = embed([a, b, c], "document")
    same, different = float(f[0] @ f[1]), float(f[0] @ f[2])
    reply = generate("Reply with exactly: ready")
    checks = [
        (f.shape[1] == 768, f"check 1  embedding length: {f.shape[1]} (expect 768)"),
        (same > different, f"check 2  same idea: {same:.2f}   different idea: {different:.2f} (expect same > different)"),
        ("ready" in reply.lower(), f"check 3  chat model says: {reply!r} (expect 'ready')"),
    ]
    for ok, line in checks:
        print(("PASS  " if ok else "FAIL  ") + line)
    passed = all(ok for ok, _ in checks)
    print("SELF-TEST PASSED" if passed else "SELF-TEST FAILED")
    return passed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ask questions about a folder of notes.")
    p.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="folder of .md/.txt notes")
    p.add_argument("--ask", help="answer one question and exit")
    p.add_argument("--check", action="store_true", help="self-test the models, no files needed")
    p.add_argument("--rebuild", action="store_true", help="re-embed even if nothing changed")
    args = p.parse_args(argv)
    try:
        return run(args)
    except ModelError as e:                 # Ollama off, model missing or out of memory: the fix, no traceback
        print(f"\n{e}")
        return 1


def run(args: argparse.Namespace) -> int:
    if args.check:
        return 0 if self_check() else 1
    index = load_index(args.folder, rebuild=args.rebuild)
    if not index["chunks"]:
        print(f"0 files found in {args.folder}. Put .md or .txt notes on any subject in that folder and run again.")
        return 0
    print(f"{index['files']} files, {len(index['chunks'])} pieces ready from {args.folder.name}/  (MIN_SCORE {MIN_SCORE})")
    if args.ask:
        print(f"you> {args.ask}")
        print_result(ask(args.ask, index))
        return 0
    print("Ask a question (blank line to quit).\n")
    while True:
        try:
            question = input("you> ").strip()
        except EOFError:
            break
        if not question:
            break
        print_result(ask(question, index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
