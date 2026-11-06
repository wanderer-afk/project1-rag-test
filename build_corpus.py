"""Step 0 - build a CLEAN copy of the class notes for the knowledge base.

Why a copy? The real notes hold classmates' names and Obsidian-only markup
(HTML <div> layout, mermaid diagrams, ![[image]] embeds). The demo video is
public, and the search should embed sentences, not HTML tags. So we copy the
finished "Week *.md" notes -> corpus/, scrub them, and REFUSE to finish if a
name is left.

Run:  python build_corpus.py "<notes folder>"   (the folder that holds the "Week *.md" notes)
      set NOTES_DIR=<notes folder>             (Windows; then just: python build_corpus.py)
      python build_corpus.py                   (no folder given and no NOTES_DIR: reads ./notes)

Only "Week *.md" files are read. raw/ (live chat logs) and drawings/ are skipped,
and nothing outside the class-notes folder (finance, health, ...) is ever touched.
"""
import os
import re
import sys
from pathlib import Path

# Where the real notes live is a private detail of each PC, so it is NOT written here:
# the command-line argument wins, then the NOTES_DIR environment variable, then ./notes.
NOTES_DIR = Path(os.environ.get("NOTES_DIR") or "notes")
CORPUS_DIR = Path(__file__).parent / "corpus"

# Real names (classmates, the instructor) live in names.txt, NOT in this file: names.txt
# is gitignored, so no real name ever reaches git. names.example.txt shows the format.
# A new name in a new week? Add it to names.txt - the gate below will tell you.
NAMES_FILE = Path(__file__).parent / "names.txt"
# Extra words to remove (places, travel, school details): private_words.txt, gitignored.
PRIVATE_WORDS_FILE = Path(__file__).parent / "private_words.txt"
DEFAULT_STAND_IN = "a classmate"


def load_name_map(path: Path = NAMES_FILE) -> dict[str, str] | None:
    """{name: stand-in}, one name per line ('#' lines are comments). A line "Name = the instructor"
    picks its own stand-in; a plain "Name" line becomes "a classmate". None = the file is missing."""
    path = Path(path)
    if not path.is_file():
        return None
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, stand_in = line.partition("=")
        name = name.strip()
        if name and not name.startswith("#"):
            found[name] = stand_in.strip() or DEFAULT_STAND_IN
    return found


def load_names(path: Path = NAMES_FILE) -> list[str] | None:
    """Just the names, longest first, so "Ann Lee" becomes one "a classmate", not "a classmate Lee"."""
    found = load_name_map(path)
    return None if found is None else sorted(found, key=len, reverse=True)


def name_pattern(name: str) -> str:
    """The whole word, ignoring capitals: a lower-case search hint ("video by ann") still names someone."""
    return rf"(?i)\b{re.escape(name)}\b"

# Second gate: words that may stand before "asked" / "(chat)" / "(spoken" / "answered".
# Any OTHER capitalised word in that spot is probably a name nobody listed yet.
SPEAKER_OK = {"He", "She", "They", "Class", "Teacher", "Student", "Students", "Someone",
              "Nobody", "Anyone", "Everyone", "Not",
              "You"}  # table header "| You asked | It did |" in the notes, not a person
SPEAKER_PATTERN = re.compile(r"\b([A-Z][a-z]+)(?:'s)? (?:asked|answered|\(chat\)|\(spoken)")

# Whole sections that are not class knowledge: personal planning, homework/schedule
# logistics, flashcards (Q::A cards), drawing lists and how-the-note-was-captured notes.
# A heading may start with a number ("## 14. To-do and homework"). A section ends at
# the next '#' or '##' heading.
DROP_SECTIONS = re.compile(
    r"^## (?:\d+\.\s*)?(To-do|To do|Homework|Flashcards|Drawings|Sources and gaps)\b.*?(?=^##? |\Z)",
    re.S | re.M)

# Personal and logistics details: the teacher's travel plans, the time zone, where
# to ask (chat app / Zoom), the class schedule, and paths inside the private vault.
# A callout (> [!type] block) that mentions one is dropped whole; any other line is dropped alone.
LOGISTICS = re.compile(
    r"time zone|your time|Zoom poll|Zoom chat|_To delete|See you tomorrow|next class"
    r"|class(?:es)? (?:may be |is |are |was |were )?(?:skipped|cancell?ed)"
    r"|To ask Wednesday|\bOct \d{1,2}\b(?!,? \d{4})"
    r"|How to study this note|\bon Wednesday\b", re.I)   # study-routine boxes point at flashcards/to-dos
# Schedule asides INSIDE a useful sentence are cut out, so the sentence itself stays:
# "(if the Oct 9-10 class runs)", "(check on Wednesday)", "class next week" -> "class",
# "... one output neuron each. Worth confirming with the instructor on Wednesday." -> "... each."
SCHEDULE_ASIDE = re.compile(r"[ \t]*\((?:[^()\n]*\bOct \d{1,2}\b(?!,? \d{4})|check on Wednesday)[^()\n]*\)"
                            r"|(?<=class) next week\b"
                            r"|[ \t]*Worth confirming[^.\n]*\bon Wednesday\.", re.I)
CALLOUT = re.compile(r"^>[ \t]?\[!\w+\][^\n]*(?:\n>[^\n]*)*", re.M)
# A flashcard line: "question::answer" outside code (x[::-1] in code is not a card).
FLASHCARD = re.compile(r"^(?![ \t]*(?:```|    ))[^`\n]*[^\[\s:`]::[^:\n][^\n]*$", re.M)


def drop_logistics(text: str) -> str:
    """A logistics callout goes whole - unless it holds a table (a Q&A log): then only the
    matching rows go, so the real questions and answers around them stay."""
    def callout(m: re.Match) -> str:
        block = m.group(0)
        is_table = re.search(r"^>\s?\|", block, re.M)
        return "" if LOGISTICS.search(block) and not is_table else block
    text = CALLOUT.sub(callout, text)
    return "\n".join(line for line in text.split("\n") if not LOGISTICS.search(line))


def clean(text: str, names: list[str] | dict[str, str]) -> str:
    """names = a list (each becomes "a classmate") or {name: stand-in} from load_name_map()."""
    stand_ins = names if isinstance(names, dict) else dict.fromkeys(names, DEFAULT_STAND_IN)
    text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)   # frontmatter (holds instructor name, tags)
    text = DROP_SECTIONS.sub("", text)                                    # to-do, homework, flashcards, sources ...
    text = re.sub(r"```mermaid.*?```", "", text, flags=re.S)              # diagrams
    text = SCHEDULE_ASIDE.sub("", text)                                   # "(if the Oct 9-10 class runs)" etc.
    text = drop_logistics(text)                                           # travel, time zone, chat apps, schedule
    text = FLASHCARD.sub("", text)                                        # stray Q::A cards outside a Flashcards section
    text = re.sub(r"\bvault notes\b", "notes", text)                      # "read your vault notes" -> "read your notes"
    text = re.sub(r"!\[\[[^\]]*\]\]", "", text)                           # image embeds
    text = re.sub(r"<[^>]+>", "", text)                                   # HTML tags
    text = re.sub(r"\[\[([^\]|]+)\|?([^\]]*)\]\]", lambda m: m.group(2) or m.group(1), text)  # [[link|alias]] -> alias
    text = re.sub(r"^>\s?\[!(\w+)\]-?\s*", r"**\1:** ", text, flags=re.M)  # > [!def] Title -> **def:** Title
    text = re.sub(r"^>\s?", "", text, flags=re.M)                          # unquote callout bodies
    for name in sorted(stand_ins, key=len, reverse=True):                 # longest first
        text = re.sub(name_pattern(name), stand_ins[name], text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def find_private_details(folder: Path) -> list[tuple[str, str]]:
    """Every (file, what) that must not be indexed: a logistics/personal detail, a vault path or a flashcard."""
    found = []
    for f in sorted(folder.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        found += [(f.name, f"logistics '{m.group(0)}'") for m in LOGISTICS.finditer(text)]
        found += [(f.name, f"flashcard '{m.group(0)[:40]}'") for m in FLASHCARD.finditer(text)]
        found += [(f.name, f"vault path '{m.group(0)}'") for m in re.finditer(r"_To delete|\d0 (?:Areas|Projects)/", text)]
    return found


def find_leaks(folder: Path, names: list[str]) -> list[tuple[str, str]]:
    """Every (file, name) that should not be there: a listed name, or an unknown speaker."""
    leaks = []
    for f in sorted(folder.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        leaks += [(f.name, n) for n in names if re.search(name_pattern(n), text)]
        leaks += [(f.name, f"unknown speaker '{m}'") for m in SPEAKER_PATTERN.findall(text)
                  if m not in SPEAKER_OK]
    return leaks


def main(notes_dir: Path = NOTES_DIR, corpus_dir: Path = CORPUS_DIR, names_file: Path = NAMES_FILE) -> int:
    sources = sorted(notes_dir.glob("Week *.md"))
    if not sources:
        print(f"No 'Week *.md' notes found in {notes_dir}")
        return 1
    name_map = load_name_map(names_file)
    names = None if name_map is None else sorted(name_map, key=len, reverse=True)
    if names is None:
        print(f"WARNING: {names_file.name} not found - no names will be scrubbed. "
              "Copy names.example.txt to names.txt and list the classmates (README, 'Privacy').")
    corpus_dir.mkdir(exist_ok=True)
    for old in corpus_dir.glob("*.md"):            # start fresh so a renamed note leaves no stale copy
        old.unlink()
    for src in sources:
        out = corpus_dir / src.name
        out.write_text(clean(src.read_text(encoding="utf-8"), name_map or {}), encoding="utf-8")
        print(f"cleaned  {src.name}  ->  {corpus_dir.name}/{out.name}  ({out.stat().st_size:,} bytes)")

    leaks = find_leaks(corpus_dir, names or [])     # safety gate: fail loudly if any name survived
    if leaks:
        print(f"NAME LEAK - {corpus_dir.name}/ is NOT safe to index or commit:", leaks)
        if names is None:
            print(f"NAME LEAK - {names_file.name} is missing, so nothing was scrubbed. Create it "
                  "(copy names.example.txt, list the real names) and run build_corpus.py again.")
        return 1
    private = find_private_details(corpus_dir)      # second gate: no travel/schedule/vault details, no flashcards
    if private:
        print(f"PRIVATE DETAIL LEFT - {corpus_dir.name}/ is NOT ready to index:", private)
        return 1
    print(f"OK - {len(sources)} notes, no classmate names in {corpus_dir.name}/ "
          f"({len(names or [])} listed names checked, capitals ignored)")
    print("OK - no logistics, personal details, vault paths or flashcards left")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])) if len(sys.argv) > 1 else main())
