# Ask My Notes

I built this for my bootcamp project. You put your own study notes in a folder, ask a question, and it answers using only those notes. The passages it used are shown on the right, so you can check the answer yourself. If the notes don't cover the question, it says so instead of guessing.

**Built with help from Claude (an AI assistant).** Claude wrote much of the first draft of the code and the tests. I ran them, checked the results, and can explain how each part works. The sample notes in this repo are about houseplants, so nothing from my classes is included.

## How it works

1. **It reads your notes.** Every `.md` or `.txt` file in the `corpus/` folder is cut into passages at its headings.
2. **It turns each passage into numbers.** An *embedding* is a list of numbers that places similar meanings close together. The model that does this, `nomic-embed-text`, runs on your computer through Ollama.
3. **It finds the closest passages to your question.** Each match gets a similarity score (higher means a closer match).
4. **It refuses if nothing is close enough.** If the best score is below 0.57, it answers "Not in my notes." The chat model is not called at all in that case.
5. **Otherwise it writes an answer with citations.** The chat model (`gemma4:26b`, also on your computer) gets the top 3 passages and two rules: answer only from them, and put the passage number after every sentence, like [1]. If the first answer has no citations, it asks once more. If it still has none, the answer is refused.

## Why I built it this way

- **Refuse before guessing.** The score check runs before the chat model, so an off-topic question can't produce an invented answer.
- **Citations are required.** Every claim points to a passage you can read.
- **The cutoff came from my own class notes.** I picked 0.57 by looking at the gap between the lowest score for a real question and the highest score for a question the notes don't cover, on a separate question set that is not included in this repo.
- **A privacy check on notes.** `build_corpus.py` copies notes into `corpus/` and stops if it finds a name from `names.txt` or an unknown speaker. It also removes schedule and Zoom details. It does not check for emails, phone numbers or addresses, so check those yourself. `names.txt` is kept private, and git ignores it.
- **Tests.** There are 63 tests. Seven are skipped in this repo: some need the Colab notebook or a question set that isn't included, and others check course-specific privacy words that were removed.

## Running it

**You need Ollama and a large model first.** Ollama runs AI models on your own computer and is free. The chat model is a large download (about 18 GB) and needs a lot of memory. Search also needs Ollama, because the embedding model runs through it. Without Ollama, only the tests that use stand-in data run.

1. Install Python 3.11 or newer, and Ollama.
2. In a Command Prompt, run `ollama pull nomic-embed-text` and `ollama pull gemma4:26b`.
3. Put your own `.md` or `.txt` notes in `corpus/`. Running `build_corpus.py` replaces everything in that folder, so keep a copy of the sample notes if you want them.
4. Double-click `RUN-CHAT.cmd`. The page opens at http://localhost:8599.

To run the tests, use the project's own Python: `.venv\Scripts\python -m unittest discover -s tests`. The `.venv` folder appears the first time you run `RUN-CHAT.cmd`. The first run also downloads the Python packages and writes a `.index/` cache.

## What it doesn't do well yet

- A question that is close to the notes' topic can still get an answer that goes slightly beyond them. The score cutoff can't catch that on its own. The chat model's rules are what catch it.
- It has only been run on one computer, and the test suite has not been run on a fresh machine.
- The 0.57 cutoff was tuned on my own class notes. Check it again with your notes and your questions.

## Files

- `app.py`: the chat page
- `rag.py`: the whole pipeline (split, embed, search, refuse, answer)
- `build_corpus.py`: copies and privacy-checks notes
- `topics.txt`: the topic buttons on the page
- `RUN-ASK.cmd`, `RUN-MY-QUESTIONS.cmd`, `run_eval.py`, `run_my_questions.py`: command-line tools. `run_eval.py` needs a question file that is not included in this repo.
- `tests/test_rag.py`: the tests
- `corpus/`: sample notes
