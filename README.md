# Ask My Notes: a local RAG chat over your own study notes

**RAG** (retrieval-augmented generation) means: search your own documents first, then have an AI model answer using only the passages it found. This project is a small, complete example of that idea, built and tested from scratch.

It runs entirely on your own computer. Your notes are never uploaded anywhere.

## What it does

1. Reads every `.md` or `.txt` file in the `corpus/` folder and splits it into passages at its headings.
2. Turns each passage into a list of 768 numbers (an **embedding**, which places similar meanings close together). The embedding model is `nomic-embed-text`, run locally through **Ollama**.
3. When you ask a question, it finds the closest passages using cosine similarity (a score from 0 to 1 for how closely two meanings match).
4. If the best match scores below a cutoff, it refuses with "Not in my notes." The model is never called in that case.
5. Otherwise it sends the top passages to a chat model (`gemma4:26b`, also local) with strict rules: answer only from the notes, and cite every sentence with the passage number it came from.
6. The page shows the answer, plus the passages behind it.

## Design choices worth a look

- **Refuse before you guess.** The cutoff gate runs before the chat model, so off-topic questions cost nothing and can't produce invented answers.
- **Citation or nothing.** An answer without `[1]`-style citations is rejected and asked again.
- **The cutoff came from the author's own notes.** It was chosen on a separate question set that is not included in this public copy, so the 0.57 value is a starting point for your own notes. Re-check it with your questions.
- **Privacy gate for the notes.** `build_corpus.py` copies notes into `corpus/` and refuses to finish if it finds a listed name, a personal detail or a private file path. Names live in `names.txt`, which git ignores.
- **Tests that do not need a model.** Chunking, retrieval, refusal and the page's empty-folder case are all tested with stand-in data (63 tests in total; the skipped ones need the original notes, the Colab notebook or the private word list). One test checks a real houseplant search with the embedding model, so it needs Ollama.

## Who can run this

Cloning the repo does not give you a working chat by itself.

- The chat page needs **Ollama** (a free program that runs AI models on your own computer) and the **gemma4:26b** model. The model download is about 18 GB and needs about 16 GB of memory.
- Search itself also needs Ollama, because the embedding model `nomic-embed-text` runs through it. Without Ollama only the tests that use stand-in data run.

## Run it (Windows)

1. Install Python 3.11 or newer and Ollama.
2. In a Command Prompt: `ollama pull nomic-embed-text` and `ollama pull gemma4:26b`.
3. Put your own `.md` or `.txt` notes in the `corpus/` folder. Replace the three sample files.
4. Double-click `RUN-CHAT.cmd`. The page opens at http://localhost:8599.

Run the tests with `python -m unittest discover -s tests`.

## Layout

- `app.py` - the Streamlit chat page
- `rag.py` - the whole pipeline: chunking, embedding, retrieval, refusal, answer
- `build_corpus.py` - copies and privacy-checks notes into `corpus/`
- `topics.txt` - the topic buttons on the page (edit to match your notes)
- `tests/test_rag.py` - the tests
- `corpus/` - sample notes (replace with your own)

## Known limits

- Questions that sit close to the notes' topic can still get an answer that goes beyond them. The cutoff alone cannot catch those; the chat model's rules are what refuse them.
- It has been run on a single PC. It has not been load-tested.
