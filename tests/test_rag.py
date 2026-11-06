"""Tests for Project 1. Run:  .venv\\Scripts\\python -m unittest discover -s tests -v

Most tests use fake vectors and a fake model, so they run in a second with no Ollama.
The LIVE tests at the bottom use the real models and are skipped if Ollama is off.
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_corpus  # noqa: E402
import rag  # noqa: E402
import run_eval  # noqa: E402
import run_my_questions  # noqa: E402

FAKE_NAMES = ["Zara Quill", "Zara", "Quill", "Orrin"]     # made-up classmates, never real ones


def fake_index(vectors: list[list[float]]) -> dict:
    m = np.array(vectors, dtype=np.float32)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    chunks = [{"source": f"note{i}.md", "heading": f"part {i}", "text": f"text {i}"} for i in range(len(m))]
    return {"chunks": chunks, "matrix": m, "files": len(m)}


def fake_embed(vector: list[float]):
    v = np.array([vector], dtype=np.float32)
    return lambda texts, kind: v / np.linalg.norm(v)


def letter_embed(texts: list[str], kind: str) -> np.ndarray:
    """A pretend embedding model: counts each letter a-z. Same idea as the real one
    (text -> numbers), but needs no Ollama, so tests run anywhere."""
    rows = np.array([[t.lower().count(ch) + 0.01 for ch in "abcdefghijklmnopqrstuvwxyz"] for t in texts],
                    dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class LoadAndChunk(unittest.TestCase):
    def test_empty_folder_loads_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(rag.load_notes(Path(d)), [])
            self.assertEqual(rag.load_notes(Path(d) / "missing"), [])

    def test_empty_folder_message(self):
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = rag.main(["--folder", d])
            self.assertEqual(code, 0)
            self.assertIn("0 files found", out.getvalue())

    def test_only_md_and_txt_files_are_read(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.md", "b.txt", "c.pdf", "d.py"):
                (Path(d) / name).write_text("## H\nbody", encoding="utf-8")
            (Path(d) / "folder.md").mkdir()                               # a FOLDER named like a note
            self.assertEqual([n["source"] for n in rag.load_notes(Path(d))], ["a.md", "b.txt"])

    def test_badly_encoded_file_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "old.txt").write_bytes(b"caf\xe9 notes")             # not valid UTF-8
            notes = rag.load_notes(Path(d))
        self.assertEqual(len(notes), 1)
        self.assertIn("notes", notes[0]["text"])

    def test_chunk_splits_on_headings_and_keeps_source(self):
        note = {"source": "x.md", "text": "intro line\n\n## First\nalpha\n\n## Second\nbeta"}
        chunks = rag.chunk(note)
        self.assertEqual([c["heading"] for c in chunks], ["(start)", "First", "Second"])
        self.assertTrue(all(c["source"] == "x.md" for c in chunks))

    def test_note_with_no_heading_is_called_start(self):
        long_para = "This note has no heading at all, just one long paragraph of plain text. " * 5
        chunks = rag.chunk({"source": "plain.txt", "text": long_para + "\n\nA second paragraph."})
        self.assertTrue(chunks)
        self.assertEqual({c["heading"] for c in chunks}, {"(start)"})   # not the whole paragraph

    def test_long_heading_is_cut_to_60(self):
        chunks = rag.chunk({"source": "x.md", "text": "## " + "Very long heading " * 10 + "\nbody"})
        self.assertLessEqual(len(chunks[0]["heading"]), 60)

    def test_chunk_respects_max_chars_and_keeps_paragraphs_whole(self):
        paras = [f"paragraph {i} " + "word " * 30 for i in range(10)]
        note = {"source": "x.md", "text": "## Long\n\n" + "\n\n".join(paras)}
        chunks = rag.chunk(note, max_chars=400)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["text"]) <= 400 for c in chunks))
        joined = " ".join(c["text"] for c in chunks)
        self.assertTrue(all(p.strip() in joined for p in paras))      # no paragraph was cut

    def test_giant_paragraph_is_hard_cut(self):
        note = {"source": "x.md", "text": "## H\n" + "x" * 2500}
        self.assertTrue(all(len(c["text"]) <= 1200 for c in rag.chunk(note)))

    def test_signature_changes_when_a_note_changes(self):
        a = rag.chunk({"source": "a.md", "text": "## H\none"})
        b = rag.chunk({"source": "a.md", "text": "## H\ntwo"})
        self.assertNotEqual(rag.signature(a), rag.signature(b))

    def test_folder_stamp_changes_on_add_edit_and_delete(self):
        """The web page re-reads the folder when this stamp changes."""
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.md", Path(d) / "b.md"
            a.write_text("one", encoding="utf-8")
            os.utime(a, (1000, 1000))
            start = rag.folder_stamp(Path(d))
            b.write_text("two", encoding="utf-8")                          # add
            os.utime(b, (500, 500))
            added = rag.folder_stamp(Path(d))
            os.utime(a, (2000, 2000))                                      # edit (newer change time)
            edited = rag.folder_stamp(Path(d))
            b.unlink()                                                     # delete an OLDER file
            deleted = rag.folder_stamp(Path(d))
        self.assertEqual(len({start, added, edited, deleted}), 4)


class SavedIndex(unittest.TestCase):
    def test_damaged_index_file_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as idx:
            (Path(d) / "a.md").write_text("## Cats\ncats purr", encoding="utf-8")
            with mock.patch.object(rag, "INDEX_DIR", Path(idx)), mock.patch.object(rag, "embed", letter_embed):
                rag.index_file(Path(d)).write_text("{half a file", encoding="utf-8")   # damaged
                with contextlib.redirect_stderr(io.StringIO()):
                    index = rag.load_index(Path(d))
                self.assertEqual(len(index["chunks"]), 1)
                saved = json.loads(rag.index_file(Path(d)).read_text(encoding="utf-8"))
                self.assertEqual(saved["signature"], rag.signature(index["chunks"]))   # a good file again


class OllamaOff(unittest.TestCase):
    """Ollama not started, or a model not downloaded: a clear fix, never a traceback."""

    def test_embed_says_how_to_fix(self):
        with mock.patch.object(rag.ollama, "embed", side_effect=ConnectionError("Failed to connect")):
            with self.assertRaises(rag.ModelError) as e:
                rag.embed(["x"], "query")
        self.assertIn("Start Ollama, then run: ollama pull nomic-embed-text", str(e.exception))

    def test_generate_says_how_to_fix(self):
        missing = rag.ollama.ResponseError("model not found", 404)
        with mock.patch.object(rag.ollama, "chat", side_effect=missing):
            with self.assertRaises(rag.ModelError) as e:
                rag.generate("hi")
        self.assertIn(f"The model {rag.CHAT_MODEL} is not downloaded yet", str(e.exception))
        self.assertIn(f"run: ollama pull {rag.CHAT_MODEL}", str(e.exception))
        self.assertNotIn("memory", str(e.exception))

    def test_missing_model_by_404_or_by_words(self):
        for err in (rag.ollama.ResponseError("pull it first", 404),
                    rag.ollama.ResponseError(f'model "{rag.CHAT_MODEL}" not found, try pulling it first', 500)):
            with mock.patch.object(rag.ollama, "chat", side_effect=err):
                with self.assertRaises(rag.ModelError) as e:
                    rag.generate("hi")
            self.assertIn(f"ollama pull {rag.CHAT_MODEL}", str(e.exception))

    def test_low_memory_is_not_told_to_pull_again(self):
        """Model downloaded but too big for free memory: say so, never 'ollama pull' (it is already there)."""
        ollama_words = "model requires more system memory (20.1 GiB) than is available (11.4 GiB)"
        low_memory = rag.ollama.ResponseError(ollama_words, 500)
        for fn, patched, model in ((lambda: rag.generate("hi"), "chat", rag.CHAT_MODEL),
                                   (lambda: rag.embed(["x"], "query"), "embed", rag.EMBED_MODEL)):
            with mock.patch.object(rag.ollama, patched, side_effect=low_memory):
                with self.assertRaises(rag.ModelError) as e:
                    fn()
            msg = str(e.exception)
            self.assertNotIn("ollama pull", msg)
            self.assertIn(f"may not have enough free memory for {model}", msg)
            self.assertIn("close other programs", msg)
            self.assertIn("Colab notebook", msg)
            self.assertIn(ollama_words, msg)                  # Ollama's own message is kept

    def test_command_line_prints_the_memory_cause(self):
        low_memory = rag.ollama.ResponseError("model requires more system memory (20.1 GiB) than is available", 500)
        out = io.StringIO()
        with mock.patch.object(rag.ollama, "embed", side_effect=low_memory), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = rag.main(["--check"])
        self.assertEqual(code, 1)
        self.assertIn("enough free memory", out.getvalue())
        self.assertNotIn("Cannot reach the model", out.getvalue())

    def test_command_line_prints_the_fix(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as idx:
            (Path(d) / "a.md").write_text("## H\nbody", encoding="utf-8")
            out = io.StringIO()
            with mock.patch.object(rag, "INDEX_DIR", Path(idx)), \
                 mock.patch.object(rag.ollama, "embed", side_effect=ConnectionError("Failed to connect")), \
                 contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = rag.main(["--folder", d, "--ask", "anything"])
        self.assertEqual(code, 1)
        self.assertIn("Start Ollama, then run: ollama pull nomic-embed-text", out.getvalue())

    def test_web_page_shows_the_fix(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as idx:
            (Path(d) / "a.md").write_text("## H\nbody", encoding="utf-8")
            with mock.patch.object(rag, "INDEX_DIR", Path(idx)), \
                 mock.patch.object(rag.ollama, "embed", side_effect=ConnectionError("Failed to connect")), \
                 contextlib.redirect_stderr(io.StringIO()):
                at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
                at.sidebar.text_input[0].set_value(d).run()
        self.assertFalse(at.exception)
        self.assertTrue(any("ollama pull nomic-embed-text" in e.value for e in at.error))


class DecideAndAnswer(unittest.TestCase):
    def setUp(self):
        self.index = fake_index([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 1]])

    def test_retrieve_ranks_by_cosine(self):
        with mock.patch.object(rag, "embed", fake_embed([1, 0, 0])):
            hits = rag.retrieve("q", self.index, k=3)
        self.assertEqual([c["source"] for _, c in hits], ["note0.md", "note1.md", "note2.md"])
        self.assertAlmostEqual(hits[0][0], 1.0, places=5)

    def test_decide(self):
        self.assertTrue(rag.decide([(0.70, {})], 0.60))
        self.assertFalse(rag.decide([(0.59, {})], 0.60))
        self.assertFalse(rag.decide([], 0.60))

    def test_trap_is_refused_and_model_never_called(self):
        never = mock.Mock(side_effect=AssertionError("the model must not be called for a trap"))
        with mock.patch.object(rag, "embed", fake_embed([0.3, 0.3, 0.9])), mock.patch.object(rag, "generate", never):
            r = rag.ask("trap", self.index, min_score=0.99)
        self.assertEqual(r["answer"], rag.REFUSAL)
        self.assertFalse(r["model_called"])
        self.assertEqual(r["sources"], [])
        never.assert_not_called()

    def test_cited_answer_keeps_sources(self):
        with mock.patch.object(rag, "embed", fake_embed([1, 0, 0])), \
             mock.patch.object(rag, "generate", return_value="It is alpha [1][2]."):
            r = rag.ask("q", self.index, min_score=0.5)
        self.assertEqual(r["cited"], [1, 2])
        self.assertEqual(len(r["sources"]), 3)
        self.assertTrue(r["model_called"])

    def test_citation_or_nothing(self):
        gen = mock.Mock(return_value="An answer with no citation at all.")
        with mock.patch.object(rag, "embed", fake_embed([1, 0, 0])), mock.patch.object(rag, "generate", gen):
            r = rag.ask("q", self.index, min_score=0.5)
        self.assertEqual(r["answer"], rag.REFUSAL)
        self.assertEqual(gen.call_count, 2)                            # asked once more, then refused

    def test_is_refusal_only_for_the_whole_reply(self):
        for text in ("Not in my notes.", "not in my notes", "  Not in my notes.  ", "**Not in my notes.**"):
            self.assertTrue(rag.is_refusal(text), text)
        self.assertFalse(rag.is_refusal("The rule is not in my notes as a formula, but it is described [1]."))

    def test_real_answer_containing_the_phrase_is_kept(self):
        text = "The exact formula is not in my notes, but the idea is to stop when the score stops improving [1]."
        with mock.patch.object(rag, "embed", fake_embed([1, 0, 0])), \
             mock.patch.object(rag, "generate", return_value=text):
            r = rag.ask("q", self.index, min_score=0.5)
        self.assertEqual(r["answer"], text)
        self.assertEqual(r["cited"], [1])

    def test_model_refusal_is_passed_through(self):
        with mock.patch.object(rag, "embed", fake_embed([1, 0, 0])), \
             mock.patch.object(rag, "generate", return_value="Not in my notes."):
            r = rag.ask("q", self.index, min_score=0.5)
        self.assertEqual(r["answer"], rag.REFUSAL)
        self.assertTrue(r["model_called"])

    def test_prompt_numbers_every_piece(self):
        hits = [(0.9, c) for c in self.index["chunks"]]
        p = rag.build_prompt("why?", hits)
        for n in (1, 2, 3):
            self.assertIn(f"[{n}] (note{n - 1} > part {n - 1})", p)
        self.assertIn(rag.REFUSAL, p)


class ColdStartRule(unittest.TestCase):
    def test_no_subject_words_in_the_code(self):
        """The assistant must work for ANY subject, so the code names none."""
        subject = re.compile(r"regression|classification|epoch|churn|transformer|word2vec|neural|"
                             r"bootcamp|class notes|week \d", re.I)
        for name in ("rag.py", "app.py", "run_my_questions.py"):
            found = subject.findall((ROOT / name).read_text(encoding="utf-8"))
            self.assertEqual(found, [], f"{name} mentions the subject: {found}")

    def test_only_embed_and_generate_need_the_model_stack(self):
        """Swap embed() and generate() for fakes, then take the Ollama library away
        completely. If the whole assistant (load, chunk, store, retrieve, decide, answer)
        still works, nothing else needs the model stack - so a new stack means
        rewriting just those two functions."""
        class NoOllama:
            def __getattr__(self, name):
                raise AssertionError(f"something other than embed()/generate() used ollama.{name}")

        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as idx:
            (Path(d) / "pets.md").write_text("## Cats\nCats purr when happy.\n\n## Dogs\nDogs bark.",
                                              encoding="utf-8")
            with mock.patch.object(rag, "ollama", NoOllama()), mock.patch.object(rag, "INDEX_DIR", Path(idx)), \
                 mock.patch.object(rag, "embed", letter_embed), \
                 mock.patch.object(rag, "generate", return_value="Cats purr [1]."), \
                 contextlib.redirect_stderr(io.StringIO()):
                index = rag.load_index(Path(d))
                r = rag.ask("Why do cats purr?", index, min_score=0.0)
        self.assertEqual(r["answer"], "Cats purr [1].")
        self.assertTrue(r["model_called"])

    def test_generate_is_repeatable_temperature_0_seed_1(self):
        """The chat model runs at temperature 0 with a fixed seed, so an eval can be re-run."""
        fake = mock.Mock(return_value={"message": {"content": "ok [1]"}})
        with mock.patch.object(rag.ollama, "chat", fake):
            self.assertEqual(rag.generate("hi"), "ok [1]")
        self.assertEqual(fake.call_args.kwargs["options"], {"temperature": 0, "seed": 1})
        self.assertIs(fake.call_args.kwargs["think"], False)     # no hidden reasoning that can eat the whole reply

    @unittest.skip("Colab notebook removed from the public copy (it held class material)")

    def test_colab_twin_uses_the_same_model_settings(self):
        """The notebook's generate() must match rag.py: temperature 0, seed 1, thinking off."""
        code = (ROOT / "colab" / "colab_style.py").read_text(encoding="utf-8")
        self.assertIn("options={'temperature': 0, 'seed': 1}", code)
        self.assertIn("think=False", code)
        self.assertIn(f"MIN_SCORE    = {rag.MIN_SCORE:.2f}", code)
        self.assertIn("for _ in range(2):", code)                 # same two tries for a blank reply
        self.assertIn("the model returned an empty reply", code)  # same honest reason as rag.py
        self.assertIn('pip -q install "ollama>=0.6" numpy', code) # pinned: think= needs a recent client

    @unittest.skip("Colab notebook removed from the public copy (it held class material)")

    def test_colab_twin_retries_a_blank_reply_then_says_empty(self):
        """Run the notebook's own generate() and ask() with a fake model that only returns blanks."""
        code = (ROOT / "colab" / "colab_style.py").read_text(encoding="utf-8")
        start, end = code.index("def embed(texts, kind):"), code.index("# Embed every chunk")
        ask_start, ask_end = code.index("def is_refusal(text):"), code.index("answer, sources = ask(question)")
        fake_chat = mock.Mock(return_value={"message": {"content": "  "}})
        ns = {"ollama": mock.Mock(chat=fake_chat), "np": np, "re": re, "CHAT_MODEL": "m", "TOP_K": 3,
              "MIN_SCORE": 0.5, "REFUSAL": rag.REFUSAL, "label": lambda c: c["heading"],
              "chunks": [{"source": "a.md", "heading": "h", "text": "t"}],
              "chunk_vectors": np.array([[1.0, 0.0]], dtype=np.float32)}
        exec(code[start:end] + code[ask_start:ask_end], ns)
        ns["embed"] = lambda texts, kind: np.array([[1.0, 0.0]], dtype=np.float32)
        self.assertEqual(ns["generate"]("hi"), "")
        self.assertEqual(fake_chat.call_count, 2)                 # asked twice, like rag.generate
        answer, sources = ns["ask"]("q")
        self.assertTrue(answer.startswith(rag.REFUSAL))
        self.assertIn("empty reply", answer)
        self.assertEqual(sources, [])

    def test_empty_reply_is_not_called_a_refusal(self):
        """A blank reply (twice) still refuses - but the reason says 'empty', so the eval cannot
        count it as the model deciding the notes lack the answer."""
        index = fake_index([[1, 0, 0]])
        with mock.patch.object(rag, "embed", fake_embed([1, 0, 0])), mock.patch.object(rag, "generate", return_value=""):
            r = rag.ask("q", index, min_score=0.5)
        self.assertEqual(r["answer"], rag.REFUSAL)
        self.assertIn("empty reply", r["why"])


class Corpus(unittest.TestCase):
    """Uses MADE-UP names only. The real list lives in the gitignored names.txt."""

    def test_clean_strips_markup_and_names(self):
        raw = ("---\ninstructor: X\n---\n# T\n<div class='a'>hi</div>\n```mermaid\ngraph\n```\n![[pic.png]]\n"
               "> [!question]- Zara Quill asked: why?\n> answer\n## To-do\n- [ ] private\n## Next\nok\n")
        out = build_corpus.clean(raw, FAKE_NAMES)
        for bad in ("instructor", "<div", "mermaid", "![[", "Zara", "Quill", "private"):
            self.assertNotIn(bad, out)
        self.assertIn("a classmate asked", out)                         # one "a classmate", not two
        self.assertIn("## Next", out)

    def test_gate_catches_listed_and_unknown_names(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("Orrin said hi. Zelda asked about loss.", encoding="utf-8")
            leaks = build_corpus.find_leaks(Path(d), FAKE_NAMES)
        self.assertIn(("a.md", "Orrin"), leaks)
        self.assertIn(("a.md", "unknown speaker 'Zelda'"), leaks)

    def test_names_file_is_read_longest_first(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "names.txt"
            f.write_text("# comment\nZara\n\nZara Quill\n", encoding="utf-8")
            self.assertEqual(build_corpus.load_names(f), ["Zara Quill", "Zara"])
            self.assertIsNone(build_corpus.load_names(Path(d) / "missing.txt"))

    def test_missing_names_file_fails_the_gate_loudly(self):
        with tempfile.TemporaryDirectory() as notes, tempfile.TemporaryDirectory() as out_dir:
            (Path(notes) / "Week 01 - x.md").write_text("Zelda asked about loss.", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = build_corpus.main(Path(notes), Path(out_dir), Path(notes) / "names.txt")
        self.assertEqual(code, 1)
        self.assertIn("NAME LEAK", out.getvalue())
        self.assertIn("names.txt is missing", out.getvalue())

    def test_real_corpus_has_no_leaks(self):
        self.assertTrue(list((ROOT / "corpus").glob("*.md")), "run build_corpus.py first")
        self.assertEqual(build_corpus.find_leaks(ROOT / "corpus", build_corpus.load_names() or []), [])

    def test_flashcard_and_logistics_sections_are_dropped(self):
        raw = ("# T\n## 3. Real topic\nKeep this.\n## 14. To-do and homework\n- [ ] secret task\n"
               "## Homework and logistics\nhand in Friday\n## Flashcards #flashcards\nWhat is X?::It is Y.\n"
               "## Drawings\n8 drawings\n## Sources and gaps (honest)\ncaptured at 2x\n# Part 5\n## 4. After\nAlso keep.\n")
        out = build_corpus.clean(raw, [])
        for bad in ("secret task", "hand in Friday", "::", "It is Y", "8 drawings", "captured at 2x"):
            self.assertNotIn(bad, out)
        for good in ("Keep this.", "# Part 5", "Also keep."):
            self.assertIn(good, out)

    @unittest.skip("course-specific privacy words were removed from the public copy")
    def test_travel_timezone_chat_and_schedule_lines_go(self):
        raw = ("Embeddings are vectors.\nThe teacher is travelling to Lisbon next month.\n"
               "Times are in UTC-3.\nAsk in the group chat #announcements.\nSee _To delete/old.md.\n"
               "The Oct 16 class may be cancelled.\nRAG beats fine-tuning.\n"
               "> [!warning] SCHEDULE\n> Classes may be skipped.\n> more of the same callout\n"
               "There is a **fine-tuning class next week (if the Oct 9-10 class runs)**, but focus on RAG.\n"
               "## Teacher's corrections (check on Wednesday)\n- To ask Wednesday: did it change?\n")
        out = build_corpus.clean(raw, [])
        for bad in ("Lisbon", "UTC-3", "group chat", "_To delete", "Oct 16", "SCHEDULE", "more of the same",
                    "Oct 9", "next week", "Wednesday"):
            self.assertNotIn(bad, out)
        for good in ("Embeddings are vectors.", "RAG beats fine-tuning.",
                     "There is a **fine-tuning class**, but focus on RAG.", "## Teacher's corrections"):
            self.assertIn(good, out)

    @unittest.skip("course-specific privacy words were removed from the public copy")

    def test_logistics_rules_do_not_eat_lesson_lines(self):
        """'next class' must be a whole phrase and a bare 'travelling' is not logistics,
        while the real travel and next-class lines still go."""
        raw = ("The next classification step is scaling.\n"
               "The travelling salesman problem is a classic search task.\n"
               "The teacher is travelling to Lisbon.\n"
               "Next class: Saturday.\n")
        out = build_corpus.clean(raw, [])
        self.assertIn("The next classification step is scaling.", out)
        self.assertIn("The travelling salesman problem is a classic search task.", out)
        self.assertNotIn("Lisbon", out)
        self.assertNotIn("Next class", out)

    def test_name_map_picks_the_stand_in_and_ignores_capitals(self):
        """'Name = Teacher' in names.txt gives that name its own stand-in; a lower-case
        mention (a search hint) is caught too, and the gate checks it the same way."""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "names.txt"
            f.write_text("# made-up names\nOrrin Bexley\nOrrin\nVashti = Teacher\n", encoding="utf-8")
            name_map = build_corpus.load_name_map(f)
            self.assertEqual(name_map, {"Orrin Bexley": "a classmate", "Orrin": "a classmate",
                                        "Vashti": "Teacher"})
            self.assertEqual(build_corpus.load_names(f), ["Orrin Bexley", "Vashti", "Orrin"])
            out = build_corpus.clean("Vashti is teaching. Orrin Bexley asked. Search 'encoding vashti'.\n",
                                     name_map)
            (Path(d) / "a.md").write_text("see the video by vashti", encoding="utf-8")
            leaks = build_corpus.find_leaks(Path(d), ["Vashti"])
        self.assertEqual(out, "Teacher is teaching. a classmate asked. Search 'encoding Teacher'.\n")
        self.assertIn(("a.md", "Vashti"), leaks)

    def test_real_corpus_token_examples_count_true(self):
        """Every word-level '**Word:** "some words" -> N tokens' example really has N words: a name
        stand-in longer than one word ('the instructor') once made "... is teaching -> 3 tokens" false."""
        found = 0
        for f in (ROOT / "corpus").glob("*.md"):
            for words, n in re.findall(r'\*\*Word:\*\*\s*"([^"]+)"\s*->\s*(\d+) tokens', f.read_text(encoding="utf-8")):
                found += 1
                self.assertEqual(len(words.split()), int(n), f"{f.name}: \"{words}\" -> {n} tokens")
        # The word-token example lived in the old Week 03 note. Every example that IS in corpus/ must count
        # true; zero examples is allowed when the corpus has none (the public copy ships sample notes only).
        self.assertGreaterEqual(found, 0)

    def test_notes_folder_is_not_hard_coded(self):
        """No PC-specific path in the code: the folder comes from the argument, NOTES_DIR, or ./notes."""
        code = (ROOT / "build_corpus.py").read_text(encoding="utf-8")
        self.assertNotRegex(code, r"[A-Z]:\\Users|OneDrive")
        self.assertEqual(build_corpus.NOTES_DIR, Path(os.environ.get("NOTES_DIR") or "notes"))

    def test_study_routine_box_and_wednesday_aside_go(self):
        raw = ("> [!info] How to study this note (15 minutes)\n> 1. Run the flashcards at the bottom.\n"
               "Each breed is its own class. Worth confirming with the instructor on Wednesday.\n")
        out = build_corpus.clean(raw, [])
        for bad in ("How to study", "flashcards", "Wednesday"):
            self.assertNotIn(bad, out)
        self.assertIn("Each breed is its own class.", out)

    @unittest.skip("course-specific privacy words were removed from the public copy")

    def test_qa_table_callout_keeps_its_good_rows(self):
        raw = ("> [!example] Q&A log\n> | Who | Q | A |\n> |---|---|---|\n"
               "> | a classmate | Is overlap good? | Small overlap. |\n> | Class | A 2-week break? | No answer. |\n")
        out = build_corpus.clean(raw, [])
        self.assertIn("Is overlap good?", out)
        self.assertNotIn("2-week break", out)

    def test_code_with_double_colon_is_not_a_flashcard(self):
        raw = "```python\nrev = word[::-1]\n```\nA real card?::yes\n"
        out = build_corpus.clean(raw, [])
        self.assertIn("word[::-1]", out)
        self.assertNotIn("A real card", out)

    def test_paper_dates_are_not_schedule(self):
        out = build_corpus.clean("The paper is dated 11 Oct 2018.\n", [])
        self.assertIn("11 Oct 2018", out)

    @unittest.skip("course-specific privacy words were removed from the public copy")

    def test_gate_catches_private_details(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("Fly to Lisbon.\nQ?::A\nsee 10 Projects/x\n", encoding="utf-8")
            found = build_corpus.find_private_details(Path(d))
        kinds = " ".join(what for _, what in found)
        for word in ("logistics", "flashcard", "vault path"):
            self.assertIn(word, kinds)

    def test_real_corpus_has_no_private_details(self):
        self.assertEqual(build_corpus.find_private_details(ROOT / "corpus"), [])


class EvalNumbers(unittest.TestCase):
    """run_eval's summary must work with any mix of questions."""

    def row(self, kind, score, rank=None):
        ok = rank == 1 if kind == "real" else score < 0.6
        return {"q": "q", "kind": kind, "score": score, "top": "t", "rank": rank, "ok": ok, "mark": "x"}

    def test_normal_mix(self):
        res = run_eval.summarise([self.row("real", 0.8, 1), self.row("real", 0.7, 2), self.row("trap", 0.5)], 0.6)
        self.assertEqual((res["hit"], res["hit3"], res["trap"]), (1, 2, 1))
        self.assertAlmostEqual(res["mrr"], 0.75)
        self.assertEqual(res["suggest"], 0.6)

    def test_no_traps_or_no_real_questions_does_not_crash(self):
        for rows in ([self.row("real", 0.8, 1)], [self.row("trap", 0.4)], []):
            res = run_eval.summarise(rows, 0.6)
            self.assertIsNone(res["suggest"])
            self.assertIn("needs at least one real AND one trap", run_eval.report(res))

    @unittest.skip("the held-out question set is not included in the public copy")

    def test_held_out_set_is_kept_exactly_and_every_question_has_an_answer_key(self):
        raw = json.loads(run_eval.HELD_OUT.read_text(encoding="utf-8"))
        self.assertEqual([len(raw[g]) for g in ("real", "off_topic_traps", "near_topic_traps")], [10, 3, 3])
        qs = run_eval.load_questions(held_out=True)
        self.assertEqual(len(qs), 16)
        for q in qs:
            self.assertIn(q["kind"], ("real", "trap", "near-topic trap"))
            self.assertEqual(q["expect_source"] is None, q["kind"] != "real", q["q"])

    def test_expected_source_can_be_one_note_or_several(self):
        self.assertTrue(run_eval.is_expected("Week 02 - x.md", "Week 02"))
        self.assertTrue(run_eval.is_expected("Week 03 - NLP.md", ["Week 02", "Week 03 - NLP"]))
        self.assertFalse(run_eval.is_expected("Week 03 Saturday.md", ["Week 03 - NLP"]))


class MyQuestions(unittest.TestCase):
    """run_my_questions.py: Bailey's own 7 real + 3 trap questions, scored once, MIN_SCORE never moved."""

    def test_template_has_7_real_and_3_trap_empty_slots_and_counts_as_empty(self):
        data = json.loads(run_my_questions.TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(data["real"], [""] * 7)
        self.assertEqual(data["trap"], [""] * 3)
        self.assertTrue(any("bailey_questions.json" in line for line in data["_instructions"]))
        with self.assertRaises(run_my_questions.QuestionsProblem) as e:
            run_my_questions.load_my_questions(run_my_questions.TEMPLATE)
        self.assertIn("every slot is empty", str(e.exception))

    def test_scorecard_table_summary_and_fixed_min_score(self):
        index = fake_index([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 1]])
        index["chunks"][0]["text"] = "the answer | with a bar"
        vectors = {"close one": [1, 0, 0], "close two": [0, 0, 1], "far trap": [0, 1, 0.6], "near trap": [1, 0.1, 0]}
        replies = {"close one": "It is one [1].", "close two": "It is | two\nlines [1].", "near trap": rag.REFUSAL}
        seen_min = []
        real_ask = rag.ask

        def spy_ask(q, idx, min_score=rag.MIN_SCORE, k=rag.TOP_K):
            seen_min.append(min_score)
            return real_ask(q, idx, min_score, k)

        with tempfile.TemporaryDirectory() as d:
            qfile = Path(d) / "q.json"
            qfile.write_text(json.dumps({"real": ["close one", "", "close two"], "trap": ["far trap", "near trap"]}),
                             encoding="utf-8")
            with mock.patch.object(rag, "load_index", return_value=index), \
                 mock.patch.object(rag, "embed", lambda texts, kind: np.array(
                     [np.array(vectors[t], dtype=np.float32) / np.linalg.norm(vectors[t]) for t in texts])), \
                 mock.patch.object(rag, "generate", side_effect=lambda p: replies[p.split("QUESTION: ")[1].split("\n")[0]]), \
                 mock.patch.object(rag, "ask", spy_ask), contextlib.redirect_stdout(io.StringIO()):
                code = run_my_questions.main(["--questions", str(qfile), "--out-dir", d])
            cards = list(Path(d).glob("my_scorecard_*.md"))
            self.assertEqual(code, 0)
            self.assertEqual(len(cards), 1)
            text = cards[0].read_text(encoding="utf-8")
        self.assertEqual(seen_min, [rag.MIN_SCORE] * 4)                # the empty slot is skipped, cutoff never moves
        self.assertIn("| right | half | wrong |", text)
        rows = [line for line in text.splitlines() if re.match(r"\| \d ", line)]
        self.assertEqual(len(rows), 4)
        for line in rows:
            self.assertTrue(line.endswith("|  |  |  |"), line)          # the three columns are left empty
            self.assertEqual(line.count("|"), 11, line)                 # a '|' in an answer cannot break the table
        self.assertIn("| far trap | trap |", text)
        self.assertIn("**2/2**", text)                                  # both real questions found a source
        self.assertIn("Traps refused (\"Not in my notes.\"): **2/2** - of those, refused before the model was called"
                      " (score under MIN_SCORE): 1", text)

    def test_missing_or_broken_file_says_what_to_do_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            broken = Path(d) / "broken.json"
            broken.write_text('{"real": ["a question" "another"]}', encoding="utf-8")
            for path, words in ((Path(d) / "none.json", "does not exist yet"), (broken, "is not valid JSON")):
                out = io.StringIO()
                with mock.patch.object(rag, "load_index", side_effect=AssertionError("must not load")), \
                     contextlib.redirect_stdout(out):
                    code = run_my_questions.main(["--questions", str(path), "--out-dir", d])
                self.assertEqual(code, 1)
                self.assertIn(words, out.getvalue())
            self.assertEqual(list(Path(d).glob("my_scorecard_*.md")), [])
        with self.assertRaises(SystemExit):                             # there is no way to pass another cutoff
            with contextlib.redirect_stderr(io.StringIO()):
                run_my_questions.main(["--min-score", "0.3"])


class Launchers(unittest.TestCase):
    """The double-click .cmd files. Batch files with Unix line endings can break 'goto', so check CRLF."""
    FILES = ["RUN-CHAT.cmd", "RUN-ASK.cmd", "RUN-MY-QUESTIONS.cmd", "scripts/checks.cmd"]

    def test_cmd_files_use_windows_line_endings(self):
        for name in self.FILES:
            data = (ROOT / name).read_bytes()
            self.assertEqual(data.count(b"\n"), data.count(b"\r\n"), f"{name} has bare LF line endings")

    def test_chat_page_is_local_only_on_port_8599(self):
        text = (ROOT / "RUN-CHAT.cmd").read_text()
        self.assertIn("--server.address 127.0.0.1 --server.port 8599", text)
        self.assertIn('cd /d "%~dp0"', text)
        self.assertIn('cd /d "%~dp0"', (ROOT / "RUN-ASK.cmd").read_text())


def ollama_up() -> bool:
    try:
        rag.ollama.list()
        return True
    except Exception:
        return False


@unittest.skipUnless(ollama_up(), "Ollama is not running")
class Live(unittest.TestCase):
    """Real models. Only the embedding model is used, so these stay fast."""

    @classmethod
    def setUpClass(cls):
        cls.index = rag.load_index(rag.DEFAULT_FOLDER)

    def test_real_question_finds_the_right_note(self):
        # Public copy: the sample notes are about houseplants. A real question should find its note and pass the cutoff.
        hits = rag.retrieve("How often should I water houseplants?", self.index)
        self.assertTrue(hits[0][1]["source"].startswith("Houseplant care basics"))
        self.assertGreaterEqual(hits[0][0], rag.MIN_SCORE)

    def test_trap_refused_without_the_model(self):
        with mock.patch.object(rag, "generate", side_effect=AssertionError("model called")):
            r = rag.ask("What is the capital of Mongolia?", self.index)
        self.assertEqual(r["answer"], rag.REFUSAL)
        self.assertFalse(r["model_called"])

    def test_streamlit_page_empty_folder_and_trap(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as d:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
            self.assertFalse(at.exception)
            at.sidebar.text_input[0].set_value(d).run()
            self.assertIn("0 files found", at.warning[0].value)
        at.sidebar.text_input[0].set_value(str(rag.DEFAULT_FOLDER)).run()
        with mock.patch.object(rag, "generate", side_effect=AssertionError("model called")):
            at.chat_input[0].set_value("What is the capital of Mongolia?").run()
        self.assertFalse(at.exception)
        self.assertIn(rag.REFUSAL, [m.markdown[0].value for m in at.chat_message if m.markdown])


if __name__ == "__main__":
    unittest.main()
