"""Streamlit chat page for the assistant: pick a notes folder, ask, see the sources.

Run:  .venv\\Scripts\\streamlit run app.py
All the thinking happens in rag.py; this file only draws the page.
"""
from pathlib import Path

import streamlit as st

import rag

st.set_page_config(page_title="Ask my notes", layout="wide")

# Notebook-paper look: cream page, blue ruled lines, red margin line, dark ink. Topic bubbles are round.
NOTEBOOK_CSS = """
<style>
.stApp {
    background-color: #fdfaf0;
    background-image:
        linear-gradient(to right, transparent 64px, #e8a0a0 64px, #e8a0a0 66px, transparent 66px),
        repeating-linear-gradient(to bottom, transparent 0, transparent 31px, #b9d3ea 31px, #b9d3ea 32px);
    color: #1f2937;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stChatInput"] { background: #ffffff; border: 1px solid #c9c2ad; }
[data-testid="stChatInput"] textarea::placeholder { color: #9ca3af; opacity: 1; }
[data-testid="stSidebar"] { background-color: #f6efdc; }
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp span, .stApp small { color: #1f2937; }
.st-key-topics button {
    border-radius: 999px !important;
    border: 2px solid #7aa7d6 !important;
    background: #ffffff !important;
    color: #1f2937 !important;
    padding: 0.35rem 1rem !important;
}
.st-key-topics button:hover { background: #e8f1fb !important; border-color: #3f74b5 !important; }
</style>
"""
st.markdown(NOTEBOOK_CSS, unsafe_allow_html=True)

def load_topics(path: Path = Path(__file__).parent / "topics.txt") -> list[tuple[str, str]]:
    """Topic bubbles from topics.txt: one 'Label | question' per line. The subject words live in that
    data file, not in this code, so the code works for any folder of notes."""
    if not path.is_file():
        return []
    topics = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "|" in line and not line.lstrip().startswith("#"):
            label, question = (part.strip() for part in line.split("|", 1))
            topics.append((label, question))
    return topics


TOPICS = load_topics()


@st.cache_resource(show_spinner="Reading and embedding the notes...")
def get_index(folder: str, stamp: tuple[int, float]) -> dict:
    """Cached per folder. `stamp` = (number of note files, sum of their change times), so adding,
    editing or deleting a note gives a new stamp and the page re-reads the folder."""
    return rag.load_index(Path(folder))


def stop_with_fix(e: rag.ModelError) -> None:
    """Ollama is off, a model is missing or cannot load: show how to fix it instead of a red traceback."""
    st.error(str(e))
    st.stop()


# ---------- sidebar: which notes, which threshold ----------
with st.sidebar:
    st.header("Knowledge base")
    folder = Path(st.text_input("Notes folder", str(rag.DEFAULT_FOLDER)))
    uploads = st.file_uploader("...or add .md / .txt files to that folder", type=["md", "txt"],
                               accept_multiple_files=True)
    if uploads and st.button("Save uploads"):
        folder.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (folder / Path(f.name).name).write_bytes(f.getvalue())
        st.success(f"Saved {len(uploads)} file(s). Re-reading the folder.")
    min_score = st.slider("Strictness (higher = refuses more often)", 0.30, 0.90, float(rag.MIN_SCORE), 0.01,
                          help="How closely a passage must match your question before the assistant uses it. "
                               f"The recommended setting is {rag.MIN_SCORE}, tuned on the sample questions.")
    st.caption(f"Two AI models run on this PC through Ollama: {rag.EMBED_MODEL} finds the matching passages, "
               f"and {rag.CHAT_MODEL} writes the answer.")
    if st.button("Clear chat"):
        st.session_state.history = []

try:
    index = get_index(str(folder), rag.folder_stamp(folder))
except rag.ModelError as e:
    stop_with_fix(e)
st.title("Ask my notes")
if not index["chunks"]:
    st.warning(f"0 files found in {folder}. Put .md or .txt notes on any subject in that folder.")
    st.stop()
st.markdown("**What this is:** a chat that answers questions about your notes. It only uses the notes "
            "in the folder below, and it shows the passages it used on the right. If the notes don't cover "
            "your question, it says so instead of guessing.")
st.caption(f"Reading {index['files']} note files, split into {len(index['chunks'])} passages, from the "
           f"`{folder.name}/` folder.")
st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", None)


def queue_topic(question: str) -> None:
    """Button callback: remember the topic question so the next run answers it like a typed question."""
    st.session_state["pending"] = question


st.caption("Tap a topic to see what the notes say about it, or type your own question below.")
with st.container(key="topics"):
    for start in range(0, len(TOPICS), 3):
        row = st.columns(3)
        for col, (label, question) in zip(row, TOPICS[start:start + 3]):
            col.button(label, key=f"topic-{label}", on_click=queue_topic, args=(question,))

chat_col, source_col = st.columns([3, 2])


def show_sources(result: dict) -> None:
    st.subheader("Sources")
    st.caption(result["why"])
    if not result["sources"]:
        st.info("No sources - the assistant refused instead of guessing.")
    for s in result["sources"]:
        used = " (cited)" if s["n"] in result["cited"] else ""
        with st.expander(f"[{s['n']}] {Path(s['source']).stem} > {s['heading']}  -  score {s['score']:.2f}{used}",
                         expanded=s["n"] in result["cited"]):
            st.markdown(s["text"])


with chat_col:
    for turn in st.session_state.history:
        st.chat_message("user").write(turn["question"])
        st.chat_message("assistant").write(turn["answer"])
    typed = st.chat_input("What do you want to know from your notes?")
    question = typed or st.session_state.pop("pending", None)
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Searching the notes..."):
            try:
                result = rag.ask(question, index, min_score=min_score)
            except rag.ModelError as e:
                stop_with_fix(e)
        st.chat_message("assistant").write(result["answer"])
        st.session_state.history.append(result)

with source_col:
    if st.session_state.history:
        show_sources(st.session_state.history[-1])
    else:
        st.subheader("Sources")
        st.caption("The notes behind each answer show up here.")
