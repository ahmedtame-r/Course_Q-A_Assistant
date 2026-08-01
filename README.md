# Multi-Course Question Answering System (RAG)

A Retrieval-Augmented Generation system that answers student questions
using only the content of university course materials. If an answer isn't
in the course material, the system says so instead of guessing.

Built as a classic RAG pipeline — no agents, no knowledge graphs, no
re-ranking, no query expansion. Just:

```
Documents -> Chunking -> Embeddings -> ChromaDB -> Retriever -> Prompt -> Ollama LLM -> Answer
```

---

## Features

- **Multi-course support** — Artificial Intelligence, Machine Learning, and
  Deep Learning, each with its own subfolder under `data/`.
- **Multi-format ingestion** — PDF, DOCX, TXT, and CSV are all discovered
  and parsed automatically. Unsupported files are skipped with a warning,
  not an error.
- **Grounded answers only** — the prompt explicitly forbids the model from
  using outside knowledge, and instructs it to say
  *"I couldn't find this information in the provided course materials."*
  when the retrieved context doesn't contain the answer.
- **Conflict detection** — if retrieved sources disagree, the model is
  instructed to say so rather than silently pick one.
- **Full source attribution** — every answer lists the course, file name,
  page number (when applicable), chunk ID, and similarity score behind it.
- **Persisted vector store** — ChromaDB is built once and reloaded on
  every subsequent run; embeddings are never recomputed unnecessarily.
- **Local, private inference** — generation runs through a locally hosted
  Ollama model (`qwen2.5:3b` by default), so no course material ever
  leaves the machine.
- **Gradio UI** — a from-scratch interface with course filtering, example
  questions, a live system-status panel, and full source inspection.

---

## Architecture

```
                         User
                          |
                          v
                   Gradio Interface  (app.py)
                          |
                          v
                    RAG Pipeline  (rag_core.pipeline.RAGPipeline)
                          |
            +-------------+--------------+
            v                             v
       Retriever                   Prompt Builder
  (rag_core.retriever)          (rag_core.prompt_manager)
            |                             |
            v                             v
        ChromaDB                    Ollama LLM
  (rag_core.vector_store)        (qwen2.5:3b, temp=0)
            ^
            |
      Embedding Model
  (rag_core.embeddings, all-MiniLM-L6-v2)
            ^
            |
     Document Processing
  (rag_core.document_processor)
            ^
            |
       Course Materials
     (data/<course>/*.{pdf,docx,txt,csv})
```

All shared logic lives in the `rag_core/` package so the notebooks and
`app.py` import the exact same classes instead of duplicating code.

---

## Folder Structure

```
Project/
├── data/
│   ├── Artificial Intelligence/
│   ├── Machine Learning/
│   └── Deep Learning/
├── chroma_db/                       # created by notebook 02 / first app.py run
├── notebooks/
│   ├── 01_Document_Preprocessing.ipynb
│   ├── 02_Vector_Database.ipynb
│   └── 03_RAG_System.ipynb
├── rag_core/                        # shared library used by notebooks + app.py
│   ├── __init__.py
│   ├── config.py                    # AppConfig — every tunable parameter
│   ├── document_processor.py        # DocumentProcessor — load/clean/chunk
│   ├── embeddings.py                # EmbeddingManager
│   ├── vector_store.py              # VectorDatabaseManager (ChromaDB)
│   ├── retriever.py                 # RetrieverEngine
│   ├── prompt_manager.py            # PromptManager
│   ├── pipeline.py                  # RAGPipeline, RAGAnswer
│   ├── formatter.py                 # AnswerFormatter
│   └── logging_setup.py
├── app.py
├── requirements.txt
└── README.md
```

> **Note on structure:** the brief lists `app.py`, `requirements.txt`, and
> `README.md` at the project root alongside `data/`, `chroma_db/`, and
> `notebooks/` — that layout is followed exactly. The `rag_core/` package
> is an addition on top of it: rather than re-implementing the `Config`,
> `DocumentProcessor`, `EmbeddingManager`, etc. classes separately inside
> each notebook and inside `app.py` (which would duplicate and drift),
> they live once in `rag_core/` and are imported everywhere. This keeps
> the notebooks focused on demonstrating the pipeline step by step while
> `app.py` stays a thin UI layer over the same code.

---

## Installation

**Requirements:**
- Python 3.10+
- [Ollama](https://ollama.com) installed locally
- ~2 GB free disk space (embedding model + vector store)

```bash
# 1. Clone / unzip the project, then cd into it
cd Project

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull the LLM used by the pipeline
ollama pull qwen2.5:3b

# 5. Start the Ollama server (leave this running in its own terminal)
ollama serve
```

---

## Running the Notebooks

Run them in order from the `notebooks/` folder (or from the project root —
each notebook resolves the project root automatically):

1. **`01_Document_Preprocessing.ipynb`** — discovers every supported file
   under `data/`, validates and cleans the text, splits it into chunks,
   attaches metadata, and saves the result to
   `chroma_db/processed_chunks.jsonl`. Runs fully offline.
2. **`02_Vector_Database.ipynb`** — loads the processed chunks, embeds
   them with `sentence-transformers/all-MiniLM-L6-v2`, and builds/persists
   the ChromaDB collection under `chroma_db/`. Requires internet access on
   first run to download the embedding model.
3. **`03_RAG_System.ipynb`** — loads the persisted vector store, wires up
   the retriever, prompt template, and Ollama LLM into the full
   `RAGPipeline`, then runs example questions, an evaluation pass, and two
   deliberate failure-case questions. Requires Ollama running locally.

Each notebook can be re-run top to bottom without modification.

## Building the Database

The database is built automatically the first time you run
`02_Vector_Database.ipynb` (or the first time `app.py` starts and finds no
existing `chroma_db/`). To rebuild it from scratch after changing course
materials:

```python
from rag_core import AppConfig, DocumentProcessor, EmbeddingManager, VectorDatabaseManager

config = AppConfig()
chunks = DocumentProcessor(config).process_all()
VectorDatabaseManager(config, EmbeddingManager(config)).build(chunks, overwrite=True)
```

## Running the Application

```bash
python app.py
```

This opens a local Gradio URL (default `http://127.0.0.1:7860`). The app
loads the persisted `chroma_db/` collection on startup — run notebooks 01
and 02 first if it doesn't exist yet.

---

## Example Questions

| Course | Example Question |
|---|---|
| Artificial Intelligence | What is the Turing Test? |
| Artificial Intelligence | What does the A* algorithm combine to find optimal paths? |
| Machine Learning | What is the difference between overfitting and underfitting? |
| Machine Learning | What clustering algorithms are commonly used? |
| Deep Learning | What is backpropagation used for? |
| Deep Learning | What does dropout do during training? |
| *(out of scope)* | What is the boiling point of mercury? → correctly returns the "not found" message |

---

## Technologies Used

- **LangChain** (`langchain-core`, `langchain-text-splitters`) — chunking and prompt templating
- **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) via `langchain-huggingface`
- **ChromaDB** via `langchain-chroma` — persisted vector storage and similarity search
- **Ollama** (`qwen2.5:3b`) via `langchain-ollama` — local LLM inference
- **Gradio** — deployment UI
- **pypdf**, **python-docx** — document loaders

---

## Future Improvements

- Add a lightweight evaluation harness (e.g. Ragas) for retrieval and
  answer-quality metrics beyond the manual checks in Notebook 3.
- Support additional formats (PPTX, Markdown) with the same
  discover-and-skip-unsupported pattern already in place.
- Add per-course upload through the Gradio UI so new material can be
  ingested without touching the notebooks.
- Cache repeated questions to reduce redundant LLM calls.
