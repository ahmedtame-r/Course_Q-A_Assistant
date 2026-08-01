<div align="center">

# Course Materials Q&A

**A retrieval-augmented tutor that only says what your course materials say.**

*No guessing. No hallucinating. Just answers, cited, from the page they came from.* 📖

[![Python](https://img.shields.io/badge/python-3.10%2B-1E2A44?style=flat-square)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-D9932E?style=flat-square)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-4C7A5A?style=flat-square)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-qwen2.5%3A3b-B14B3B?style=flat-square)](https://ollama.com/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-1E2A44?style=flat-square)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-33456B?style=flat-square)](#license)

</div>

<br>

<p align="center">
  <a href="#the-idea">The Idea</a> •
  <a href="#how-it-thinks">How It Thinks</a> •
  <a href="#features">Features</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## The Idea

Ask a general-purpose chatbot a question about your coursework and you get a plausible-sounding answer stitched together from the internet at large — half right, confidently wrong, or quietly making things up.

> **Course Materials Q&A** takes the opposite approach. Every answer is retrieved directly from the PDFs, slides, and notes you actually studied from, then generated with a strict no-hallucination prompt template. If the material doesn't cover it, the system says so — instead of inventing something that sounds plausible.

Think of it less like a chatbot and more like a study partner who has actually done the reading, keeps a finger on the exact page, and refuses to bluff.

## How It Thinks

```
   Student Question
          │
          ▼
   Retriever              top-K similarity search over your course chunks
          │
          ▼
   Prompt Builder          grounded, no-hallucination template
          │
          ▼
   Ollama LLM               qwen2.5:3b · temperature 0 (deterministic, no improvising)
          │
          ▼
   Answer + Sources + Confidence
```

Every response comes back with the course, file, page, and chunk it was pulled from, so you can check the primary source yourself instead of taking the model's word for it.

## Features

| | |
|---|---|
| **Multi-course, filterable** | Scope a question to a single course or search across all of them at once |
| **Full source attribution** | Every answer links back to course, file, page, chunk ID, and similarity score |
| **Grounded by design** | The prompt template refuses to answer beyond what was retrieved — no silent hallucination |
| **Format-agnostic ingestion** | PDF, DOCX, TXT, and CSV course materials all feed the same pipeline |
| **Raw chunk inspection** | Expand the retrieved passages to see exactly what the model was working from |
| **Local-first** | Runs entirely on Ollama, so your course materials never leave your machine |
| **A frontend with its own identity** | An "annotated textbook" visual theme — ink and amber highlighter tones — with a dark mode built in |

## Tech Stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain |
| Vector store | ChromaDB (persisted locally) |
| Embeddings | HuggingFace sentence embeddings |
| LLM | Ollama — `qwen2.5:3b`, temperature `0` |
| UI | Gradio |

## Getting Started

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/course-materials-qa.git
cd course-materials-qa
pip install -r requirements.txt

# 2. Pull the local model
ollama pull qwen2.5:3b

# 3. Build the vector database from your course materials
#    (run notebooks 01 and 02 — ingestion + embedding)
jupyter notebook

# 4. Launch the app
python app.py
```

The app looks for a persisted database under `chroma_db/`. If it isn't there yet, the **System Status** tab will tell you exactly which notebooks to run first instead of failing silently.

## Project Structure

```
.
├── app.py                  # Gradio frontend
├── rag_core/                # shared pipeline package
│   ├── config.py             # AppConfig — models, chunking, paths
│   ├── document_processor.py
│   ├── embeddings.py
│   ├── vector_db.py
│   ├── retriever.py
│   ├── prompt_manager.py
│   ├── pipeline.py           # RAGPipeline — ties retrieval + generation together
│   └── formatter.py          # AnswerFormatter — markdown, source table, chunk preview
├── notebooks/
│   ├── 01_ingest.ipynb       # load & chunk course materials
│   └── 02_embed.ipynb        # build the persisted vector store
└── chroma_db/                # persisted vector database (generated)
```

## Roadmap

- [ ] Streaming token-by-token answers
- [ ] Per-course confidence calibration
- [ ] Swap-in support for larger local models
- [ ] Export a conversation + sources as a study sheet (PDF)

## Contributing

Issues and pull requests are welcome, especially around retrieval quality, prompt grounding, and new source formats.

## License

MIT — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
<sub>Built with LangChain, ChromaDB, HuggingFace Embeddings, and Ollama.</sub>
</div>
