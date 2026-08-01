"""
Multi-Course Question Answering System - Gradio deployment.

Launches a browser UI on top of the shared `rag_core` pipeline: a student
picks a course (or "All Courses"), asks a question in plain English, and
receives an answer that is grounded exclusively in the retrieved course
material, together with full source attribution.

Run with:  python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_core import (
    AppConfig,
    AnswerFormatter,
    DocumentProcessor,
    EmbeddingManager,
    PromptManager,
    RAGPipeline,
    RetrieverEngine,
    VectorDatabaseManager,
)
from rag_core.logging_setup import get_logger

logger = get_logger("app")

# ---------------------------------------------------------------------------
# Pipeline bootstrap
# ---------------------------------------------------------------------------
config = AppConfig()
config.ensure_directories()

embedding_manager = EmbeddingManager(config)
vector_db_manager = VectorDatabaseManager(config, embedding_manager)

VECTOR_DB_READY = vector_db_manager.exists()
BOOTSTRAP_ERROR: str | None = None

if VECTOR_DB_READY:
    try:
        vector_store = vector_db_manager.load()
        retriever = RetrieverEngine(config, vector_store)
        prompt_manager = PromptManager(config)
        pipeline: RAGPipeline | None = RAGPipeline(config, retriever, prompt_manager)
    except Exception as exc:  # pragma: no cover - defensive bootstrap
        logger.error("Failed to load the vector database: %s", exc)
        BOOTSTRAP_ERROR = str(exc)
        pipeline = None
else:
    BOOTSTRAP_ERROR = (
        "No persisted vector database was found under 'chroma_db/'. "
        "Run notebooks 01 and 02 first to build it."
    )
    pipeline = None

COURSE_OPTIONS = ["All Courses"] + list(config.course_names)

EXAMPLE_QUESTIONS = [
    "What is the Turing Test?",
    "What is the difference between overfitting and underfitting?",
    "What does backpropagation do in a neural network?",
    "What evaluation metrics are used for classification tasks?",
]


# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------
def handle_question(question: str, course: str):
    """Run a question through the RAG pipeline and format every UI output."""
    if pipeline is None:
        message = BOOTSTRAP_ERROR or "The system is not ready yet."
        return (
            f"### Answer\n{message}",
            [],
            "No chunks retrieved.",
            "",
        )

    course_filter = None if course == "All Courses" else course
    result = pipeline.answer(question, course_filter=course_filter)

    answer_md = AnswerFormatter.to_markdown(result)
    source_rows = AnswerFormatter.to_source_table(result)
    chunks_preview = AnswerFormatter.to_chunks_preview(result)
    timing = f"Generated in {result.generation_seconds}s" if result.generation_seconds else ""

    return answer_md, source_rows, chunks_preview, timing


def clear_fields():
    return "", "All Courses", "", [], "", ""


def fill_example(example_text: str):
    return example_text


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------
def db_status_badge() -> str:
    if pipeline is None:
        return '<span class="status-badge status-bad">&#9679; Not Ready</span>'
    count = vector_db_manager.count()
    return f'<span class="status-badge status-good">&#9679; Ready &middot; {count} vectors</span>'


PIPELINE_DIAGRAM_HTML = """
<div class="pipeline-diagram">
  <div class="pipe-node"><span class="pipe-tag">01</span>Student Question</div>
  <div class="pipe-arrow">&#8595;</div>
  <div class="pipe-node"><span class="pipe-tag">02</span>Retriever<br><span class="pipe-sub">Top-K similarity search</span></div>
  <div class="pipe-arrow">&#8595;</div>
  <div class="pipe-node"><span class="pipe-tag">03</span>Prompt Builder<br><span class="pipe-sub">Grounded, no-hallucination template</span></div>
  <div class="pipe-arrow">&#8595;</div>
  <div class="pipe-node"><span class="pipe-tag">04</span>Ollama LLM<br><span class="pipe-sub">qwen2.5:3b &middot; temperature 0</span></div>
  <div class="pipe-arrow">&#8595;</div>
  <div class="pipe-node pipe-final"><span class="pipe-tag pipe-tag-final">05</span>Answer + Sources + Confidence</div>
</div>
"""

# ---------------------------------------------------------------------------
# Palette — "annotated textbook": ink navy + warm amber highlighter on
# parchment, evoking a well-marked-up course reader rather than a generic
# AI gradient. Serif display type for the scholarly voice, clean sans for
# data and UI chrome.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');

:root, .dark {
    --ink: #ECE7D8;          /* primary text — warm parchment-white on dark */
    --ink-soft: #A9B3D0;     /* secondary text / labels */
    --parchment: #1A2032;    /* card surface (was light paper, now dark surface) */
    --surface-raised: #212940; /* inputs, table rows, raised chrome */
    --rule: #2E3856;         /* hairline borders */
    --amber: #E7A63E;        /* highlighter accent, brightened for dark bg */
    --amber-soft: #362A15;   /* dark amber wash for tag backgrounds */
    --sage: #6FB98A;         /* verified / good */
    --sage-soft: #16261D;
    --rust: #E2806B;         /* error / not ready */
    --rust-soft: #331F1B;

    /* Re-point Gradio's own theme tokens so built-in inputs, tabs, and
       tables follow the same dark palette instead of staying light. */
    --body-background-fill: #10141F;
    --background-fill-primary: #10141F;
    --background-fill-secondary: #161C2C;
    --block-background-fill: var(--parchment);
    --block-border-color: var(--rule);
    --block-label-text-color: var(--ink-soft);
    --border-color-primary: var(--rule);
    --border-color-accent: var(--amber);
    --body-text-color: var(--ink);
    --body-text-color-subdued: var(--ink-soft);
    --input-background-fill: var(--surface-raised);
    --input-border-color: var(--rule);
    --input-placeholder-color: #6C7594;
    --button-secondary-background-fill: var(--surface-raised);
    --button-secondary-background-fill-hover: #29334F;
    --button-secondary-text-color: var(--ink);
    --button-secondary-border-color: var(--rule);
    --table-even-background-fill: var(--parchment);
    --table-odd-background-fill: #161C2C;
    --table-border-color: var(--rule);
    --panel-background-fill: var(--parchment);
}

.gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #10141F !important;
}

h1, h2, h3, .hero-card h1 {
    font-family: 'Source Serif 4', Georgia, serif !important;
}

.hero-card {
    position: relative;
    background: #0C0F1A;
    background-image: radial-gradient(circle at 88% 18%, rgba(231, 166, 62, 0.22), transparent 55%);
    border-radius: 6px;
    padding: 30px 34px;
    color: var(--ink);
    margin-bottom: 18px;
    border-left: 5px solid var(--amber);
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45);
}
.hero-card h1 {
    margin: 0 0 8px 0;
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    color: var(--ink);
}
.hero-card p {
    margin: 0;
    color: var(--ink-soft);
    font-size: 0.98rem;
}

.info-card {
    border-radius: 8px;
    padding: 20px 22px;
    background: var(--parchment);
    border: 1px solid var(--rule);
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.3);
}
.info-card h3, .info-card strong {
    color: var(--ink);
}
.info-card, .info-card p, .info-card li, .info-card td {
    color: var(--ink-soft);
}
.info-card table {
    color: var(--ink-soft);
}

.answer-card {
    border-radius: 8px;
    padding: 22px 24px;
    background: var(--parchment);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--amber);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    color: var(--ink-soft);
}
.answer-card h3 {
    color: var(--ink);
}

/* Pipeline diagram — a citation trail with numbered tabs on dark cards. */
.pipeline-diagram {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 14px 0 6px 0;
}
.pipe-node {
    position: relative;
    background: var(--surface-raised);
    border: 1px solid var(--rule);
    border-radius: 6px;
    padding: 10px 22px 10px 44px;
    text-align: left;
    font-weight: 600;
    color: var(--ink);
    min-width: 250px;
}
.pipe-tag {
    position: absolute;
    left: -1px;
    top: -1px;
    bottom: -1px;
    width: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--amber-soft);
    color: var(--amber);
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 0.78rem;
    border-right: 1px solid var(--rule);
    border-radius: 6px 0 0 6px;
}
.pipe-tag-final {
    background: var(--amber);
    color: #10141F;
    border-right: none;
}
.pipe-final {
    background: #0C0F1A;
    color: var(--ink);
    border: 1px solid var(--amber);
}
.pipe-sub {
    display: block;
    font-weight: 400;
    font-size: 0.78rem;
    color: var(--ink-soft);
    margin-top: 2px;
}
.pipe-final .pipe-sub {
    color: #B9C3D6;
}
.pipe-arrow {
    font-size: 1rem;
    color: var(--amber);
    letter-spacing: -1px;
}

.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.status-good {
    background: var(--sage-soft);
    color: var(--sage);
}
.status-bad {
    background: var(--rust-soft);
    color: var(--rust);
}

/* Buttons: amber-on-dark primary, ink hover */
button.primary, .gr-button-primary {
    background: var(--amber) !important;
    border-color: var(--amber) !important;
    color: #10141F !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #F2B95A !important;
    border-color: #F2B95A !important;
    color: #10141F !important;
}

footer.app-footer {
    text-align: center;
    padding: 16px 0 4px 0;
    font-size: 0.8rem;
    color: var(--ink-soft);
    opacity: 0.7;
}
"""

# Forces dark mode on load, independent of the visitor's OS preference, so
# the app always renders the dark "annotated textbook at night" palette.
FORCE_DARK_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.replace(url.toString());
    }
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Course Materials Q&A", js=FORCE_DARK_JS) as demo:

    gr.HTML(
        """
        <div class="hero-card">
            <h1>Multi-Course Question Answering System</h1>
            <p>Ask questions about your course materials. Answers are grounded strictly in retrieved content — no hallucinations.</p>
        </div>
        """
    )

    with gr.Tabs():
        # ------------------------------------------------------------
        # HOME TAB
        # ------------------------------------------------------------
        with gr.Tab("Home"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        ### Project Overview

                        This system implements a classic **Retrieval-Augmented
                        Generation (RAG)** pipeline over university course
                        materials spanning three courses: **Artificial
                        Intelligence**, **Machine Learning**, and **Deep
                        Learning**.

                        Supported source formats: **PDF · DOCX · TXT · CSV**.

                        Every answer is generated **only** from the retrieved
                        chunks. If the material does not contain the answer,
                        the system says so explicitly instead of guessing.
                        """,
                        elem_classes=["info-card"],
                    )
                with gr.Column(scale=1):
                    gr.HTML(PIPELINE_DIAGRAM_HTML, elem_classes=["info-card"])

        # ------------------------------------------------------------
        # Q&A TAB
        # ------------------------------------------------------------
        with gr.Tab("Ask a Question"):
            with gr.Row():
                with gr.Column(scale=2):
                    course_dropdown = gr.Dropdown(
                        choices=COURSE_OPTIONS,
                        value="All Courses",
                        label="Course",
                    )
                    question_box = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. What is the Turing Test?",
                        lines=3,
                    )
                    with gr.Row():
                        ask_btn = gr.Button("Ask", variant="primary")
                        clear_btn = gr.Button("Clear")

                    gr.Markdown("**Example questions**")
                    with gr.Row():
                        example_buttons = [gr.Button(q, size="sm") for q in EXAMPLE_QUESTIONS]

                with gr.Column(scale=3):
                    answer_output = gr.Markdown(
                        value="### Answer\nAsk a question to see the answer here.",
                        elem_classes=["answer-card"],
                    )
                    timing_output = gr.Markdown(value="")

                    with gr.Accordion("Retrieved Sources", open=True):
                        sources_table = gr.Dataframe(
                            headers=["Course", "File", "Page", "Chunk ID", "Similarity"],
                            datatype=["str", "str", "str", "str", "number"],
                            value=[],
                            interactive=False,
                        )

                    with gr.Accordion("Retrieved Chunks (raw text)", open=False):
                        chunks_output = gr.Textbox(
                            value="",
                            lines=10,
                            interactive=False,
                            show_label=False,
                        )

            ask_btn.click(
                fn=handle_question,
                inputs=[question_box, course_dropdown],
                outputs=[answer_output, sources_table, chunks_output, timing_output],
            )
            question_box.submit(
                fn=handle_question,
                inputs=[question_box, course_dropdown],
                outputs=[answer_output, sources_table, chunks_output, timing_output],
            )
            clear_btn.click(
                fn=clear_fields,
                inputs=[],
                outputs=[question_box, course_dropdown, answer_output, sources_table, chunks_output, timing_output],
            )
            for btn in example_buttons:
                btn.click(fn=fill_example, inputs=[btn], outputs=[question_box])

        # ------------------------------------------------------------
        # SYSTEM STATUS TAB
        # ------------------------------------------------------------
        with gr.Tab("System Status"):
            gr.HTML(f"<div class='info-card'>Vector Database: {db_status_badge()}</div>")
            gr.Markdown(
                f"""
                <div class="info-card">

                | Setting | Value |
                |---|---|
                | Embedding Model | `{config.embedding_model_name}` |
                | LLM Model (Ollama) | `{config.llm_model_name}` |
                | Temperature | `{config.llm_temperature}` |
                | Top-K | `{config.top_k}` |
                | Chunk Size / Overlap | `{config.chunk_size} / {config.chunk_overlap}` |
                | Persist Directory | `{config.persist_directory}` |
                | Supported Courses | {", ".join(config.course_names)} |
                | Supported Formats | {", ".join(config.supported_extensions)} |

                </div>
                """
            )
            if BOOTSTRAP_ERROR:
                gr.Markdown(f"**Note:** {BOOTSTRAP_ERROR}")

    gr.HTML(
        """
        <footer class="app-footer">
            Multi-Course Question Answering System · Built with LangChain, ChromaDB, HuggingFace Embeddings and Ollama
        </footer>
        """
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue=gr.themes.Color(
                name="ink",
                c50="#EEF1F6", c100="#DCE2ED", c200="#B9C5DA", c300="#93A5C4",
                c400="#5D719D", c500="#33456B", c600="#283A5B", c700="#1E2A44",
                c800="#182036", c900="#101828", c950="#0B0F1A",
            ),
            secondary_hue=gr.themes.Color(
                name="amber",
                c50="#FCF6E9", c100="#F6E4BC", c200="#EFD08E", c300="#E8BC60",
                c400="#E0A83E", c500="#D9932E", c600="#B87824", c700="#93601D",
                c800="#6E4816", c900="#49300F", c950="#241804",
            ),
        ),
        css=CUSTOM_CSS,
    )
