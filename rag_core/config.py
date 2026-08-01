"""
Central configuration for the Multi-Course Question Answering System.

Every tunable value used across document processing, embedding,
vector storage, retrieval and generation is declared here so that
nothing is hard-coded inside the notebooks or ``app.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Single source of truth for every configurable parameter.

    Instances can be created with defaults, or overridden field by
    field, e.g. ``AppConfig(chunk_size=800, top_k=6)``.
    """

    # ------------------------------------------------------------------
    # Filesystem locations
    # ------------------------------------------------------------------
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    persist_directory: Path = field(init=False)
    processed_chunks_path: Path = field(init=False)

    # ------------------------------------------------------------------
    # Supported course materials
    # ------------------------------------------------------------------
    supported_extensions: tuple = (".pdf", ".docx", ".txt", ".csv")
    course_names: tuple = ("Artificial Intelligence", "Machine Learning", "Deep Learning")

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunk_separators: tuple = ("\n\n", "\n", ". ", " ", "")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_normalize: bool = True

    # ------------------------------------------------------------------
    # Vector database
    # ------------------------------------------------------------------
    collection_name: str = "course_materials"

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = 4
    score_threshold: float | None = None  # None disables filtering by score

    # ------------------------------------------------------------------
    # LLM (Ollama)
    # ------------------------------------------------------------------
    llm_model_name: str = "qwen2.5:3b"
    llm_temperature: float = 0.0
    llm_top_p: float = 0.9
    llm_num_ctx: int = 4096
    llm_base_url: str = "http://localhost:11434"
    llm_request_timeout: int = 120

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    no_answer_message: str = "I couldn't find this information in the provided course materials."
    conflict_message: str = "The course materials contain conflicting information on this topic."

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.persist_directory = self.project_root / "chroma_db"
        self.processed_chunks_path = self.project_root / "chroma_db" / "processed_chunks.jsonl"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build a config, letting environment variables override defaults.

        Recognised variables: ``RAG_CHUNK_SIZE``, ``RAG_CHUNK_OVERLAP``,
        ``RAG_TOP_K``, ``RAG_EMBEDDING_MODEL``, ``RAG_LLM_MODEL``,
        ``RAG_LLM_TEMPERATURE``, ``RAG_OLLAMA_BASE_URL``.
        """
        cfg = cls()
        cfg.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", cfg.chunk_size))
        cfg.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", cfg.chunk_overlap))
        cfg.top_k = int(os.getenv("RAG_TOP_K", cfg.top_k))
        cfg.embedding_model_name = os.getenv("RAG_EMBEDDING_MODEL", cfg.embedding_model_name)
        cfg.llm_model_name = os.getenv("RAG_LLM_MODEL", cfg.llm_model_name)
        cfg.llm_temperature = float(os.getenv("RAG_LLM_TEMPERATURE", cfg.llm_temperature))
        cfg.llm_base_url = os.getenv("RAG_OLLAMA_BASE_URL", cfg.llm_base_url)
        return cfg

    def ensure_directories(self) -> None:
        """Create the data and persistence directories if missing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        for course in self.course_names:
            (self.data_dir / course).mkdir(parents=True, exist_ok=True)
