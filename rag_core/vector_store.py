"""Persisted ChromaDB vector store management."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

from rag_core.config import AppConfig
from rag_core.document_processor import ProcessedChunk
from rag_core.embeddings import EmbeddingManager
from rag_core.logging_setup import get_logger


class VectorDatabaseManager:
    """Creates, persists and reloads the ChromaDB collection."""

    def __init__(self, config: AppConfig, embedding_manager: EmbeddingManager) -> None:
        self.config = config
        self.embedding_manager = embedding_manager
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        self._store: Chroma | None = None

    def exists(self) -> bool:
        """Whether a persisted Chroma database is already present on disk.

        Checks specifically for Chroma's own SQLite file rather than "any
        file in the directory" — `persist_directory` also holds
        `processed_chunks.jsonl` (written by `DocumentProcessor`), which
        would otherwise cause a false positive and make `build()` skip
        embedding entirely, loading an empty collection instead.
        """
        sqlite_file = Path(self.config.persist_directory) / "chroma.sqlite3"
        return sqlite_file.exists() and sqlite_file.stat().st_size > 0

    def build(self, chunks: list[ProcessedChunk], overwrite: bool = False) -> Chroma:
        """Embed and persist a fresh collection from processed chunks.

        If a database already exists and ``overwrite`` is False, the
        existing database is loaded instead of being recomputed, per
        the "never recreate embeddings unnecessarily" requirement.
        """
        if self.exists() and not overwrite:
            self.logger.info("Existing vector database found. Loading it instead of rebuilding.")
            return self.load()

        if not chunks:
            raise ValueError("Cannot build a vector database from an empty chunk list.")

        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        self.logger.info("Embedding and indexing %d chunk(s)...", len(chunks))
        try:
            self._store = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding_manager.model,
                metadatas=metadatas,
                ids=ids,
                collection_name=self.config.collection_name,
                persist_directory=str(self.config.persist_directory),
            )
        except Exception as exc:
            self.logger.error("Failed to build the vector database: %s", exc)
            raise

        self.logger.info("Vector database persisted to %s", self.config.persist_directory)
        return self._store

    def load(self) -> Chroma:
        """Load a previously persisted ChromaDB collection from disk."""
        if not self.exists():
            raise FileNotFoundError(
                f"No vector database found at {self.config.persist_directory}. Run the build step first."
            )

        self.logger.info("Loading vector database from %s", self.config.persist_directory)
        self._store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embedding_manager.model,
            persist_directory=str(self.config.persist_directory),
        )
        return self._store

    def get_or_build(self, chunks: list[ProcessedChunk] | None = None) -> Chroma:
        """Load the database if it exists, otherwise build it from ``chunks``."""
        if self.exists():
            return self.load()
        if not chunks:
            raise ValueError("No persisted database found and no chunks were provided to build one.")
        return self.build(chunks)

    @property
    def store(self) -> Chroma:
        if self._store is None:
            return self.get_or_build()
        return self._store

    def count(self) -> int:
        """Number of vectors currently stored in the collection."""
        try:
            return self.store._collection.count()
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Could not read collection count: %s", exc)
            return 0
