"""HuggingFace embedding model wrapper."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from rag_core.config import AppConfig
from rag_core.logging_setup import get_logger


class EmbeddingManager:
    """Lazily instantiates and caches a HuggingFace embedding model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        self._model: HuggingFaceEmbeddings | None = None

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Return the underlying embedding model, creating it on first use."""
        if self._model is None:
            self.logger.info("Loading embedding model '%s' on '%s'...", self.config.embedding_model_name, self.config.embedding_device)
            try:
                self._model = HuggingFaceEmbeddings(
                    model_name=self.config.embedding_model_name,
                    model_kwargs={"device": self.config.embedding_device},
                    encode_kwargs={"normalize_embeddings": self.config.embedding_normalize},
                )
            except Exception as exc:
                self.logger.error("Failed to load embedding model: %s", exc)
                raise
            self.logger.info("Embedding model ready.")
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings."""
        if not texts:
            return []
        return self.model.embed_documents(texts)
