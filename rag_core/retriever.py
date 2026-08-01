"""Top-K similarity search retriever."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma

from rag_core.config import AppConfig
from rag_core.logging_setup import get_logger


@dataclass
class RetrievedChunk:
    """A single retrieved chunk enriched with its similarity score."""

    text: str
    course: str
    file_name: str
    doc_type: str
    page_number: int | None
    chunk_id: str
    similarity_score: float


class RetrieverEngine:
    """Wraps a Chroma store to perform configurable top-K similarity search."""

    def __init__(self, config: AppConfig, vector_store: Chroma) -> None:
        self.config = config
        self.vector_store = vector_store
        self.logger = get_logger(self.__class__.__name__, config.log_level)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        course_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-K most similar chunks to ``query``.

        Args:
            query: The user's natural-language question.
            top_k: Overrides ``config.top_k`` when provided.
            course_filter: If given, restrict the search to a single course.
        """
        k = top_k or self.config.top_k
        where = {"course": course_filter} if course_filter else None

        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k, filter=where
            )
        except Exception as exc:
            self.logger.error("Retrieval failed for query '%s': %s", query, exc)
            return []

        retrieved: list[RetrievedChunk] = []
        for document, score in results:
            if self.config.score_threshold is not None and score < self.config.score_threshold:
                continue
            metadata = document.metadata
            page_number = metadata.get("page_number", -1)
            retrieved.append(
                RetrievedChunk(
                    text=document.page_content,
                    course=metadata.get("course", "Unknown"),
                    file_name=metadata.get("file_name", "Unknown"),
                    doc_type=metadata.get("doc_type", "Unknown"),
                    page_number=page_number if page_number and page_number > 0 else None,
                    chunk_id=metadata.get("chunk_id", "Unknown"),
                    similarity_score=round(float(score), 4),
                )
            )

        self.logger.info("Retrieved %d chunk(s) for query.", len(retrieved))
        return retrieved
