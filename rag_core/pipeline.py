"""End-to-end Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from langchain_ollama import OllamaLLM

from rag_core.config import AppConfig
from rag_core.logging_setup import get_logger
from rag_core.prompt_manager import PromptManager
from rag_core.retriever import RetrievedChunk, RetrieverEngine


@dataclass
class RAGAnswer:
    """The full result of running a question through the pipeline."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    confidence_note: str = ""
    generation_seconds: float = 0.0
    error: str | None = None


class RAGPipeline:
    """Wires the retriever, prompt manager and Ollama LLM together."""

    def __init__(
        self,
        config: AppConfig,
        retriever: RetrieverEngine,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.prompt_manager = prompt_manager or PromptManager(config)
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        self._llm: OllamaLLM | None = None

    @property
    def llm(self) -> OllamaLLM:
        """Lazily instantiate the Ollama client."""
        if self._llm is None:
            self.logger.info("Connecting to Ollama model '%s' at %s", self.config.llm_model_name, self.config.llm_base_url)
            self._llm = OllamaLLM(
                model=self.config.llm_model_name,
                base_url=self.config.llm_base_url,
                temperature=self.config.llm_temperature,
                top_p=self.config.llm_top_p,
                num_ctx=self.config.llm_num_ctx,
            )
        return self._llm

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        course_filter: str | None = None,
    ) -> RAGAnswer:
        """Run the full RAG flow for a single question.

        Receive question -> retrieve top-K chunks -> build prompt ->
        generate answer -> format response -> return.
        """
        question = (question or "").strip()
        if not question:
            return RAGAnswer(
                question=question,
                answer="Please enter a question.",
                error="empty_question",
            )

        chunks = self.retriever.retrieve(question, top_k=top_k, course_filter=course_filter)

        if not chunks:
            return RAGAnswer(
                question=question,
                answer=self.config.no_answer_message,
                sources=[],
                confidence_note="No matching course material was retrieved.",
            )

        prompt = self.prompt_manager.build_prompt(question, chunks)

        start = perf_counter()
        try:
            raw_answer = self.llm.invoke(prompt)
        except Exception as exc:
            self.logger.error("LLM generation failed: %s", exc)
            return RAGAnswer(
                question=question,
                answer=(
                    "The language model could not be reached. Please confirm that Ollama is "
                    f"running locally and that the model '{self.config.llm_model_name}' has "
                    "been pulled (ollama pull <model>)."
                ),
                sources=chunks,
                error=str(exc),
            )
        elapsed = perf_counter() - start

        answer_text = raw_answer.strip()
        confidence_note = self._build_confidence_note(chunks)

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=chunks,
            confidence_note=confidence_note,
            generation_seconds=round(elapsed, 2),
        )

    @staticmethod
    def _build_confidence_note(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No supporting material found."
        best_score = max(chunk.similarity_score for chunk in chunks)
        if best_score >= 0.75:
            return f"High confidence - top match similarity {best_score:.2f}."
        if best_score >= 0.5:
            return f"Moderate confidence - top match similarity {best_score:.2f}."
        return f"Low confidence - top match similarity {best_score:.2f}. Answer may be incomplete."
