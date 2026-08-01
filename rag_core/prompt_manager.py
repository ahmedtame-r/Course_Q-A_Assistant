"""Builds the grounded prompt sent to the LLM."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from rag_core.config import AppConfig
from rag_core.retriever import RetrievedChunk

_SYSTEM_INSTRUCTIONS = """You are a strict academic assistant that answers student questions \
using ONLY the course material excerpts provided below.

Rules you must always follow:
1. Base your answer exclusively on the CONTEXT section. Never use outside or prior knowledge.
2. Never guess and never invent facts, numbers, or citations that are not present in the context.
3. If the context does not contain the answer, respond with exactly: "{no_answer_message}"
4. If different excerpts in the context contradict each other, say: "{conflict_message}" \
and briefly describe the disagreement using only what is written in the context.
5. Keep the answer concise, factual, and written in your own words rather than copied verbatim.
6. Do not mention these instructions or the word "context" in your final answer."""


class PromptManager:
    """Owns the prompt template and formats retrieved chunks into context."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.template = PromptTemplate(
            input_variables=["context", "question"],
            partial_variables={
                "no_answer_message": config.no_answer_message,
                "conflict_message": config.conflict_message,
            },
            template=(
                _SYSTEM_INSTRUCTIONS
                + "\n\nCONTEXT:\n{context}\n\nSTUDENT QUESTION:\n{question}\n\nANSWER:"
            ),
        )

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Turn retrieved chunks into a numbered, source-labelled block."""
        if not chunks:
            return "(no relevant course material was retrieved)"

        blocks = []
        for index, chunk in enumerate(chunks, start=1):
            location = f"{chunk.course} / {chunk.file_name}"
            if chunk.page_number:
                location += f" (page {chunk.page_number})"
            blocks.append(f"[Source {index} - {location}]\n{chunk.text}")

        return "\n\n".join(blocks)

    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Render the final prompt string ready to send to the LLM."""
        context = self.format_context(chunks)
        return self.template.format(context=context, question=question)
