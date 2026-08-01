"""Formats a `RAGAnswer` for display in notebooks and the Gradio UI."""

from __future__ import annotations

from rag_core.pipeline import RAGAnswer


class AnswerFormatter:
    """Turns a `RAGAnswer` into human-readable text and structured tables."""

    @staticmethod
    def to_markdown(result: RAGAnswer) -> str:
        """Render the answer, confidence note and sources as Markdown."""
        lines = [f"### Answer\n{result.answer}"]

        if result.confidence_note:
            lines.append(f"\n*{result.confidence_note}*")

        if result.sources:
            lines.append("\n### Sources")
            for index, source in enumerate(result.sources, start=1):
                location = f"{source.course} / {source.file_name}"
                if source.page_number:
                    location += f", page {source.page_number}"
                lines.append(
                    f"{index}. **{location}** "
                    f"(chunk `{source.chunk_id}`, similarity `{source.similarity_score}`)"
                )

        return "\n".join(lines)

    @staticmethod
    def to_source_table(result: RAGAnswer) -> list[list]:
        """Return sources as rows for a Gradio Dataframe: course, file, page, chunk, score."""
        rows = []
        for source in result.sources:
            rows.append(
                [
                    source.course,
                    source.file_name,
                    source.page_number if source.page_number else "-",
                    source.chunk_id,
                    source.similarity_score,
                ]
            )
        return rows

    @staticmethod
    def to_chunks_preview(result: RAGAnswer, max_chars: int = 400) -> str:
        """Render the raw retrieved chunk text, truncated for readability."""
        if not result.sources:
            return "No chunks were retrieved for this question."

        blocks = []
        for index, source in enumerate(result.sources, start=1):
            snippet = source.text[:max_chars]
            if len(source.text) > max_chars:
                snippet += "..."
            blocks.append(f"[{index}] {source.file_name}\n{snippet}")

        return "\n\n".join(blocks)
