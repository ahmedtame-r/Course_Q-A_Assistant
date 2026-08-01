"""
rag_core
========

Shared library powering the Multi-Course Question Answering System.

This package is imported by every notebook (01, 02, 03) and by ``app.py``
so that the document-processing, embedding, vector-store, retrieval,
prompting and generation logic lives in exactly one place instead of
being copy-pasted between notebooks and the deployment script.
"""

from rag_core.config import AppConfig
from rag_core.document_processor import DocumentProcessor, ProcessedChunk
from rag_core.embeddings import EmbeddingManager
from rag_core.vector_store import VectorDatabaseManager
from rag_core.retriever import RetrieverEngine, RetrievedChunk
from rag_core.prompt_manager import PromptManager
from rag_core.pipeline import RAGPipeline, RAGAnswer
from rag_core.formatter import AnswerFormatter

__all__ = [
    "AppConfig",
    "DocumentProcessor",
    "ProcessedChunk",
    "EmbeddingManager",
    "VectorDatabaseManager",
    "RetrieverEngine",
    "RetrievedChunk",
    "PromptManager",
    "RAGPipeline",
    "RAGAnswer",
    "AnswerFormatter",
]

__version__ = "1.0.0"
