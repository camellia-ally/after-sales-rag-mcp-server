"""Data contracts for the internal RAG answer layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.response.citation_generator import Citation


@dataclass
class RAGAnswerResponse:
    """Provider-independent result returned by the answer service."""

    status: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

