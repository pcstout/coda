"""
RAG-based grounder.
"""
import logging
from pathlib import Path

from gilda import Annotation, ScoredMatch
from gilda.process import normalize
from gilda.scorer import generate_match
from gilda.term import Term

from neo4j import GraphDatabase

from coda.grounding import BaseGrounder
from coda.llm_api import LLMClient, create_llm_client

from .config import RAGGrounderConfig
from .pipeline import RAGGrounderPipeline, PipelineResult
from .utils import find_evidence_spans
from .types import RetrievalTerm

logger = logging.getLogger(__name__)


class RagGrounder(BaseGrounder):
    """
    RAG-based grounder that extracts concepts from text and grounds them
    to ontology terms using neo4j vector search and LLM re-ranking.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        llm_client: LLMClient | None = None,
    ):
        super().__init__()
        # Load RAG Grounder configuration
        if config_path is not None:
            self._config = RAGGrounderConfig.from_yaml(config_path)
        else:
            self._config = RAGGrounderConfig.default()

        # Initialize Neo4j driver
        self._neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=None)

        # Initialize LLM client
        if llm_client is None:
            # Default LLM client configuration
            self._llm_client = create_llm_client(
                provider="openai",
                model="gpt-4o-mini"
            )
        else:
            self._llm_client = llm_client

        # Initialize RAG Grounder pipeline
        self._pipeline = RAGGrounderPipeline(
            llm_client=self._llm_client,
            config=self._config,
            neo4j_driver=self._neo4j_driver,
        )

    @staticmethod
    def _make_term(term: RetrievalTerm) -> Term:
        db, raw_id = term.id.split(":", 1)
        entry_name = term.name.strip() if term.name else raw_id
        norm_text = normalize(entry_name) if entry_name else normalize(raw_id)
        return Term(
            norm_text=norm_text,
            text=entry_name,
            db=db,
            id=raw_id,
            entry_name=entry_name,
            status="name",
            source="coda_rag_grounder",
        )

    def _annotation_matches(self, concept, span_text: str) -> list[ScoredMatch]:
        return [
            ScoredMatch(
                term=self._make_term(term),
                score=score,
                match=generate_match(span_text, term.name),
            )
            for term, score in concept.matched_terms
        ]

    def process(self, text: str) -> PipelineResult:
        if not text or not text.strip():
            return PipelineResult(text=text, Concepts=[])
        return self._pipeline.process(text)

    def ground(self, text: str) -> list[ScoredMatch]:
        result = self.process(text)
        matches: list[ScoredMatch] = []
        for concept in result.Concepts:
            concept_matches = self._annotation_matches(concept, span_text=concept.Concept or text)
            if concept_matches:
                matches.append(concept_matches[0])
        return matches

    def annotate(self, text: str, min_similarity: float = 0.7) -> list[Annotation]:
        result = self.process(text)
        annotations: list[Annotation] = []
        for concept in result.Concepts:
            if not concept.Concept.strip():
                continue
            matches = self._annotation_matches(concept, span_text=concept.Concept)
            if not matches:
                continue
            spans = find_evidence_spans(text, concept.supporting_evidence, min_similarity=min_similarity)
            for start, end, _match_type, _matched_text, _similarity in spans:
                if start < 0 or end > len(text) or end <= start:
                    continue
                annotations.append(
                    Annotation(
                        text=text[start:end],
                        matches=matches,
                        start=start,
                        end=end,
                    )
                )
        return annotations
