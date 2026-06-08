"""
Main pipeline orchestrator for RAG grounder.

Combines LLM extraction, semantic retrieval, and re-ranking for any concept type.
"""
import logging
import time
from typing import List, Tuple

from neo4j import Driver
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from coda.llm_api import LLMClient

from .types import RetrievalTerm
from .config import RAGGrounderConfig
from .extractor import Extractor
from .reranker import Reranker
from .retriever import Retriever

logger = logging.getLogger(__name__)


class ProcessedConcept(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    Concept: str
    supporting_evidence: List[str] = []
    matched_terms: List[Tuple[RetrievalTerm, float]] = []
    synonym_curies: List[str] = []


class PipelineResult(BaseModel):
    text: str
    Concepts: List[ProcessedConcept]


class RAGGrounderPipeline:
    """
    Complete pipeline for extracting concepts and grounding them to retrieval terms.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: RAGGrounderConfig,
        neo4j_driver: Driver,
    ):
        self.llm_client = llm_client
        self.config = config

        self.extractor = Extractor(
            llm_client=llm_client,
            concept_type=config.extractor.concept_type,
        )
        self.retriever = Retriever(
            driver=neo4j_driver,
            ontology=config.retriever.ontology,
            model_name=config.retriever.embedding_model,
        )
        self.reranker = Reranker(llm_client=llm_client)

    def process(self, text: str) -> PipelineResult:
        if not text or not text.strip():
            return PipelineResult(text=text, Concepts=[])

        logger.info("Starting RAG grounder pipeline")
        total_start = time.time()
        step_times = {}

        # Step 1: Extract concepts using LLM
        step1_start = time.time()
        extraction_result = self.extractor.extract(text)
        step_times["extraction"] = time.time() - step1_start

        concepts_raw = extraction_result.get("Concepts", [])
        logger.info(f"Extraction completed in {step_times['extraction']:.2f}s, found {len(concepts_raw)} concept(s)")

        if not concepts_raw:
            return PipelineResult(text=text, Concepts=[])

        concepts = [
            {
                "Concept": c.get("Concept", ""),
                "supporting_evidence": c.get("Supporting_Evidence", []),
                "matched_terms": [],
            }
            for c in concepts_raw
        ]

        # Step 2: Retrieve terms using semantic search
        step2_start = time.time()
        for concept in tqdm(concepts, desc="Retrieving terms", leave=False, disable=len(concepts) <= 1):
            concept_name = concept["Concept"]
            evidence_text = "\n".join(concept["supporting_evidence"])
            retrieval_text = f"{concept_name}\n\n{evidence_text}" if evidence_text else concept_name
            concept["matched_terms"] = self.retriever.retrieve(
                retrieval_text,
                top_k=self.config.retriever.top_k,
                min_similarity=self.config.retriever.min_similarity,
            )
            logger.debug(f"Retrieved {len(concept['matched_terms'])} terms for: {concept_name}")
        step_times["retrieval"] = time.time() - step2_start
        logger.info(f"Retrieval completed in {step_times['retrieval']:.2f}s")

        # Step 3: Re-rank terms using LLM, preserving cosine scores
        step3_start = time.time()
        for concept in tqdm(concepts, desc="Reranking terms", leave=False, disable=len(concepts) <= 1):
            concept_name = concept["Concept"]
            score_by_id = {term.id: score for term, score in concept["matched_terms"]}
            reranked = self.reranker.rerank(
                concept=concept_name,
                evidences=concept["supporting_evidence"],
                retrieved_terms=[term for term, _ in concept["matched_terms"]],
            )
            concept["matched_terms"] = [(term, score_by_id.get(term.id, 0.0)) for term in reranked]
            logger.debug(f"Re-ranked {len(concept['matched_terms'])} terms for: {concept_name}")
        step_times["reranking"] = time.time() - step3_start
        logger.info(f"Re-ranking completed in {step_times['reranking']:.2f}s")

        total_time = time.time() - total_start
        logger.info(
            f"Pipeline timing: "
            f"Extraction={step_times['extraction']:.2f}s, "
            f"Retrieval={step_times['retrieval']:.2f}s, "
            f"Re-ranking={step_times['reranking']:.2f}s, "
            f"Total={total_time:.2f}s"
        )

        processed_concepts = []
        for concept in concepts:
            synonym_curies: List[str] = []
            if concept["matched_terms"]:
                synonym_curies = concept["matched_terms"][0][0].synonyms or []
            processed_concepts.append(
                ProcessedConcept(
                    Concept=concept["Concept"],
                    supporting_evidence=concept["supporting_evidence"],
                    matched_terms=concept["matched_terms"],
                    synonym_curies=synonym_curies,
                )
            )
        return PipelineResult(text=text, Concepts=processed_concepts)
