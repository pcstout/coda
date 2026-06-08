"""
LLM-based re-ranking of retrieved terms.
"""
import logging
from typing import List, Optional

from pydantic import ValidationError

from coda.llm_api import LLMClient

from .types import RetrievalTerm
from .schemas import RERANKING_SCHEMA, RerankingResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-rank retrieved terms using LLM reasoning.
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize reranker.

        Parameters
        ----------
        llm_client : LLMClient
            LLM client instance for making API calls.
        """
        self.llm_client = llm_client
        self.schema = RERANKING_SCHEMA

    def rerank(
        self,
        concept: str,
        evidences: List[str],
        retrieved_terms: List[RetrievalTerm],
        system_prompt: Optional[str] = None,
    ) -> List[RetrievalTerm]:
        """Re-rank retrieved terms based on concept and evidence.

        Parameters
        ----------
        concept : str
            Concept name.
        evidences : list of str
            List of supporting evidence strings.
        retrieved_terms : list of RetrievalTerm
            List of retrieved terms to rerank.
        system_prompt : str, optional
            Optional custom system prompt.

        Returns
        -------
        list of RetrievalTerm
            Reranked list of RetrievalTerm objects, ordered from most to least appropriate.
        """
        if not retrieved_terms:
            return []

        # Format retrieved terms for prompt
        retrieved_terms_formatted = []
        for term in retrieved_terms:
            term_info = f"  - Identifier: {term.id}, Name: {term.name}"
            if term.definition:
                term_info += f", Definition: {term.definition}"
            retrieved_terms_formatted.append(term_info)

        if system_prompt is None:
            system_prompt = """You are an expert that re-ranks retrieved terms based on how well they match a concept and its supporting evidence.

Consider these factors (in order of importance):
1. **Concept alignment**: Does the term accurately represent the concept?
2. **Evidence alignment**: Does the term match the supporting evidence?
3. **Specificity**: Prefer more specific terms over general ones when appropriate
4. **Retrieval relevance**: Consider how well the term matches semantically

Return ONLY JSON that matches the provided schema, ordered from most to least appropriate."""

        evidence_text = (
            "\n".join(f"  - {e}" for e in evidences) if evidences else "  (No specific evidence provided)"
        )

        user_prompt = f"""Given this concept and evidences, rerank these terms:

Concept:
{concept}

Supporting evidence:
{evidence_text}

Retrieved candidate terms:
{"\n".join(retrieved_terms_formatted)}

Re-rank these terms based on how well they match the concept and evidence."""

        try:
            response_json = self.llm_client.call_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=self.schema,
                schema_name="reranking",
                max_retries=3,
                retry_delay=1.0,
            )

            if response_json.get("api_failed", False):
                logger.error("LLM API call failed")
                return []

            try:
                validated_result = RerankingResult.model_validate(response_json)
                result_dict = validated_result.model_dump()
            except ValidationError as e:
                logger.warning(f"Invalid response structure from LLM: {e}")
                return []

            term_id_to_term = {term.id: term for term in retrieved_terms}
            reranked_terms = []
            for term_info in result_dict["Reranked_Terms"]:
                term_id = term_info.get("Term_Identifier", "")
                if term_id in term_id_to_term:
                    reranked_terms.append(term_id_to_term[term_id])
                else:
                    logger.warning(
                        f"Term ID '{term_id}' from reranking not found in retrieved terms"
                    )

            return reranked_terms

        except Exception as e:
            logger.error(f"Failed to rerank terms: {e}", exc_info=True)
            return []

