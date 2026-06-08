"""
Term retrieval using neo4j vector search.
"""
import logging
from typing import List, Tuple

from neo4j import Driver
from sentence_transformers import SentenceTransformer

from .types import RetrievalTerm

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves ontology terms from neo4j using vector similarity search.
    """

    def __init__(
        self,
        driver: Driver,
        ontology: str,
        model_name: str,
    ):
        self.driver = driver
        self.ontology = ontology
        self.model_name = model_name
        self._model = None
        self._ensure_vector_index()

    def _ensure_vector_index(self) -> None:
        model = SentenceTransformer(self.model_name)
        dim = model.get_sentence_embedding_dimension()
        self._model = model

        index_name = f"{self.ontology}_embedding"
        # Label names cannot be parameterised in Cypher, so we use an f-string.
        # ontology is an internal config value, not user input.
        with self.driver.session() as session:
            session.run(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{self.ontology}) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
            )
        logger.info("Vector index '%s' ready (dim=%d)", index_name, dim)

    def retrieve(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[RetrievalTerm, float]]:
        """Retrieve top-k most similar terms for query text.

        Parameters
        ----------
        query_text : str
            Query text to search for.
        top_k : int
            Number of top terms to return.
        min_similarity : float
            Minimum similarity threshold (0.0 to 1.0).

        Returns
        -------
        list of tuple (RetrievalTerm, float)
            Terms ordered by similarity descending.
        """
        if not query_text or not query_text.strip():
            return []

        query_embedding = self._model.encode(
            query_text,
            normalize_embeddings=True,
        ).tolist()

        index_name = f"{self.ontology}_embedding"

        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                YIELD node, score
                WHERE score >= $min_similarity
                RETURN node.id AS id, node.name AS name, score
                """,
                index_name=index_name,
                top_k=top_k,
                query_embedding=query_embedding,
                min_similarity=min_similarity,
            )
            return [
                (RetrievalTerm(id=record["id"], name=record["name"]), record["score"])
                for record in result
            ]
