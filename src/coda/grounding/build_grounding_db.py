"""Build a namespace-filtered Gilda SQLite grounding database.

Gilda's full term set spans ~2M terms across many namespaces (genes, proteins,
chemicals, gene ontology, ...). CODA only grounds to a subset of medical
namespaces (see :data:`coda.grounding.gilda_grounder.DEFAULT_NAMESPACES`).
Filtering the database down to the namespaces CODA actually
uses cuts its size with no change to grounding results.

This is used at Docker build time to produce a smaller ``grounding_terms.db``.

Usage::

    python -m coda.grounding.build_grounding_db <output_db_path> [NS1 NS2 ...]

If no namespaces are given, ``DEFAULT_NAMESPACES`` is used.
"""
import logging
import sys

from gilda.grounder import Grounder
from gilda.resources.sqlite_adapter import build

from coda.grounding.gilda_grounder import DEFAULT_NAMESPACES

logger = logging.getLogger(__name__)


def build_filtered_db(path: str, namespaces=None) -> None:
    """Build a SQLite grounding db containing only the given namespaces.

    Parameters
    ----------
    path :
        Output path for the ``.db`` file.
    namespaces :
        Iterable of Gilda namespace names (e.g. ``MESH``) to keep. Defaults to
        :data:`coda.grounding.gilda_grounder.DEFAULT_NAMESPACES`.
    """
    namespaces = set(namespaces) if namespaces else set(DEFAULT_NAMESPACES)
    logger.info("Loading full Gilda grounder")
    grounder = Grounder()
    filtered = {}
    for norm_text, terms in grounder.entries.items():
        kept = [t for t in terms if t.db in namespaces]
        if kept:
            filtered[norm_text] = kept
    logger.info("Building filtered SQLite db at %s (namespaces=%s, %d rows)",
                path, sorted(namespaces), len(filtered))
    build(filtered, path=path)


def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        sys.exit("usage: python -m coda.grounding.build_grounding_db "
                 "<output_db_path> [NAMESPACE ...]")
    build_filtered_db(sys.argv[1], sys.argv[2:] or None)


if __name__ == "__main__":
    main()
