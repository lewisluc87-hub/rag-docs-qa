"""MCP server exposing the RAG retrieval pipeline as tools.

Wraps the existing ingest/retrieval code (ingest.py, chunking.py) rather
than reimplementing it -- a bugfix to the chunker or the embedding
function automatically applies here too, with no duplicated logic.

This is retrieval-only: `ask_portfolio` returns the matched source
passages, not a generated answer. Wiring in a live LLM call for
generation is a natural v2 (mirroring the eval harness's own
--with-generation flag, not yet implemented there either) -- it needs an
ANTHROPIC_API_KEY available to this process, which isn't something the
sandbox this was built in could verify end-to-end.

Usage (stdio transport, what Claude Desktop expects):
    python mcp_server.py                # production embeddings (needs a
                                         #   Chroma DB already built with
                                         #   `python ingest.py`)
    python mcp_server.py --dev-tfidf    # offline dev embedder (needs a DB
                                         #   built with `python ingest.py
                                         #   --dev-tfidf` -- the embedder
                                         #   MUST match how the DB was built)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from mcp.server.mcpserver import MCPServer

from ingest import (
    COLLECTION_NAME,
    DEFAULT_DB_PATH,
    DOCS_DIR,
    get_default_embedding_function,
    load_and_chunk,
)

server = MCPServer(
    "rag-docs-qa",
    description=(
        "Retrieval over this portfolio's own project docs (5 repos, "
        "chunked and embedded). Retrieval-only -- returns matched source "
        "passages, not a generated answer."
    ),
)

# Populated once at startup by _init_resources(), before the server starts
# serving requests. Tool functions read these; nothing here is mutated per
# request, so there's no need for per-call locking.
_collection = None
_repo_chunk_counts: dict[str, int] = {}
EVAL_REPORT_PATH = Path(__file__).parent / "eval" / "eval_report.md"


def _init_resources(dev_tfidf: bool) -> None:
    global _collection, _repo_chunk_counts

    chunks = load_and_chunk(DOCS_DIR)
    _repo_chunk_counts = {}
    for c in chunks:
        repo = c.doc_id.split("/")[0]
        _repo_chunk_counts[repo] = _repo_chunk_counts.get(repo, 0) + 1

    if dev_tfidf:
        from dev_embeddings import DevTfidfEmbeddingFunction

        embedding_function = DevTfidfEmbeddingFunction(fit_corpus=[c.text for c in chunks])
    else:
        embedding_function = get_default_embedding_function()

    client = chromadb.PersistentClient(path=str(DEFAULT_DB_PATH))
    try:
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_function)
    except chromadb.errors.NotFoundError as e:
        raise RuntimeError(
            f"No Chroma collection found at {DEFAULT_DB_PATH}. Run "
            f"`python ingest.py{' --dev-tfidf' if dev_tfidf else ''}` first."
        ) from e


@server.tool()
def ask_portfolio(question: str, k: int = 3) -> str:
    """Retrieve passages from this portfolio's project docs relevant to a question.

    Returns the top-k matched source passages (not a generated answer) --
    each with its source repo/file, section heading, and text, so the
    caller can see exactly what was found and where it came from.
    """
    if _collection is None:
        return "Error: server resources not initialized."

    res = _collection.query(query_texts=[question], n_results=k)
    ids = res["ids"][0]
    if not ids:
        return "No passages found."

    parts = []
    for doc_id, dist, meta, text in zip(
        ids, res["distances"][0], res["metadatas"][0], res["documents"][0]
    ):
        parts.append(
            f"[{meta['doc_id']} \u00bb {meta['heading_path']}] (distance: {dist:.3f})\n{text}"
        )
    return "\n\n---\n\n".join(parts)


@server.tool()
def list_indexed_repos() -> str:
    """List which repos are indexed in this corpus and how many chunks each contributes."""
    if not _repo_chunk_counts:
        return "No repos indexed."
    lines = [f"- {repo}: {count} chunks" for repo, count in sorted(_repo_chunk_counts.items())]
    return "\n".join(lines)


@server.tool()
def get_eval_summary() -> str:
    """Return the most recent eval harness report (retrieval hit rate, known limitations).

    Reflects whatever embedder the eval was last run with -- check the
    report's own text for whether that was the offline dev embedder
    (not representative of production quality) or the real one.
    """
    if not EVAL_REPORT_PATH.exists():
        return (
            "No eval report found. Run `python eval/run_eval.py` "
            "(or `--dev-tfidf`) to generate one."
        )
    return EVAL_REPORT_PATH.read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-tfidf",
        action="store_true",
        help="Use the offline TF-IDF embedder. MUST match how the Chroma DB was built.",
    )
    args = parser.parse_args()

    _init_resources(dev_tfidf=args.dev_tfidf)
    server.run()


if __name__ == "__main__":
    sys.exit(main())
