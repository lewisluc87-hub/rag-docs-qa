"""Ingest the docs/ corpus into a local persistent Chroma collection.

Usage:
    python ingest.py                # rebuild the collection from docs/
    python ingest.py --db ./my_db   # custom Chroma path

Embedding function: this defaults to Chroma's built-in ONNX MiniLM
embedding function (`chromadb.utils.embedding_functions.DefaultEmbeddingFunction`),
which downloads a small model checkpoint from Hugging Face the first time
it runs and then works fully offline. That download requires normal
internet access -- see README for the one dev-sandbox exception (a
TF-IDF fallback used only where model downloads are blocked).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction

from chunking import chunk_markdown, split_oversized

DOCS_DIR = Path(__file__).parent / "docs"
DEFAULT_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "repo_docs"
MAX_CHUNK_CHARS = 800


def load_and_chunk(docs_dir: Path) -> list:
    """Chunk every .md file under docs_dir, tagging doc_id as 'repo/file.md'."""
    all_chunks = []
    for md_file in sorted(docs_dir.rglob("*.md")):
        doc_id = str(md_file.relative_to(docs_dir))
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, doc_id)
        all_chunks.extend(chunks)
    return split_oversized(all_chunks, max_chars=MAX_CHUNK_CHARS)


def get_default_embedding_function() -> EmbeddingFunction:
    """The production embedding function: Chroma's built-in ONNX MiniLM.

    Requires internet access to Hugging Face on first run (downloads
    ~80MB once, then cached locally). Use this on a normal dev machine.
    """
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()


def build_collection(
    chunks: list,
    db_path: Path,
    embedding_function: EmbeddingFunction,
    collection_name: str = COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=str(db_path))
    # Drop and recreate so re-running ingest.py never leaves stale chunks
    # (e.g. from a doc that was later deleted or restructured) alongside new ones.
    try:
        client.delete_collection(collection_name)
    except chromadb.errors.NotFoundError:
        pass  # first run -- nothing to drop yet
    collection = client.create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    if not chunks:
        raise ValueError("No chunks to ingest -- check docs_dir contains .md files")

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "heading_path": c.heading_path,
                "level": c.level,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in chunks
        ],
    )
    return collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--docs", type=Path, default=DOCS_DIR)
    parser.add_argument(
        "--dev-tfidf",
        action="store_true",
        help="Use the offline TF-IDF embedder instead of downloading the real "
        "embedding model. For environments without Hugging Face access. "
        "NOT representative of production retrieval quality.",
    )
    args = parser.parse_args()

    chunks = load_and_chunk(args.docs)
    print(f"Chunked {len(chunks)} sections from {args.docs}")

    if args.dev_tfidf:
        from dev_embeddings import DevTfidfEmbeddingFunction

        print("WARNING: using dev-only TF-IDF embeddings, not the production model.")
        embedding_function = DevTfidfEmbeddingFunction(fit_corpus=[c.text for c in chunks])
    else:
        embedding_function = get_default_embedding_function()
    collection = build_collection(chunks, args.db, embedding_function)
    print(f"Ingested {collection.count()} chunks into '{args.db}' (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    sys.exit(main())
