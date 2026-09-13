"""Integration tests for the ingest pipeline.

Unlike test_chunking.py / test_scoring.py (pure logic, synthetic
fixtures), these exercise the real docs/ corpus and a real (temporary)
Chroma collection end-to-end, using the offline dev embedder so they run
without network access in CI.
"""

from __future__ import annotations

import pytest

from dev_embeddings import DevTfidfEmbeddingFunction
from ingest import DOCS_DIR, build_collection, load_and_chunk


@pytest.fixture
def real_chunks():
    chunks = load_and_chunk(DOCS_DIR)
    assert chunks, "docs/ must contain at least one .md file for these tests to be meaningful"
    return chunks


def test_load_and_chunk_produces_chunks_from_every_doc(real_chunks):
    doc_ids = {c.doc_id for c in real_chunks}
    # Every repo currently in docs/ should contribute at least one chunk.
    expected_repos = {
        "flagship-pipeline",
        "video-to-prompt",
        "waveform-generator",
        "web-scraper",
        "youtube-transcriber",
    }
    covered_repos = {doc_id.split("/")[0] for doc_id in doc_ids}
    assert expected_repos <= covered_repos


def test_no_chunk_exceeds_max_chars_after_split(real_chunks):
    # split_oversized runs inside load_and_chunk via ingest's pipeline --
    # confirm on the REAL corpus, not just the synthetic fixture in
    # test_chunking.py, since that's what actually gets embedded.
    oversized = [c for c in real_chunks if len(c.text) > 1600]
    assert not oversized, f"{len(oversized)} chunks exceed 1600 chars even after splitting"


def test_all_chunk_ids_are_unique_across_the_whole_corpus(real_chunks):
    ids = [c.chunk_id for c in real_chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id would silently overwrite in Chroma"


def test_build_collection_end_to_end_with_real_corpus(tmp_path, real_chunks):
    """Real Chroma, real corpus text, dev embedder -- no mocks. Verifies
    the add() call's metadata wiring matches what run_eval.py later reads
    (doc_id, heading_path) rather than trusting the API contract blind.
    """
    ef = DevTfidfEmbeddingFunction(fit_corpus=[c.text for c in real_chunks])
    collection = build_collection(real_chunks, tmp_path / "test_db", ef, collection_name="test")

    assert collection.count() == len(real_chunks)

    # Spot check: query for known content and confirm metadata survives the round trip.
    res = collection.query(query_texts=["key-field scraper monitor"], n_results=1)
    assert res["metadatas"][0][0]["doc_id"] == "web-scraper/README.md"
    assert "heading_path" in res["metadatas"][0][0]


def test_build_collection_is_idempotent_on_rerun(tmp_path, real_chunks):
    # Regression guard: re-running ingest (e.g. after editing a doc)
    # must not leave stale chunks from the previous run alongside new
    # ones -- this is exactly the kind of bug that would silently corrupt
    # eval results without raising an error.
    ef = DevTfidfEmbeddingFunction(fit_corpus=[c.text for c in real_chunks])
    db_path = tmp_path / "test_db"

    build_collection(real_chunks, db_path, ef, collection_name="test")
    second = build_collection(real_chunks, db_path, ef, collection_name="test")

    assert second.count() == len(real_chunks), "re-ingesting must not duplicate or leak old chunks"


def test_build_collection_raises_on_empty_chunk_list(tmp_path):
    ef = DevTfidfEmbeddingFunction(fit_corpus=["placeholder"])
    with pytest.raises(ValueError):
        build_collection([], tmp_path / "empty_db", ef, collection_name="test")
