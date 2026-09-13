"""Run the eval suite against an ingested Chroma collection.

Retrieval evaluation runs fully offline against whatever collection you
point it at (no API key needed). Generation-quality / hallucination
checks require a live LLM call and are OFF by default -- see
--with-generation below.

Usage:
    python ingest.py --dev-tfidf          # build the collection first
    python eval/run_eval.py --dev-tfidf   # score it (must match ingest's embedder)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest import (
    COLLECTION_NAME,
    DEFAULT_DB_PATH,
    DOCS_DIR,
    get_default_embedding_function,
    load_and_chunk,
)
from scoring import RetrievalScore, summarize

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
DEFAULT_K = 3
# Retrieval-distance threshold below which a match is considered "confident".
# NOT calibrated for the dev TF-IDF embedder (see README) -- recalibrate
# against real embedding-model distances before trusting this on adversarial
# questions in production.
CONFIDENCE_THRESHOLD = 1.5


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    out = []
    for set_name, questions in data.items():
        if set_name.startswith("_"):
            continue
        for q in questions:
            out.append({**q, "set": set_name})
    return out


def run(collection, questions: list[dict], k: int = DEFAULT_K) -> list[dict]:
    results = []
    for q in questions:
        res = collection.query(query_texts=[q["question"]], n_results=k)
        retrieved_ids = res["ids"][0]
        distances = res["distances"][0]
        heading_paths = [m["heading_path"] for m in res["metadatas"][0]]

        score = RetrievalScore(expected_ids=q["expected_chunk_ids"], retrieved_ids=retrieved_ids)
        results.append(
            {
                "id": q["id"],
                "set": q["set"],
                "question": q["question"],
                "expected_chunk_ids": q["expected_chunk_ids"],
                "retrieved_ids": retrieved_ids,
                "retrieved_headings": heading_paths,
                "distances": distances,
                "hit": score.hit,
                "recall": score.recall,
                "is_adversarial": score.is_adversarial,
                "min_distance": min(distances) if distances else None,
            }
        )
    return results


def render_report(results: list[dict], k: int) -> str:
    scores = [
        RetrievalScore(expected_ids=r["expected_chunk_ids"], retrieved_ids=r["retrieved_ids"])
        for r in results
    ]
    by_set: dict[str, list[dict]] = {}
    for r in results:
        by_set.setdefault(r["set"], []).append(r)

    lines = ["# Eval report", "", f"k={k}", ""]

    overall = summarize(scores)
    lines += [
        "## Overall (answerable questions only)",
        f"- Hit rate (recall@{k} > 0): {overall['hit_rate']:.2f}"
        if overall["hit_rate"] is not None
        else "- Hit rate: n/a",
        f"- Mean recall: {overall['mean_recall']:.2f}"
        if overall["mean_recall"] is not None
        else "- Mean recall: n/a",
        f"- Full recall on {overall['n_answerable_full_recall']}/{overall['n_answerable']} questions",
        f"- Zero recall on {overall['n_answerable_zero_recall']}/{overall['n_answerable']} questions",
        "",
    ]

    for set_name, rows in by_set.items():
        lines.append(f"## {set_name} ({len(rows)} questions)")
        for r in rows:
            marker = "✅" if (r["hit"] or r["is_adversarial"]) else "❌"
            lines.append(f"### {marker} {r['id']}: {r['question']}")
            if r["is_adversarial"]:
                lines.append(
                    f"- Adversarial (expected: no good match). "
                    f"Min distance: {r['min_distance']:.3f} "
                    f"({'CONFIDENT MATCH -- investigate' if r['min_distance'] is not None and r['min_distance'] <= CONFIDENCE_THRESHOLD else 'appropriately unconfident'})"
                )
            else:
                lines.append(f"- Recall: {r['recall']:.2f} | Expected: {r['expected_chunk_ids']}")
            lines.append(f"- Retrieved: {list(zip(r['retrieved_ids'], [round(d, 3) for d in r['distances']]))}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "eval_report.md")
    parser.add_argument(
        "--dev-tfidf",
        action="store_true",
        help="Use the offline TF-IDF embedder -- MUST match how the collection was ingested.",
    )
    args = parser.parse_args()

    if args.dev_tfidf:
        from dev_embeddings import DevTfidfEmbeddingFunction

        chunks = load_and_chunk(DOCS_DIR)
        embedding_function = DevTfidfEmbeddingFunction(fit_corpus=[c.text for c in chunks])
    else:
        embedding_function = get_default_embedding_function()

    client = chromadb.PersistentClient(path=str(args.db))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_function)

    questions = load_questions()
    results = run(collection, questions, k=args.k)
    report = render_report(results, k=args.k)

    args.out.write_text(report)
    print(f"Wrote {args.out}")

    overall = summarize(
        [RetrievalScore(r["expected_chunk_ids"], r["retrieved_ids"]) for r in results]
    )
    print(f"Hit rate: {overall['hit_rate']:.2f}" if overall["hit_rate"] is not None else "n/a")


if __name__ == "__main__":
    sys.exit(main())
