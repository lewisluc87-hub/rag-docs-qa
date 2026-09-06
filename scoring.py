"""Pure scoring logic for the eval harness.

Kept separate from run_eval.py's orchestration so these can be unit
tested against plain lists of IDs, with no Chroma instance, no
embedding model, and no network required.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalScore:
    expected_ids: list[str]
    retrieved_ids: list[str]

    @property
    def hit(self) -> bool:
        """True if at least one expected chunk was retrieved (recall@k > 0).

        For adversarial questions expected_ids is empty by design, in
        which case `hit` is not a meaningful concept -- see
        `is_adversarial` and `false_positive` instead.
        """
        if not self.expected_ids:
            return False
        return any(e in self.retrieved_ids for e in self.expected_ids)

    @property
    def recall(self) -> float:
        """Fraction of expected chunks that appear anywhere in retrieved_ids.

        Meaningful for cross-doc questions with multiple expected chunks
        (e.g. cd-03 expects one chunk from each of 3 docs) -- a single-hit
        question scores 1.0 or 0.0 same as `hit`, but a 3-expected
        question distinguishes "found 1 of 3" from "found 3 of 3".
        """
        if not self.expected_ids:
            return 0.0
        found = sum(1 for e in self.expected_ids if e in self.retrieved_ids)
        return found / len(self.expected_ids)

    @property
    def is_adversarial(self) -> bool:
        return len(self.expected_ids) == 0


def false_positive_rate(retrieved_distances: list[float], threshold: float) -> float:
    """Fraction of retrieved results at or under `threshold` (i.e. the
    retriever is confident about a match). For an adversarial question
    this fraction SHOULD be 0 -- any confident match on an out-of-scope
    question is a retrieval-level false positive, upstream of whatever
    the generation step does with it.
    """
    if not retrieved_distances:
        return 0.0
    confident = sum(1 for d in retrieved_distances if d <= threshold)
    return confident / len(retrieved_distances)


def summarize(scores: list[RetrievalScore]) -> dict:
    answerable = [s for s in scores if not s.is_adversarial]
    adversarial = [s for s in scores if s.is_adversarial]

    return {
        "n_answerable": len(answerable),
        "n_adversarial": len(adversarial),
        "hit_rate": (sum(s.hit for s in answerable) / len(answerable)) if answerable else None,
        "mean_recall": (
            sum(s.recall for s in answerable) / len(answerable) if answerable else None
        ),
        "n_answerable_full_recall": sum(1 for s in answerable if s.recall == 1.0),
        "n_answerable_zero_recall": sum(1 for s in answerable if s.recall == 0.0),
    }
