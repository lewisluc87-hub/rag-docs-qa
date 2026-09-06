"""Offline embedding function for dev/testing only -- NOT used in production.

The real embedding function (chromadb's built-in ONNX MiniLM, see
ingest.get_default_embedding_function) downloads a model checkpoint from
Hugging Face on first use. In network-restricted environments where that
download isn't reachable, this TF-IDF-based stand-in lets the ingest ->
retrieval pipeline still be exercised end-to-end with real corpus text,
so pipeline bugs (metadata wiring, Chroma API usage, chunk boundaries)
get caught without depending on network access.

TF-IDF is not a semantic embedding -- it won't generalize past exact/
overlapping vocabulary the way a real embedding model does. Retrieval
quality numbers produced with this embedder are NOT representative of
the shipped system and should not be reported as such.
"""

from __future__ import annotations

from chromadb import EmbeddingFunction
from sklearn.feature_extraction.text import TfidfVectorizer


class DevTfidfEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by a fitted TF-IDF vectorizer.

    Must be constructed with `fit_corpus` containing (at minimum) the same
    texts that will later be added to the collection, since TF-IDF's
    vocabulary and IDF weights are fixed at fit time -- unlike a neural
    embedder, it cannot meaningfully embed a document outside that
    vocabulary in a way comparable to the fitted space.

    Note: this intentionally doesn't implement get_config()/build_from_config(),
    which are Chroma's mechanism for reloading a persisted collection's
    embedding function from disk without re-specifying it in code. Since
    this class is fit on a corpus (not just constructed with fixed
    parameters), that reload path doesn't apply cleanly anyway -- callers
    always re-fit and pass the embedding function explicitly (see
    ingest.py / run_eval.py). Chroma logs a DeprecationWarning for this;
    it's expected and harmless for dev/test use.
    """

    def __init__(self, fit_corpus: list[str], max_features: int = 2048):
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._vectorizer.fit(fit_corpus)

    def __call__(self, input: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(input)
        return matrix.toarray().tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        # Same vector space for queries and documents -- TF-IDF has no
        # separate query/document training distinction the way some
        # neural embedders do.
        return self.__call__(input)

    def name(self) -> str:
        return "dev-tfidf-embedding-function"
