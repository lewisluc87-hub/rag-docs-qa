# rag-docs-qa
[![CI](https://github.com/lewisluc87-hub/rag-docs-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/lewisluc87-hub/rag-docs-qa/actions/workflows/ci.yml)

A retrieval-augmented Q&A system over this portfolio's own project docs,
built with an eval harness that actually measures retrieval quality and
checks for hallucination on out-of-scope questions -- not just "it runs."

Ingests the README (and SPEC, where present) from five other repos in
this portfolio, chunks them by heading, embeds them, and answers
questions against the resulting corpus via retrieval + an LLM.

## Why this exists

RAG demos are usually built on a public corpus (Wikipedia, arXiv) the
builder doesn't know cold, which makes it hard to write a real eval set
-- you don't actually know the ground truth. Using this portfolio's own
docs means every eval question's expected answer is independently
verifiable by reading the source file, which is what makes the eval
harness (the actual point of this project) trustworthy rather than
decorative.

## Architecture

```
docs/*.md → chunk by heading (fence-aware) → split oversized sections
                                                        ↓
                                              embed → Chroma (local, persistent)
                                                        ↓
                                   query → embed → retrieve top-k → [LLM answer -- not yet wired]
```

- **Chunking** (`chunking.py`): splits markdown at heading boundaries
  (levels 1-3), tracking a breadcrumb (e.g. `"Flagship pipeline > Usage >
  Flags"`) so a retrieved chunk carries its own context. Fence-aware: a
  `#`-prefixed shell comment inside a \`\`\`bash block is not mistaken
  for a heading -- several of these READMEs have exactly that shape, and
  a naive line-scanner would silently corrupt those chunks.
- **Oversized-chunk splitting** (`chunking.split_oversized`): some
  sections (SPEC.md's numbered sections especially) run 2000-3000+
  chars, well past what an embedding model actually encodes before
  silently truncating. This sub-splits on paragraph boundaries, keeping
  the heading breadcrumb, rather than letting the back half of a long
  section become invisible to retrieval with no warning.
- **Vector store**: Chroma, local persistent client -- no server to run.
- **Embeddings**: see [Embedding function](#embedding-function) below --
  there are two, for a specific reason.
- **Generation**: not yet implemented. Retrieval is fully built and
  evaluated; wiring a real Claude call for answer generation (plus an
  LLM-as-judge grounding check) is the next step, gated on having a live
  `ANTHROPIC_API_KEY` available, which this dev sandbox didn't have.

## Embedding function

Two embedding functions exist, and this split is deliberate, not a
shortcut:

- **Production** (`ingest.get_default_embedding_function`): Chroma's
  built-in ONNX MiniLM embedder. This is what you should use for real
  retrieval quality. It downloads a small model checkpoint from Hugging
  Face on first run, then works offline.
- **Dev/test** (`dev_embeddings.DevTfidfEmbeddingFunction`): a TF-IDF
  vectorizer, used only because the sandbox this was built in couldn't
  reach Hugging Face (network allowlist). It let the *entire* pipeline
  --chunking, Chroma wiring, metadata round-tripping, the eval
  harness-- actually be run and verified against real corpus text
  instead of trusting untested code. It is explicitly **not**
  representative of production retrieval quality (see
  [Known limitations](#known-limitations)).

Run with `--dev-tfidf` to use the offline embedder; omit it for the real
one (needs normal internet access once, to fetch the model).

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Build the vector store (production embeddings, needs internet once)
python ingest.py

# Or, offline / no model download:
python ingest.py --dev-tfidf
```

## MCP server

`mcp_server.py` exposes the retrieval pipeline as an MCP server, so any
MCP client (Claude Desktop, etc.) can query this portfolio's docs
directly. It wraps `ingest.py`/`chunking.py` rather than reimplementing
retrieval -- a fix to the chunker applies here automatically.

**Tools exposed:**
- `ask_portfolio(question, k=3)` -- retrieval-only: returns the top-k
  matched source passages (repo, section heading, text), not a
  generated answer. Wiring in a live Claude call for generation is a
  natural v2, gated on an `ANTHROPIC_API_KEY` being available to this
  process, which this dev sandbox didn't have -- mirrors the eval
  harness's own not-yet-built `--with-generation` flag.
- `list_indexed_repos()` -- which of the 5 repos are indexed and how
  many chunks each contributes.
- `get_eval_summary()` -- the most recent `eval/eval_report.md`, so a
  live MCP client can see the eval harness's own results rather than
  needing someone to go find the file.

**Run it** (stdio transport, what Claude Desktop expects):
```bash
python ingest.py --dev-tfidf        # build the DB first
python mcp_server.py --dev-tfidf    # embedder flag MUST match ingest's
```

**Claude Desktop config** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "rag-docs-qa": {
      "command": "python",
      "args": ["/absolute/path/to/rag-docs-qa/mcp_server.py"]
    }
  }
}
```
Omit `--dev-tfidf` here once you've built the DB with the real embedder
(`python ingest.py`, no flag) -- the two must match.

### A breaking SDK change past this project's knowledge cutoff

The installed `mcp` package (2.2.0) renamed `FastMCP` to `MCPServer`
(`mcp.server.mcpserver.MCPServer`) at some point after this assistant's
training data -- the old `from mcp.server.fastmcp import FastMCP` import
that most existing tutorials use raises `ModuleNotFoundError` on this
version, with the package's own error message pointing to the rename.
`mcp_server.py` uses the current `MCPServer` API. If you're on an older
`mcp<2` install, either upgrade or use the migration guide the error
message links to.

### Verified

- A real client (the `mcp` SDK's own `ClientSession`) driving the actual
  server as a subprocess over real stdio -- not a mock of the protocol --
  confirmed: all three tools are correctly listed, `list_indexed_repos`
  reports all five repos, `ask_portfolio` retrieves the correct passage
  for a known question (the same one eval question `sd-01` uses),
  `get_eval_summary` returns a real report once one exists.
- The server fails fast with a clear error message (not a silent
  broken server) when the Chroma DB hasn't been built yet.
- **Found and fixed a real async bug** while writing the integration
  test: a shared `pytest` fixture that opened the stdio client/session
  once and `yield`ed it to multiple tests hit `RuntimeError: Attempted
  to exit cancel scope in a different task than it was entered in`.
  `mcp`'s stdio client holds an `anyio` `TaskGroup` open for the
  connection's lifetime, and `anyio` cancel scopes are task-bound --
  under `pytest-asyncio`, a shared async-generator fixture's teardown
  can run in a different task than its setup, which `anyio` rejects.
  Fixed by having each test open and fully close its own session
  within one task (`tests/test_mcp_server.py`'s `open_session()`
  helper) instead of sharing one across tests.

## Running the tests

```bash
pytest -q
```

34 tests:
- `test_chunking.py` -- pure logic, synthetic fixtures. Includes the
  hash-comment-in-fence regression case and the oversized-paragraph
  edge case.
- `test_scoring.py` -- pure logic for the eval harness's scoring
  functions (recall, hit rate, adversarial handling), no DB dependency.
- `test_ingest.py` -- integration tests against the **real** corpus and
  a real (temporary) Chroma collection with the dev embedder: verifies
  every repo contributes chunks, no chunk exceeds the size cap after
  splitting, chunk IDs are unique across the whole corpus, metadata
  survives the round-trip through Chroma, and re-running ingest doesn't
  leave stale chunks behind.
- `test_mcp_server.py` -- drives the real MCP server as a subprocess
  over real stdio with the official `mcp` client, not a protocol mock.
  Requires the Chroma DB to already be built (`python ingest.py
  --dev-tfidf`) before running -- see [MCP server](#mcp-server).

**Note:** `test_mcp_server.py` needs `chroma_db/` to already exist, so
run `python ingest.py --dev-tfidf` (and, for the full report path,
`python eval/run_eval.py --dev-tfidf`) before `pytest -q` -- this is
also why CI builds those first and runs `pytest` last, not the reverse.

CI (`ruff check .` + build the offline DB/eval report + `pytest -q`)
runs on every push, so a broken end-to-end pipeline fails CI, not just
broken unit tests.

## Eval harness

`eval/questions.json` has three sets:

1. **`answerable_single_doc`** (12 questions) -- ground truth is one
   chunk in one doc.
2. **`answerable_cross_doc`** (5 questions) -- ground truth spans 2-3
   chunks across different repos' docs (e.g. "which three tools all
   require ffmpeg, and what does each use it for").
3. **`adversarial_out_of_scope`** (8 questions) -- things genuinely not
   in the corpus (a different repo entirely, a detail that only exists
   in prior session history, a deliberately subtle "batch" wording
   trap). A correct system declines rather than confabulates.

Every `expected_chunk_ids` entry is checked by a script against the real
ingested corpus before being trusted (see "Building the eval set"
below) -- none of it is asserted from memory of what the source repos
"probably" say.

```bash
python ingest.py --dev-tfidf
python eval/run_eval.py --dev-tfidf
```

writes `eval/eval_report.md` with per-question hit/recall and, for
adversarial questions, the retrieval distance (flagging any
suspiciously confident match on a question that shouldn't have one).

### Building the eval set: two questions that didn't survive contact with the real corpus

Two originally-planned eval questions were dropped or moved after
actually checking them against the ingested text, rather than against
memory of the source projects:

- A planned cross-doc question asked *why* `flagship-pipeline` switched
  from `node-canvas` to `@napi-rs/canvas`. Grepping the real corpus
  showed `waveform-generator`'s README documents *that* `@napi-rs/canvas`
  is used, but the reasoning for the switch only exists in prior session
  history -- it was never actually committed to any README. That
  question moved to the adversarial set (`adv-02`) instead.
- A planned adversarial question asked what object-detection model
  `video-to-prompt` uses, assuming it wasn't documented. It is --
  `SPEC.md`'s Tech Stack section names YOLOv8n via `ultralytics`
  explicitly. That question moved to the answerable set (`sd-03`)
  instead.

This is the eval-harness equivalent of "test with real data before
trusting it": an eval set that hasn't been checked against the actual
corpus can silently grade a system on the wrong ground truth.

## Verified

- Ran the full pipeline end-to-end against the real 6-file, 86-chunk
  corpus (offline dev embedder): ingestion completes, all 25 ground-truth
  `expected_chunk_ids` in `questions.json` resolve to real chunks, and
  `run_eval.py` produces a real, non-trivial report (0.59 hit rate on
  answerable questions -- see [Known limitations](#known-limitations)
  for why that number is expected and not a pipeline bug).
- Metadata round-trip (doc_id, heading_path) verified against Chroma's
  actual query response shape, not assumed from the API docs.
- Re-running ingestion is idempotent -- verified by a test that runs
  `build_collection` twice against the same DB path and checks the
  count doesn't grow.

## Known limitations

- **Dev-embedder eval numbers are not production numbers.** The 0.59
  hit rate / 0.49 mean recall reported by `eval/eval_report.md` (offline
  mode) reflects TF-IDF's exact-vocabulary matching, not real semantic
  retrieval. Re-run without `--dev-tfidf` (needs Hugging Face access) to
  get numbers that mean something for the shipped system, before citing
  this eval harness's results as representative.
- **License sections are near-duplicate boilerplate across all five
  repos** ("MIT -- see LICENSE"), which makes any question mentioning
  "license" score misleadingly close in vector space regardless of
  which repo is actually being asked about -- observed directly in the
  dev-embedder eval run (`adv-07`). This may persist even with a real
  embedding model, since the underlying text really is near-identical;
  worth re-checking once real embeddings are in place rather than
  assuming it's TF-IDF-specific.
- **No generation or hallucination-check step yet.** Everything above is
  retrieval-only. The eval harness's adversarial set currently only
  checks retrieval-distance confidence as a proxy for "the system should
  decline to answer" -- it does not yet verify that a generated answer
  actually declines. That requires a live LLM call and is the next
  piece to build (`--with-generation` flag, not yet implemented).
- **Small corpus.** 86 chunks across 5 repos is enough to hand-verify
  every ground-truth answer, which was the point, but it also means
  retrieval looks easier here than it would on a larger, messier corpus.
