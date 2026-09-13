from chunking import Chunk, chunk_markdown, split_oversized

# --- Regression fixture -----------------------------------------------
# Mirrors the exact shape found in web-scraper/README.md and
# video-to-prompt/README.md: a "## Usage" section whose fenced bash
# example contains lines starting with '#' (shell comments). A naive
# line-scan chunker would misread these as headings and silently split
# the code block into multiple bogus sections.
DOC_WITH_HASH_COMMENTS_IN_FENCE = """\
# Configurable web scraper

A CLI tool that scrapes listing-style pages.

## Usage

```bash
# Static HTML site
scraper run configs/site.json

# JS-rendered site (uses headless Chromium via Playwright)
scraper run configs/site.json --fetch-mode rendered

# Override page count / delay from the command line
scraper run configs/site.json --max-pages 5
```

## Known limitations (v1)

Network-dependent; no retry logic yet.
"""


def test_hash_comments_inside_fence_are_not_treated_as_headings():
    chunks = chunk_markdown(DOC_WITH_HASH_COMMENTS_IN_FENCE, "web-scraper/README.md")
    headings = [c.heading_path for c in chunks]

    # Exactly the real headings -- none of the "# Static HTML site" etc.
    # shell comments should have produced their own chunk.
    assert headings == [
        "Configurable web scraper",
        "Configurable web scraper > Usage",
        "Configurable web scraper > Known limitations (v1)",
    ]

    usage_chunk = chunks[1]
    # The fenced code block must survive intact as one contiguous block,
    # including every '#'-prefixed comment line inside it.
    assert "# Static HTML site" in usage_chunk.text
    assert "# JS-rendered site (uses headless Chromium via Playwright)" in usage_chunk.text
    assert "# Override page count / delay from the command line" in usage_chunk.text
    assert "```bash" in usage_chunk.text


def test_preamble_before_first_heading_becomes_its_own_chunk():
    chunks = chunk_markdown(DOC_WITH_HASH_COMMENTS_IN_FENCE, "web-scraper/README.md")
    assert chunks[0].heading_path == "Configurable web scraper"
    assert "A CLI tool that scrapes listing-style pages." in chunks[0].text


def test_breadcrumb_reflects_nesting():
    doc = """\
# Flagship pipeline

## Usage

### Flags

--dry-run does nothing destructive.
"""
    chunks = chunk_markdown(doc, "flagship-pipeline/README.md")
    flags_chunk = next(c for c in chunks if c.level == 3)
    assert flags_chunk.heading_path == "Flagship pipeline > Usage > Flags"


def test_sibling_heading_resets_breadcrumb_stack():
    doc = """\
# Root

## Section A

### Sub A1

text a1

## Section B

text b
"""
    chunks = chunk_markdown(doc, "doc.md")
    section_b = next(c for c in chunks if c.heading_path == "Root > Section B")
    assert "text b" in section_b.text
    # Section B must NOT inherit "Sub A1" in its breadcrumb
    assert "Sub A1" not in section_b.heading_path


def test_heading_immediately_followed_by_heading_produces_no_empty_chunk():
    doc = "# Title\n## Empty\n## Next\nsome text\n"
    chunks = chunk_markdown(doc, "doc.md")
    # "## Empty" has no body before "## Next" -- should be dropped, not
    # emitted as a zero-content chunk that would pollute retrieval.
    assert all(c.text.strip() for c in chunks)
    headings = [c.heading_path for c in chunks]
    assert "Title > Empty" not in headings
    assert "Title > Next" in headings


def test_chunk_id_is_stable_and_unique():
    chunks = chunk_markdown(DOC_WITH_HASH_COMMENTS_IN_FENCE, "web-scraper/README.md")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique for use as vector-DB keys"
    assert ids[0].startswith("web-scraper/README.md#")


def test_fence_toggle_state_does_not_leak_across_documents():
    # Regression guard: chunk_markdown must not use module-level state for
    # fence tracking, or an odd number of ``` in one doc would corrupt
    # heading detection in the next doc processed in the same ingest run.
    odd_fence_doc = "# A\n```\nunclosed fence\n"
    normal_doc = "# B\n\n## Real Heading\n\ntext\n"

    chunk_markdown(odd_fence_doc, "a.md")
    chunks_b = chunk_markdown(normal_doc, "b.md")

    headings = [c.heading_path for c in chunks_b]
    assert "B > Real Heading" in headings


# --- split_oversized ----------------------------------------------------

def _mk_chunk(text: str, doc_id: str = "doc.md", start=1, end=None) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        heading_path="Doc > Section",
        level=2,
        start_line=start,
        end_line=end if end is not None else start + text.count("\n"),
        text=text,
    )


def test_chunk_under_limit_is_left_untouched():
    small = _mk_chunk("short paragraph, well under the limit.")
    result = split_oversized([small], max_chars=800)
    assert result == [small]
    assert result[0].part is None


def test_oversized_chunk_is_split_on_paragraph_boundaries():
    paragraphs = [f"Paragraph {i}. " + ("word " * 30) for i in range(6)]
    big_text = "\n\n".join(paragraphs)
    assert len(big_text) > 800  # sanity check on the fixture itself

    big = _mk_chunk(big_text, start=100, end=140)
    parts = split_oversized([big], max_chars=800)

    assert len(parts) > 1
    for p in parts:
        assert len(p.text) <= 800 or "\n\n" not in p.text  # allow lone oversized paragraph
        assert p.heading_path == "Doc > Section"  # context preserved on every sub-chunk
        assert p.doc_id == "doc.md"  # source file attribution NOT corrupted

    # reassembling all parts should recover every paragraph, none dropped
    reassembled = "\n\n".join(p.text for p in parts)
    for para in paragraphs:
        assert para in reassembled


def test_split_oversized_produces_unique_chunk_ids():
    paragraphs = [f"Paragraph {i}. " + ("word " * 30) for i in range(6)]
    big = _mk_chunk("\n\n".join(paragraphs), start=100, end=140)
    parts = split_oversized([big], max_chars=800)

    ids = [p.chunk_id for p in parts]
    assert len(ids) == len(set(ids))
    assert all("::part" in i for i in ids)


def test_single_oversized_paragraph_is_kept_whole_not_mangled():
    # A single paragraph with no blank-line breaks longer than max_chars
    # can't be split without cutting mid-sentence -- verify it's kept
    # intact rather than silently truncated or corrupted.
    one_giant_paragraph = "word " * 300  # no \n\n at all
    big = _mk_chunk(one_giant_paragraph, start=1, end=1)
    parts = split_oversized([big], max_chars=800)

    assert len(parts) == 1
    assert parts[0].text == one_giant_paragraph
