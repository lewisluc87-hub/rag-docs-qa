"""Markdown chunking for the RAG ingestion pipeline.

Splits a markdown document into sections at heading boundaries (levels 1-3).
Deliberately fence-aware: a line starting with '#' inside a ``` code block
(e.g. a bash/python comment) is NOT treated as a heading. Naively splitting
on any line starting with '#' silently corrupts chunks on docs that contain
shell comments in fenced examples -- which several of these READMEs do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*```")


@dataclass
class Chunk:
    doc_id: str  # e.g. "video-to-prompt/README.md" -- always the real source file
    heading_path: str  # e.g. "video2prompt > Running the tests"
    level: int
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    text: str
    part: int | None = None  # set when split_oversized() sub-divides this section

    @property
    def chunk_id(self) -> str:
        base = f"{self.doc_id}#{self.start_line}-{self.end_line}"
        return f"{base}::part{self.part}" if self.part is not None else base


@dataclass
class _OpenSection:
    level: int
    title: str
    start_line: int
    body_lines: list[str] = field(default_factory=list)


def chunk_markdown(text: str, doc_id: str) -> list[Chunk]:
    """Split markdown into heading-bounded chunks, ignoring headings inside code fences.

    A new chunk starts at every heading of level 1-3 encountered outside a
    fenced code block. Each chunk's heading_path is a breadcrumb built from
    the active heading stack (e.g. "Flagship pipeline > Usage > Flags"),
    so retrieval results carry enough context to be understood standalone.
    """
    lines = text.splitlines()
    in_fence = False
    # stack of (level, title) for breadcrumb construction
    heading_stack: list[tuple[int, str]] = []
    chunks: list[Chunk] = []
    current: _OpenSection | None = None

    # We need the heading_stack *as it was when the section opened* for its
    # own breadcrumb, so snapshot it at open time rather than at close time.
    heading_stack_snapshot: list[tuple[int, str]] = []

    for i, line in enumerate(lines, start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            if current is not None:
                current.body_lines.append(line)
            continue

        m = None if in_fence else HEADING_RE.match(line)
        if m:
            # close out previous section ending at the line before this heading
            if current is not None:
                body = "\n".join(current.body_lines).strip("\n")
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        heading_path=" > ".join(t for _, t in heading_stack_snapshot),
                        level=current.level,
                        start_line=current.start_line,
                        end_line=i - 1,
                        text=body,
                    )
                )
            level = len(m.group(1))
            title = m.group(2).strip()
            # pop stack to this level, then push
            heading_stack = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, title))
            heading_stack_snapshot = list(heading_stack)
            current = _OpenSection(level=level, title=title, start_line=i)
        else:
            if current is None:
                # content before the first heading -- start an untitled preamble section
                current = _OpenSection(level=0, title="", start_line=i)
                heading_stack_snapshot = []
            current.body_lines.append(line)

    if current is not None:
        body = "\n".join(current.body_lines).strip("\n")
        if body.strip():
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    heading_path=" > ".join(t for _, t in heading_stack_snapshot),
                    level=current.level,
                    start_line=current.start_line,
                    end_line=len(lines),
                    text=body,
                )
            )

    # drop chunks with empty bodies (e.g. a heading immediately followed by another heading)
    return [c for c in chunks if c.text.strip()]


def split_oversized(chunks: list[Chunk], max_chars: int = 800) -> list[Chunk]:
    """Sub-split any chunk longer than max_chars on paragraph boundaries.

    Embedding models truncate silently past their max sequence length
    (e.g. all-MiniLM-L6-v2 caps around ~256 tokens, roughly 1000 chars).
    Left alone, a long section like SPEC.md's "Breakdown Mode" (3000+
    chars) would only ever be embedded from its first paragraph or two --
    every paragraph after the truncation point becomes invisible to
    retrieval with no error or warning. This greedily packs paragraphs
    into sub-chunks under max_chars, keeping the same heading_path so
    retrieved results are still identifiable, and gives each part a
    stable, unique chunk_id via an appended "part" number.
    """
    out: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= max_chars:
            out.append(c)
            continue

        paragraphs = [p for p in re.split(r"\n\s*\n", c.text) if p.strip()]
        parts: list[str] = []
        buf = ""
        for p in paragraphs:
            candidate = f"{buf}\n\n{p}" if buf else p
            if len(candidate) > max_chars and buf:
                parts.append(buf)
                buf = p
            else:
                buf = candidate
        if buf:
            parts.append(buf)

        # A single paragraph longer than max_chars on its own can't be
        # split further without breaking mid-sentence context; keep it
        # whole rather than mangling it -- truncation on a single
        # oversized paragraph is a smaller, known cost than reflowing text.
        n_lines = c.end_line - c.start_line + 1
        for idx, part in enumerate(parts, start=1):
            # Approximate line offsets proportionally so start/end stay
            # monotonic and chunk_id stays unique; exactness isn't needed
            # since heading_path (not line numbers) is what's shown to users.
            frac_start = (idx - 1) / len(parts)
            frac_end = idx / len(parts)
            out.append(
                Chunk(
                    doc_id=c.doc_id,
                    heading_path=c.heading_path,
                    level=c.level,
                    start_line=c.start_line + int(frac_start * n_lines),
                    end_line=c.start_line + int(frac_end * n_lines) - 1,
                    text=part,
                    part=idx,
                )
            )
    return out
