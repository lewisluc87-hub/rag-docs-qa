"""Integration tests for mcp_server.py.

These drive the ACTUAL server as a subprocess over real stdio, using the
official mcp SDK's client -- not a mock of the protocol. This is the same
"real, not stubbed" philosophy as test_ingest.py: it proves the tool
schemas, the stdio handshake, and each tool's execution path all
genuinely work together, not just that the underlying functions do.

Requires a Chroma DB already built with the dev embedder:
    python ingest.py --dev-tfidf
before running these tests (matching what CI's smoke-test step does).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

SERVER_PATH = Path(__file__).parent.parent / "mcp_server.py"


@asynccontextmanager
async def open_session():
    """Start the real server as a subprocess and open a real MCP session against it.

    Deliberately NOT a pytest fixture: mcp's stdio_client holds an anyio
    TaskGroup open across the connection's lifetime, and anyio's cancel
    scopes are task-bound -- a shared async-generator fixture's teardown
    can run in a different task than its setup under pytest-asyncio,
    which anyio rejects with "cancel scope in a different task than it
    was entered in". Opening and closing the session fully within each
    test's own task sidesteps that entirely.
    """
    params = StdioServerParameters(command="python3", args=[str(SERVER_PATH), "--dev-tfidf"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


async def test_lists_all_three_tools():
    async with open_session() as session:
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert names == {"ask_portfolio", "list_indexed_repos", "get_eval_summary"}


async def test_list_indexed_repos_reports_all_five_repos():
    async with open_session() as session:
        res = await session.call_tool("list_indexed_repos", {})
        text = res.content[0].text
        for repo in [
            "flagship-pipeline",
            "video-to-prompt",
            "waveform-generator",
            "web-scraper",
            "youtube-transcriber",
        ]:
            assert repo in text


async def test_ask_portfolio_retrieves_relevant_passage():
    # Same query used in the eval set (sd-01) with a known-correct answer --
    # verifies real retrieval through the real server, not just that some
    # text comes back.
    async with open_session() as session:
        res = await session.call_tool(
            "ask_portfolio",
            {
                "question": "What flag identifies a stable, unique item across scraper monitor runs?",
                "k": 3,
            },
        )
        text = res.content[0].text
        assert "web-scraper/README.md" in text
        assert "--key-field" in text


async def test_ask_portfolio_respects_k_parameter():
    async with open_session() as session:
        res_k1 = await session.call_tool("ask_portfolio", {"question": "ffmpeg", "k": 1})
        res_k3 = await session.call_tool("ask_portfolio", {"question": "ffmpeg", "k": 3})
        # Each passage block is separated by "---"; k=1 should yield exactly
        # one block, k=3 up to three.
        assert res_k1.content[0].text.count("---") == 0
        assert res_k3.content[0].text.count("---") <= 2


async def test_get_eval_summary_returns_real_report_or_clear_absence_message():
    async with open_session() as session:
        res = await session.call_tool("get_eval_summary", {})
        text = res.content[0].text
        # Either a real report (has this exact heading) or the documented
        # "not generated yet" message -- never a raw traceback or empty string.
        assert text.startswith("# Eval report") or "Run `python eval/run_eval.py`" in text
