"""Regression tests: scrape_page retries once on transient browser errors.

LinkedIn (and other JS-heavy hosts) sometimes crash the camoufox subprocess
mid-scrape — Playwright surfaces the failure as
"Page.goto: Target page, context or browser has been closed". Before the
retry was added, the scrape was marked failed and the user had to wait for
the next poller tick. Now we retry once with a fresh launch.

The imports inside the autouse fixture (and the sys.modules cleanup) keep
this test from polluting sys.modules with lib.browser.*, which would
break tests/test_toolsets.py::test_toolsets_does_not_import_browser.
"""

import json
import sys
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _quarantine_browser_imports():
    """Per-test snapshot/restore of sys.modules so importing
    mcp_servers.browser_server inside the test doesn't leak."""
    before = set(sys.modules)
    yield
    for name in list(sys.modules):
        if name in before:
            continue
        if name.startswith("lib.browser") or name.startswith("mcp_servers.browser_server"):
            sys.modules.pop(name, None)


def _import_under_test():
    """Defer import until inside a test so the autouse cleanup applies."""
    from mcp_servers.browser_server import (
        _is_transient_browser_error,
        scrape_page,
    )
    return _is_transient_browser_error, scrape_page


class TestTransientBrowserErrorMatcher:
    def test_matches_known_signatures(self):
        _is_transient, _ = _import_under_test()
        assert _is_transient(
            "Page.goto: Target page, context or browser has been closed"
        )
        assert _is_transient("Browser has been closed")
        assert _is_transient("WebSocket Connection closed unexpectedly")

    def test_does_not_match_unrelated_errors(self):
        _is_transient, _ = _import_under_test()
        assert not _is_transient("blocked_page_detected")
        assert not _is_transient("login_wall_detected")
        assert not _is_transient(
            "TimeoutError: Navigation timeout of 60000 ms exceeded"
        )
        assert not _is_transient("")


@pytest.mark.asyncio
class TestScrapePageRetry:
    async def test_retries_once_on_transient_error_then_succeeds(self):
        _, scrape_page = _import_under_test()
        once = AsyncMock(
            side_effect=[
                {"error": "Target page, context or browser has been closed"},
                {"content": "ok", "url": "https://x", "title": "t"},
            ]
        )
        with patch(
            "mcp_servers.browser_server._resolve_scrape_inputs",
            return_value=("x.com", "x.com", [], {}),
        ), patch(
            "mcp_servers.browser_server._scrape_page_once", once
        ):
            result = await scrape_page("https://x.com/job/1")

        parsed = json.loads(result)
        assert parsed.get("error") is None
        assert parsed["content"] == "ok"
        assert once.await_count == 2

    async def test_does_not_retry_on_terminal_error(self):
        _, scrape_page = _import_under_test()
        once = AsyncMock(return_value={"error": "blocked_page_detected"})
        with patch(
            "mcp_servers.browser_server._resolve_scrape_inputs",
            return_value=("x.com", "x.com", [], {}),
        ), patch(
            "mcp_servers.browser_server._scrape_page_once", once
        ):
            result = await scrape_page("https://x.com/job/1")

        parsed = json.loads(result)
        assert parsed["error"] == "blocked_page_detected"
        assert once.await_count == 1

    async def test_no_retry_on_success_first_try(self):
        _, scrape_page = _import_under_test()
        once = AsyncMock(
            return_value={"content": "ok", "url": "https://x", "title": "t"}
        )
        with patch(
            "mcp_servers.browser_server._resolve_scrape_inputs",
            return_value=("x.com", "x.com", [], {}),
        ), patch(
            "mcp_servers.browser_server._scrape_page_once", once
        ):
            result = await scrape_page("https://x.com/job/1")

        assert json.loads(result)["content"] == "ok"
        assert once.await_count == 1

    async def test_propagates_persistent_transient_error(self):
        """Two crashes in a row should still surface the error rather
        than loop. The poller will mark the scrape failed and move on."""
        _, scrape_page = _import_under_test()
        once = AsyncMock(
            return_value={"error": "Target page, context or browser has been closed"}
        )
        with patch(
            "mcp_servers.browser_server._resolve_scrape_inputs",
            return_value=("x.com", "x.com", [], {}),
        ), patch(
            "mcp_servers.browser_server._scrape_page_once", once
        ):
            result = await scrape_page("https://x.com/job/1")

        assert "Target page" in json.loads(result)["error"]
        assert once.await_count == 2
