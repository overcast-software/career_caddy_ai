"""Obstacle agent — resolves UI blockers on a live Playwright page.

Invoked only when the fast deterministic paths (`_try_rememberme_reauth`,
`_try_expand_truncations`) fail. Gets the current page text + free-text
hints from the ScrapeProfile and picks clicks to reach real content.

Scoped to `try_click` + `get_text` only — no fill/submit — to keep the
blast radius tiny. Rejects clicks that look like sign-out or cancel.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext

from agents.agent_factory import get_model

logger = logging.getLogger(__name__)

OBSTACLE_MODEL_ENV = "OBSTACLE_AGENT_MODEL"

SYSTEM_PROMPT = """\
You are resolving an obstacle blocking a web scraper from reaching real page content.

You have two tools:
- try_click(selector): click a CSS selector on the live page.
- get_text(): return the current visible body text to confirm the obstacle cleared.

Given the page text and site-specific hints, pick the CSS selector most likely
to resolve the obstacle (e.g. a specific account's "Continue" button, a
"See more" expander, a cookie accept button). After clicking, call get_text()
to confirm — if the obstacle is gone, stop. Otherwise try once more.

Never click anything labelled sign out, log out, cancel, back, or close.
Prefer stable selectors (attributes, classes) over bare tags. Stop after at
most 3 clicks total — over-clicking is worse than failing.
"""

DISALLOWED_ACTION_WORDS = (
    "sign out", "signout", "log out", "logout",
    "cancel", " back", "close", "dismiss",
)


@dataclass
class _PageDeps:
    page: Any  # Playwright Page; kept untyped to avoid forcing the import here.


async def run_obstacle_agent(
    page, hints: str, page_text: str, max_clicks: int = 3,
) -> dict:
    """Run the obstacle agent against a live Playwright page.

    Returns {"resolved": bool, "actions": [selector, ...], "note": str}.
    "resolved" reflects whether any click succeeded — the caller must
    re-check the page state to confirm the real obstacle cleared.
    """
    model = os.environ.get(OBSTACLE_MODEL_ENV) or get_model("browser_scraper")
    agent: Agent[_PageDeps, str] = Agent(
        model, deps_type=_PageDeps, system_prompt=SYSTEM_PROMPT,
    )
    clicks: list[str] = []

    @agent.tool
    async def try_click(ctx: RunContext[_PageDeps], selector: str) -> str:
        """Attempt to click the element matching `selector`."""
        if len(clicks) >= max_clicks:
            return "click_limit_reached"
        lowered = selector.lower()
        for word in DISALLOWED_ACTION_WORDS:
            if word in lowered:
                return f"rejected: selector contains {word!r}"
        try:
            el = await ctx.deps.page.query_selector(selector)
        except Exception as exc:
            return f"selector_error: {exc}"
        if not el:
            return "not_found"
        try:
            text = ((await el.inner_text()) or "").strip().lower()
        except Exception:
            text = ""
        for word in DISALLOWED_ACTION_WORDS:
            if word in text:
                return f"rejected: element text contains {word!r}"
        try:
            await el.scroll_into_view_if_needed(timeout=2_000)
            await el.click(timeout=5_000)
            try:
                await ctx.deps.page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass
            await asyncio.sleep(1)
            clicks.append(selector)
            return f"clicked ({text or 'no-text'})"
        except Exception as exc:
            return f"click_error: {exc}"

    @agent.tool
    async def get_text(ctx: RunContext[_PageDeps]) -> str:
        """Return current visible page text."""
        try:
            return await ctx.deps.page.inner_text("body")
        except Exception as exc:
            return f"error: {exc}"

    # Keep prompt compact — most obstacles are decidable from a few kB.
    snippet = page_text.strip()
    if len(snippet) > 4000:
        snippet = snippet[:4000] + "\n...[truncated]"

    user_prompt = (
        f"Site-specific hints for resolving obstacles:\n{hints}\n\n"
        f"Current page visible text:\n---\n{snippet}\n---\n\n"
        "Resolve the obstacle."
    )

    try:
        result = await agent.run(user_prompt, deps=_PageDeps(page=page))
        note = getattr(result, "output", None) or getattr(result, "data", "") or ""
        return {"resolved": bool(clicks), "actions": clicks, "note": str(note)[:500]}
    except Exception as exc:
        logger.warning("obstacle agent run failed: %s", exc, exc_info=True)
        return {"resolved": False, "actions": clicks, "note": f"agent_error: {exc}"}
