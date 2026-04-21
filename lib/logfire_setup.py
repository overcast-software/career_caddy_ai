"""Central logfire wiring for the ai services.

Every ai entry point (chat_server, browser_server, career_caddy_server,
public_server, hold_poller, agent scripts) calls ``setup_logfire`` at
startup. When ``LOGFIRE_TOKEN`` is set the helper:

  1. configures logfire with a service-scoped name,
  2. attaches ``LogfireLoggingHandler`` to the stdlib root logger so
     ``logger.info(...)`` calls flow to logfire (not just explicit
     ``logfire.info(...)`` / ``logfire.span(...)`` calls), and
  3. auto-instruments LLM + HTTP clients so every call gets a span
     with prompt/response/token counts without per-callsite
     ``logfire.span`` boilerplate.

When the token is unset the call is a silent no-op — handy for local
dev that shouldn't ship spans unless the operator opts in.
"""
from __future__ import annotations

import logging
import os

_SETUP_DONE_KEY = "_cc_logfire_setup_done"


def setup_logfire(service_name: str, *, instrument_llm: bool = True) -> bool:
    """Configure logfire for this process.

    Returns ``True`` if logfire was actually configured, ``False`` when
    the token was missing and the call no-op'd. Idempotent — safe to
    call from multiple entry points within the same process.
    """
    if not os.environ.get("LOGFIRE_TOKEN"):
        return False
    if os.environ.get(_SETUP_DONE_KEY) == "1":
        return True

    try:
        import logfire
    except ImportError:
        logging.getLogger(__name__).warning(
            "LOGFIRE_TOKEN set but logfire package not installed; skipping setup"
        )
        return False

    # scrubbing=False: we trust the messages we emit and the default
    # scrubber eats too much useful metadata (token counts, user ids).
    # console=False: don't double-print to stdout — the existing
    # logging.StreamHandler already writes to the terminal.
    logfire.configure(
        service_name=service_name,
        scrubbing=False,
        console=False,
    )

    # Bridge stdlib logging → logfire. Without this, `logger.info(...)`
    # is invisible to logfire; only direct `logfire.info(...)` calls
    # ship spans. INFO-level default so DEBUG noise doesn't flood.
    from logfire.integrations.logging import LogfireLoggingHandler

    handler = LogfireLoggingHandler()
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    # Avoid duplicates on idempotent re-entry.
    already = any(isinstance(h, LogfireLoggingHandler) for h in root.handlers)
    if not already:
        root.addHandler(handler)
    # Root needs to be at INFO or lower for our INFO handler to see anything.
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)

    if instrument_llm:
        # Every LLM call becomes a span with prompt/response/tokens.
        # These are safe to call even if the underlying library isn't
        # imported in this process — logfire patches on first use.
        for fn_name in (
            "instrument_pydantic_ai",
            "instrument_openai",
            "instrument_anthropic",
            "instrument_httpx",
        ):
            fn = getattr(logfire, fn_name, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — instrumentation is best-effort
                logging.getLogger(__name__).warning(
                    "logfire.%s failed: %s", fn_name, exc
                )

    os.environ[_SETUP_DONE_KEY] = "1"
    logging.getLogger(__name__).info(
        "logfire configured — service=%s instrument_llm=%s",
        service_name,
        instrument_llm,
    )
    return True
