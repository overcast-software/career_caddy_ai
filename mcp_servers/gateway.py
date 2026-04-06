"""
Career Caddy MCP Gateway — all servers on one port.

single /sse endpoint, tools namespaced by prefix
    email_*    → email_server.py tools (notmuch search, tag, etc.)
    caddy_*    → career_caddy_server.py tools (jobs, companies, applications)
    browser_*  → browser_server.py tools (scrape_page, camofox raw tools)

    Connect at: http://localhost:3002/sse

    Run with: python mcp_servers/gateway.py --multi-path

All sub-servers are started lazily as stdio subprocesses when the first
client connects, so startup is fast and unused servers never run.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastmcp import FastMCP
from mcp_servers.email_server import server as email_server
from mcp_servers.career_caddy_server import server as caddy_server
from mcp_servers.browser_server import server as browser_server

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
# Keep noisy low-level loggers at INFO to avoid overwhelming output
for _noisy in ("httpcore", "httpx", "urllib3", "anyio", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"
PORT = 3002


# ---------------------------------------------------------------------------
# Option A — single endpoint, all tools prefixed
# ---------------------------------------------------------------------------


def make_gateway() -> FastMCP:
    gateway = FastMCP("career-caddy-gateway")
    gateway.mount(email_server, namespace="email")
    gateway.mount(caddy_server, namespace="caddy")
    gateway.mount(browser_server, namespace="browser")
    return gateway


# ---------------------------------------------------------------------------
# Option B — one Starlette app, each server mounted at its own SSE path
# ---------------------------------------------------------------------------


def make_multi_path_app():
    try:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount
    except ImportError:
        logger.error("starlette and uvicorn are required for --multi-path mode")
        sys.exit(1)

    email_server = _proxy("email_server.py", "email")
    caddy_server = _proxy("career_caddy_server.py", "caddy")
    browser_server = _proxy("browser_server.py", "browser")

    # Each FastMCP exposes an ASGI app via http_app()
    app = Starlette(
        routes=[
            Mount("/email", app=email_server.http_app()),
            Mount("/caddy", app=caddy_server.http_app()),
            Mount("/browser", app=browser_server.http_app()),
        ]
    )
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Career Caddy MCP Gateway")
    parser.add_argument(
        "--multi-path",
        action="store_true",
        help=(
            "Serve each MCP server at its own SSE path "
            "(/email/sse, /caddy/sse, /browser/sse) "
            "instead of a single /sse endpoint with prefixed tools."
        ),
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.multi_path:
        import uvicorn

        logger.info(f"Starting multi-path gateway on {args.host}:{args.port}")
        logger.info(f"  Email:   http://{args.host}:{args.port}/email/sse")
        logger.info(f"  Caddy:   http://{args.host}:{args.port}/caddy/sse")
        logger.info(f"  Browser: http://{args.host}:{args.port}/browser/sse")
        app = make_multi_path_app()
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        logger.info(f"Starting single-endpoint gateway on {args.host}:{args.port}")
        logger.info(f"  Endpoint: http://{args.host}:{args.port}/mcp")
        logger.info("  Tools: email_*, caddy_*, browser_*")
        gateway = make_gateway()

        async def _log_tools():
            tools = await gateway.list_tools()
            logger.info(f"Registered tools ({len(tools)}):")
            for t in sorted(tools, key=lambda t: t.name):
                logger.info(f"  {t.name}")

        asyncio.run(_log_tools())
        gateway.run(transport="streamable-http", host=args.host, port=args.port)
        # gateway.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
