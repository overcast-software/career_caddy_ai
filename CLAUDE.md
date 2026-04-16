# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Career Caddy AI provides browser automation, job extraction, and chat agents for the Career Caddy backend API. Email-based workflows (notmuch classification, email pipeline) have been moved to `career_caddy_automation`.

## Environment Setup

Dependencies are managed via `pyproject.toml` with `uv`. Environment variables come from `.envrc` (use `direnv` or `source .envrc`):

| Variable | Required | Purpose |
|----------|----------|---------|
| `CC_API_TOKEN` | Yes | Career Caddy backend auth token |
| `OPENAI_API_KEY` | Yes* | LLM provider (* or use ANTHROPIC_API_KEY) |
| `ANTHROPIC_API_KEY` | No | Alternative LLM provider |
| `FASTMCP_HOST` | No | MCP server bind host (default: `0.0.0.0`) |
| `FASTMCP_PORT` | No | MCP server port (default: `3002`) |
| `CAMOUFOX_DATA_DIR` | No | Where camoufox stores its browser binary |
| `LOGFIRE_TOKEN` | No | Observability / tracing |
| `OLLAMA_API_BASE` | No | Local Ollama endpoint (default: `http://127.0.0.1:11434`) |

```bash
# 1. Install dependencies (requires Python 3.13+)
pip install uv
uv sync

# 2. Download the Camoufox Firefox binary (one-time, ~200MB)
python -m camoufox fetch

# 3. Configure environment
cp .envrc.example .envrc    # Required: CC_API_TOKEN, OPENAI_API_KEY
source .envrc               # or: use direnv

# 4. Set up browser credentials (needed for browser automation)
cp secrets.yml.example secrets.yml   # fill in your job site credentials
```

**Creating `CC_API_TOKEN`**: After initializing the Career Caddy API (see `api/CLAUDE.md`), create a long-lived key:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/token/ \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"secret"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access'])")

curl -X POST http://localhost:8000/api/v1/api-keys/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"ai-agent"}'
# Copy the "key" field from the response → CC_API_TOKEN
```

## Running the Pipeline

```bash
# Scrape a single job URL and add it to Career Caddy
uv run caddy-pipeline https://example.com/job/posting
```

## Architecture

### Core Pattern: Pydantic-AI Agents + MCP Servers

Agents (in `agents/`) use the `pydantic-ai` framework. They access tools through **MCP servers** (in `mcp_servers/`) using two transport types:
- `MCPServerStdio` — launches the server as a subprocess (used by the pipeline; no external service needed)
- `MCPServerSSE` — connects to a running HTTP/SSE server (used when the `browser-mcp` docker service is running)

**Default LLM model**: `openai:gpt-4o-mini`. Change via the `model=` argument to `Agent(...)`.

### MCP Servers

| Server | Transport | Port | Purpose |
|--------|-----------|------|---------|
| `career_caddy_server.py` | stdio | — | CRUD on Career Caddy REST API (jobs, companies, applications) |
| `browser_server.py` | stdio or SSE | 3004 | Browser automation via `camoufox` |
| `public_server.py` | SSE | 8000 | Public MCP endpoint at `mcp.careercaddy.online` |
| `chat_server.py` | SSE | 8000 | Frontend chat via SSE streaming |

### Agent Responsibilities

- **`career_caddy_agent.py`** — Validates and posts jobs to Career Caddy API; checks for duplicates before creating
- **`job_extractor_agent.py`** — Extracts structured `JobPostData` from raw job posting text
- **`job_email_to_caddy.py`** — URL scraping pipeline: browser scrape → extract → Career Caddy post

**Critical rules in `career_caddy_agent.py`** (enforced via system prompt):
1. Always call `find_job_post_by_link(url)` before creating — never create duplicates
2. Use `create_job_post_with_company_check(company_name)`, not `create_job_post()` (avoids FK errors)
3. Stop immediately if any tool returns `{"success": false, ...}`
4. Never retry failed tool calls or scan by incrementing IDs

### Data Models

- `lib/models/job_models.py` — `JobPostData`, `CompanyData` (primary DTOs between agents)
- `lib/models/career_caddy.py` — API-specific models (`JobPostCreate`, `APIResponse`, `APICredentials`)

### Credentials & Browser Auth

`lib/browser/credentials.py` loads two YAML files:

**`secrets.yml`** (gitignored — create from `secrets.yml.example`):
```yaml
linkedin.com:
  username: your_email@example.com
  password: your_password
```

**`sites.yml`** (versioned — add login automation config for new sites):
```yaml
linkedin.com:
  login_url: https://www.linkedin.com/login
  username_selector: "#username"
  password_selector: "#password"
  submit_selector: ".login__form_action_container button"
  post_login_check: ".global-nav__me"
```

Domain lookup normalizes subdomains automatically (`www.linkedin.com` → `linkedin.com`).

## LLM Configuration

Per-agent model overrides are controlled via environment variables in `agents/agent_factory.py:get_model()`.

Resolution order: role-specific env var → `CADDY_DEFAULT_MODEL` → hardcoded `openai:gpt-4o-mini`.

| Env Var | Agent Role | Recommended |
|---------|------------|-------------|
| `CADDY_MODEL` | career_caddy_agent (CRUD) | gpt-4o-mini |
| `CHAT_MODEL` | chat_server (user-facing) | claude-haiku-4-5 |
| `JOB_EXTRACTOR_MODEL` | job_extractor | claude-haiku-4-5 |
| `BROWSER_SCRAPER_MODEL` | browser scraper | gpt-4o-mini |
| `CADDY_DEFAULT_MODEL` | fallback for all roles | gpt-4o-mini |

The hold poller (`scripts/hold_poller.py`) skips the browser_scraper LLM entirely — it calls `scrape_page()` directly as a Python function, then hands content to the job extractor.

## Tests

```bash
uv run pytest tests/
```
