"""Message history utilities for pydantic-ai agents."""

import json
from dataclasses import replace as dc_replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

# Keep history tight so large tool responses (e.g. get_career_data) don't
# accumulate across turns.  The full response is always delivered — old turns
# are dropped to make room.  gpt-4o-mini has a 128k context window.
# Use 3 chars/token (JSON is denser than prose) to avoid underestimating.
_DEFAULT_MAX_TOKENS = 20_000
# Conservative estimate: 1 token ≈ 3 characters for JSON-heavy content
_CHARS_PER_TOKEN = 3
# Emergency cap: only applied explicitly — never auto-applied in truncate_message_history.
# Prevents a response so large it blows the model's context window entirely.
# gpt-4o-mini: 128k tokens ≈ 384k chars at 3 chars/token.
_MAX_TOOL_RESPONSE_CHARS = 300_000


def _estimate_tokens(msg: ModelMessage) -> int:
    """Rough token estimate for a message based on JSON character count."""
    try:
        text = json.dumps(msg, default=str)
    except Exception:
        text = str(msg)
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _cap_tool_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Truncate oversized tool responses in-place to prevent context blowout."""
    result = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            new_parts = []
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and isinstance(part.content, str)
                    and len(part.content) > _MAX_TOOL_RESPONSE_CHARS
                ):
                    truncated = part.content[:_MAX_TOOL_RESPONSE_CHARS] + "\n... [truncated]"
                    part = dc_replace(part, content=truncated)
                new_parts.append(part)
            if new_parts != list(msg.parts):
                msg = dc_replace(msg, parts=new_parts)
        result.append(msg)
    return result


def truncate_message_history(
    messages: list[ModelMessage],
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list[ModelMessage]:
    """Drop old messages when the history exceeds the token budget.

    Keeps as many recent messages as fit within max_tokens, always preserving
    complete tool call / tool return pairs. The resulting history is guaranteed
    to start with a user-prompt message (never a tool return or assistant turn).
    """
    total = sum(_estimate_tokens(m) for m in messages)
    if total <= max_tokens:
        return messages

    # Walk backwards from the end, accumulating messages until we hit the budget.
    kept: list[ModelMessage] = []
    budget = max_tokens
    for msg in reversed(messages):
        cost = _estimate_tokens(msg)
        if cost > budget and kept:
            break
        kept.append(msg)
        budget -= cost

    kept.reverse()

    # Ensure we don't start with a tool-return or a pure-tool-call response —
    # both require a preceding message that we may have just dropped.
    while kept:
        first = kept[0]
        if isinstance(first, ModelRequest):
            # Drop if it starts with a ToolReturnPart (no preceding tool_calls)
            has_only_returns = all(isinstance(p, ToolReturnPart) for p in first.parts)
            has_any_return = any(isinstance(p, ToolReturnPart) for p in first.parts)
            if has_only_returns:
                kept.pop(0)
                continue
            if has_any_return:
                # Strip the leading ToolReturnParts, keep user content
                stripped = [p for p in first.parts if not isinstance(p, ToolReturnPart)]
                kept[0] = dc_replace(first, parts=stripped)
        elif isinstance(first, ModelResponse):
            # An assistant message with no preceding user turn — drop it
            kept.pop(0)
            continue
        break

    # Never return an empty list — pydantic-ai raises UserError if we do.
    # If we've stripped everything (e.g. history was all tool returns), fall back
    # to returning the original messages unchanged rather than blowing up.
    return kept if kept else messages


def sanitize_orphaned_tool_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove tool call/response pairs where not all parallel tool calls were answered.

    Handles three failure modes:

    1. Single unanswered tool call (MCP server crash before any response):
       [assistant: tool_call=A] [user: new turn]
       → drop the assistant message

    2. Partial parallel tool call response (one of N parallel calls fails):
       [assistant: tool_calls=[A, B]] [user: tool_return=A only] [assistant: ...]
       → drop the assistant message AND the partial pure-tool-return message

    3. Mixed message after partial failure:
       [assistant: tool_calls=[A, B]] [user: tool_return=A + user_text] [...]
       → drop the assistant message AND strip the ToolReturnPart(s) from the
         mixed message (keeping its user-text content), because those returns
         now have no preceding tool_calls and would cause OpenAI to reject the
         history with "messages with role 'tool' must be a response to a
         preceding message with 'tool_calls'".

    The check is sequence-local: for each ModelResponse with tool_calls, only the
    immediately-following ModelRequest(s) count as responses. This avoids
    false-positives from reused tool_call_ids across turns.
    """
    cleaned: list[ModelMessage] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, ModelResponse):
            tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
            if tool_calls:
                pending_ids = {p.tool_call_id for p in tool_calls}
                responded_ids: set[str] = set()

                # Scan immediately-following ModelRequests for tool returns
                j = i + 1
                while j < len(messages) and isinstance(messages[j], ModelRequest):
                    req = messages[j]
                    non_return_parts = [
                        p for p in req.parts if not isinstance(p, ToolReturnPart)
                    ]
                    responded_ids |= {
                        p.tool_call_id
                        for p in req.parts
                        if isinstance(p, ToolReturnPart)
                    }
                    if non_return_parts:
                        # Mixed message — stop scanning but remember position
                        break
                    j += 1

                if pending_ids - responded_ids:
                    # Drop this ModelResponse and all pure-tool-return messages after it.
                    # If the message at j is a mixed request, strip its orphaned
                    # ToolReturnParts so it doesn't become an invalid role='tool' message.
                    i = j
                    if i < len(messages) and isinstance(messages[i], ModelRequest):
                        req = messages[i]
                        # Strip returns for ALL pending_ids — the entire ModelResponse
                        # was dropped, so every tool_call from it is now orphaned,
                        # including the ones that did get a response.
                        stripped = [
                            p for p in req.parts
                            if not (
                                isinstance(p, ToolReturnPart)
                                and p.tool_call_id in pending_ids
                            )
                        ]
                        if len(stripped) != len(req.parts):
                            if stripped:
                                messages[i] = dc_replace(req, parts=stripped)
                            else:
                                # Nothing left — skip this message too
                                i += 1
                    continue

        cleaned.append(msg)
        i += 1

    # Second pass: remove ToolReturnParts that have no matching ToolCallPart in
    # the immediately-preceding message (as tracked in the *output* list, since
    # earlier passes may have already dropped some messages).
    result: list[ModelMessage] = []
    for msg in cleaned:
        if isinstance(msg, ModelRequest):
            return_parts = [p for p in msg.parts if isinstance(p, ToolReturnPart)]
            if return_parts:
                prev = result[-1] if result else None
                if isinstance(prev, ModelResponse):
                    prev_call_ids = {
                        p.tool_call_id
                        for p in prev.parts
                        if isinstance(p, ToolCallPart)
                    }
                else:
                    prev_call_ids = set()

                orphaned_ids = {p.tool_call_id for p in return_parts} - prev_call_ids
                if orphaned_ids:
                    stripped = [
                        p
                        for p in msg.parts
                        if not (
                            isinstance(p, ToolReturnPart)
                            and p.tool_call_id in orphaned_ids
                        )
                    ]
                    if stripped:
                        result.append(dc_replace(msg, parts=stripped))
                    # if nothing left, skip the message entirely
                    continue
        result.append(msg)

    # Never return an empty list — pydantic-ai raises UserError if we do.
    return result if result else messages
