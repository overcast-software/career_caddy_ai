#!/usr/bin/env python3
import logfire


def scrubbing_callback(m: logfire.ScrubMatch):
    if (
        m.path == ("attributes", "tool_response", "data", "data")
        and m.pattern_match.group(0) == "JWT"
    ):
        return m.value

    if m.path == ("attributes", "mcp.session.id"):
        return m.value

    if (
        m.path == ("attributes", "tool_arguments", "url")
        and m.pattern_match.group(0) == "auth"
    ):
        return m.value

    if (
        m.path == ("attributes", "tool_response", "url")
        and m.pattern_match.group(0) == "auth"
    ):
        return m.value

    if (
        m.path == ("attributes", "tool_response", "content")
        and m.pattern_match.group(0) == "Cookie"
    ):
        return m.value
