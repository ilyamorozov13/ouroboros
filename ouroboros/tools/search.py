"""Web search tool — Tavily primary, OpenAI fallback."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry


def _search_tavily(query: str) -> str:
    """Search via Tavily REST API (no external dependencies)."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return ""  # signal: key not available

    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        answer = data.get("answer", "")
        results = data.get("results", [])
        sources = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:300]} for r in results]
        return json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False, indent=2)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return json.dumps({"error": f"Tavily HTTP {e.code}: {body}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Tavily error: {repr(e)}"}, ensure_ascii=False)


def _search_openai(query: str) -> str:
    """Search via OpenAI Responses API (fallback)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "OPENAI_API_KEY not set; web_search unavailable."})
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=os.environ.get("OUROBOROS_WEBSEARCH_MODEL", "gpt-4o"),
            tools=[{"type": "web_search_preview"}],
            tool_choice="auto",
            input=query,
        )
        d = resp.model_dump()
        text = ""
        for item in d.get("output", []) or []:
            if item.get("type") == "message":
                for block in item.get("content", []) or []:
                    if block.get("type") in ("output_text", "text"):
                        text += block.get("text", "")
        return json.dumps({"answer": text or "(no answer)"}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def _web_search(ctx: ToolContext, query: str) -> str:
    # Try Tavily first
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        result = _search_tavily(query)
        # If result is non-empty and not an error, return it
        if result and '"error"' not in result:
            return result
        # If Tavily returned an error, still return it (don't silently fall through)
        if result and '"error"' in result:
            return result

    # Fall back to OpenAI
    return _search_openai(query)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": "Search the web via OpenAI Responses API. Returns JSON with answer + sources.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
        }, _web_search),
    ]
