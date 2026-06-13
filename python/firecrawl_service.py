#!/usr/bin/env python3
"""Firecrawl proxy — keeps the API key off the browser. Port 8769."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIR = Path(__file__).resolve().parent
ENV_FILE = DIR.parent / ".env"
PORT = int(os.environ.get("FIRECRAWL_PORT", "8769"))
FIRECRAWL_URL = os.environ.get(
    "FIRECRAWL_API_URL", "https://api.firecrawl.dev/v2/search"
)


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k:
            out[k] = v
    return out


def _merge_firecrawl_env_from_file() -> None:
    parsed = _parse_env_file(ENV_FILE)
    key = (parsed.get("FIRECRAWL_API_KEY") or "").strip()
    if key:
        os.environ["FIRECRAWL_API_KEY"] = key
        print("firecrawl_service: applied FIRECRAWL_API_KEY from %s" % (ENV_FILE,), file=sys.stderr)


def cors(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def _extract_web_rows(api_json: dict) -> list[dict]:
    data = api_json.get("data")
    if isinstance(data, dict):
        web = data.get("web")
        if isinstance(web, list):
            return [x for x in web if isinstance(x, dict)]
        news = data.get("news")
        if isinstance(news, list):
            return [x for x in news if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _rows_to_markdown(rows: list[dict], max_chars: int = 14000) -> str:
    parts: list[str] = []
    used = 0
    for i, row in enumerate(rows, 1):
        title = str(row.get("title") or row.get("name") or "Untitled").strip()
        url = str(row.get("url") or "").strip()
        desc = str(row.get("description") or "").strip()
        md = str(row.get("markdown") or "").strip()
        summary = str(row.get("summary") or "").strip()
        chunk = f"### {i}. {title}\n"
        if url:
            chunk += f"URL: {url}\n"
        body = md or summary or desc
        if body:
            chunk += body + "\n"
        elif not md and not desc and not summary:
            chunk += "(no snippet)\n"
        if used + len(chunk) > max_chars:
            room = max_chars - used - 40
            if room > 120:
                parts.append(chunk[:room] + "\n…(truncated)")
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n".join(parts).strip()


def firecrawl_search(query: str, limit: int = 5) -> tuple[bool, str, str | None]:
    key = (os.environ.get("FIRECRAWL_API_KEY") or "").strip()
    if not key:
        return False, "", "no_key"
    q = (query or "").strip()
    if not q:
        return False, "", "empty_query"
    lim = max(1, min(int(limit) if limit else 5, 10))
    payload = json.dumps(
        {
            "query": q,
            "limit": lim,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        FIRECRAWL_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            detail = str(e)
        return False, "", detail
    except Exception as e:
        return False, "", str(e)
    try:
        api_json = json.loads(raw)
    except Exception:
        return False, "", "invalid_upstream_json"
    if not api_json.get("success", True) and not _extract_web_rows(api_json):
        err = api_json.get("error") or api_json.get("message") or "upstream_error"
        return False, "", str(err)
    rows = _extract_web_rows(api_json)
    if not rows:
        return True, "(No web results returned for this query.)", None
    return True, _rows_to_markdown(rows), None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        cors(self)
        self.end_headers()

    def _json(self, code: int, obj: dict) -> None:
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        cors(self)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        from urllib.parse import urlparse

        u = urlparse(self.path)
        if u.path in ("/", "/api/health"):
            key = (os.environ.get("FIRECRAWL_API_KEY") or "").strip()
            self._json(
                200,
                {
                    "ok": True,
                    "service": "mozg-firecrawl",
                    "configured": bool(len(key) > 8),
                },
            )
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        from urllib.parse import urlparse

        u = urlparse(self.path)
        if u.path != "/api/search":
            self._json(404, {"error": "not_found"})
            return
        ln = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(ln) if ln else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json(400, {"error": "invalid_json"})
            return
        query = (body.get("query") or "").strip()
        limit = body.get("limit")
        try:
            lim = int(limit) if limit is not None else 5
        except Exception:
            lim = 5
        ok, md, err = firecrawl_search(query, lim)
        if not ok:
            code = 401 if err == "no_key" else 400 if err in ("empty_query",) else 502
            self._json(code, {"ok": False, "error": err or "search_failed"})
            return
        self._json(200, {"ok": True, "markdown": md})


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE)
    except Exception:
        pass
    _merge_firecrawl_env_from_file()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"firecrawl_service listening on http://127.0.0.1:{PORT}", file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
