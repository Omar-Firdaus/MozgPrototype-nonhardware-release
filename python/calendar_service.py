#!/usr/bin/env python3
"""Google Calendar plugin — OAuth + events. Port 8768."""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
except ImportError:
    print(
        "Missing Google libraries. Install:\n"
        "  pip install google-api-python-client google-auth-oauthlib google-auth-httplib2",
        file=sys.stderr,
    )
    raise

DIR = Path(__file__).resolve().parent
ENV_FILE = DIR.parent / ".env"
TOKEN_FILE = DIR / "calendar_token.json"
PORT = int(os.environ.get("GOOGLE_CALENDAR_PORT", "8768"))
REDIRECT_URI = f"http://127.0.0.1:{PORT}/oauth/callback"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# PKCE: same Flow instance must finish the token exchange
_pending_oauth_lock = threading.Lock()
_pending_oauth_flows: dict[str, Flow] = {}


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


def _merge_google_calendar_env_from_file() -> None:
    parsed = _parse_env_file(ENV_FILE)
    merged = False
    for key in ("GOOGLE_CALENDAR_CLIENT_ID", "GOOGLE_CALENDAR_CLIENT_SECRET"):
        val = (parsed.get(key) or "").strip()
        if val:
            os.environ[key] = val
            merged = True
    if merged:
        print(
            "calendar_service: applied GOOGLE_CALENDAR_* from %s" % (ENV_FILE,),
            file=sys.stderr,
        )


def _client_config() -> dict | None:
    cid = os.environ.get("GOOGLE_CALENDAR_CLIENT_ID", "").strip()
    sec = os.environ.get("GOOGLE_CALENDAR_CLIENT_SECRET", "").strip()
    if not cid or not sec:
        return None
    return {
        "installed": {
            "client_id": cid,
            "client_secret": sec,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }
    }


def load_credentials() -> Credentials | None:
    if not TOKEN_FILE.is_file():
        return None
    try:
        return Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    except Exception:
        return None


def save_credentials(creds: Credentials) -> None:
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")


def get_calendar_service():
    creds = load_credentials()
    if not creds:
        return None, "not_connected"
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                save_credentials(creds)
            except Exception as e:
                return None, f"refresh_failed: {e}"
        else:
            return None, "invalid_credentials"
    try:
        return build("calendar", "v3", credentials=creds, cache_discovery=False), None
    except Exception as e:
        return None, str(e)


def cors(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


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

    def _html(self, code: int, body: str) -> None:
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        cors(self)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)

        if u.path in ("/", "/api/health"):
            self._json(200, {"ok": True, "service": "mozg-calendar"})
            return

        if u.path == "/api/status":
            cfg = _client_config()
            creds = load_credentials()
            connected = False
            if creds is not None:
                if getattr(creds, "valid", False):
                    connected = True
                elif getattr(creds, "refresh_token", None):
                    connected = True
            self._json(
                200,
                {"connected": connected, "has_client_config": bool(cfg)},
            )
            return

        if u.path == "/api/oauth/url":
            cfg = _client_config()
            if not cfg:
                self._json(
                    400,
                    {"error": "missing_env", "detail": "Set GOOGLE_CALENDAR_CLIENT_ID and GOOGLE_CALENDAR_CLIENT_SECRET"},
                )
                return
            try:
                flow = Flow.from_client_config(cfg, scopes=SCOPES, redirect_uri=REDIRECT_URI)
                authorization_url, state = flow.authorization_url(
                    access_type="offline",
                    include_granted_scopes="true",
                    prompt="consent",
                )
                with _pending_oauth_lock:
                    _pending_oauth_flows[state] = flow
                    # avoid unbounded growth if Connect is clicked many times
                    while len(_pending_oauth_flows) > 16:
                        try:
                            _pending_oauth_flows.pop(next(iter(_pending_oauth_flows)))
                        except StopIteration:
                            break
                self._json(200, {"authorization_url": authorization_url})
            except Exception as e:
                self._json(500, {"error": str(e)})
            return

        if u.path == "/oauth/callback":
            err = (q.get("error") or [None])[0]
            if err:
                self._html(
                    200,
                    f"<html><body><p>Authorization failed: {err}</p></body></html>",
                )
                return
            code = (q.get("code") or [None])[0]
            if not code:
                self._html(200, "<html><body><p>Missing code.</p></body></html>")
                return
            state = (q.get("state") or [None])[0]
            if not state:
                self._html(
                    200,
                    "<html><body><p>Missing OAuth state. Close this tab and click Connect again.</p></body></html>",
                )
                return
            with _pending_oauth_lock:
                flow = _pending_oauth_flows.pop(state, None)
            if not flow:
                self._html(
                    200,
                    "<html><body><p>OAuth session expired (restart calendar_service.py, then click Connect again).</p></body></html>",
                )
                return
            try:
                flow.fetch_token(code=code)
                save_credentials(flow.credentials)
                self._html(
                    200,
                    "<html><body><p>Google Calendar connected. You can close this window.</p></body></html>",
                )
            except Exception as e:
                self._html(200, f"<html><body><p>Error: {e}</p></body></html>")
            return

        if u.path == "/api/events":
            days_s = (q.get("days") or ["7"])[0]
            try:
                days = max(1, min(30, int(days_s)))
            except ValueError:
                days = 7
            svc, err = get_calendar_service()
            if not svc:
                self._json(401, {"error": err or "auth"})
                return
            now = datetime.now(timezone.utc)
            tmin = now.isoformat()
            tmax = (now + timedelta(days=days)).isoformat()
            try:
                ev = (
                    svc.events()
                    .list(
                        calendarId="primary",
                        timeMin=tmin,
                        timeMax=tmax,
                        singleEvents=True,
                        orderBy="startTime",
                        maxResults=50,
                    )
                    .execute()
                )
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            items = ev.get("items") or []
            out = []
            for it in items:
                s = it.get("start") or {}
                e = it.get("end") or {}
                start = s.get("dateTime") or s.get("date") or ""
                end = e.get("dateTime") or e.get("date") or ""
                out.append(
                    {
                        "id": it.get("id"),
                        "summary": it.get("summary") or "(no title)",
                        "start": start,
                        "end": end,
                    }
                )
            self._json(200, {"events": out})
            return

        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/events":
            self._json(404, {"error": "not_found"})
            return
        ln = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(ln) if ln else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json(400, {"error": "invalid_json"})
            return
        action = (body.get("action") or "create").strip().lower()
        svc, err = get_calendar_service()
        if not svc:
            self._json(401, {"error": err or "auth"})
            return
        if action == "delete":
            event_id = (body.get("id") or "").strip()
            if not event_id:
                self._json(400, {"error": "need_id"})
                return
            try:
                svc.events().delete(
                    calendarId="primary", eventId=event_id, sendUpdates="all"
                ).execute()
                self._json(200, {"ok": True})
            except Exception as e:
                self._json(500, {"error": str(e)})
            return
        if action == "add_attendees":
            event_id = (body.get("id") or "").strip()
            emails = body.get("emails")
            if not event_id or not isinstance(emails, list) or not emails:
                self._json(400, {"error": "need_id_and_emails"})
                return
            try:
                ev = (
                    svc.events()
                    .get(calendarId="primary", eventId=event_id)
                    .execute()
                )
                existing = list(ev.get("attendees") or [])
                seen = {
                    (a.get("email") or "").lower()
                    for a in existing
                    if a.get("email")
                }
                for em in emails:
                    em = str(em).strip()
                    if not em or "@" not in em:
                        continue
                    low = em.lower()
                    if low not in seen:
                        seen.add(low)
                        existing.append({"email": em})
                svc.events().patch(
                    calendarId="primary",
                    eventId=event_id,
                    body={"attendees": existing},
                    sendUpdates="all",
                ).execute()
                self._json(200, {"ok": True})
            except Exception as e:
                self._json(500, {"error": str(e)})
            return
        if action != "create":
            self._json(400, {"error": "unknown_action"})
            return
        summary = (body.get("summary") or "").strip()
        start = (body.get("start") or "").strip()
        end = (body.get("end") or "").strip()
        raw_emails = body.get("emails")
        emails = (
            [str(x).strip() for x in raw_emails]
            if isinstance(raw_emails, list)
            else []
        )
        emails = [e for e in emails if "@" in e]
        if not summary or not start or not end:
            self._json(400, {"error": "need_summary_start_end"})
            return
        event = {
            "summary": summary,
            "start": {"dateTime": start, "timeZone": "Etc/UTC"},
            "end": {"dateTime": end, "timeZone": "Etc/UTC"},
        }
        if emails:
            event["attendees"] = [{"email": e} for e in emails]
        try:
            created = (
                svc.events()
                .insert(
                    calendarId="primary",
                    body=event,
                    sendUpdates="all" if emails else "none",
                )
                .execute()
            )
            created_id = str(created.get("id") or "").strip()
            # Some calendars can drop attendees on insert depending on account/domain policy.
            # Force-apply attendees via patch so guest list is persisted when emails were supplied.
            patch_error = ""
            created_id_usable = (
                bool(created_id)
                and "<" not in created_id
                and ">" not in created_id
                and "event_id_from_create" not in created_id.lower()
            )
            if emails and created_id_usable:
                try:
                    ev = (
                        svc.events()
                        .get(calendarId="primary", eventId=created_id)
                        .execute()
                    )
                    existing = list(ev.get("attendees") or [])
                    seen = {
                        (a.get("email") or "").lower()
                        for a in existing
                        if a.get("email")
                    }
                    for em in emails:
                        low = em.lower()
                        if low not in seen:
                            seen.add(low)
                            existing.append({"email": em})
                    created = (
                        svc.events()
                        .patch(
                            calendarId="primary",
                            eventId=created_id,
                            body={"attendees": existing},
                            sendUpdates="all",
                        )
                        .execute()
                    )
                except Exception as e:
                    patch_error = str(e)
            elif emails and not created_id_usable:
                patch_error = "created_event_id_missing_or_invalid"
            self._json(
                200,
                {
                    "ok": True,
                    "id": created.get("id"),
                    "htmlLink": created.get("htmlLink"),
                    "attendee_count": len(created.get("attendees") or []),
                    "attendee_patch_error": patch_error,
                },
            )
        except Exception as e:
            self._json(500, {"error": str(e)})


def main() -> None:
    # Load .env from smart-glasses root (parent of python/)
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE)
    except Exception:
        pass
    _merge_google_calendar_env_from_file()
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"calendar_service listening on http://127.0.0.1:{PORT}", file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
