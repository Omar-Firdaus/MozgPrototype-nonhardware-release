"""Apollo contact search — turns a name into profile tiles for the HUD."""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

_CONTACT_SEARCH_URL = "https://api.apollo.io/api/v1/contacts/search"

# urllib gets blocked sometimes; browser UA helps
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _clean_key(raw: str) -> str:
    s = (raw or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _headers(api_key: str) -> Dict[str, str]:
    ua = (os.environ.get("APOLLO_USER_AGENT") or "").strip() or _DEFAULT_UA
    return {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "Accept": "application/json",
        "X-Api-Key": api_key,
        "User-Agent": ua,
    }


def _ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _short(s: str, n: int = 480) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _tile(out: List[Dict[str, str]], key: str, label: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        v = "Yes" if value else "No"
    elif isinstance(value, (int, float)):
        v = str(value)
    elif isinstance(value, str):
        v = value.strip()
    else:
        return
    if v:
        out.append({"key": key, "label": label, "value": _short(v)})


def _split_name(display_name: str) -> Optional[Dict[str, str]]:
    parts = [p for p in (display_name or "").strip().split() if p]
    if len(parts) < 2:
        return None
    return {"first_name": parts[0], "last_name": parts[-1]}


def _api_key() -> str:
    return _clean_key(
        os.environ.get("APOLLO_CONTACT_SEARCH_API_KEY")
        or os.environ.get("APOLLO_API_KEY")
        or ""
    )


def _contact_tiles(contact: Dict[str, Any], total: int) -> List[Dict[str, str]]:
    tiles: List[Dict[str, str]] = []

    def section(title: str) -> None:
        tiles.append({"key": "__section__", "label": title, "value": ""})

    section("Profile")
    _tile(tiles, "result_count", "Matches", total)
    _tile(tiles, "name", "Name", contact.get("name"))
    _tile(tiles, "headline", "Headline", contact.get("headline"))

    city = (contact.get("city") or "").strip() if isinstance(contact.get("city"), str) else ""
    state = (contact.get("state") or "").strip() if isinstance(contact.get("state"), str) else ""
    country = (contact.get("country") or "").strip() if isinstance(contact.get("country"), str) else ""
    location = ", ".join(x for x in (city, state, country) if x)
    if not location:
        location = (contact.get("formatted_address") or contact.get("present_raw_address") or "").strip()
    _tile(tiles, "location", "Location", location)
    _tile(tiles, "time_zone", "Time zone", contact.get("time_zone"))

    section("Employment")
    _tile(tiles, "title", "Current title", contact.get("title"))
    _tile(tiles, "organization_name", "Current company", contact.get("organization_name"))

    roles = contact.get("contact_roles")
    if isinstance(roles, list) and roles:
        role0 = roles[0] if isinstance(roles[0], dict) else {}
        _tile(tiles, "function", "Function", role0.get("function") or role0.get("name"))
        _tile(tiles, "seniority", "Seniority", role0.get("seniority"))
        _tile(tiles, "subdepartment", "Department", role0.get("subdepartment"))

    jce = contact.get("contact_job_change_event")
    if isinstance(jce, dict):
        _tile(tiles, "job_event_type", "Job change", jce.get("event_type"))
        _tile(tiles, "job_event_at", "Job change date", jce.get("event_date"))

    _tile(tiles, "created_at", "Contact created", contact.get("created_at"))
    _tile(tiles, "updated_at", "Contact updated", contact.get("updated_at"))

    section("Contact")
    _tile(tiles, "email", "Primary email", contact.get("email"))
    _tile(tiles, "email_status", "Email status", contact.get("email_status"))

    emails = contact.get("contact_emails")
    if isinstance(emails, list) and emails:
        for i, em in enumerate(emails[:3]):
            if not isinstance(em, dict):
                continue
            _tile(tiles, f"email_{i}", f"Email ({i + 1})", em.get("email"))
            _tile(
                tiles,
                f"email_{i}_status",
                f"Email ({i + 1}) status",
                em.get("email_true_status") or em.get("email_status"),
            )

    phones = contact.get("phone_numbers")
    if isinstance(phones, list) and phones:
        for i, ph in enumerate(phones[:3]):
            if not isinstance(ph, dict):
                continue
            _tile(
                tiles,
                f"phone_{i}",
                f"Phone ({i + 1})",
                ph.get("raw_number") or ph.get("sanitized_number"),
            )
            _tile(tiles, f"phone_{i}_type", f"Phone ({i + 1}) type", ph.get("type"))

    _tile(tiles, "linkedin_url", "LinkedIn", contact.get("linkedin_url"))
    _tile(tiles, "twitter_url", "X / Twitter", contact.get("twitter_url"))
    _tile(tiles, "facebook_url", "Facebook", contact.get("facebook_url"))

    section("Company")
    org = contact.get("organization")
    if not isinstance(org, dict):
        org = {}
    _tile(tiles, "org_name", "Name", org.get("name") or contact.get("organization_name"))
    _tile(tiles, "org_domain", "Domain", org.get("primary_domain"))
    _tile(tiles, "org_website", "Website", org.get("website_url"))
    _tile(tiles, "org_phone", "Phone", org.get("primary_phone") or org.get("phone"))
    _tile(tiles, "org_founded", "Founded", org.get("founded_year"))
    _tile(tiles, "org_linkedin", "LinkedIn", org.get("linkedin_url"))
    _tile(tiles, "org_twitter", "X / Twitter", org.get("twitter_url"))
    _tile(tiles, "org_facebook", "Facebook", org.get("facebook_url"))
    return tiles


def apollo_people_search_tiles(display_name: str) -> Dict[str, Any]:
    name = (display_name or "").strip()
    if not _split_name(name):
        return {"ok": False, "error": "needs_full_name", "detail": "Use a first and last name."}
    key = _api_key()
    if not key:
        return {"ok": False, "error": "missing_contact_search_key"}

    req = urllib.request.Request(
        _CONTACT_SEARCH_URL,
        data=json.dumps({"q_keywords": name, "page": 1, "per_page": 5}).encode("utf-8"),
        headers=_headers(key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45, context=_ssl_ctx()) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:1200]
        if e.code in (401, 403):
            return {
                "ok": False,
                "error": "contact_search_not_allowed",
                "detail": f"{e.code}: {err_body}",
            }
        return {"ok": False, "error": "http_error", "detail": f"{e.code}: {err_body}"}
    except Exception as exc:
        return {"ok": False, "error": "request_failed", "detail": str(exc)[:500]}

    contacts = raw.get("contacts") if isinstance(raw.get("contacts"), list) else []
    if not contacts:
        return {"ok": False, "error": "no_match", "detail": "No contacts found for this name.", "raw": raw}

    top = contacts[0] if isinstance(contacts[0], dict) else {}
    return {
        "ok": True,
        "tiles": _contact_tiles(top, total=len(contacts)),
        "person": top,
        "raw": raw,
    }


try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass
