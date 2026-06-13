"""Find social profile URLs from a display name via social-analyzer."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DIR = Path(__file__).resolve().parent
_DEFAULT_SITES = "linkedin instagram github twitter facebook tiktok youtube reddit medium"

_DOMAIN_TO_ID: List[Tuple[str, str]] = [
    ("linkedin.com", "linkedin"),
    ("instagram.com", "instagram"),
    ("github.com", "github"),
    ("twitter.com", "twitter"),
    ("mobile.twitter.com", "twitter"),
    ("x.com", "twitter"),
    ("facebook.com", "facebook"),
    ("tiktok.com", "tiktok"),
    ("youtube.com", "youtube"),
    ("reddit.com", "reddit"),
    ("medium.com", "medium"),
]


def _sa_root() -> Path:
    env = (os.environ.get("SOCIAL_ANALYZER_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _DIR / "social-analyzer"


def _import_analyzer():
    root = _sa_root()
    if not (root / "app.py").is_file():
        raise FileNotFoundError(
            f"social-analyzer not found at {root} — "
            "git clone https://github.com/qeeqbox/social-analyzer.git"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app import SocialAnalyzer  # type: ignore

    return SocialAnalyzer


def _normalize_url(url: str) -> str:
    return url.strip().split("?")[0].rstrip("/")


def _classify_url(url: str) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    low = url.lower()
    for domain, pid in _DOMAIN_TO_ID:
        if domain in low:
            return pid
    return None


def _grab_url(url: str, found: Dict[str, str]) -> None:
    pid = _classify_url(url)
    if pid and pid not in found:
        found[pid] = _normalize_url(url)


def extract_profiles_from_sa_result(obj: Any) -> Dict[str, str]:
    found: Dict[str, str] = {}

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
            link = o.get("link") or o.get("url")
            if isinstance(link, str):
                _grab_url(link, found)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(obj)
    for m in re.finditer(r"https?://[^\s\"'<>]+", json.dumps(obj, default=str)):
        _grab_url(m.group(0), found)
    return found


def _username_guesses(display_name: str, max_variants: int) -> str:
    name = (display_name or "").strip()
    parts = [p for p in re.split(r"\s+", name) if p]
    if not parts:
        return ""
    first = re.sub(r"[^a-zA-Z0-9]", "", parts[0]).lower()
    last = re.sub(r"[^a-zA-Z0-9]", "", parts[-1]).lower() if len(parts) > 1 else ""
    compact = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
    seen: list[str] = []

    def add(s: str) -> None:
        s = (s or "").strip().lower()
        if len(s) >= 2 and s not in seen:
            seen.append(s)

    add(compact)
    if first and last:
        add(first + last)
        add(first + "-" + last)
        add(first + "_" + last)
    if first:
        add(first)
    if last and len(last) > 2:
        add(last)
    return ",".join(seen[: max(1, max_variants)])


def lookup_social_profiles(display_name: str) -> Dict[str, Any]:
    name = (display_name or "").strip()
    out: Dict[str, Any] = {"profiles": {}, "error": None, "raw": None}
    if len(name) < 2:
        out["error"] = "empty_name"
        return out

    n_sites = len((os.environ.get("SOCIAL_ANALYZER_WEBSITES") or _DEFAULT_SITES).split())
    max_var = int(os.environ.get("SOCIAL_ANALYZER_MAX_USERNAME_VARIANTS", "3" if n_sites > 4 else "5"))
    username = _username_guesses(name, max_var)
    if not username:
        out["error"] = "empty_name"
        return out

    websites = (os.environ.get("SOCIAL_ANALYZER_WEBSITES") or _DEFAULT_SITES).strip()
    SocialAnalyzer = _import_analyzer()

    try:
        sa = SocialAnalyzer(silent=True)
        ret = sa.run_as_object(
            username=username,
            websites=websites,
            mode="fast",
            output="json",
            silent=True,
            timeout=int(os.environ.get("SOCIAL_ANALYZER_TIMEOUT", "90")),
            filter="all",
            profiles="all",
            metadata=False,
        )
        out["raw"] = ret
        found = extract_profiles_from_sa_result(ret)
        out["profiles"] = found
        if not found:
            out["error"] = "no_profiles_detected"
    except Exception as e:
        out["error"] = str(e)[:500]
    return out
