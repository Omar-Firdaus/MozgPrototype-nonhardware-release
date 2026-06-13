"""
Local face library — embeddings.npy + labels.json, cosine match, unknown enroll.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "face_data"
EMB_FILE = DATA_DIR / "embeddings.npy"
LAB_FILE = DATA_DIR / "labels.json"
CROPS_DIR = DATA_DIR / "unknown_crops"
SOCIAL_FILE = DATA_DIR / "social_profiles_by_label.json"
DEDUPE_PENDING_CLIENT = DATA_DIR / "dedupe_pending_client.json"

# cosine = dot product on L2-normalized vectors; tune via env if needed
SAME_PERSON_THRESHOLD = float(os.environ.get("FACE_SAME_PERSON_COSINE", "0.70"))
TEMPLATE_EMA = float(os.environ.get("FACE_TEMPLATE_EMA", "0.08"))
FACE_DEDUPE_COSINE = float(os.environ.get("FACE_DEDUPE_COSINE", "0.62"))

_embeddings: Optional[np.ndarray] = None  # (N, D), row-wise L2-normalized
_labels: list[str] = []
_unknown_counter: int = 0
_ema_last_disk_save: float = 0.0
_dedupe_lock = threading.Lock()


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return (mat / norms).astype(np.float32)


def load_database() -> None:
    global _embeddings, _labels, _unknown_counter
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    if EMB_FILE.exists():
        _embeddings = np.load(EMB_FILE).astype(np.float32)
        if _embeddings.ndim == 1:
            _embeddings = _embeddings.reshape(1, -1)
        if _embeddings.shape[0] == 0:
            d = int(_embeddings.shape[1]) if _embeddings.ndim == 2 else 0
            _embeddings = np.zeros((0, d), dtype=np.float32) if d else np.zeros((0, 0), dtype=np.float32)
        else:
            _embeddings = _l2_normalize_rows(_embeddings)
    else:
        _embeddings = np.zeros((0, 0), dtype=np.float32)

    if LAB_FILE.exists():
        with open(LAB_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "labels" in raw:
            m = raw["labels"]
            if isinstance(m, dict):
                keys = sorted(m.keys(), key=lambda k: int(k))
                _labels = [m[k] for k in keys]
            else:
                _labels = list(m)
            _unknown_counter = int(raw.get("unknown_counter", 0))
        elif isinstance(raw, dict):
            keys = sorted((k for k in raw.keys() if k.isdigit()), key=int)
            _labels = [raw[k] for k in keys]
            _unknown_counter = 0
        else:
            _labels = []
            _unknown_counter = 0
    else:
        _labels = []
        _unknown_counter = 0

    n = 0 if _embeddings is None else _embeddings.shape[0]
    if n != len(_labels):
        raise ValueError(f"embeddings rows ({n}) and labels ({len(_labels)}) mismatch")


def save_database() -> None:
    if _embeddings is None:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_FILE, _embeddings)
    payload = {
        "unknown_counter": _unknown_counter,
        "labels": {str(i): lab for i, lab in enumerate(_labels)},
    }
    with open(LAB_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Social links by label (linkedin, instagram, …)

_SOCIAL_META = frozenset({"updatedAt"})


def _normalize_social_url(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = v.strip().split("?")[0].rstrip("/")
    if not s.startswith("http"):
        return None
    return s


def _social_load() -> Dict[str, Any]:
    if not SOCIAL_FILE.is_file():
        return {"byLabel": {}}
    try:
        with open(SOCIAL_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and isinstance(raw.get("byLabel"), dict):
            return raw
    except Exception:
        pass
    return {"byLabel": {}}


def _social_save(data: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOCIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def social_get_for_label(label: str) -> Dict[str, Any]:
    return dict(_social_load().get("byLabel", {}).get(label, {}))


def social_public_links(label: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in social_get_for_label(label).items():
        if k in _SOCIAL_META:
            continue
        u = _normalize_social_url(v)
        if u:
            out[k] = u
    return out


def social_set_profiles(label: str, profiles: Optional[Dict[str, Any]]) -> None:
    """Merge non-empty http(s) URLs into the store (platform id → URL)."""
    if not profiles:
        return
    data = _social_load()
    m = data.setdefault("byLabel", {})
    cur = dict(m.get(label, {}))
    for k, v in profiles.items():
        if not k or k in _SOCIAL_META:
            continue
        u = _normalize_social_url(v)
        if u:
            cur[k] = u
    if any(k for k in cur if k not in _SOCIAL_META):
        cur["updatedAt"] = int(time.time())
        m[label] = cur
        _social_save(data)


def social_set_for_label(
    label: str, linkedin: Optional[str], instagram: Optional[str]
) -> None:
    """Backward-compatible: only linkedin + instagram."""
    d: Dict[str, Any] = {}
    if linkedin:
        d["linkedin"] = linkedin
    if instagram:
        d["instagram"] = instagram
    social_set_profiles(label, d)


def social_rename_label(old: str, new: str) -> None:
    if old == new:
        return
    data = _social_load()
    m = data.get("byLabel", {})
    if old in m:
        m[new] = m.pop(old)
        _social_save(data)


def social_merge_labels(keep: str, remove: str) -> None:
    if keep == remove:
        return
    data = _social_load()
    m = data.get("byLabel", {})
    a = dict(m.get(keep, {}))
    b = m.pop(remove, None) or {}
    for key, val in b.items():
        if key in _SOCIAL_META:
            continue
        if a.get(key):
            continue
        u = _normalize_social_url(val)
        if u:
            a[key] = u
    if any(k for k in a if k not in _SOCIAL_META):
        a["updatedAt"] = int(time.time())
        m[keep] = a
    _social_save(data)


def social_delete_labels(labels: List[str]) -> None:
    if not labels:
        return
    data = _social_load()
    m = data.get("byLabel", {})
    for lab in labels:
        m.pop(lab, None)
    _social_save(data)


def social_merge_cluster(merged_labels: List[str], canonical: str) -> None:
    """Fold social entries for merged face rows into the canonical label."""
    if not merged_labels or not canonical:
        return
    data = _social_load()
    m = data.get("byLabel", {})
    accumulated: Dict[str, str] = {}
    for lab in merged_labels:
        o = m.pop(lab, None)
        if not o:
            continue
        for key, val in o.items():
            if key in _SOCIAL_META:
                continue
            if key in accumulated:
                continue
            u = _normalize_social_url(val)
            if u:
                accumulated[key] = u
    ent = dict(m.get(canonical, {}))
    for key, val in accumulated.items():
        if not ent.get(key):
            ent[key] = val
    if any(k for k in ent if k not in _SOCIAL_META):
        ent["updatedAt"] = int(time.time())
        m[canonical] = ent
    _social_save(data)


def is_full_name_for_social(label: str) -> bool:
    s = (label or "").strip()
    if not s or _is_unknown_label(s):
        return False
    return len(s.split()) >= 2


def add_embedding(embedding: np.ndarray, label: str) -> int:
    """Append one L2-normalized embedding and label. Returns row index."""
    global _embeddings, _labels
    e = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    e = e / (np.linalg.norm(e) + 1e-12)
    empty = _embeddings is None or _embeddings.shape[0] == 0
    if empty:
        _embeddings = e.astype(np.float32)
    else:
        if e.shape[1] != _embeddings.shape[1]:
            raise ValueError(f"embedding dim {e.shape[1]} != db dim {_embeddings.shape[1]}")
        _embeddings = np.vstack([_embeddings, e])
    _labels.append(label)
    idx = len(_labels) - 1
    save_database()
    return idx


def update_label(index: int, new_name: str) -> None:
    if index < 0 or index >= len(_labels):
        raise IndexError("label index out of range")
    _labels[index] = new_name
    save_database()


def match_face(embedding: np.ndarray) -> Tuple[str, float]:
    """
    Best cosine similarity vs all stored embeddings (L2-normalized rows vs vector).
    Returns (label, score). Empty DB → ("", -1.0).
    """
    if _embeddings is None or _embeddings.shape[0] == 0:
        return "", -1.0
    v = np.asarray(embedding, dtype=np.float32).reshape(-1)
    v = v / (np.linalg.norm(v) + 1e-12)
    sims = _embeddings @ v
    i = int(np.argmax(sims))
    return _labels[i], float(sims[i])


def _is_unknown_label(lab: str) -> bool:
    return lab.startswith("unknown_")


def _pick_canonical_label(labs: List[str]) -> str:
    """When merging duplicate rows: prefer a named label over unknown_*."""
    named = [x for x in labs if not _is_unknown_label(x)]
    if named:
        return sorted(named)[0]
    return sorted(labs)[0]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        p = self._p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


def _components_uf(uf: _UnionFind, n: int) -> List[List[int]]:
    buckets: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        buckets.setdefault(r, []).append(i)
    out = [sorted(v) for v in buckets.values()]
    out.sort(key=min)
    return out


def _sync_unknown_counter_from_labels() -> None:
    global _unknown_counter
    m = 0
    for lab in _labels:
        if _is_unknown_label(lab):
            try:
                x = int(lab.split("_", 1)[1])
                m = max(m, x)
            except (IndexError, ValueError):
                pass
    _unknown_counter = m


def dedupe_embedding_rows(threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Find pairwise cosine-similar rows (≥ threshold), cluster transitively, merge each
    cluster into one row (mean embedding, canonical label). Renumbers crops.

    Default threshold (0.62) is lenient vs typical ArcFace dup pairs; tune FACE_DEDUPE_COSINE if needed.
    """
    with _dedupe_lock:
        return _dedupe_embedding_rows_unlocked(threshold)


def _dedupe_embedding_rows_unlocked(threshold: Optional[float] = None) -> Dict[str, Any]:
    global _embeddings, _labels
    load_database()
    thr = float(threshold) if threshold is not None else FACE_DEDUPE_COSINE
    n = 0 if _embeddings is None else _embeddings.shape[0]
    if n < 2:
        return {"mergedGroups": 0, "rowsBefore": n, "rowsAfter": n, "threshold": thr, "details": []}

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(_embeddings[i], _embeddings[j]))
            if sim >= thr:
                uf.union(i, j)

    comps = _components_uf(uf, n)
    multi = [c for c in comps if len(c) > 1]
    if not multi:
        return {"mergedGroups": 0, "rowsBefore": n, "rowsAfter": n, "threshold": thr, "details": []}

    new_rows: List[np.ndarray] = []
    new_labs: List[str] = []
    map_old_to_new: Dict[int, int] = {}
    details: List[Dict[str, Any]] = []
    new_idx = 0

    for comp in comps:
        if len(comp) == 1:
            j = comp[0]
            new_rows.append(_embeddings[j].copy())
            new_labs.append(_labels[j])
            map_old_to_new[j] = new_idx
            new_idx += 1
        else:
            mat = _embeddings[comp]
            merged = np.mean(mat, axis=0)
            merged = merged / (np.linalg.norm(merged) + 1e-12)
            new_rows.append(merged.astype(np.float32))
            labs_in = [_labels[j] for j in comp]
            canon = _pick_canonical_label(labs_in)
            new_labs.append(canon)
            g = _embeddings[comp]
            gram = g @ g.T
            pair_sims: List[float] = []
            for a in range(len(comp)):
                for b in range(a + 1, len(comp)):
                    pair_sims.append(float(gram[a, b]))
            details.append(
                {
                    "indices": comp,
                    "labels": labs_in,
                    "keptLabel": canon,
                    "minPairwiseCosine": min(pair_sims) if pair_sims else 1.0,
                }
            )
            for j in comp:
                map_old_to_new[j] = new_idx
            new_idx += 1

    _embeddings = np.vstack([r.reshape(1, -1) for r in new_rows]).astype(np.float32)
    _labels = new_labs
    _sync_unknown_counter_from_labels()

    staging: Dict[int, bytes] = {}
    for new_i in range(len(new_labs)):
        olds = [o for o, nn in map_old_to_new.items() if nn == new_i]
        rep = min(olds)
        p = CROPS_DIR / f"{rep}.jpg"
        if p.is_file():
            staging[new_i] = p.read_bytes()

    for p in CROPS_DIR.glob("*.jpg"):
        try:
            p.unlink()
        except OSError:
            pass
    for ni, data in staging.items():
        (CROPS_DIR / f"{ni}.jpg").write_bytes(data)

    save_database()
    for d in details:
        labs_in = d.get("labels") or []
        canon = d.get("keptLabel")
        if labs_in and canon and len(labs_in) > 1:
            social_merge_cluster(labs_in, str(canon))
    if details:
        try:
            DEDUPE_PENDING_CLIENT.write_text(
                json.dumps({"ok": True, "details": details, "threshold": thr}),
                encoding="utf-8",
            )
        except OSError:
            pass
    return {
        "mergedGroups": len(multi),
        "rowsBefore": n,
        "rowsAfter": len(new_labs),
        "threshold": thr,
        "details": details,
    }


def consume_dedupe_pending_for_client() -> Dict[str, Any]:
    """Read and remove pending dedupe details so the UI can merge localStorage transcripts."""
    if not DEDUPE_PENDING_CLIENT.is_file():
        return {}
    try:
        raw = DEDUPE_PENDING_CLIENT.read_text(encoding="utf-8")
        DEDUPE_PENDING_CLIENT.unlink()
        return json.loads(raw)
    except Exception:
        try:
            if DEDUPE_PENDING_CLIENT.is_file():
                DEDUPE_PENDING_CLIENT.unlink()
        except OSError:
            pass
        return {}


def api_list_people() -> List[Dict[str, Any]]:
    """Rows for the Electron People page (same process as cv_engine)."""
    out: List[Dict[str, Any]] = []
    n = 0 if _embeddings is None else _embeddings.shape[0]
    for i in range(n):
        lab = _labels[i] if i < len(_labels) else ""
        needs = _is_unknown_label(lab)
        thumb_b64: Optional[str] = None
        p = CROPS_DIR / f"{i}.jpg"
        if p.is_file():
            thumb_b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        soc = social_public_links(lab)
        out.append(
            {
                "index": i,
                "label": lab,
                "needsName": needs,
                "thumbnail": thumb_b64,
                "socialLinks": soc,
            }
        )
    return out


def delete_face_rows(indices: List[int]) -> Dict[str, Any]:
    """
    Remove face rows by slot index. Remaps embeddings, labels, and unknown_crops/*.jpg.
    """
    global _embeddings, _labels
    load_database()
    n = 0 if _embeddings is None else _embeddings.shape[0]
    if n == 0:
        return {"deleted": 0, "removedLabels": [], "rows": 0}

    kill = {int(i) for i in indices if 0 <= int(i) < n}
    if not kill:
        return {"deleted": 0, "removedLabels": [], "rows": n}

    removed_labels = [_labels[i] for i in sorted(kill)]

    new_rows: List[np.ndarray] = []
    new_labs: List[str] = []
    map_old_to_new: Dict[int, int] = {}
    ni = 0
    for i in range(n):
        if i in kill:
            continue
        new_rows.append(_embeddings[i].reshape(1, -1).copy())
        new_labs.append(_labels[i])
        map_old_to_new[i] = ni
        ni += 1

    if new_rows:
        _embeddings = np.vstack(new_rows).astype(np.float32)
    else:
        d = int(_embeddings.shape[1]) if _embeddings.ndim == 2 and _embeddings.shape[0] else 0
        _embeddings = np.zeros((0, d), dtype=np.float32) if d else np.zeros((0, 0), dtype=np.float32)
    _labels = new_labs
    _sync_unknown_counter_from_labels()

    staging: Dict[int, bytes] = {}
    for old_i, new_i in map_old_to_new.items():
        p = CROPS_DIR / f"{old_i}.jpg"
        if p.is_file():
            staging[new_i] = p.read_bytes()

    for p in CROPS_DIR.glob("*.jpg"):
        try:
            p.unlink()
        except OSError:
            pass
    for j, data in staging.items():
        (CROPS_DIR / f"{j}.jpg").write_bytes(data)

    save_database()
    social_delete_labels(removed_labels)
    return {
        "deleted": len(kill),
        "removedLabels": removed_labels,
        "rows": len(_labels),
    }


def merge_face_rows(keep_index: int, remove_index: int) -> Dict[str, Any]:
    """
    Merge two DB rows that refer to the same person (duplicate enrollments).
    Embeddings are averaged (L2-normalized). The row at remove_index is dropped;
    label and slot kept are those at keep_index. Thumbnail files are renumbered.
    """
    global _embeddings, _labels
    load_database()
    n = 0 if _embeddings is None else _embeddings.shape[0]
    if n == 0:
        raise ValueError("empty database")
    if keep_index == remove_index:
        raise ValueError("keep_index and remove_index must differ")
    if keep_index < 0 or remove_index < 0 or keep_index >= n or remove_index >= n:
        raise ValueError("index out of range")

    a = _embeddings[keep_index].astype(np.float32)
    b = _embeddings[remove_index].astype(np.float32)
    merged = a + b
    merged = merged / (np.linalg.norm(merged) + 1e-12)

    keep_label = _labels[keep_index]
    remove_label = _labels[remove_index]

    new_rows: List[np.ndarray] = []
    new_labels: List[str] = []
    for i in range(n):
        if i == remove_index:
            continue
        if i == keep_index:
            new_rows.append(merged.astype(np.float32))
        else:
            new_rows.append(_embeddings[i].copy())
        new_labels.append(_labels[i])

    _embeddings = np.vstack(new_rows) if new_rows else np.zeros((0, merged.shape[0]), dtype=np.float32)
    _labels = new_labels

    # Remap crop files to contiguous indices after row removal
    old_n = n
    mapping: Dict[int, int] = {}
    ni = 0
    for i in range(old_n):
        if i == remove_index:
            continue
        mapping[i] = ni
        ni += 1

    staging: Dict[int, bytes] = {}
    for old_i, new_i in mapping.items():
        p = CROPS_DIR / f"{old_i}.jpg"
        if p.is_file():
            staging[new_i] = p.read_bytes()

    for p in CROPS_DIR.glob("*.jpg"):
        try:
            p.unlink()
        except OSError:
            pass
    for new_i, data in staging.items():
        (CROPS_DIR / f"{new_i}.jpg").write_bytes(data)

    save_database()
    social_merge_labels(keep_label, remove_label)
    return {
        "keepLabel": keep_label,
        "removeLabel": remove_label,
        "rows": int(_embeddings.shape[0]),
    }


def rename_unknown(old_label: str, new_name: str) -> int:
    """Rename every row whose label equals old_label. Returns number of rows updated."""
    n = 0
    for i, lab in enumerate(_labels):
        if lab == old_label:
            _labels[i] = new_name
            n += 1
    if n:
        social_rename_label(old_label, new_name)
        save_database()
    return n


def _ema_update_row(index: int, new_vec: np.ndarray) -> None:
    """Blend matched template toward latest observation (same row, no new slot)."""
    global _embeddings
    if _embeddings is None or index < 0 or index >= _embeddings.shape[0]:
        return
    a = float(np.clip(TEMPLATE_EMA, 0.0, 0.5))
    if a <= 0:
        return
    row = _embeddings[index] * (1.0 - a) + new_vec * a
    _embeddings[index] = (row / (np.linalg.norm(row) + 1e-12)).astype(np.float32)


def resolve_face(
    embedding: np.ndarray,
    face_crop_bgr: Optional[np.ndarray] = None,
    save_unknown_crop: bool = True,
) -> Tuple[str, float, bool]:
    """
    If best cosine vs any stored row >= SAME_PERSON_THRESHOLD → same identity (no duplicate row).
    Otherwise enroll unknown_{n}. Template EMA on match reduces embedding drift.

    Returns (label, score, enrolled_new).
    """
    global _unknown_counter

    v = np.asarray(embedding, dtype=np.float32).reshape(-1)
    v = v / (np.linalg.norm(v) + 1e-12)

    if _embeddings is None or _embeddings.shape[0] == 0:
        _unknown_counter += 1
        lab = f"unknown_{_unknown_counter}"
        idx = add_embedding(v, lab)
        if save_unknown_crop and face_crop_bgr is not None and face_crop_bgr.size:
            p = CROPS_DIR / f"{idx}.jpg"
            cv2.imwrite(str(p), face_crop_bgr)
        return lab, 0.0, True

    sims = _embeddings @ v
    i_best = int(np.argmax(sims))
    best = float(sims[i_best])

    if best >= SAME_PERSON_THRESHOLD:
        _ema_update_row(i_best, v)
        global _ema_last_disk_save
        now = time.time()
        if now - _ema_last_disk_save >= 4.0:
            save_database()
            _ema_last_disk_save = now
        return _labels[i_best], best, False

    _unknown_counter += 1
    lab = f"unknown_{_unknown_counter}"
    idx = add_embedding(v, lab)
    if save_unknown_crop and face_crop_bgr is not None and face_crop_bgr.size:
        p = CROPS_DIR / f"{idx}.jpg"
        cv2.imwrite(str(p), face_crop_bgr)
    return lab, best, True
