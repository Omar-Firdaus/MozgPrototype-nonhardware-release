import asyncio
import base64
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import websockets

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

PEOPLE_PORT = 8767

_face_reco_on = None
_reco_lock = threading.Lock()
_reco_job = None  # tuple(job_id, face_bgr)
_reco_done = (0, None, None)  # (job_id, label, score)
_reco_worker_started = False


def _safe_crop_bgr(frame_bgr, x, y, fw, fh):
    h, w = frame_bgr.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x + fw)), min(h, int(y + fh))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1]


def _run_embedding_recognition(face_bgr):
    global _face_reco_on
    if _face_reco_on is False:
        return None, None
    try:
        from database import load_database, resolve_face
        from embedder import get_embedding

        if _face_reco_on is None:
            load_database()
            _face_reco_on = True
        emb = get_embedding(face_bgr)
        label, score, _ = resolve_face(
            emb, face_crop_bgr=face_bgr, save_unknown_crop=True
        )
        return label, score
    except Exception as exc:
        if _face_reco_on is None:
            print(f"[cv] face reco off: {exc}")
        _face_reco_on = False
        return None, None

WS_PORT = 8765
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0") or "0")
VERTICAL_FLIP = True
TARGET_FPS = 24
FACE_RECO_MIN_INTERVAL_S = float(os.environ.get("FACE_RECO_MIN_INTERVAL_S", "0.55") or "0.55")
DETECT_SCALE = 0.5
JPEG_MAX_WIDTH = 854
JPEG_QUALITY = 68

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _reco_worker_loop():
    global _reco_job, _reco_done
    while True:
        task = None
        with _reco_lock:
            if _reco_job is not None:
                task = _reco_job
                _reco_job = None
        if task is None:
            time.sleep(0.008)
            continue
        job_id, face_bgr = task
        label, score = _run_embedding_recognition(face_bgr)
        with _reco_lock:
            _reco_done = (job_id, label, score)


def _ensure_reco_worker():
    global _reco_worker_started
    with _reco_lock:
        if _reco_worker_started:
            return
        _reco_worker_started = True
    threading.Thread(target=_reco_worker_loop, daemon=True).start()


def _queue_reco_job(job_id, face_bgr):
    # Keep only the latest face crop; dropping stale work avoids backlog lag.
    global _reco_job
    with _reco_lock:
        _reco_job = (job_id, face_bgr.copy())


def _latest_reco_done():
    with _reco_lock:
        return _reco_done


def _open_camera_with_fallback():
    """Try preferred CAMERA_INDEX first, then common alternates."""
    candidates = [CAMERA_INDEX]
    for idx in (0, 1, 2, 3):
        if idx not in candidates:
            candidates.append(idx)

    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Reduce internal backlog so boxes track the latest frame (platform-dependent).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = cap.read()
        if ok and frame is not None:
            if idx != CAMERA_INDEX:
                print(
                    f"[cv] CAMERA_INDEX={CAMERA_INDEX} unavailable; using camera index {idx}"
                )
            return cap
        cap.release()

    print(
        f"[cv] No usable camera found. Tried indices: {', '.join(str(i) for i in candidates)}"
    )
    return None


def pick_best_face_full_frame(face_cascade, small_gray):
    """
    Return a single (x, y, w, h) in **full-resolution** frame coordinates, or None.

    Uses cascade level weights when available (detectMultiScale3); otherwise
    picks the largest detection as a proxy for the primary face.
    """
    inv = 1.0 / DETECT_SCALE
    faces = ()
    weights = None
    try:
        faces, _reject, weights = face_cascade.detectMultiScale3(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(int(40 * DETECT_SCALE), int(40 * DETECT_SCALE)),
            outputRejectLevels=True,
        )
    except Exception:
        faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(40 * DETECT_SCALE), int(40 * DETECT_SCALE)),
        )

    rects = np.asarray(faces, dtype=np.float64)
    if rects.size == 0:
        return None
    if rects.ndim == 1:
        rects = rects.reshape(1, -1)

    n = rects.shape[0]
    if weights is not None:
        warr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if warr.size == n:
            i = int(np.argmax(warr))
        else:
            areas = rects[:, 2] * rects[:, 3]
            i = int(np.argmax(areas))
    else:
        areas = rects[:, 2] * rects[:, 3]
        i = int(np.argmax(areas))

    x, y, fw, fh = rects[i]
    return int(x * inv), int(y * inv), int(fw * inv), int(fh * inv)


class _PeopleHandler(BaseHTTPRequestHandler):
    """GET /api/people; POST /api/people/rename; POST /api/people/merge — {keepIndex, removeIndex}"""

    def log_message(self, format, *args):
        return

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, code: int, obj: dict) -> None:
        """JSON response with CORS (browsers reject bare send_error from file:// or other origins)."""
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/people/dedupe-pending":
            try:
                from database import consume_dedupe_pending_for_client

                pending = consume_dedupe_pending_for_client()
                if not pending:
                    self._json(200, {"ok": True, "details": []})
                else:
                    self._json(200, pending)
            except Exception as exc:
                self.send_error(500, str(exc))
            return
        if path != "/api/people":
            self.send_error(404)
            return
        try:
            from database import api_list_people

            body = json.dumps({"people": api_list_people()}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(500, str(exc))

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            ln = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(ln).decode("utf-8")
            data = json.loads(raw) if raw else {}
        except Exception:
            self.send_error(400, "invalid JSON")
            return

        if path == "/api/people/rename":
            try:
                old = str(data.get("from", "")).strip()
                new = str(data.get("to", "")).strip()
                if not old or not new:
                    self.send_error(400, "JSON body needs {from, to}")
                    return
                from database import rename_unknown

                updated = rename_unknown(old, new)
                body = json.dumps({"ok": True, "updated": updated}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._cors()
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_error(500, str(exc))
            return

        if path == "/api/people/merge":
            try:
                ki = data.get("keepIndex")
                ri = data.get("removeIndex")
                if ki is None or ri is None:
                    self.send_error(400, "JSON body needs {keepIndex, removeIndex}")
                    return
                keep_index = int(ki)
                remove_index = int(ri)
                from database import merge_face_rows

                result = merge_face_rows(keep_index, remove_index)
                body = json.dumps({"ok": True, **result}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._cors()
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_error(500, str(exc))
            return

        if path == "/api/people/deduplicate":
            try:
                thr = data.get("threshold")
                t = float(thr) if thr is not None else None
                from database import dedupe_embedding_rows

                result = dedupe_embedding_rows(t)
                body = json.dumps({"ok": True, **result}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._cors()
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_error(500, str(exc))
            return

        if path == "/api/people/delete":
            try:
                raw_ix = data.get("indices")
                if not isinstance(raw_ix, list) or len(raw_ix) < 1:
                    self.send_error(400, "JSON body needs indices: [0, 1, ...]")
                    return
                ix_list = [int(x) for x in raw_ix]
                from database import delete_face_rows

                result = delete_face_rows(ix_list)
                body = json.dumps({"ok": True, **result}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._cors()
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_error(500, str(exc))
            return

        if path == "/api/people/social-lookup":
            try:
                ix = data.get("index")
                if ix is None:
                    self._json(
                        400,
                        {"ok": False, "error": "bad_request", "detail": "JSON body needs index"},
                    )
                    return
                index = int(ix)
                from database import (
                    _labels,
                    is_full_name_for_social,
                    load_database,
                    social_public_links,
                    social_set_profiles,
                )

                load_database()
                if index < 0 or index >= len(_labels):
                    self._json(
                        400,
                        {"ok": False, "error": "bad_request", "detail": "index out of range"},
                    )
                    return
                lab = _labels[index]
                if not is_full_name_for_social(lab):
                    self._json(
                        200,
                        {
                            "ok": False,
                            "error": "needs_full_name",
                            "detail": "Set a first and last name (not unknown_*) to look up social profiles.",
                        },
                    )
                    return
                try:
                    from social_analyzer_runner import lookup_social_profiles
                except Exception as imp_exc:
                    self._json(
                        200,
                        {
                            "ok": False,
                            "error": "import_failed",
                            "detail": str(imp_exc)[:400],
                        },
                    )
                    return

                try:
                    r = lookup_social_profiles(lab)
                except Exception as run_exc:
                    self._json(
                        200,
                        {
                            "ok": False,
                            "error": "lookup_failed",
                            "detail": str(run_exc)[:500],
                        },
                    )
                    return

                err = r.get("error")
                profiles = r.get("profiles") or {}
                if isinstance(profiles, dict) and profiles:
                    social_set_profiles(lab, profiles)
                links = social_public_links(lab)
                self._json(
                    200,
                    {
                        "ok": True,
                        "label": lab,
                        "profiles": profiles,
                        "socialLinks": links,
                        "error": err,
                    },
                )
            except Exception as exc:
                self._json(
                    200,
                    {"ok": False, "error": "server_exception", "detail": str(exc)[:500]},
                )
            return

        if path == "/api/people/apollo-search":
            try:
                ix = data.get("index")
                if ix is None:
                    self._json(
                        400,
                        {"ok": False, "error": "bad_request", "detail": "JSON body needs index"},
                    )
                    return
                index = int(ix)
                from database import _labels, is_full_name_for_social, load_database

                load_database()
                if index < 0 or index >= len(_labels):
                    self._json(
                        400,
                        {"ok": False, "error": "bad_request", "detail": "index out of range"},
                    )
                    return
                lab = _labels[index]
                if not is_full_name_for_social(lab):
                    self._json(
                        200,
                        {
                            "ok": False,
                            "error": "needs_full_name",
                            "detail": "Set a first and last name to search Apollo.",
                            "tiles": [],
                            "label": lab,
                        },
                    )
                    return

                from apollo_enrichment import apollo_people_search_tiles

                out = apollo_people_search_tiles(lab)
                if out.get("ok"):
                    self._json(
                        200,
                        {
                            "ok": True,
                            "label": lab,
                            "tiles": out.get("tiles") or [],
                        },
                    )
                else:
                    self._json(
                        200,
                        {
                            "ok": False,
                            "error": out.get("error"),
                            "detail": out.get("detail"),
                            "tiles": [],
                            "label": lab,
                        },
                    )
            except Exception as exc:
                self._json(
                    200,
                    {"ok": False, "error": "server_exception", "detail": str(exc)[:500], "tiles": []},
                )
            return

        self.send_error(404)


def _dedupe_poll_loop():
    from database import EMB_FILE, dedupe_embedding_rows

    poll = float(os.environ.get("CV_ENGINE_DEDUPE_POLL_SEC", "25") or "25")
    if poll <= 0:
        return
    seen_mtime = 0.0
    while True:
        time.sleep(poll)
        try:
            mtime = EMB_FILE.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime <= seen_mtime:
            continue
        try:
            r = dedupe_embedding_rows()
            mg = int(r.get("mergedGroups", 0))
            if mg:
                print(
                    f"[cv] Auto-dedupe: merged {mg} duplicate group(s), "
                    f"{r.get('rowsBefore')} → {r.get('rowsAfter')} rows (cosine ≥ {r.get('threshold')})"
                )
        except Exception as exc:
            print(f"[cv] Auto-dedupe skipped: {exc}")
        try:
            seen_mtime = EMB_FILE.stat().st_mtime
        except FileNotFoundError:
            seen_mtime = 0.0


def _start_people_http():
    try:
        from database import dedupe_embedding_rows, load_database

        load_database()
        flag = os.environ.get("CV_ENGINE_AUTO_DEDUPE", "").strip().lower()
        if flag in ("1", "true", "yes"):
            try:
                r = dedupe_embedding_rows()
                mg = int(r.get("mergedGroups", 0))
                if mg:
                    print(
                        f"[cv] Auto-dedupe: merged {mg} duplicate group(s), "
                        f"{r.get('rowsBefore')} → {r.get('rowsAfter')} rows (cosine ≥ {r.get('threshold')})"
                    )
            except Exception as exc:
                print(f"[cv] Auto-dedupe skipped: {exc}")
    except Exception as exc:
        print(f"[cv] People API: database init: {exc}")
    try:
        # threaded so slow Apollo/social calls don't block people list reads
        httpd = ThreadingHTTPServer(("127.0.0.1", PEOPLE_PORT), _PeopleHandler)
    except OSError as exc:
        print(f"[cv] People API skipped (port {PEOPLE_PORT}): {exc}")
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"  People API http://127.0.0.1:{PEOPLE_PORT}/api/people")
    poll_sec = float(os.environ.get("CV_ENGINE_DEDUPE_POLL_SEC", "25") or "25")
    if poll_sec > 0:
        threading.Thread(target=_dedupe_poll_loop, daemon=True).start()
        print(
            f"  Face dedupe watcher: every {poll_sec:.0f}s when embeddings change (set CV_ENGINE_DEDUPE_POLL_SEC=0 to disable)"
        )


def encode_jpeg_b64(frame_bgr):
    h0, w0 = frame_bgr.shape[:2]
    if w0 > JPEG_MAX_WIDTH:
        s = JPEG_MAX_WIDTH / w0
        small = cv2.resize(
            frame_bgr,
            (int(w0 * s), int(h0 * s)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = frame_bgr
    ok, buf = cv2.imencode(
        ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def stream_faces(websocket):
    cap = _open_camera_with_fallback()
    if cap is None:
        await asyncio.sleep(0.8)
        return
    _ensure_reco_worker()

    frame_delay = 1.0 / max(8, TARGET_FPS)
    reco_state = {
        "t": 0.0,
        "label": None,
        "score": None,
        "need_fresh_reco": True,
        "job_id": 0,
    }

    try:
        while True:
            t_loop = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            if VERTICAL_FLIP:
                frame = cv2.flip(frame, 0)

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
            best = pick_best_face_full_frame(face_cascade, small)
            face_list = []
            if best is None:
                reco_state["need_fresh_reco"] = True
                reco_state["label"] = None
                reco_state["score"] = None
            else:
                x, y, fw, fh = best
                face_obj = {"x": x, "y": y, "w": fw, "h": fh}
                crop = _safe_crop_bgr(frame, x, y, fw, fh)
                if crop is not None:
                    now = time.monotonic()
                    run_reco = reco_state["need_fresh_reco"] or (
                        now - reco_state["t"] >= FACE_RECO_MIN_INTERVAL_S
                    )
                    if run_reco:
                        reco_state["job_id"] += 1
                        _queue_reco_job(reco_state["job_id"], crop)
                        reco_state["t"] = now
                        reco_state["need_fresh_reco"] = False
                    done_id, label, score = _latest_reco_done()
                    if done_id == reco_state["job_id"] and label is not None:
                        reco_state["label"] = label
                        reco_state["score"] = score
                    label, score = reco_state["label"], reco_state["score"]
                    if label is not None:
                        face_obj["name"] = label
                        face_obj["score"] = round(float(score), 4)
                face_list.append(face_obj)

            jpeg_b64 = encode_jpeg_b64(frame)
            payload = {
                "faces": face_list,
                "frame": [w, h],
            }
            if jpeg_b64:
                payload["jpeg"] = jpeg_b64

            try:
                await websocket.send(json.dumps(payload))
            except websockets.exceptions.ConnectionClosed:
                break

            elapsed = time.perf_counter() - t_loop
            await asyncio.sleep(max(0.0, frame_delay - elapsed))
    finally:
        cap.release()


async def main():
    _start_people_http()
    # Port scans / HTTP probes on 8765 cause handshake failures; avoid spamming the console.
    for _name in ("websockets.server", "websockets.asyncio.server", "websockets.protocol"):
        logging.getLogger(_name).setLevel(logging.CRITICAL)
    # Default max message ~1 MiB; JPEG payloads need more headroom.
    async with websockets.serve(
        stream_faces,
        "127.0.0.1",
        WS_PORT,
        max_size=8 * 1024 * 1024,
    ):
        print(f"CV running on ws://127.0.0.1:{WS_PORT}")
        print("  (single camera: Python streams video + faces — close other apps using the camera)")
        await asyncio.Future()


asyncio.run(main())
