"""
Verificación automática de la Fase 2 (sin intervención del usuario).

1. Cruce local (0 tokens): compara los productos dichos, el inventario visual que devolvió Spark
   y la transcripción para detectar anomalías:
   - UNMATCHED_VISUAL_OBJECT: objeto mostrado a propósito que no corresponde a ningún producto dicho.
   - UNTRANSCRIBED_SPEECH: hay voz en el audio fuera de los turnos transcritos.
   - LOW_ASR_DENSITY: turno largo con muy pocas palabras (posibles palabras perdidas).
   - PRODUCT_WITHOUT_VISUAL: producto dicho sin ningún objeto visto en ese momento.
2. Segunda pasada dirigida (solo si hay anomalías): re-transcribe únicamente esos tramos con el
   vocabulario de lo que se ve como pistas y hace UNA llamada a Spark con los clips de esos tramos.
3. Aplica automáticamente las decisiones con confianza suficiente y deja un informe en el estado.
"""
import base64
import hashlib
import io
import json
import subprocess
import shutil
import unicodedata
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.core.config import (
    MUSE_MISSING_PRICE_POLICY,
    MUSE_PHASE2_VERIFY_MAX_WINDOWS,
    MUSE_PHASE2_VERIFY_MIN_CONFIDENCE,
    MUSE_SPARK_MODEL,
    MUSE_VOICE_LANGUAGES,
    MUSE_VOICE_MODEL,
    STORAGE_DIR,
)
from backend.core import prompts
from backend.core.muse_api import chat_json, transcribe_wav
from backend.core.muse_client import (
    NOT_MENTIONED,
    SALES_KEYWORDS_BIAS,
    build_video_proxy,
    get_cached_result,
    locate_source_video,
    set_cached_result,
)
from backend.core.token_tracker import TokenTracker

VERIFIER_PROMPT_VERSION = "verify_v2"
VISUAL_MATCH_PAD_SEC = 1.5
LOW_DENSITY_WORDS_PER_SEC = 1.2
LOW_DENSITY_MIN_TURN_SEC = 2.0
CHECK_PAD_SEC = 1.0
CLIP_FPS = 2

ANOMALY_PRIORITY = {
    "UNMATCHED_VISUAL_OBJECT": 0,
    "UNTRANSCRIBED_SPEECH": 1,
    "LOW_ASR_DENSITY": 2,
    "PRODUCT_WITHOUT_VISUAL": 3,
}

VERIFIER_SYSTEM_PROMPT = """You are the Phase 2 quality verifier of a pipeline that turns sales videos into a catalog.
A first pass extracted products by combining the automatic speech transcript (ASR) and the video. An automatic cross-check found anomalies in certain segments. The seller may speak Spanish or English.
For each segment (check) you receive: the anomalies, the original ASR text, a re-transcription of the segment made with vocabulary hints, the products already detected in that video and a clip of the video for that segment.

Give one or more decisions per check:
- ADD_PRODUCT: the seller offers an object that is NOT in the product list. It is an offer if they say a price or clearly present it for sale (hold it or point at it while talking about selling). If no price is heard, price = null.
- UPDATE_PRODUCT: an existing product has the wrong name or price according to the evidence; target_product_id says which one.
- NO_CHANGE: what was detected is correct and the anomaly is a false alarm (for example, the seller speaks slowly).
- DISMISS: the object is not offered for sale (it passes in the background, is part of the surroundings or the seller uses it for something else).

Rules:
- Do not duplicate products: an object already in the list under another name is UPDATE_PRODUCT or NO_CHANGE, never ADD_PRODUCT.
- item_name: specific commercial name in English of what is seen (type, color, visible brand).
- spoken_quote: what is heard about that object, literal and untranslated; empty string if nothing is said.
- start_sec / end_sec: seconds of the full video in which the object is shown (the clip starts at window.start_sec).
- confidence: between 0 and 1 based on the evidence you hear and see. evidence: briefly explain, in English, what is heard and what is seen.
- target_product_id: empty string except for UPDATE_PRODUCT and NO_CHANGE."""
prompts.register(
    "p2_verify", VERIFIER_SYSTEM_PROMPT, phase=2, model="Muse Spark", group="advanced",
    title="Audio ↔ video verifier",
    description="Reviews doubtful segments (objects shown without a price, untranscribed speech) and decides whether to add, fix or keep products.",
)

VERIFIER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "check_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["ADD_PRODUCT", "UPDATE_PRODUCT", "NO_CHANGE", "DISMISS"],
                    },
                    "target_product_id": {"type": "string"},
                    "item_name": {"type": "string"},
                    "spoken_quote": {"type": "string"},
                    "price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "currency": {
                        "type": "string",
                        "enum": ["USD", "PEN", "EUR", "MXN", "COP", "ARS", "CLP", "BOB", "OTHER"],
                    },
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "check_id",
                    "action",
                    "target_product_id",
                    "item_name",
                    "spoken_quote",
                    "price",
                    "currency",
                    "start_sec",
                    "end_sec",
                    "confidence",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


# =============================================================================
# Utilidades
# =============================================================================
def _fold(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(ch) != "Mn"
    ).strip()


def name_similarity(a: str, b: str) -> float:
    """Similitud entre nombres: máximo entre parecido de caracteres y palabras compartidas."""
    fa, fb = _fold(a), _fold(b)
    if not fa or not fb:
        return 0.0
    ratio = SequenceMatcher(None, fa, fb).ratio()
    wa = {w for w in fa.split() if len(w) > 2}
    wb = {w for w in fb.split() if len(w) > 2}
    overlap = len(wa & wb) / max(1, min(len(wa), len(wb))) if wa and wb else 0.0
    return max(ratio, overlap)


def _overlaps(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _read_wav(wav_path: Path) -> Tuple[np.ndarray, int, int, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16), sr, channels, width


def _cut_wav_bytes(wav_path: Path, start_sec: float, end_sec: float) -> bytes:
    samples, sr, channels, width = _read_wav(wav_path)
    segment = samples[int(start_sec * sr) * channels : int(end_sec * sr) * channels]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(sr)
        out.writeframes(segment.tobytes())
    return buf.getvalue()


def find_untranscribed_speech(wav_path: Path, turns: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Tramos de >= 0.5 s con energía de voz que no caen dentro de ningún turno del ASR."""
    samples, sr, channels, _ = _read_wav(wav_path)
    signal = samples.astype(np.float32)[::channels]
    win = max(1, int(sr * 0.1))
    if len(signal) < win * 5:
        return []
    rms = np.array([np.sqrt(np.mean(signal[i : i + win] ** 2)) for i in range(0, len(signal) - win, win)])
    threshold = max(300.0, float(np.percentile(rms, 30)) * 2.5)
    covered = np.zeros(len(rms), dtype=bool)
    for t in turns:
        covered[int(t["start_sec"] * 10) : int(t["end_sec"] * 10) + 1] = True
    loud = (rms > threshold) & ~covered

    spans: List[Tuple[float, float]] = []
    start: Optional[int] = None
    for i, flag in enumerate(np.append(loud, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 5:
                spans.append((round(start / 10, 2), round(i / 10, 2)))
            start = None
    return spans


def _cut_video_clip(proxy_path: Path, start_sec: float, end_sec: float) -> Optional[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    clip_path = proxy_path.parent / f"{proxy_path.stem}_clip_{start_sec:.1f}-{end_sec:.1f}.mp4"
    if clip_path.exists():
        return clip_path
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.2f}", "-to", f"{end_sec:.2f}", "-i", str(proxy_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-c:a", "aac", "-b:a", "48k", "-ac", "1",
        str(clip_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return clip_path if result.returncode == 0 and clip_path.exists() else None


def _candidate_frames(video_meta: Dict[str, Any], start_sec: float, end_sec: float) -> List[Dict[str, Any]]:
    retained = video_meta.get("retained_frames", [])
    matching = [f for f in retained if start_sec <= float(f.get("timestamp_sec", 0.0)) <= end_sec]
    if not matching and retained:
        mid = (start_sec + end_sec) / 2.0
        matching = sorted(retained, key=lambda f: abs(float(f.get("timestamp_sec", 0.0)) - mid))[:3]
    return [
        {
            "frame_id": f["frame_id"],
            "timestamp_sec": f["timestamp_sec"],
            "url": f["url"],
            "laplacian_score": f["laplacian_score"],
        }
        for f in matching[:3]
    ]


# =============================================================================
# 1. Cruce local de evidencias (0 tokens)
# =============================================================================
def detect_anomalies(
    products: List[Dict[str, Any]],
    visual_inventory: List[Dict[str, Any]],
    transcriptions: List[Dict[str, Any]],
    videos_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    matched_inventory: set = set()
    videos_with_inventory = {inv["video_id"] for inv in visual_inventory}

    for prod in products:
        spoken = prod["spoken_timestamps"]
        window = (spoken["start_sec"] - VISUAL_MATCH_PAD_SEC, spoken["end_sec"] + VISUAL_MATCH_PAD_SEC)
        candidates = [
            idx
            for idx, inv in enumerate(visual_inventory)
            if inv["video_id"] == prod["video_id"] and _overlaps((inv["start_sec"], inv["end_sec"]), window)
        ]
        if candidates:
            scored = [(name_similarity(visual_inventory[i]["object_name"], prod["detected_name"]), i) for i in candidates]
            best_score, best_idx = max(scored)
            matched_inventory.add(best_idx)
            matched_inventory.update(i for score, i in scored if score >= 0.5)
        elif prod["video_id"] in videos_with_inventory:
            anomalies.append(
                {
                    "type": "PRODUCT_WITHOUT_VISUAL",
                    "video_id": prod["video_id"],
                    "start_sec": spoken["start_sec"],
                    "end_sec": spoken["end_sec"],
                    "detail": f"{prod['product_id']} ('{prod['detected_name']}') does not match any object seen.",
                    "related_product_id": prod["product_id"],
                }
            )

    for idx, inv in enumerate(visual_inventory):
        if idx in matched_inventory:
            continue
        anomalies.append(
            {
                "type": "UNMATCHED_VISUAL_OBJECT",
                "video_id": inv["video_id"],
                "start_sec": float(inv["start_sec"]),
                "end_sec": float(inv["end_sec"]),
                "detail": (
                    f"'{inv['object_name']}' is shown ({inv.get('showcase_evidence', '')}) "
                    f"but no spoken offer was detected for that object."
                ),
                "related_product_id": "",
            }
        )

    for trans in transcriptions:
        vid_id = trans["video_id"]
        for turn in trans.get("turns", []):
            duration = turn["end_sec"] - turn["start_sec"]
            words = len(turn["text"].split())
            if duration >= LOW_DENSITY_MIN_TURN_SEC and words / max(duration, 0.1) < LOW_DENSITY_WORDS_PER_SEC:
                anomalies.append(
                    {
                        "type": "LOW_ASR_DENSITY",
                        "video_id": vid_id,
                        "start_sec": turn["start_sec"],
                        "end_sec": turn["end_sec"],
                        "detail": (
                            f"{duration:.1f}s turn with only {words} words "
                            f"({words / max(duration, 0.1):.2f} words/s): '{turn['text']}'."
                        ),
                        "related_product_id": "",
                    }
                )
        rel_audio = videos_by_id.get(vid_id, {}).get("audio", {}).get("relative_path")
        if rel_audio and (STORAGE_DIR / rel_audio).exists():
            for start, end in find_untranscribed_speech(STORAGE_DIR / rel_audio, trans.get("turns", [])):
                anomalies.append(
                    {
                        "type": "UNTRANSCRIBED_SPEECH",
                        "video_id": vid_id,
                        "start_sec": start,
                        "end_sec": end,
                        "detail": f"There is speech between {start}s and {end}s that is not in any transcribed turn.",
                        "related_product_id": "",
                    }
                )

    for n, anomaly in enumerate(anomalies, start=1):
        anomaly["anomaly_id"] = f"anom_{n}"
    return anomalies


def group_anomalies(
    anomalies: List[Dict[str, Any]], videos_by_id: Dict[str, Dict[str, Any]], transcriptions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Une anomalías del mismo video que se solapan en una ventana a verificar."""
    durations = {
        t["video_id"]: float(
            videos_by_id.get(t["video_id"], {}).get("quality_assessment", {}).get("duration_sec")
            or t.get("audio_duration_sec")
            or 60.0
        )
        for t in transcriptions
    }
    ordered = sorted(anomalies, key=lambda a: (a["video_id"], a["start_sec"]))
    groups: List[Dict[str, Any]] = []
    for anomaly in ordered:
        vid = anomaly["video_id"]
        start = max(0.0, anomaly["start_sec"] - CHECK_PAD_SEC)
        end = min(durations.get(vid, 60.0), anomaly["end_sec"] + CHECK_PAD_SEC)
        if groups and groups[-1]["video_id"] == vid and start <= groups[-1]["end_sec"]:
            groups[-1]["end_sec"] = max(groups[-1]["end_sec"], end)
            groups[-1]["anomalies"].append(anomaly)
        else:
            groups.append({"video_id": vid, "start_sec": start, "end_sec": end, "anomalies": [anomaly]})

    groups.sort(key=lambda g: min(ANOMALY_PRIORITY[a["type"]] for a in g["anomalies"]))
    groups = groups[:MUSE_PHASE2_VERIFY_MAX_WINDOWS]
    for n, group in enumerate(sorted(groups, key=lambda g: (g["video_id"], g["start_sec"])), start=1):
        group["check_id"] = f"chk_{n}"
        group["start_sec"] = round(group["start_sec"], 2)
        group["end_sec"] = round(group["end_sec"], 2)
    return sorted(groups, key=lambda g: g["check_id"])


# =============================================================================
# 2. Segunda pasada dirigida
# =============================================================================
def _retranscribe_window(
    tracker: TokenTracker,
    group: Dict[str, Any],
    wav_path: Path,
    keywords: List[str],
) -> str:
    segment = _cut_wav_bytes(wav_path, group["start_sec"], group["end_sec"])
    cache_key = hashlib.md5(
        segment + f"|{MUSE_VOICE_MODEL}|PUSH_TO_TALK|{','.join(keywords)}|{','.join(MUSE_VOICE_LANGUAGES)}".encode("utf-8")
    ).hexdigest()
    cached = get_cached_result("voice_retranscribe", cache_key)
    window_label = f"{group['video_id']} {group['start_sec']}-{group['end_sec']}s"
    if cached is not None:
        tracker.record_event(
            phase=2,
            model="Muse Voice Transcribe",
            operation=f"Verifier: MD5 cache HIT for the re-transcription of segment {window_label}.",
            status="CACHE_HIT_MD5",
            details={"check_id": group["check_id"]},
        )
        return cached["transcript"]

    response = transcribe_wav(
        segment,
        model=MUSE_VOICE_MODEL,
        mode="PUSH_TO_TALK",
        language_bias=MUSE_VOICE_LANGUAGES,
        keywords=keywords,
    )
    transcript = str(response.get("transcript", "")).strip()
    set_cached_result("voice_retranscribe", cache_key, {"transcript": transcript})
    tracker.record_event(
        phase=2,
        model="Muse Voice Transcribe",
        operation=(
            f"Verifier: re-transcription of segment {window_label} "
            f"({group['end_sec'] - group['start_sec']:.1f}s of audio) with {len(keywords)} vocabulary hints."
        ),
        status="API_CALL",
        details={"check_id": group["check_id"], "transcript": transcript},
    )
    return transcript


def _keywords_for_video(vid_id: str, visual_inventory: List[Dict[str, Any]], products: List[Dict[str, Any]]) -> List[str]:
    words = list(SALES_KEYWORDS_BIAS)
    names = [inv["object_name"] for inv in visual_inventory if inv["video_id"] == vid_id]
    names += [p["detected_name"] for p in products if p["video_id"] == vid_id]
    for name in names:
        for word in name.split()[:3]:
            clean = word.strip(",.()!¡¿?").lower()
            if len(clean) > 3 and clean not in words:
                words.append(clean)
    return words[:30]


def _new_product_from_decision(
    decision: Dict[str, Any],
    video_meta: Dict[str, Any],
    group: Dict[str, Any],
) -> Dict[str, Any]:
    start_s = round(float(decision.get("start_sec") or group["start_sec"]), 2)
    end_s = round(float(decision.get("end_sec") or group["end_sec"]), 2)
    if end_s <= start_s:
        start_s, end_s = group["start_sec"], group["end_sec"]
    duration = float(video_meta.get("quality_assessment", {}).get("duration_sec") or end_s)
    expanded_start = round(max(0.0, start_s - 1.5), 2)
    expanded_end = round(min(duration, end_s + 1.5), 2)
    price = decision.get("price")
    quote = decision.get("spoken_quote", "")

    return {
        "product_id": "",
        "video_id": group["video_id"],
        "video_filename": video_meta.get("filename", group["video_id"]),
        "mention_type": "EXPLICIT" if quote else "VISUAL_ONLY",
        "detected_name": decision["item_name"],
        "spoken_item_name": "",
        "visual_item_name": decision["item_name"],
        "name_source": "VERIFIER",
        "raw_spoken_phrase": quote,
        "interpreted_phrase": quote,
        "price": float(price) if price is not None else None,
        "price_status": "SPOKEN" if price is not None else "CONSULTAR_PRECIO",
        "original_price_mentioned": None,
        "price_corrected_live": False,
        "currency": decision.get("currency", "USD"),
        "condition": NOT_MENTIONED,
        "features": [],
        "spoken_timestamps": {"start_sec": start_s, "end_sec": end_s, "timing_source": "VERIFIER"},
        "expanded_visual_window": {
            "start_sec": expanded_start,
            "end_sec": expanded_end,
            "desync_margin_sec": 1.5,
        },
        "candidate_frames_preview": _candidate_frames(video_meta, expanded_start, expanded_end),
        "user_verified": False,
        "added_by": "VERIFIER",
        "verification_confidence": decision.get("confidence"),
        "verification_evidence": decision.get("evidence", ""),
    }


def _apply_decisions(
    decisions: List[Dict[str, Any]],
    products: List[Dict[str, Any]],
    groups_by_id: Dict[str, Dict[str, Any]],
    videos_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    products = [dict(p) for p in products]
    by_id = {p["product_id"]: p for p in products}
    report: List[Dict[str, Any]] = []

    for decision in decisions:
        action = decision.get("action")
        confidence = float(decision.get("confidence") or 0.0)
        group = groups_by_id.get(decision.get("check_id", ""))
        applied, note = False, ""

        if group is None:
            note = "unknown check_id"
        elif confidence < MUSE_PHASE2_VERIFY_MIN_CONFIDENCE and action in ("ADD_PRODUCT", "UPDATE_PRODUCT"):
            note = f"confidence {confidence:.2f} below the minimum {MUSE_PHASE2_VERIFY_MIN_CONFIDENCE:.2f}: not applied"
        elif action == "ADD_PRODUCT":
            window = (float(decision.get("start_sec", 0)), float(decision.get("end_sec", 0)))
            duplicate = next(
                (
                    p
                    for p in products
                    if p["video_id"] == group["video_id"]
                    and _overlaps(
                        window,
                        (p["spoken_timestamps"]["start_sec"] - 1.5, p["spoken_timestamps"]["end_sec"] + 1.5),
                    )
                    and name_similarity(p["detected_name"], decision.get("item_name", "")) >= 0.6
                ),
                None,
            )
            if duplicate is not None:
                note = f"already exists as {duplicate['product_id']} ('{duplicate['detected_name']}')"
            elif decision.get("price") is None and MUSE_MISSING_PRICE_POLICY == "EXCLUDE":
                note = "no spoken price: excluded by MUSE_MISSING_PRICE_POLICY=EXCLUDE"
            else:
                products.append(_new_product_from_decision(decision, videos_by_id.get(group["video_id"], {}), group))
                applied = True
                note = "product added" + (" with 'Ask for price'" if decision.get("price") is None else "")
        elif action == "UPDATE_PRODUCT":
            target = by_id.get(decision.get("target_product_id", ""))
            if target is None:
                note = "target_product_id does not exist"
            else:
                if decision.get("item_name"):
                    target["name_before_verification"] = target["detected_name"]
                    target["detected_name"] = decision["item_name"]
                    target["visual_item_name"] = decision["item_name"]
                    target["name_source"] = "VERIFIER"
                if decision.get("price") is not None and float(decision["price"]) != target.get("price"):
                    target["price_before_verification"] = target.get("price")
                    target["price"] = float(decision["price"])
                    target["price_status"] = "SPOKEN"
                target["verification_evidence"] = decision.get("evidence", "")
                target["verification_confidence"] = confidence
                applied = True
                note = f"{target['product_id']} updated"
        else:
            note = "no changes" if action == "NO_CHANGE" else "discarded: not a product for sale"

        report.append({**decision, "applied": applied, "note": note})
    return products, report


def _renumber(products: List[Dict[str, Any]], video_order: Dict[str, int]) -> List[Dict[str, Any]]:
    products = sorted(
        products, key=lambda p: (video_order.get(p["video_id"], 99), p["spoken_timestamps"]["start_sec"])
    )
    for n, prod in enumerate(products, start=1):
        prod["product_id"] = f"prod_{n}"
    return products


# =============================================================================
# Orquestador
# =============================================================================
def run_phase2_verification(
    tracker: TokenTracker,
    products: List[Dict[str, Any]],
    visual_inventory: List[Dict[str, Any]],
    transcriptions: List[Dict[str, Any]],
    videos_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    for prod in products:
        prod.setdefault("price_status", "SPOKEN")

    anomalies = detect_anomalies(products, visual_inventory, transcriptions, videos_by_id)
    groups = group_anomalies(anomalies, videos_by_id, transcriptions)
    report: Dict[str, Any] = {
        "visual_inventory": visual_inventory,
        "anomalies": anomalies,
        "checks": [],
        "decisions": [],
        "status": "NO_ANOMALIES",
        "min_confidence": MUSE_PHASE2_VERIFY_MIN_CONFIDENCE,
        "missing_price_policy": MUSE_MISSING_PRICE_POLICY,
    }

    if not groups:
        tracker.record_event(
            phase=2,
            model="Phase 2 verifier",
            operation=(
                f"Audio/video cross-check without anomalies: {len(products)} products and "
                f"{len(visual_inventory)} shown objects match. No second pass needed."
            ),
            status="ZERO_TOKEN_LOCAL",
        )
        return products, report

    tracker.record_event(
        phase=2,
        model="Phase 2 verifier",
        operation=(
            f"Audio/video cross-check: {len(anomalies)} anomaly(ies) in {len(groups)} segment(s) "
            f"({', '.join(sorted({a['type'] for a in anomalies}))}). Launching the targeted second pass."
        ),
        status="ZERO_TOKEN_LOCAL",
        details={"anomalies": anomalies},
    )

    user_content: List[Dict[str, Any]] = []
    clip_signature: List[str] = []
    for group in groups:
        vid_id = group["video_id"]
        video_meta = videos_by_id.get(vid_id, {})
        trans = next(t for t in transcriptions if t["video_id"] == vid_id)
        wav_path = STORAGE_DIR / video_meta["audio"]["relative_path"]

        retranscript = _retranscribe_window(
            tracker, group, wav_path, _keywords_for_video(vid_id, visual_inventory, products)
        )
        check_payload = {
            "check_id": group["check_id"],
            "video_id": vid_id,
            "window": {"start_sec": group["start_sec"], "end_sec": group["end_sec"]},
            "anomalies": [{"type": a["type"], "detail": a["detail"]} for a in group["anomalies"]],
            "asr_turns_in_window": [
                t
                for t in trans.get("turns", [])
                if _overlaps((t["start_sec"], t["end_sec"]), (group["start_sec"], group["end_sec"]))
            ],
            "retranscription_of_window": retranscript,
            "existing_products_in_video": [
                {
                    "product_id": p["product_id"],
                    "name": p["detected_name"],
                    "price": p.get("price"),
                    "start_sec": p["spoken_timestamps"]["start_sec"],
                    "end_sec": p["spoken_timestamps"]["end_sec"],
                    "quote": p.get("raw_spoken_phrase", ""),
                }
                for p in products
                if p["video_id"] == vid_id
            ],
        }
        group["retranscription"] = retranscript
        report["checks"].append({k: v for k, v in check_payload.items() if k != "existing_products_in_video"})
        user_content.append({"type": "text", "text": json.dumps(check_payload, ensure_ascii=False)})

        source = locate_source_video(video_meta)
        proxy = build_video_proxy(source) if source else None
        clip = _cut_video_clip(proxy, group["start_sec"], group["end_sec"]) if proxy else None
        if clip:
            encoded = base64.b64encode(clip.read_bytes()).decode("ascii")
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"Clip for {group['check_id']} ({CLIP_FPS} fps; starts at second "
                        f"{group['start_sec']} of the video):"
                    ),
                }
            )
            user_content.append(
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}", "fps": CLIP_FPS}}
            )
            clip_signature.append(hashlib.md5(clip.read_bytes()).hexdigest())

    cache_key = hashlib.md5(
        (
            f"{MUSE_SPARK_MODEL}|{VERIFIER_PROMPT_VERSION}|{','.join(clip_signature)}|"
            + json.dumps([c for c in user_content if c["type"] == "text"], ensure_ascii=False)
            + prompts.signature("p2_verify")
        ).encode("utf-8")
    ).hexdigest()
    cached = get_cached_result("phase2_verifier", cache_key)
    if cached is not None:
        decisions, usage = cached["decisions"], cached["usage"]
        tracker.record_event(
            phase=2,
            model="Muse Spark (Verifier)",
            operation=f"MD5 cache HIT for the verifier ({len(decisions)} decisions) without calling the API.",
            tokens_saved=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            status="CACHE_HIT_MD5",
        )
    else:
        parsed, usage, _ = chat_json(
            model=MUSE_SPARK_MODEL,
            messages=[
                {"role": "system", "content": prompts.get("p2_verify")},
                {"role": "user", "content": user_content},
            ],
            schema_name="Phase2VerifierDecisions",
            schema=VERIFIER_SCHEMA,
            reasoning_effort="low",
        )
        decisions = parsed.get("decisions", [])
        set_cached_result("phase2_verifier", cache_key, {"decisions": decisions, "usage": usage})
        tracker.record_event(
            phase=2,
            model="Muse Spark (Verifier)",
            operation=(
                f"{MUSE_SPARK_MODEL}: 1 call for {len(groups)} segment(s) with video clip: "
                f"{len(decisions)} decisions (reasoning 'low': {usage['reasoning_tokens']} tokens)."
            ),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            status="API_CALL",
        )

    groups_by_id = {g["check_id"]: g for g in groups}
    updated, decision_report = _apply_decisions(decisions, products, groups_by_id, videos_by_id)
    video_order = {t["video_id"]: i for i, t in enumerate(transcriptions)}
    updated = _renumber(updated, video_order)

    applied = [d for d in decision_report if d["applied"]]
    report["decisions"] = decision_report
    report["status"] = "VERIFIED_WITH_CHANGES" if applied else "VERIFIED_NO_CHANGES"
    tracker.record_event(
        phase=2,
        model="Phase 2 verifier",
        operation=(
            f"Decisions applied automatically: {len(applied)} of {len(decision_report)} "
            f"({', '.join(d['action'] + ': ' + d['item_name'] for d in applied) or 'none'}). "
            f"Final products: {len(updated)}."
        ),
        status="ZERO_TOKEN_LOCAL",
    )
    return updated, report
