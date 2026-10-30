import base64
import difflib
import hashlib
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.core.config import (
    ESTIMATED_VISION_TOKENS_PER_FRAME,
    META_MUSE_API_KEY,
    MUSE_SPARK_MODEL,
    MUSE_SPARK_PHASE2_VIDEO,
    MUSE_SPARK_PHASE2_VIDEO_FPS,
    MUSE_SPARK_REASONING_PHASE2,
    MUSE_VOICE_LANGUAGES,
    MUSE_VOICE_MODEL,
    STORAGE_DIR,
)
from backend.core import prompts
from backend.core.muse_api import chat_json, transcribe_wav
from backend.core.token_tracker import TokenTracker

CACHE_DIR = STORAGE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(namespace: str, key_md5: str) -> Path:
    return CACHE_DIR / f"{namespace}_{key_md5}.json"


def get_cached_result(namespace: str, key_md5: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(namespace, key_md5)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def set_cached_result(namespace: str, key_md5: str, payload: Dict[str, Any]) -> None:
    path = _cache_path(namespace, key_md5)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")



SPARK_PROMPT_VERSION = "phase2_v6"
SALES_KEYWORDS_BIAS = ["vendo", "vender", "precio", "cuesta", "dólares", "soles"]
DEICTIC_PLACEHOLDER_NAME = "Objeto por identificar en video (Mención Deíctica)"

SPARK_SYSTEM_PROMPT = """Eres Muse Spark dentro de un pipeline que convierte videos de venta en un catálogo e-commerce.
Recibes transcripciones automáticas (ASR) de 1 a 3 videos, divididas en turnos con su tiempo. En cada video un vendedor recorre un espacio mostrando objetos y diciendo su precio.
La transcripción tiene errores fonéticos y a veces se come palabras (una palabra que suena parecido en lugar del verbo o del nombre del objeto, o el nombre del objeto omitido antes del precio): interpreta la intención real.
Si además recibes el video, úsalo para ver qué objeto muestra el vendedor en el momento de cada turno y así corregir o completar los nombres que el ASR entendió mal u omitió. Los tiempos de los turnos del ASR son la referencia temporal.

Reglas:
1. Un producto es cada objeto distinto que el vendedor ofrece con un precio en el audio. No inventes productos que no se ofrezcan en lo hablado, aunque aparezcan otros objetos en el video.
2. quote: copia LITERAL, carácter por carácter y con los errores del ASR, del fragmento del turno que habla de ese producto: desde la primera palabra que lo introduce ("este", "esto", el nombre del objeto...) hasta el precio final o la última característica. No parafrasees. Los fragmentos de productos distintos no se solapan y van en el orden en que se dicen.
3. mention_type: EXPLICIT si el vendedor nombra qué es el objeto ("este juguete", "estos auriculares"); DEICTIC si solo lo señala sin nombrarlo ("esto", "esto de acá", "este de acá").
4. spoken_item_name: el sustantivo que dijo, corregido solo si el ASR lo deformó y el contexto lo deja claro; cadena vacía si es DEICTIC.
5. price: el precio final con el que se queda el vendedor; interpreta números dichos en palabras. Si corrige el precio en vivo ("a 10... mejor a 8"), original_price es el primero que dijo; si no hay corrección, original_price es null.
6. currency: según lo dicho (dólares -> USD, soles -> PEN, euros -> EUR, pesos según contexto); si no menciona moneda usa USD.
7. condition: el estado que mencione ("casi nuevo", "tiene un detalle aquí") o cadena vacía. features: características que diga ("inalámbrico", "talla M"); lista vacía si no dice ninguna. No inventes datos.
8. interpreted_phrase: la frase del producto corregida y legible, completando las palabras que el ASR omitió si el video lo deja claro.
9. visual_item_name: nombre comercial específico del objeto que se ve en el video durante ese turno (tipo, color, marca visible; por ejemplo "Taza de cerámica blanca con asa"). Cadena vacía si no recibiste video o no se distingue.
10. name_source: AUDIO si el nombre sale tal cual del audio; AUDIO_CORRECTED_BY_VIDEO si el video corrigió una palabra mal transcrita; VIDEO_ONLY si en el audio no se nombró (deíctico u omitido) y el nombre sale solo de lo que se ve; UNKNOWN si no se puede saber.
11. discarded_chatter: muletillas, saludos o comentarios sin intención de venta, con el motivo.
12. visual_inventory: si recibiste video, lista TODOS los objetos que el vendedor muestra a propósito (lo sostiene en la mano, lo señala, lo acerca o lo enfoca en primer plano), se haya dicho o no algo sobre ellos, con el intervalo aproximado en segundos del video en que se muestra. No incluyas objetos que solo pasan por el fondo. Lista vacía si no recibiste video."""
prompts.register(
    "p2_extract", SPARK_SYSTEM_PROMPT, phase=2, model="Muse Spark", group="creative",
    title="Extractor de productos",
    description="Lee la transcripción y el video y decide qué productos se ofrecen, su nombre, precio, moneda, estado y características.",
    hint="Ej.: 'Si el vendedor no dice moneda, asume PEN.' o 'Incluye la talla en el nombre cuando la diga.'",
)

SPARK_PRODUCTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "turn_id": {"type": "integer"},
                    "quote": {"type": "string"},
                    "interpreted_phrase": {"type": "string"},
                    "mention_type": {"type": "string", "enum": ["EXPLICIT", "DEICTIC"]},
                    "spoken_item_name": {"type": "string"},
                    "visual_item_name": {"type": "string"},
                    "name_source": {
                        "type": "string",
                        "enum": ["AUDIO", "AUDIO_CORRECTED_BY_VIDEO", "VIDEO_ONLY", "UNKNOWN"],
                    },
                    "price": {"type": "number"},
                    "original_price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "currency": {
                        "type": "string",
                        "enum": ["USD", "PEN", "EUR", "MXN", "COP", "ARS", "CLP", "BOB", "OTHER"],
                    },
                    "condition": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "video_id",
                    "turn_id",
                    "quote",
                    "interpreted_phrase",
                    "mention_type",
                    "spoken_item_name",
                    "visual_item_name",
                    "name_source",
                    "price",
                    "original_price",
                    "currency",
                    "condition",
                    "features",
                ],
                "additionalProperties": False,
            },
        },
        "discarded_chatter": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["video_id", "text", "reason"],
                "additionalProperties": False,
            },
        },
        "visual_inventory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "object_name": {"type": "string"},
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "showcase_evidence": {"type": "string"},
                },
                "required": ["video_id", "object_name", "start_sec", "end_sec", "showcase_evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["products", "discarded_chatter", "visual_inventory"],
    "additionalProperties": False,
}


def locate_source_video(video_meta: Dict[str, Any]) -> Optional[Path]:
    """Encuentra el video original subido en la Fase 1 (storage/<sesión>/videos/<video_id>_*)."""
    rel_audio = video_meta.get("audio", {}).get("relative_path", "")
    if not rel_audio:
        return None
    session_dir = STORAGE_DIR / Path(rel_audio).parts[0]
    matches = sorted((session_dir / "videos").glob(f"{video_meta.get('video_id')}_*"))
    return matches[0] if matches else None


def build_video_proxy(video_path: Path, short_side: int = 480, fps: int = 2) -> Optional[Path]:
    """Copia liviana del video (lado corto 480 px, 2 fps, audio mono) para enviarla a Spark."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    proxy_dir = video_path.parent.parent / "video_proxies"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = proxy_dir / f"{video_path.stem}_{short_side}p.mp4"
    if proxy_path.exists():
        return proxy_path

    scale = f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path),
        "-vf", scale, "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-c:a", "aac", "-b:a", "48k", "-ac", "1",
        str(proxy_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not proxy_path.exists():
        return None
    return proxy_path


def _fold_for_match(text: str) -> str:
    """Minúsculas y sin tildes, conservando la longitud para mapear posiciones de caracteres."""
    return "".join(unicodedata.normalize("NFD", ch)[0].lower() for ch in text)


def locate_quote_span(turn_text: str, quote: str, start_from: int = 0) -> Optional[Tuple[int, int]]:
    """Ubica la cita de Spark dentro del texto del turno (exacta o aproximada)."""
    haystack = _fold_for_match(turn_text)
    needle = _fold_for_match(quote).strip(" .,;:!?¿¡\"'")
    if not needle:
        return None

    idx = haystack.find(needle, start_from)
    if idx >= 0:
        return idx, idx + len(needle)

    matcher = difflib.SequenceMatcher(None, haystack[start_from:], needle, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 3]
    if not blocks or sum(b.size for b in blocks) < len(needle) * 0.6:
        return None
    end = start_from + blocks[-1].a + blocks[-1].size
    start = max(start_from + blocks[0].a, end - int(len(needle) * 1.3))
    return start, end


class MuseClient:
    """
    Cliente unificado para Meta Muse API (Muse Voice Transcribe, Muse Spark, Muse Glimmer,
    SAM 3.1, Muse Image, Muse Code) con Caché MD5 en disco y auditoría en TokenTracker.
    """

    def __init__(self, tracker: TokenTracker) -> None:
        self.tracker = tracker
        self.api_key = META_MUSE_API_KEY


    # =========================================================================
    # FASE 2: Muse Voice Transcribe + Muse Spark (Meta Model API real)
    # =========================================================================
    def transcribe_video_audio(
        self,
        video_id: str,
        filename: str,
        wav_path: Path,
        audio_md5: str,
        duration_sec: float,
    ) -> Dict[str, Any]:
        mode = "ENDPOINTING"
        cache_key = hashlib.md5(f"{MUSE_VOICE_MODEL}|{mode}|{audio_md5}".encode("utf-8")).hexdigest()
        cached = get_cached_result("voice_transcribe_api", cache_key)

        if cached is not None:
            self.tracker.record_event(
                phase=2,
                model="Muse Voice Transcribe",
                operation=(
                    f"Caché MD5 HIT para '{filename}' ({video_id}): se reutiliza la transcripción de "
                    f"{cached.get('audio_duration_sec', 0):.1f}s de audio sin llamar a la API."
                ),
                status="CACHE_HIT_MD5",
                details={"video_id": video_id, "audio_md5": audio_md5, "model": MUSE_VOICE_MODEL},
            )
            cached["cache_hit"] = True
            cached["segments"] = []
            cached["filtered_chatter"] = []
            # La caché es por contenido del audio: el mismo video puede volver en otra posición (vid_3 que antes fue
            # vid_1). Sin esto Spark recibía el id viejo y asignaba los productos al video equivocado.
            cached["video_id"] = video_id
            cached["filename"] = filename
            return cached

        response = transcribe_wav(
            wav_path,
            model=MUSE_VOICE_MODEL,
            mode=mode,
            language_bias=MUSE_VOICE_LANGUAGES,
            keywords=SALES_KEYWORDS_BIAS,
        )
        transcript = str(response.get("transcript", "")).strip()
        audio_duration_sec = round(float(response.get("audioDurationMs", duration_sec * 1000)) / 1000.0, 2)

        turns = [
            {
                "turn_id": int(t.get("turnId", idx)),
                "start_sec": round(float(t.get("startMs", 0)) / 1000.0, 2),
                "end_sec": round(float(t.get("endMs", 0)) / 1000.0, 2),
                "text": str(t.get("transcript", "")).strip(),
            }
            for idx, t in enumerate(response.get("turns", []))
            if str(t.get("transcript", "")).strip()
        ]
        if not turns and transcript:
            turns = [{"turn_id": 0, "start_sec": 0.0, "end_sec": audio_duration_sec, "text": transcript}]

        result = {
            "video_id": video_id,
            "filename": filename,
            "audio_md5": audio_md5,
            "duration_sec": duration_sec,
            "audio_duration_sec": audio_duration_sec,
            "model": MUSE_VOICE_MODEL,
            "asr_session_id": response.get("sessionId"),
            "raw_asr_transcript": transcript,
            "turns": turns,
            "cache_hit": False,
        }
        set_cached_result("voice_transcribe_api", cache_key, result)

        self.tracker.record_event(
            phase=2,
            model="Muse Voice Transcribe",
            operation=(
                f"Audio de '{filename}' ({video_id}) transcrito por {MUSE_VOICE_MODEL}: "
                f"{audio_duration_sec:.1f}s, {len(turns)} turno(s) de voz. La API cobra por audio y no reporta tokens."
            ),
            status="API_CALL",
            details={
                "video_id": video_id,
                "audio_md5": audio_md5,
                "audio_duration_sec": audio_duration_sec,
                "turns": len(turns),
                "asr_session_id": response.get("sessionId"),
            },
        )
        return {**result, "segments": [], "filtered_chatter": []}

    def analyze_semantics_with_spark(
        self,
        transcriptions: List[Dict[str, Any]],
        videos_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        spark_input = [
            {
                "video_id": t["video_id"],
                "duration_sec": t.get("audio_duration_sec") or t.get("duration_sec"),
                "turns": t.get("turns", []),
            }
            for t in transcriptions
        ]
        canonical_payload = json.dumps(spark_input, sort_keys=True, ensure_ascii=False)

        # Videos livianos para que Spark vea los objetos y corrija nombres mal transcritos
        video_proxies: List[Tuple[str, Path]] = []
        if MUSE_SPARK_PHASE2_VIDEO:
            for t in transcriptions:
                source = locate_source_video(videos_by_id.get(t["video_id"], {}))
                proxy = build_video_proxy(source) if source else None
                if proxy:
                    video_proxies.append((t["video_id"], proxy))
        video_signature = ",".join(
            f"{vid}:{hashlib.md5(p.read_bytes()).hexdigest()}@{MUSE_SPARK_PHASE2_VIDEO_FPS}"
            for vid, p in video_proxies
        )

        batch_md5 = hashlib.md5(
            (
                f"{MUSE_SPARK_MODEL}|{MUSE_SPARK_REASONING_PHASE2}|{SPARK_PROMPT_VERSION}|"
                f"{video_signature}|{canonical_payload}{prompts.signature('p2_extract')}"
            ).encode("utf-8")
        ).hexdigest()

        cached = get_cached_result("muse_spark_batch_api", batch_md5)
        if cached is not None:
            usage = cached.get("usage", {})
            self.tracker.record_event(
                phase=2,
                model="Muse Spark",
                operation=(
                    f"Caché MD5 HIT del análisis semántico ({batch_md5[:10]}...): "
                    f"{len(cached.get('products', []))} productos sin llamar a la API."
                ),
                tokens_saved=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                status="CACHE_HIT_MD5",
                details={"batch_md5": batch_md5},
            )
            self._apply_segments_to_transcriptions(transcriptions, cached)
            cached["cache_hit"] = True
            return cached

        user_content: Any = canonical_payload
        if video_proxies:
            user_content = []
            for vid_id, proxy in video_proxies:
                encoded = base64.b64encode(proxy.read_bytes()).decode("ascii")
                user_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Video {vid_id} (muestreado a {MUSE_SPARK_PHASE2_VIDEO_FPS:g} fps: la imagen N "
                            f"corresponde aproximadamente al segundo (N-1)/{MUSE_SPARK_PHASE2_VIDEO_FPS:g} del video):"
                        ),
                    }
                )
                user_content.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{encoded}", "fps": MUSE_SPARK_PHASE2_VIDEO_FPS},
                    }
                )
            user_content.append({"type": "text", "text": f"Transcripciones ASR por turnos:\n{canonical_payload}"})

        parsed, usage, raw = chat_json(
            model=MUSE_SPARK_MODEL,
            messages=[
                {"role": "system", "content": prompts.get("p2_extract")},
                {"role": "user", "content": user_content},
            ],
            schema_name="SalesProductsBatch",
            schema=SPARK_PRODUCTS_SCHEMA,
            reasoning_effort=MUSE_SPARK_REASONING_PHASE2,
        )

        products = self._build_products_from_spark(parsed.get("products", []), transcriptions, videos_by_id)
        discarded = [
            {"video_id": c["video_id"], "discarded_text": c["text"], "reason": c["reason"]}
            for c in parsed.get("discarded_chatter", [])
        ]

        batch_output = {
            "batch_md5": batch_md5,
            "model": MUSE_SPARK_MODEL,
            "cache_hit": False,
            "total_products_detected": len(products),
            "discarded_chatter": discarded,
            "products": products,
            "usage": usage,
            "videos_sent_to_spark": [vid for vid, _ in video_proxies],
            "visual_inventory": parsed.get("visual_inventory", []),
            "spark_raw_output": parsed,
        }
        set_cached_result("muse_spark_batch_api", batch_md5, batch_output)
        self._apply_segments_to_transcriptions(transcriptions, batch_output)

        self.tracker.record_event(
            phase=2,
            model="Muse Spark",
            operation=(
                f"{MUSE_SPARK_MODEL}: 1 llamada con salida JSON Schema para {len(transcriptions)} transcripción(es)"
                f" + {len(video_proxies)} video(s) a {MUSE_SPARK_PHASE2_VIDEO_FPS:g} fps: "
                f"{len(products)} productos "
                f"({sum(1 for p in products if p['mention_type'] == 'EXPLICIT')} EXPLICIT, "
                f"{sum(1 for p in products if p['mention_type'] == 'DEICTIC')} DEICTIC) y "
                f"{len(discarded)} fragmentos de charla descartados "
                f"(razonamiento '{MUSE_SPARK_REASONING_PHASE2}': {usage['reasoning_tokens']} de los tokens de salida)."
            ),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            status="API_CALL",
            details={
                "batch_md5": batch_md5,
                "reasoning_effort": MUSE_SPARK_REASONING_PHASE2,
                "reasoning_tokens": usage["reasoning_tokens"],
                "cached_prompt_tokens": usage["cached_tokens"],
                "response_id": raw.get("id"),
            },
        )
        return batch_output

    def _build_products_from_spark(
        self,
        spark_products: List[Dict[str, Any]],
        transcriptions: List[Dict[str, Any]],
        videos_by_id: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convierte la salida de Spark en productos con ventanas de tiempo calculadas sobre los turnos."""
        trans_by_id = {t["video_id"]: t for t in transcriptions}

        # Agrupar por (video, turno) respetando el orden en que Spark los listó
        grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for item in spark_products:
            trans = trans_by_id.get(item.get("video_id"))
            if trans is None or not trans.get("turns"):
                continue
            turn_ids = [t["turn_id"] for t in trans["turns"]]
            turn_id = item.get("turn_id") if item.get("turn_id") in turn_ids else turn_ids[0]
            grouped.setdefault((trans["video_id"], turn_id), []).append(item)

        timed: List[Tuple[str, float, Dict[str, Any], float, str]] = []
        for (vid_id, turn_id), items in grouped.items():
            turn = next(t for t in trans_by_id[vid_id]["turns"] if t["turn_id"] == turn_id)
            text_len = max(1, len(turn["text"]))
            turn_span = max(0.1, turn["end_sec"] - turn["start_sec"])

            cursor = 0
            spans: List[Optional[Tuple[int, int]]] = []
            for item in items:
                span = locate_quote_span(turn["text"], item.get("quote", ""), cursor)
                spans.append(span)
                if span:
                    cursor = span[1]

            for idx, (item, span) in enumerate(zip(items, spans)):
                if span:
                    frac_start, frac_end = span[0] / text_len, span[1] / text_len
                    timing_source = "QUOTE_ALIGNED_IN_TURN"
                else:
                    frac_start, frac_end = idx / len(items), (idx + 1) / len(items)
                    timing_source = "TURN_EVEN_SPLIT"
                start_s = round(turn["start_sec"] + frac_start * turn_span, 2)
                end_s = round(turn["start_sec"] + frac_end * turn_span, 2)
                timed.append((vid_id, start_s, item, end_s, timing_source))

        video_order = {t["video_id"]: i for i, t in enumerate(transcriptions)}
        timed.sort(key=lambda row: (video_order.get(row[0], 99), row[1]))

        products: List[Dict[str, Any]] = []
        for counter, (vid_id, start_s, item, end_s, timing_source) in enumerate(timed, start=1):
            video_meta = videos_by_id.get(vid_id, {})
            trans = trans_by_id[vid_id]
            vid_duration = float(
                video_meta.get("quality_assessment", {}).get("duration_sec")
                or trans.get("audio_duration_sec")
                or end_s
            )
            expanded_start = round(max(0.0, start_s - 1.5), 2)
            expanded_end = round(min(vid_duration, end_s + 1.5), 2)

            retained_frames = video_meta.get("retained_frames", [])
            matching_frames = [
                f
                for f in retained_frames
                if expanded_start <= float(f.get("timestamp_sec", 0.0)) <= expanded_end
            ]
            if not matching_frames and retained_frames:
                mid_t = (expanded_start + expanded_end) / 2.0
                matching_frames = sorted(
                    retained_frames,
                    key=lambda f: abs(float(f.get("timestamp_sec", 0.0)) - mid_t),
                )[:3]

            original_price = item.get("original_price")
            price = float(item.get("price") or 0.0)
            price_corrected = original_price is not None and float(original_price) != price
            spoken_name = str(item.get("spoken_item_name", "")).strip()
            visual_name = str(item.get("visual_item_name", "")).strip()
            mention_type = item.get("mention_type", "DEICTIC")
            if visual_name:
                detected_name = visual_name
            elif mention_type == "EXPLICIT" and spoken_name:
                detected_name = spoken_name.capitalize()
            else:
                detected_name = DEICTIC_PLACEHOLDER_NAME

            products.append(
                {
                    "product_id": f"prod_{counter}",
                    "video_id": vid_id,
                    "video_filename": trans.get("filename", vid_id),
                    "mention_type": mention_type,
                    "detected_name": detected_name,
                    "spoken_item_name": spoken_name,
                    "visual_item_name": visual_name,
                    "name_source": item.get("name_source", "UNKNOWN") if visual_name else "AUDIO",
                    "raw_spoken_phrase": item.get("quote", ""),
                    "interpreted_phrase": item.get("interpreted_phrase", ""),
                    "price": price,
                    "original_price_mentioned": float(original_price) if price_corrected else None,
                    "price_corrected_live": price_corrected,
                    "currency": item.get("currency", "USD"),
                    "condition": item.get("condition") or "No mencionado",
                    "features": item.get("features", []),
                    "spoken_timestamps": {
                        "start_sec": start_s,
                        "end_sec": end_s,
                        "timing_source": timing_source,
                    },
                    "expanded_visual_window": {
                        "start_sec": expanded_start,
                        "end_sec": expanded_end,
                        "desync_margin_sec": 1.5,
                    },
                    "candidate_frames_preview": [
                        {
                            "frame_id": f["frame_id"],
                            "timestamp_sec": f["timestamp_sec"],
                            "url": f["url"],
                            "laplacian_score": f["laplacian_score"],
                        }
                        for f in matching_frames[:3]
                    ],
                    "user_verified": False,
                }
            )
        return products

    @staticmethod
    def _apply_segments_to_transcriptions(
        transcriptions: List[Dict[str, Any]], batch_output: Dict[str, Any]
    ) -> None:
        """Rellena `segments` y `filtered_chatter` de cada transcripción para la UI."""
        for trans in transcriptions:
            vid_id = trans["video_id"]
            trans["segments"] = [
                {
                    "segment_id": p["product_id"],
                    "start_sec": p["spoken_timestamps"]["start_sec"],
                    "end_sec": p["spoken_timestamps"]["end_sec"],
                    "text": p["raw_spoken_phrase"],
                }
                for p in batch_output.get("products", [])
                if p["video_id"] == vid_id
            ]
            trans["filtered_chatter"] = [
                c["discarded_text"]
                for c in batch_output.get("discarded_chatter", [])
                if c["video_id"] == vid_id
            ]

    # La Fase 3 real vive en backend/services/phase3_vision_sam.py (muse-spark-1.3 + sam-3.1).

    # La Fase 4 real vive en backend/services/phase4_studio.py (muse-image-1.0 + control de fidelidad).

    # La Fase 5 real vive en backend/services/phase5_codegen.py (textos con muse-spark-1.3 + plantilla del catálogo).
