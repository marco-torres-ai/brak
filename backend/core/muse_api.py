"""
Cliente HTTP mínimo para la Meta Model API (https://api.meta.ai/v1).

Docs: https://dev.meta.ai/docs/models.md
- Voice Transcribe: POST /asr/transcribe (multipart: `request` JSON + `audio` WAV mono 16/24 kHz).
- Muse Spark: POST /chat/completions (compatible con OpenAI) con `response_format: json_schema`.
- SAM 3.1: POST /responses con `input_text` (frase corta o prompt de caja) + `input_image`; la salida son
  tokens especiales `<0f>id<|box;...|><|mask;...|>` que decodifica el parser oficial `meta-sam-parser`.
"""
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from backend.core import progress
from backend.core.config import META_MUSE_API_KEY, MUSE_API_BASE_URL

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class MuseAPIError(RuntimeError):
    pass


def _auth_headers(accept: str = "application/json") -> Dict[str, str]:
    if not META_MUSE_API_KEY:
        raise MuseAPIError("META_MUSE_API_KEY is missing from the .env file")
    return {"Authorization": f"Bearer {META_MUSE_API_KEY}", "Accept": accept}


def _post(
    path: str, timeout: float = 120.0, attempts: int = 4, max_timeout: float = 300.0, label: str = "", **kwargs: Any
) -> httpx.Response:
    """POST con reintentos. Si Meta no responde dentro de `timeout`, el siguiente intento espera el doble (hasta
    `max_timeout`): una petición trabada del lado de Meta se recupera rápido y una respuesta lenta pero legítima
    igual alcanza a llegar. Las esperas y reintentos se informan al progreso en vivo."""
    url = f"{MUSE_API_BASE_URL}{path}"
    name = label or path
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            resp = httpx.post(url, timeout=timeout, **kwargs)
        except httpx.TimeoutException as exc:
            if attempt == attempts:
                raise MuseAPIError(f"{name}: Meta did not respond after {attempts} attempts ({exc})") from exc
            progress.note_wait(
                f"{name}: Meta did not respond within {timeout:.0f} s; retrying automatically "
                f"(attempt {attempt + 1} of {attempts}, waiting up to {min(max_timeout, timeout * 2):.0f} s)."
            )
            timeout = min(max_timeout, timeout * 2)
            continue
        except httpx.TransportError as exc:
            if attempt == attempts:
                raise MuseAPIError(f"Network error calling {path}: {exc}") from exc
            progress.note_wait(f"{name}: network error ({type(exc).__name__}); retry {attempt + 1} of {attempts} in {delay:.0f} s.")
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code in RETRYABLE_STATUS and attempt < attempts:
            progress.note_wait(f"{name}: Meta returned HTTP {resp.status_code}; retry {attempt + 1} of {attempts} in {delay:.0f} s.")
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code >= 400:
            raise MuseAPIError(f"{path} returned HTTP {resp.status_code}: {resp.text[:800]}")
        return resp
    raise MuseAPIError(f"{path}: retries exhausted")


def normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Unifica el objeto `usage` (estilo OpenAI o Anthropic) a input/output/cached."""
    usage = usage or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cached = prompt_details.get("cached_tokens", usage.get("cache_read_input_tokens", 0)) or 0
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cached_tokens": int(cached),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0),
    }


def transcribe_wav(
    wav: Union[Path, bytes],
    model: str,
    mode: str = "ENDPOINTING",
    language_bias: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Transcribe un WAV (ruta o bytes). Devuelve {sessionId, transcript, audioDurationMs, turns[]}."""
    request_body: Dict[str, Any] = {"model": model, "audioEncoding": "WAV", "mode": mode}
    if language_bias:
        request_body["languageBias"] = language_bias
    if keywords:
        request_body["keywords"] = keywords

    audio_bytes = wav if isinstance(wav, bytes) else Path(wav).read_bytes()
    resp = _post(
        "/asr/transcribe",
        timeout=60.0,
        max_timeout=180.0,
        label="Muse Voice Transcribe",
        headers=_auth_headers(),
        files={
            "request": (None, json.dumps(request_body), "application/json"),
            "audio": ("audio.wav", audio_bytes, "audio/wav"),
        },
    )
    return resp.json()


def chat_json(
    model: str,
    messages: List[Dict[str, Any]],
    schema_name: str,
    schema: Dict[str, Any],
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
    """Llama a /chat/completions con salida estructurada estricta. Devuelve (json, usage, respuesta cruda).

    reasoning_effort: none | minimal | low | medium | high | xhigh | max (los tokens de razonamiento
    se facturan como salida, así que conviene el mínimo suficiente para cada tarea).
    """
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    }
    if max_completion_tokens:
        body["max_completion_tokens"] = max_completion_tokens
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    # Límite según el tamaño de la petición: ~90 s para llamadas chicas (suelen responder en 10-30 s) y hasta 300 s
    # con 3 videos de 60 s. Si se agota, _post reintenta con el doble.
    size_mb = len(json.dumps(body)) / 1_000_000
    timeout = min(300.0, 90.0 + 20.0 * size_mb)
    resp = _post("/chat/completions", timeout=timeout, max_timeout=300.0, label=f"Muse Spark ({size_mb:.1f} MB)", headers=_auth_headers(), json=body)
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise MuseAPIError(f"Unexpected response from {model}: {json.dumps(data)[:800]}") from exc
    return parsed, normalize_usage(data.get("usage")), data


def segment_image_sam(model: str, image_jpeg: bytes, prompt_text: str) -> Tuple[str, Dict[str, Any]]:
    """Segmenta una imagen con SAM 3.1. Devuelve (texto de salida con tokens box/mask, usage crudo).

    prompt_text es una frase nominal corta ("bottle") o un prompt de caja armado con
    `meta_sam_parser.build_object_prompt`. Cero coincidencias = texto vacío (no es error).
    """
    data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "metadata": {"include_confidence": "true"},
    }
    resp = _post("/responses", timeout=60.0, max_timeout=180.0, label="SAM 3.1", headers=_auth_headers(), json=body)
    data = resp.json()
    texts = [
        part.get("text", "")
        for item in data.get("output") or []
        for part in (item.get("content") or [])
        if isinstance(part, dict)
    ]
    return "\n".join(t for t in texts if t), data.get("usage") or {}


def edit_image(
    model: str,
    images: List[bytes],
    prompt: str,
    size: Optional[str] = None,
    output_format: str = "jpeg",
    mime: str = "image/png",
) -> Tuple[bytes, Dict[str, Any]]:
    """Edita/compone con muse-image (POST /images/edits, JSON). Devuelve (bytes de la imagen, usage).

    No admite máscara ni fondo transparente; la fidelidad al producto se pide en el prompt.
    Se cobra por imagen generada (no por token); `usage` es solo referencial.
    """
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "images": [{"image_url": f"data:{mime};base64," + base64.b64encode(img).decode("ascii")} for img in images],
        "n": 1,
        "output_format": output_format,
        "response_format": "b64_json",
    }
    if size:
        body["size"] = size
    resp = _post("/images/edits", timeout=120.0, max_timeout=300.0, label="Muse Image", headers=_auth_headers(), json=body)
    data = resp.json()
    try:
        image_bytes = base64.b64decode(data["data"][0]["b64_json"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise MuseAPIError(f"Unexpected response from {model}: {json.dumps(data)[:800]}") from exc
    return image_bytes, data.get("usage") or {}
