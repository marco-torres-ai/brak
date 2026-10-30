"""
Progreso en vivo de la fase que se está ejecutando (en memoria, para el frontend).

Las fases devuelven su estado recién al terminar; mientras tanto el frontend consulta GET /api/v1/progress.
Lo alimentan el TokenTracker (cada evento) y el cliente de la API (esperas largas y reintentos).
La app atiende una sesión a la vez (ver AGENT_CONTEXT), así que se guarda solo la sesión en curso.
"""
import threading
import time
from typing import Any, Dict, Optional

MAX_EVENTS = 40

_lock = threading.Lock()
_state: Dict[str, Any] = {}


def begin(session_id: Optional[str], phase: int) -> None:
    """Marca el inicio de una fase (o del flujo completo) para una sesión."""
    with _lock:
        _state.clear()
        _state.update(
            {
                "session_id": session_id,
                "phase": phase,
                "running": True,
                "started_at": time.time(),
                "updated_at": time.time(),
                "message": f"Fase {phase} iniciada",
                "warning": None,
                "tokens_total": None,
                "tokens_saved_since_start": 0,
                "images_since_start": 0,
                "api_waits": 0,
                "events": [],
            }
        )


def end(error: Optional[str] = None) -> None:
    with _lock:
        if not _state:
            return
        _state["running"] = False
        _state["updated_at"] = time.time()
        _state["ended_at"] = time.time()
        _state["error"] = error
        if not error:
            _state["warning"] = None


def note_event(phase: int, model: str, operation: str, status: str, tokens_total: int, tokens_saved: int) -> None:
    """Lo llama el TokenTracker en cada evento."""
    with _lock:
        if not _state.get("running"):
            return
        _state["phase"] = phase
        _state["updated_at"] = time.time()
        _state["message"] = f"{model}: {operation}"
        _state["tokens_total"] = tokens_total
        if status == "CACHE_HIT_MD5":
            _state["tokens_saved_since_start"] += tokens_saved
        if model == "Muse Image" and status == "API_CALL":
            _state["images_since_start"] += 1
        # Una respuesta que llega limpia el aviso de espera larga.
        if status in ("API_CALL", "CACHE_HIT_MD5", "ZERO_TOKEN_LOCAL"):
            _state["warning"] = None
        _state["events"] = (_state["events"] + [{"t": time.time(), "phase": phase, "model": model, "status": status, "text": operation[:240]}])[-MAX_EVENTS:]


def note_wait(text: str) -> None:
    """Lo llama el cliente de la API cuando Meta tarda o falla y se reintenta."""
    with _lock:
        if not _state.get("running"):
            return
        _state["updated_at"] = time.time()
        _state["warning"] = text
        _state["api_waits"] += 1
        _state["events"] = (_state["events"] + [{"t": time.time(), "phase": _state.get("phase"), "model": "Meta API", "status": "WAIT", "text": text}])[-MAX_EVENTS:]


def snapshot(session_id: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        if not _state or (session_id and _state.get("session_id") not in (None, session_id)):
            return {"session_id": session_id, "running": False}
        now = time.time()
        return {
            **{k: v for k, v in _state.items() if k != "events"},
            "elapsed_sec": round((_state.get("ended_at") or now) - _state["started_at"], 1),
            "seconds_since_update": round(now - _state["updated_at"], 1),
            "events": list(_state["events"][-15:]),
        }
