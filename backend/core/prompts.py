"""Registro de prompts configurables de los agentes.

Cada módulo registra su prompt original al importarse (`register`). Las fases piden el texto
efectivo con `get(key)` y agregan `signature(key)` a sus claves de caché.

Regla de oro: sin perfil activo, `get` devuelve exactamente el texto original y `signature`
devuelve "", así que el pipeline y su caché quedan idénticos.

Persistencia (storage/prompts/, fuera de git):
  - profiles/<slug>.json = {"name", "created_at", "overrides": {key: {"extra", "full_text"}}}
  - active.json          = {"profile": slug | null}
"""

import hashlib
import json
import re
import string
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.core.config import STORAGE_DIR

PROMPTS_DIR = STORAGE_DIR / "prompts"
PROFILES_DIR = PROMPTS_DIR / "profiles"
ACTIVE_FILE = PROMPTS_DIR / "active.json"
MAX_TEXT_CHARS = 20000
MAX_EXTRA_CHARS = 4000
DEFAULT_PROFILE_NAME = "Original de Meta Muse"
AUTO_PROFILE_NAME = "Mi perfil"

_REGISTRY: Dict[str, Dict[str, Any]] = {}
_ORDER: List[str] = []
_LOCK = threading.Lock()
_FILE_CACHE: Dict[str, Tuple[float, Any]] = {}


class PromptError(ValueError):
    """Entrada inválida para el registro (se traduce a HTTP 400)."""


# ----------------------------------------------------------------------------- registro

def register(
    key: str,
    default: str,
    *,
    phase: int,
    model: str,
    group: str,
    title: str,
    description: str,
    kind: str = "system",
    required: Tuple[str, ...] = (),
    optional: Tuple[str, ...] = (),
    hint: str = "",
) -> None:
    if key not in _REGISTRY:
        _ORDER.append(key)
    _REGISTRY[key] = {
        "key": key,
        "default": default,
        "phase": phase,
        "model": model,
        "group": group,
        "title": title,
        "description": description,
        "kind": kind,
        "required": tuple(required),
        "optional": tuple(optional),
        "hint": hint,
    }


def _meta(key: str) -> Dict[str, Any]:
    if key not in _REGISTRY:
        raise PromptError(f"Agente desconocido: {key}")
    return _REGISTRY[key]


# ----------------------------------------------------------------------------- archivos

def _read_json(path: Path) -> Any:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _FILE_CACHE.pop(str(path), None)
        return None
    hit = _FILE_CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = None
    _FILE_CACHE[str(path)] = (mtime, data)
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    _FILE_CACHE.pop(str(path), None)


def _slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:48] or "perfil"


def _profile_path(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]{1,48}", slug or ""):
        raise PromptError("Nombre de perfil inválido.")
    return PROFILES_DIR / f"{slug}.json"


def active_slug() -> Optional[str]:
    data = _read_json(ACTIVE_FILE) or {}
    slug = data.get("profile")
    if slug and _read_json(_profile_path(slug)) is not None:
        return slug
    return None


def _active_profile() -> Optional[Dict[str, Any]]:
    slug = active_slug()
    return _read_json(_profile_path(slug)) if slug else None


def _override(key: str) -> Dict[str, Any]:
    prof = _active_profile()
    if not prof:
        return {}
    return (prof.get("overrides") or {}).get(key) or {}


# ----------------------------------------------------------------------------- lectura

def _clean_override(key: str, o: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    meta = _meta(key)
    extra = (o.get("extra") or "").strip()
    full = o.get("full_text")
    if full is not None and (not full.strip() or full == meta["default"]):
        full = None
    return extra, full


def is_custom(key: str) -> bool:
    extra, full = _clean_override(key, _override(key))
    return bool(extra) or full is not None


def get(key: str) -> str:
    """Texto efectivo del agente (original, reescrito y/o con instrucciones extra)."""
    meta = _meta(key)
    extra, full = _clean_override(key, _override(key))
    text = full if full is not None else meta["default"]
    if extra:
        if meta["kind"] == "image":
            # Los prompts de imagen pasan por .format(): las llaves del usuario se escapan.
            text = text + " Additional user instructions: " + extra.replace("{", "{{").replace("}", "}}")
        else:
            text = text + "\n\nInstrucciones adicionales del usuario (respeta siempre el formato de salida exigido): " + extra
    return text


def signature(*keys: str) -> str:
    """'' si todos los agentes están en su versión original; si no, un hash corto del texto efectivo."""
    if not any(is_custom(k) for k in keys):
        return ""
    joined = "\x00".join(get(k) for k in keys)
    return "|p:" + hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


def describe_phase(phase: int) -> Optional[str]:
    """Texto para el registro de tokens cuando la fase usa prompts personalizados; None si todo es original."""
    custom = [_REGISTRY[k]["title"] for k in _ORDER if _REGISTRY[k]["phase"] == phase and is_custom(k)]
    if not custom:
        return None
    prof = _active_profile() or {}
    return f"Prompts: perfil '{prof.get('name', '?')}' ({len(custom)} agente(s) personalizado(s) en esta fase: {', '.join(custom)})."


def active_profile_name() -> str:
    prof = _active_profile()
    return prof.get("name", DEFAULT_PROFILE_NAME) if prof else DEFAULT_PROFILE_NAME


# ----------------------------------------------------------------------------- validación

def _fields(text: str) -> List[str]:
    try:
        return [f for _, f, _, _ in string.Formatter().parse(text) if f is not None]
    except ValueError as exc:
        raise PromptError(f"Llaves desbalanceadas en el texto: {exc}. Usa solo los marcadores permitidos.")


def validate(key: str, extra: str, full_text: Optional[str]) -> None:
    meta = _meta(key)
    if len(extra or "") > MAX_EXTRA_CHARS:
        raise PromptError(f"Las instrucciones adicionales superan {MAX_EXTRA_CHARS} caracteres.")
    if full_text is None:
        return
    if not full_text.strip():
        raise PromptError("El texto completo no puede quedar vacío. Usa 'Restaurar original'.")
    if len(full_text) > MAX_TEXT_CHARS:
        raise PromptError(f"El texto completo supera {MAX_TEXT_CHARS} caracteres.")
    if meta["kind"] != "image":
        return
    fields = _fields(full_text)
    allowed = set(meta["required"]) | set(meta["optional"])
    unknown = sorted({f for f in fields if f not in allowed})
    if unknown:
        raise PromptError(
            "Marcador(es) no permitido(s): " + ", ".join("{" + f + "}" for f in unknown)
            + ". Permitidos: " + ", ".join("{" + f + "}" for f in sorted(allowed))
        )
    missing = [f for f in meta["required"] if f not in fields]
    if missing:
        raise PromptError("Falta(n) el/los marcador(es) obligatorio(s): " + ", ".join("{" + f + "}" for f in missing))


# ----------------------------------------------------------------------------- escritura

def _profiles() -> List[Dict[str, Any]]:
    out = []
    if PROFILES_DIR.exists():
        for p in sorted(PROFILES_DIR.glob("*.json")):
            data = _read_json(p)
            if isinstance(data, dict):
                out.append({"slug": p.stem, "name": data.get("name", p.stem), "created_at": data.get("created_at"),
                            "custom_agents": len([k for k, v in (data.get("overrides") or {}).items() if k in _REGISTRY and (v.get("extra") or v.get("full_text"))])})
    return out


RESERVED_SLUGS = {"active", "import"}  # chocarían con las rutas /profiles/active y /profiles/import


def _unique_slug(name: str) -> str:
    base = _slugify(name)
    slug, n = base, 2
    while (PROFILES_DIR / f"{slug}.json").exists() or slug in RESERVED_SLUGS:
        slug = f"{base}-{n}"[:48]
        n += 1
    return slug


def _set_active(slug: Optional[str]) -> None:
    _write_json(ACTIVE_FILE, {"profile": slug})


def create_profile(name: str, copy_from: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None, activate: bool = True) -> str:
    name = (name or "").strip()[:60]
    if not name:
        raise PromptError("El perfil necesita un nombre.")
    with _LOCK:
        if overrides is None:
            overrides = {}
            if copy_from:
                src = _read_json(_profile_path(copy_from))
                if src is None:
                    raise PromptError("El perfil a duplicar no existe.")
                overrides = json.loads(json.dumps(src.get("overrides") or {}))
        clean: Dict[str, Any] = {}
        for k, v in (overrides or {}).items():
            if k not in _REGISTRY or not isinstance(v, dict):
                continue
            extra = str(v.get("extra") or "")
            full = v.get("full_text")
            full = str(full) if full is not None else None
            validate(k, extra, full)
            clean[k] = {"extra": extra, "full_text": full}
        slug = _unique_slug(name)
        _write_json(_profile_path(slug), {"name": name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "overrides": clean})
        if activate:
            _set_active(slug)
        return slug


def set_active(slug: Optional[str]) -> None:
    with _LOCK:
        if slug and _read_json(_profile_path(slug)) is None:
            raise PromptError("Ese perfil no existe.")
        _set_active(slug or None)


def rename_profile(slug: str, name: str) -> None:
    """Cambia solo el nombre visible; el slug (nombre del archivo) y los prompts no cambian."""
    name = (name or "").strip()[:60]
    if not name:
        raise PromptError("El perfil necesita un nombre.")
    with _LOCK:
        path = _profile_path(slug)
        data = _read_json(path)
        if data is None:
            raise PromptError("Ese perfil no existe.")
        data = json.loads(json.dumps(data))
        data["name"] = name
        _write_json(path, data)


def delete_profile(slug: str) -> None:
    with _LOCK:
        path = _profile_path(slug)
        if not path.exists():
            raise PromptError("Ese perfil no existe.")
        if active_slug() == slug:
            _set_active(None)
        path.unlink()
        _FILE_CACHE.pop(str(path), None)


def export_profile(slug: str) -> Dict[str, Any]:
    data = _read_json(_profile_path(slug))
    if data is None:
        raise PromptError("Ese perfil no existe.")
    return {"brak_prompt_profile": 1, "name": data.get("name"), "overrides": data.get("overrides") or {}}


def import_profile(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("overrides"), dict):
        raise PromptError("El archivo no es un perfil de prompts de Brak Shop.")
    return create_profile(str(payload.get("name") or "Perfil importado"), overrides=payload["overrides"])


def update_agent(key: str, extra: str, full_text: Optional[str]) -> str:
    """Guarda en el perfil activo; si está activo el original, crea y activa 'Mi perfil'. Devuelve el slug."""
    _meta(key)
    extra = (extra or "").strip()
    validate(key, extra, full_text)
    slug = active_slug()
    if not slug:
        slug = create_profile(AUTO_PROFILE_NAME, overrides={}, activate=True)
    with _LOCK:
        path = _profile_path(slug)
        data = _read_json(path) or {"name": slug, "overrides": {}}
        data = json.loads(json.dumps(data))
        overrides = data.setdefault("overrides", {})
        if not extra and (full_text is None or full_text == _REGISTRY[key]["default"]):
            overrides.pop(key, None)
        else:
            overrides[key] = {"extra": extra, "full_text": full_text}
        _write_json(path, data)
    return slug


def reset_agent(key: str) -> None:
    _meta(key)
    slug = active_slug()
    if not slug:
        return
    with _LOCK:
        path = _profile_path(slug)
        data = json.loads(json.dumps(_read_json(path) or {"overrides": {}}))
        (data.setdefault("overrides", {})).pop(key, None)
        _write_json(path, data)


# ----------------------------------------------------------------------------- API

def snapshot() -> Dict[str, Any]:
    agents = []
    for key in _ORDER:
        m = _REGISTRY[key]
        extra, full = _clean_override(key, _override(key))
        agents.append({
            "key": key,
            "phase": m["phase"],
            "model": m["model"],
            "group": m["group"],
            "title": m["title"],
            "description": m["description"],
            "kind": m["kind"],
            "hint": m["hint"],
            "placeholders": {"required": list(m["required"]), "optional": list(m["optional"])},
            "default_text": m["default"],
            "extra": extra,
            "full_text": full,
            "effective_text": get(key),
            "is_custom": bool(extra) or full is not None,
        })
    return {
        "active_profile": active_slug(),
        "active_profile_name": active_profile_name(),
        "default_profile_name": DEFAULT_PROFILE_NAME,
        "profiles": _profiles(),
        "agents": agents,
    }
