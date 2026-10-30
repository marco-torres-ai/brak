"""Flujos configurables: qué pasos opcionales corren y con qué parámetros.

El orden de las fases es fijo (cada una necesita el resultado de la anterior); un flujo solo
enciende/apaga pasos opcionales y ajusta parámetros. Las fases leen sus opciones con `get(key)`.

Regla de oro: sin flujo activo ("Completo de Meta Muse") cada opción vale lo mismo que su
variable del .env, así que el pipeline queda idéntico.

Persistencia (storage/flows/, fuera de git):
  - flows/<slug>.json = {"name", "created_at", "options": {key: valor}}
  - active.json       = {"flow": slug | null}
"""

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import (
    MUSE_IMAGE_PARALLEL,
    MUSE_PHASE2_VERIFY,
    MUSE_PHASE3_VERIFY_CUTOUTS,
    MUSE_PHASE4_LIFESTYLE,
    MUSE_SPARK_PHASE2_VIDEO,
    STORAGE_DIR,
)
from backend.core.prompts import PromptError, _read_json, _slugify, _write_json

FLOWS_DIR = STORAGE_DIR / "flows"
FLOW_FILES_DIR = FLOWS_DIR / "flows"
ACTIVE_FILE = FLOWS_DIR / "active.json"
DEFAULT_FLOW_NAME = "Completo de Meta Muse"
AUTO_FLOW_NAME = "Mi flujo"
RESERVED_SLUGS = {"active", "options"}  # chocarían con /flows/active y /flows/options
_LOCK = threading.Lock()

# node = id de la bolita en el frontend. Los valores por defecto salen del .env (comportamiento actual).
OPTIONS: Dict[str, Dict[str, Any]] = {
    "extract.video": {"node": "extract", "phase": 2, "type": "bool", "default": MUSE_SPARK_PHASE2_VIDEO,
                      "label": "Spark también mira el video",
                      "description": "Además de la transcripción, Spark ve el video para corregir nombres mal oídos. Apagado: más rápido y barato, pero menos preciso."},
    "verify2.enabled": {"node": "verify2", "phase": 2, "type": "bool", "default": MUSE_PHASE2_VERIFY,
                        "label": "Verificar audio ↔ video",
                        "description": "Revisa tramos dudosos y agrega productos mostrados que no quedaron en la lista."},
    "verify3.enabled": {"node": "verify3", "phase": 3, "type": "bool", "default": MUSE_PHASE3_VERIFY_CUTOUTS,
                        "label": "Verificar recortes",
                        "description": "Revisa cada recorte de SAM y prueba alternativas si salió mal."},
    "images.parallel": {"node": "master", "phase": 4, "type": "int", "default": MUSE_IMAGE_PARALLEL, "min": 1, "max": 8,
                        "label": "Imágenes en paralelo",
                        "description": "Cuántas fotos pide Muse Image a la vez. Más = más rápido; si Meta se satura, baja este número."},
    "view.left": {"node": "v_left", "phase": 4, "type": "bool", "default": True, "label": "Vista izquierda (IA)",
                  "description": "Genera el lado izquierdo del producto."},
    "view.right": {"node": "v_right", "phase": 4, "type": "bool", "default": True, "label": "Vista derecha (IA)",
                   "description": "Genera el lado derecho del producto."},
    "view.back": {"node": "v_back", "phase": 4, "type": "bool", "default": True, "label": "Vista trasera (IA)",
                  "description": "Genera la parte de atrás del producto."},
    "view.lifestyle": {"node": "v_life", "phase": 4, "type": "bool", "default": MUSE_PHASE4_LIFESTYLE, "label": "Foto de ambiente",
                       "description": "Coloca el producto en un ambiente realista."},
    "fidelity.enabled": {"node": "fidelity", "phase": 4, "type": "bool", "default": True, "label": "Control de fidelidad",
                         "description": "Spark compara cada foto con el producto real y repite las infieles. Apagado: más rápido y barato, pero pueden pasar fotos con partes inventadas."},
}


def _path(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]{1,48}", slug or ""):
        raise PromptError("Nombre de flujo inválido.")
    return FLOW_FILES_DIR / f"{slug}.json"


def active_slug() -> Optional[str]:
    slug = (_read_json(ACTIVE_FILE) or {}).get("flow")
    if slug and _read_json(_path(slug)) is not None:
        return slug
    return None


def _active() -> Optional[Dict[str, Any]]:
    slug = active_slug()
    return _read_json(_path(slug)) if slug else None


def get(key: str) -> Any:
    """Valor efectivo de una opción en el flujo activo (o su valor por defecto)."""
    meta = OPTIONS[key]
    flow = _active()
    if flow and key in (flow.get("options") or {}):
        return flow["options"][key]
    return meta["default"]


def active_name() -> str:
    flow = _active()
    return flow.get("name", DEFAULT_FLOW_NAME) if flow else DEFAULT_FLOW_NAME


def changed_options() -> Dict[str, Any]:
    """Opciones del flujo activo que difieren del valor por defecto."""
    return {k: get(k) for k in OPTIONS if get(k) != OPTIONS[k]["default"]}


def describe() -> Optional[Dict[str, Any]]:
    """Para la trazabilidad de la sesión; None si el flujo equivale al completo."""
    changed = changed_options()
    if not changed:
        return None
    return {"flow": active_name(), "changed": {OPTIONS[k]["label"]: v for k, v in changed.items()}}


def _validate(options: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in (options or {}).items():
        if key not in OPTIONS:
            raise PromptError(f"Opción desconocida: {key}")
        meta = OPTIONS[key]
        if meta["type"] == "bool":
            if not isinstance(value, bool):
                raise PromptError(f"'{meta['label']}' debe ser verdadero o falso.")
        else:
            if isinstance(value, bool) or not isinstance(value, int) or not meta["min"] <= value <= meta["max"]:
                raise PromptError(f"'{meta['label']}' debe ser un número entero entre {meta['min']} y {meta['max']}.")
        clean[key] = value
    return clean


def _unique_slug(name: str) -> str:
    base = _slugify(name)
    slug, n = base, 2
    while (FLOW_FILES_DIR / f"{slug}.json").exists() or slug in RESERVED_SLUGS:
        slug = f"{base}-{n}"[:48]
        n += 1
    return slug


def create_flow(name: str, copy_from: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> str:
    name = (name or "").strip()[:60]
    if not name:
        raise PromptError("El flujo necesita un nombre.")
    with _LOCK:
        base: Dict[str, Any] = {}
        if copy_from:
            src = _read_json(_path(copy_from))
            if src is None:
                raise PromptError("El flujo a copiar no existe.")
            base = dict(src.get("options") or {})
        base.update(_validate(options or {}))
        slug = _unique_slug(name)
        _write_json(_path(slug), {"name": name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "options": _validate(base)})
        _write_json(ACTIVE_FILE, {"flow": slug})
        return slug


def update_options(options: Dict[str, Any]) -> str:
    """Guarda en el flujo activo; si está activo el completo, crea y activa 'Mi flujo'."""
    clean = _validate(options)
    slug = active_slug()
    if not slug:
        return create_flow(AUTO_FLOW_NAME, options=clean)
    with _LOCK:
        data = json.loads(json.dumps(_read_json(_path(slug)) or {"name": slug, "options": {}}))
        opts = data.setdefault("options", {})
        for k, v in clean.items():
            if v == OPTIONS[k]["default"]:
                opts.pop(k, None)
            else:
                opts[k] = v
        _write_json(_path(slug), data)
    return slug


def reset_options() -> None:
    slug = active_slug()
    if not slug:
        return
    with _LOCK:
        data = json.loads(json.dumps(_read_json(_path(slug)) or {}))
        data["options"] = {}
        _write_json(_path(slug), data)


def set_active(slug: Optional[str]) -> None:
    with _LOCK:
        if slug and _read_json(_path(slug)) is None:
            raise PromptError("Ese flujo no existe.")
        _write_json(ACTIVE_FILE, {"flow": slug or None})


def rename_flow(slug: str, name: str) -> None:
    name = (name or "").strip()[:60]
    if not name:
        raise PromptError("El flujo necesita un nombre.")
    with _LOCK:
        data = _read_json(_path(slug))
        if data is None:
            raise PromptError("Ese flujo no existe.")
        data = json.loads(json.dumps(data))
        data["name"] = name
        _write_json(_path(slug), data)


def delete_flow(slug: str) -> None:
    with _LOCK:
        path = _path(slug)
        if not path.exists():
            raise PromptError("Ese flujo no existe.")
        if active_slug() == slug:
            _write_json(ACTIVE_FILE, {"flow": None})
        path.unlink()


def snapshot() -> Dict[str, Any]:
    flows: List[Dict[str, Any]] = []
    if FLOW_FILES_DIR.exists():
        for p in sorted(FLOW_FILES_DIR.glob("*.json")):
            data = _read_json(p)
            if isinstance(data, dict):
                flows.append({"slug": p.stem, "name": data.get("name", p.stem), "changed": len(data.get("options") or {})})
    return {
        "active_flow": active_slug(),
        "active_flow_name": active_name(),
        "default_flow_name": DEFAULT_FLOW_NAME,
        "flows": flows,
        "options": [
            {"key": k, **{f: m[f] for f in ("node", "phase", "type", "label", "description")}, "default": m["default"],
             "value": get(k), "min": m.get("min"), "max": m.get("max")}
            for k, m in OPTIONS.items()
        ],
    }
