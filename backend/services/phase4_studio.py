"""
FASE 4: Estudio fotográfico con muse-image-1.0 (API real) + control automático de fidelidad con muse-spark-1.3.

Por producto:
1. Máster de estudio (siempre): a partir del recorte de la Fase 3 y su frame original, muse-image entrega el producto
   completo (incluidas las partes que quedaron fuera del encuadre o tapadas por la mano), nítido y bien iluminado
   sobre fondo blanco. De él se saca un PNG maestro transparente con sam-3.1.
2. Vistas desde el máster: frente (fondo blanco), izquierda, derecha y atrás (vistas deducidas por IA, se marcan como
   tales) y ambiente (lifestyle).
3. Fidelidad: Spark compara el máster con la toma real y luego cada vista con el máster (mismo producto, nada agregado,
   ángulo correcto, encargo cumplido). Lo rechazado se regenera 1 vez con los problemas señalados; si sigue mal:
   frente → recorte real compuesto sobre blanco (sin IA), resto de vistas → se omiten.
   Si el máster se rechaza y la toma era mala (recorte rectangular o calidad < 4), no se inventa nada: se muestra la
   foto real y se recomienda volver a grabar.
Llamadas a Spark: como máximo 4 (máster, reintento de máster, vistas, reintento de vistas).
"""
import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.core.config import (
    MUSE_IMAGE_COST_USD,
    MUSE_IMAGE_MODEL,
    MUSE_IMAGE_SIZE,
    MUSE_SPARK_MODEL,
    STORAGE_DIR,
    load_session_state,
    save_session_state,
)
from backend.core import flows, prompts
from backend.core.muse_api import MuseAPIError, chat_json, edit_image
from backend.core.muse_client import CACHE_DIR, get_cached_result, set_cached_result
from backend.core.token_tracker import TokenTracker

PHASE4_PROMPT_VERSION = "phase4_v4"
FIDELITY_VERSION = "fidelity_v4"
NEUTRAL_DESC = "the product shown in the input image"
TILE = 340

ONLY_VISIBLE = (
    " Depict only the product: the product name may mention items that are not shown (for example an included accessory); "
    "do not add them. No hands, no people. The whole product must be fully visible inside the frame with some margin, never cropped."
)
MASTER_PROMPT = (
    "The first image is a product cut out from a phone video frame for an online store; the second image is the original "
    "frame for reference. Known problems: {problems}. Create a professional studio master photo of the SAME product, {desc}, "
    "centered on a seamless pure white background, front view as in the photo. Complete ONLY the parts that, in the original "
    "frame, are cut off by the frame edge or covered by the seller's hand or fingers; where the product itself ends (its base, "
    "bottom, top or sides), keep it exactly as it is: never extend it or add body parts, legs, bases or sections that the frame "
    "does not show. Remove any hand, fingers or background remnants, make it sharp with even studio lighting and accurate colors. "
    "Keep the product identical: same shape, proportions, colors, materials, printed text and logos." + ONLY_VISIBLE
)
VIEW_PROMPTS = {
    "front": (
        "Professional e-commerce catalog photo of EXACTLY this product, {desc}, front view, on a seamless pure white background, "
        "soft natural shadow underneath, centered, square framing, studio lighting. Keep the product identical: same shape, "
        "proportions, colors, materials, printed text and logos. Do not add or remove parts." + ONLY_VISIBLE
    ),
    "left": (
        "Professional e-commerce catalog photo of EXACTLY this product, {desc}, rotated to show its LEFT SIDE (camera moved about "
        "90 degrees to the product's left), on a seamless pure white background with a soft shadow, studio lighting. Keep the same "
        "design, colors, materials and proportions; infer the unseen side conservatively from the visible design, without adding "
        "new text, logos or features." + ONLY_VISIBLE
    ),
    "right": (
        "Professional e-commerce catalog photo of EXACTLY this product, {desc}, rotated to show its RIGHT SIDE (camera moved about "
        "90 degrees to the product's right), on a seamless pure white background with a soft shadow, studio lighting. Keep the same "
        "design, colors, materials and proportions; infer the unseen side conservatively from the visible design, without adding "
        "new text, logos or features." + ONLY_VISIBLE
    ),
    "back": (
        "Professional e-commerce catalog photo of EXACTLY this product, {desc}, rotated 180 degrees to show its BACK, on a seamless "
        "pure white background with a soft shadow, studio lighting. Keep the same design, colors, materials and proportions; infer "
        "the back conservatively from the visible design (plain where unknown), without adding new text, logos or features." + ONLY_VISIBLE
    ),
    "lifestyle": (
        "Place EXACTLY this product, {desc}, in a realistic, tasteful lifestyle setting that fits the store category '{category}', "
        "photographed for an online store with natural light; the product is the clear focus. Keep the product identical: same "
        "shape, proportions, colors, materials, printed text and logos. Scene props must not look like they are included with the "
        "product (no extra units, accessories of the same kind or packaging)." + ONLY_VISIBLE
    ),
}
RETRY_SUFFIX = " A previous attempt was rejected because: {issues}. Correct that and stay faithful to the real product."

VIEW_KINDS = ("front", "left", "right", "back", "lifestyle")
AI_ANGLE_KINDS = ("left", "right", "back")
KIND_TO_LABEL = {"master": "MASTER", "front": "FRONT", "left": "LEFT", "right": "RIGHT", "back": "BACK", "lifestyle": "LIFESTYLE"}
KIND_NAME_ES = {"master": "máster de estudio", "front": "frente", "left": "lado izquierdo", "right": "lado derecho", "back": "atrás", "lifestyle": "ambiente"}
FILE_NAMES = {
    "master": "{pid}_master_studio.png",
    "front": "{pid}_studio_white.jpg",
    "left": "{pid}_angle_left.jpg",
    "right": "{pid}_angle_right.jpg",
    "back": "{pid}_angle_back.jpg",
    "lifestyle": "{pid}_lifestyle.jpg",
}

FIDELITY_SYSTEM_PROMPT = """Controlas la calidad y fidelidad de fotos de catálogo generadas por IA. Por cada producto recibes una imagen con paneles rotulados. REFERENCE es el producto real (o su máster ya aprobado); los demás paneles fueron generados a partir de él. FRAME, si aparece, es el fotograma original completo del video: úsalo para distinguir las partes tapadas por la mano o cortadas por el borde del encuadre (que MASTER debe completar) del lugar donde el producto realmente termina (que MASTER no debe extender).
Para cada panel generado decide faithful = true solo si cumple TODO lo que aplica a su rótulo:
- Todos: es el mismo producto (misma forma, proporciones, colores, materiales, textos y logos visibles); no agrega accesorios, otra unidad ni objetos como parte del producto aunque el nombre los mencione; no hay manos ni personas; el producto entero se ve dentro del encuadre, sin quedar cortado.
- MASTER y FRONT: vista frontal sobre fondo blanco liso, nítida y bien iluminada. MASTER debe completar lo que en FRAME está tapado por la mano o cortado por el borde (un hueco con forma de dedo NO es la forma del producto), pero NO debe extender el producto más allá de lo que muestra FRAME (por ejemplo, agregar piernas, una base o secciones a un objeto que termina ahí).
- LEFT, RIGHT, BACK: vistas deducidas desde otro ángulo. Deben mostrar de verdad ese ángulo (no repetir ni espejar la vista frontal), sobre fondo blanco, y ser coherentes con el producto: mismos colores, materiales y estilo; pueden mostrar caras que no se ven en REFERENCE, pero sin textos, logos ni rasgos nuevos que lo contradigan.
- LIFESTYLE: el producto en un ambiente realista; los objetos del ambiente no parecen incluidos con el producto.
issues: qué falla y cómo, EN INGLÉS (se le pasará al generador para corregir); cadena vacía si faithful es true."""

IMAGE_HINT = "Escribe en inglés para mejores resultados. Ej.: 'Warm golden-hour light.' o 'Minimalist Scandinavian interior.'"
prompts.register(
    "p4_master", MASTER_PROMPT, phase=4, model="Muse Image", group="creative", kind="image",
    title="Foto máster de estudio",
    description="Limpia el recorte: completa lo que tapó la mano o cortó el borde y lo deja sobre fondo blanco. Es la base de todas las vistas.",
    required=("problems", "desc"), hint=IMAGE_HINT,
)
for _kind, _title, _desc in (
    ("front", "Vista frontal", "Foto de catálogo de frente, fondo blanco."),
    ("left", "Vista lado izquierdo", "Rota el producto ~90° para mostrar su lado izquierdo (deducido por IA)."),
    ("right", "Vista lado derecho", "Rota el producto ~90° para mostrar su lado derecho (deducido por IA)."),
    ("back", "Vista trasera", "Rota el producto 180° para mostrar la parte de atrás (deducida por IA)."),
    ("lifestyle", "Foto de ambiente", "Coloca el producto en un ambiente realista acorde a su categoría."),
):
    prompts.register(
        f"p4_view_{_kind}", VIEW_PROMPTS[_kind], phase=4, model="Muse Image", group="creative", kind="image",
        title=_title, description=_desc, required=("desc",), optional=("category",), hint=IMAGE_HINT,
    )
prompts.register(
    "p4_fidelity", FIDELITY_SYSTEM_PROMPT, phase=4, model="Muse Spark", group="advanced",
    title="Control de fidelidad",
    description="Compara cada imagen generada con el producto real y rechaza las que inventan partes, cambian colores o cortan el producto.",
)

FIDELITY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "checks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "enum": list(KIND_TO_LABEL.values())},
                                "faithful": {"type": "boolean"},
                                "issues": {"type": "string"},
                            },
                            "required": ["label", "faithful", "issues"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["product_id", "checks"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["products"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Utilidades de imagen (0 tokens)
# ---------------------------------------------------------------------------


def _on_white(rgba: np.ndarray, size: int = 1024, pad: float = 0.1) -> np.ndarray:
    """Producto centrado en un lienzo cuadrado blanco."""
    h, w = rgba.shape[:2]
    scale = size * (1 - 2 * pad) / max(h, w)
    img = cv2.resize(rgba, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.float32)
    oy, ox = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[oy : oy + img.shape[0], ox : ox + img.shape[1]]
    canvas[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img[:, :, :3] * alpha + roi * (1 - alpha)
    return canvas.astype(np.uint8)


def _local_studio_white(rgba: np.ndarray, size: int = 1024) -> np.ndarray:
    """Respaldo 100 % fiel: el recorte real sobre blanco con una sombra suave (sin IA)."""
    if rgba.ndim == 3 and rgba.shape[2] == 3:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA)
    h, w = rgba.shape[:2]
    scale = size * 0.78 / max(h, w)
    img = cv2.resize(rgba, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.float32)
    oy, ox = (size - img.shape[0]) // 2 - int(size * 0.03), (size - img.shape[1]) // 2
    shadow = np.zeros((size, size), dtype=np.float32)
    cy = min(size - 1, oy + img.shape[0])
    cv2.ellipse(shadow, (size // 2, cy), (int(img.shape[1] * 0.42), max(6, int(size * 0.025))), 0, 0, 360, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), size * 0.02) * 0.35
    canvas *= (1 - shadow)[..., None]
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[oy : oy + img.shape[0], ox : ox + img.shape[1]]
    canvas[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img[:, :, :3] * alpha + roi * (1 - alpha)
    return canvas.astype(np.uint8)


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def _jpeg(img: np.ndarray, q: int = 90) -> bytes:
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1].tobytes()


def _tile(img_bgr: np.ndarray, label: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = (TILE - 12) / max(h, w)
    img = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    board = np.full((TILE, TILE, 3), 235, dtype=np.uint8)
    oy, ox = (TILE - img.shape[0]) // 2, (TILE - img.shape[1]) // 2
    board[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img
    head = np.full((34, TILE, 3), (42, 23, 15), dtype=np.uint8)
    cv2.putText(head, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 189, 56), 2, cv2.LINE_AA)
    return np.vstack([head, board])


def _cut_edges(item: Dict[str, Any]) -> List[str]:
    """Bordes del frame que toca el producto (si los toca, quedó cortado por el encuadre)."""
    gl = item.get("glimmer_selection") or {}
    bbox = gl.get("bbox_pixels") or []
    frame_rel = gl.get("best_frame_rel_path")
    if len(bbox) != 4 or not frame_rel:
        return []
    frame = cv2.imread(str(STORAGE_DIR / frame_rel))
    if frame is None:
        return []
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    return [n for n, hit in (("left", x1 <= 2), ("top", y1 <= 2), ("right", x2 >= w - 2), ("bottom", y2 >= h - 2)) if hit]


def _problems_text(item: Dict[str, Any], edges: List[str]) -> str:
    # Solo hechos medibles: las notas libres del verificador de la Fase 3 ("falta la parte inferior") hacían que el
    # generador inventara partes (piernas a un busto de esqueleto).
    gl = item.get("glimmer_selection") or {}
    parts = []
    if edges:
        parts.append(f"the product touches the {', '.join(edges)} edge of the video frame and may be cut off there")
    diag = (gl.get("quality_diagnosis") or "").split(" | Recorte:")[0]
    if diag and diag != "Óptimo":
        parts.append(diag)
    parts.append("the seller's hand or fingers may cover part of it")
    return "; ".join(parts)


def _poor_source(item: Dict[str, Any]) -> bool:
    gl = item.get("glimmer_selection") or {}
    status = ((item.get("sam31_cutout") or {}).get("verification") or {}).get("status")
    return status == "RECTANGULAR_FALLBACK" or float(gl.get("quality_score") or 10) < 4


# ---------------------------------------------------------------------------
# muse-image con caché
# ---------------------------------------------------------------------------


def _generate(images: List[bytes], prompt: str, out_path: Path, output_format: str) -> Dict[str, Any]:
    """1 imagen con muse-image (caché MD5 por prompt + entradas). Nunca lanza: devuelve {'error'} si falla."""
    key = hashlib.md5(
        "|".join([MUSE_IMAGE_MODEL, PHASE4_PROMPT_VERSION, MUSE_IMAGE_SIZE, output_format, prompt] + [hashlib.md5(b).hexdigest() for b in images]).encode("utf-8")
    ).hexdigest()
    ext = "png" if output_format == "png" else "jpg"
    cached_file = CACHE_DIR / f"muse_image_{key}.{ext}"
    meta = get_cached_result("muse_image", key)
    t0 = time.time()
    if meta and cached_file.exists():
        out_path.write_bytes(cached_file.read_bytes())
        return {"path": out_path, "usage": meta.get("usage", {}), "from_cache": True, "seconds": 0.0}
    try:
        data, usage = edit_image(MUSE_IMAGE_MODEL, images, prompt, size=MUSE_IMAGE_SIZE, output_format=output_format)
    except MuseAPIError as exc:
        return {"error": str(exc)[:400], "seconds": round(time.time() - t0, 1)}
    cached_file.write_bytes(data)
    set_cached_result("muse_image", key, {"usage": usage, "prompt": prompt})
    out_path.write_bytes(data)
    return {"path": out_path, "usage": usage, "from_cache": False, "seconds": round(time.time() - t0, 1)}


def _run_jobs(tracker: TokenTracker, jobs: List[Dict[str, Any]]) -> None:
    """Ejecuta trabajos de imagen en paralelo y registra cada uno en el TokenTracker."""
    if not jobs:
        return

    def run(job: Dict[str, Any]) -> Dict[str, Any]:
        res = _generate(job["images"], job["prompt"], job["out_path"], job["format"])
        if "content_policy_violation" in (res.get("error") or "") and job.get("neutral_prompt"):
            # Marcas o personajes en el nombre pueden activar el filtro: se reintenta sin nombrar el producto.
            first_error = res["error"]
            res = _generate(job["images"], job["neutral_prompt"], job["out_path"], job["format"])
            res["policy_retry"] = True
            if res.get("error"):
                res["error"] = f"{first_error[:200]} | sin nombre: {res['error'][:200]}"
        return res

    with ThreadPoolExecutor(max_workers=max(1, int(flows.get("images.parallel")))) as pool:
        results = list(pool.map(run, jobs))
    for job, res in zip(jobs, results):
        job["result"] = res
        usage = res.get("usage") or {}
        name = f"{job['pid']} {KIND_NAME_ES[job['kind']]}{' (reintento)' if job.get('retry') else ''}"
        if res.get("error"):
            tracker.record_event(phase=4, model="Muse Image", operation=f"{name}: falló ({res['error'][:160]}).", status="API_ERROR")
        elif res["from_cache"]:
            tracker.record_event(
                phase=4,
                model="Muse Image",
                operation=f"Caché MD5 HIT: {name} sin llamar a la API (ahorra US${MUSE_IMAGE_COST_USD:.2f}).",
                status="CACHE_HIT_MD5",
                details={"cost_usd_saved": MUSE_IMAGE_COST_USD},
            )
        else:
            tracker.record_event(
                phase=4,
                model="Muse Image",
                operation=(
                    f"{MUSE_IMAGE_MODEL}: {name}{' (sin nombre: la política de contenido bloqueó el nombre)' if res.get('policy_retry') else ''} "
                    f"en {res['seconds']}s. Cobra US${MUSE_IMAGE_COST_USD:.2f} por imagen, no por token "
                    f"(tokens de referencia: in={usage.get('input_tokens', 0)}, out={usage.get('output_tokens', 0)})."
                ),
                status="API_CALL",
                details={"cost_usd": MUSE_IMAGE_COST_USD, "usage_reference_only": usage},
            )


# ---------------------------------------------------------------------------
# Fidelidad con muse-spark-1.3
# ---------------------------------------------------------------------------


def _check_fidelity(tracker: TokenTracker, entries: List[Dict[str, Any]], round_label: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """entries: [{pid, name, reference (BGR), generated: {kind: path}}]. Devuelve {(pid, kind): {faithful, issues}}."""
    entries = [e for e in entries if e["generated"]]
    if not entries:
        return {}
    content: List[Dict[str, Any]] = []
    digests = []
    for e in entries:
        tiles = [_tile(e["reference"], "REFERENCE")]
        if e.get("frame") is not None:
            tiles.append(_tile(e["frame"], "FRAME"))
        for kind, path in e["generated"].items():
            tiles.append(_tile(cv2.imread(str(path)), KIND_TO_LABEL[kind]))
        strip = _jpeg(np.hstack(tiles), 88)
        digests.append([e["pid"], hashlib.md5(strip).hexdigest()])
        panels = ["REFERENCE"] + (["FRAME"] if e.get("frame") is not None else []) + [KIND_TO_LABEL[k] for k in e["generated"]]
        content.append({"type": "text", "text": f"product_id: {e['pid']}\nProducto: {e['name']}\nPaneles: " + ", ".join(panels)})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(strip).decode("ascii")}})

    cache_key = hashlib.md5((json.dumps([MUSE_SPARK_MODEL, FIDELITY_VERSION, digests]) + prompts.signature("p4_fidelity")).encode("utf-8")).hexdigest()
    cached = get_cached_result("phase4_fidelity", cache_key)
    n_images = sum(len(e["generated"]) for e in entries)
    if cached:
        parsed, usage = cached["parsed"], cached["usage"]
        tracker.record_event(
            phase=4,
            model="Muse Spark (Fidelidad)",
            operation=f"Caché MD5 HIT: {round_label} sin llamar a la API.",
            tokens_saved=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            status="CACHE_HIT_MD5",
        )
    else:
        try:
            parsed, usage, _ = chat_json(
                model=MUSE_SPARK_MODEL,
                messages=[{"role": "system", "content": prompts.get("p4_fidelity")}, {"role": "user", "content": content}],
                schema_name="Phase4Fidelity",
                schema=FIDELITY_SCHEMA,
                reasoning_effort="low",
            )
        except MuseAPIError as exc:
            # Sin control no se pierde lo generado: las imágenes quedan marcadas como "sin verificar".
            tracker.record_event(phase=4, model="Muse Spark (Fidelidad)", operation=f"{round_label}: falló ({str(exc)[:160]}); las imágenes quedan sin verificar.", status="API_ERROR")
            return {(e["pid"], k): {"faithful": True, "issues": "", "unverified": True} for e in entries for k in e["generated"]}
        set_cached_result("phase4_fidelity", cache_key, {"parsed": parsed, "usage": usage})
        tracker.record_event(
            phase=4,
            model="Muse Spark (Fidelidad)",
            operation=f"{MUSE_SPARK_MODEL}: {round_label} ({n_images} imagen(es) de {len(entries)} producto(s); razonamiento 'low': {usage['reasoning_tokens']} tokens).",
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            status="API_CALL",
            details={"usage": usage},
        )
    label_to_kind = {v: k for k, v in KIND_TO_LABEL.items()}
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in parsed.get("products", []):
        for c in p.get("checks", []):
            kind = label_to_kind.get(c.get("label"))
            if kind:
                out[(p["product_id"], kind)] = {"faithful": bool(c.get("faithful")), "issues": c.get("issues", "")}
    return out


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def master_png_job(args: Tuple[np.ndarray, Dict[str, Any], Path]) -> bool:
    return _master_png(*args)


def _master_png(master_bgr: np.ndarray, item: Dict[str, Any], out_path: Path) -> bool:
    """PNG transparente del máster (muse-image no da transparencia): sam-3.1 sobre el fondo blanco."""
    from backend.services.phase3_vision_sam import cutout_rgba, segment_product

    non_white = np.any(master_bgr < 235, axis=2)
    if not non_white.any():
        return False
    ys, xs = np.nonzero(non_white)
    box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    used = (item.get("sam31_cutout") or {}).get("sam_prompt_used") or ""
    prompts = [p.strip() for p in used.split("+") if p.strip()] or ["object"]
    seg = segment_product(master_bgr, box, prompts)
    if seg["instance"] is None:
        return False
    rgba, _ = cutout_rgba(master_bgr, seg["instance"]["mask"], None)
    cv2.imwrite(str(out_path), rgba)
    return True


async def run_phase4_studio(session_id: Optional[str] = None) -> Dict[str, Any]:
    state = load_session_state(session_id)
    if not state or not state.get("session_id"):
        raise ValueError("No existe una sesión activa. Ejecuta las Fases 1, 2 y 3 primero.")
    if int(state.get("current_phase_completed", 0)) < 3:
        raise ValueError("La Fase 3 debe completarse antes de ejecutar la Fase 4.")
    grounded: List[Dict[str, Any]] = state.get("grounded_products", [])
    if not grounded:
        raise ValueError("No hay productos recortados en la Fase 3 para el estudio.")

    sid = state["session_id"]
    studio_dir = STORAGE_DIR / sid / "studio"
    studio_dir.mkdir(parents=True, exist_ok=True)
    tracker = TokenTracker(state.get("token_audit"))

    ctx: Dict[str, Dict[str, Any]] = {}
    for item in grounded:
        pid = item["product_id"]
        cut_rel = (item.get("sam31_cutout") or {}).get("cutout_rel_path") or f"{sid}/cutouts/{pid}_cutout.png"
        rgba = cv2.imread(str(STORAGE_DIR / cut_rel), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            tracker.record_event(phase=4, model="Muse Image", operation=f"{pid}: no se encontró su recorte de la Fase 3; se omite.", status="SKIPPED")
            continue
        if rgba.shape[2] == 3:
            rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA)
        frame_rel = (item.get("glimmer_selection") or {}).get("best_frame_rel_path")
        edges = _cut_edges(item)
        if _poor_source(item):
            # La Fase 3 ya juzgó que la toma no deja ver el producto: cualquier imagen generada sería inventada.
            tracker.record_event(
                phase=4,
                model="Estudio Fase 4",
                operation=f"{pid}: toma insuficiente según la Fase 3 (recorte rectangular o calidad < 4); no se genera ninguna imagen con IA, se muestra la foto real.",
                status="ZERO_TOKEN_LOCAL",
            )
        ctx[pid] = {
            "item": item,
            "rgba": rgba,
            "original": _on_white(rgba),
            "frame": cv2.imread(str(STORAGE_DIR / frame_rel)) if frame_rel else None,
            # Solo el nombre: visual_description describe la escena del video (mano, pared...) y el generador la copiaba.
            "desc": f"\"{item.get('resolved_name', pid)}\"",
            "category": item.get("visual_category", "General"),
            "cut_rel": cut_rel,
            "edges": edges,
            "problems": _problems_text(item, edges),
            "paths": {},
            "fidelity": {},
            "low_quality_source": _poor_source(item),
        }

    def path_for(pid: str, kind: str) -> Path:
        return studio_dir / FILE_NAMES[kind].format(pid=pid)

    def make_job(pid: str, kind: str, issues: str = "") -> Dict[str, Any]:
        c = ctx[pid]
        if kind == "master":
            images = [_png(c["original"])] + ([_jpeg(c["frame"], 92)] if c["frame"] is not None else [])
            template = prompts.get("p4_master")
            prompt = template.format(problems=c["problems"], desc=c["desc"])
            neutral = template.format(problems=c["problems"], desc=NEUTRAL_DESC)
            fmt = "png"
        else:
            images = [_png(c["base"])]
            template = prompts.get(f"p4_view_{kind}")
            prompt = template.format(desc=c["desc"], category=c["category"])
            neutral = template.format(desc=NEUTRAL_DESC, category=c["category"])
            fmt = "jpeg"
        if issues:
            prompt += RETRY_SUFFIX.format(issues=issues)
            neutral += RETRY_SUFFIX.format(issues=issues)
        return {"pid": pid, "kind": kind, "images": images, "prompt": prompt, "neutral_prompt": neutral, "out_path": path_for(pid, kind), "format": fmt, "retry": bool(issues)}

    def collect(jobs: List[Dict[str, Any]]) -> None:
        for job in jobs:
            ctx[job["pid"]]["paths"][job["kind"]] = None if job["result"].get("error") else job["out_path"]

    def store_verdicts(verdicts: Dict[Tuple[str, str], Dict[str, Any]], round_no: int) -> None:
        for (pid, kind), v in verdicts.items():
            prev = ctx[pid]["fidelity"].get(kind) or {}
            entry = {**v, "round": round_no}
            if round_no == 2:
                entry["round1_issues"] = prev.get("issues") or "error de generación"
            ctx[pid]["fidelity"][kind] = entry

    def failed(pid: str, kind: str) -> bool:
        return not ctx[pid]["paths"].get(kind) or ctx[pid]["fidelity"].get(kind, {}).get("faithful") is False

    all_jobs: List[Dict[str, Any]] = []
    # Flujo activo: control de fidelidad opcional y vistas encendidas/apagadas (el frente siempre se genera).
    fidelity_on = bool(flows.get("fidelity.enabled"))
    view_options = {"left": "view.left", "right": "view.right", "back": "view.back", "lifestyle": "view.lifestyle"}

    # 1. Máster de estudio para todos los productos
    jobs = [make_job(pid, "master") for pid, c in ctx.items() if not c["low_quality_source"]]
    _run_jobs(tracker, jobs)
    collect(jobs)
    all_jobs += jobs
    entries = [
        {"pid": pid, "name": c["item"].get("resolved_name", pid), "reference": c["original"], "frame": c["frame"], "generated": {"master": c["paths"]["master"]} if c["paths"].get("master") else {}}
        for pid, c in ctx.items()
        if not c["low_quality_source"]
    ]
    if fidelity_on:
        store_verdicts(_check_fidelity(tracker, entries, "control del máster de estudio (ronda 1)"), 1)
    retry = [make_job(pid, "master", ctx[pid]["fidelity"].get("master", {}).get("issues") or "generation failed") for pid, c in ctx.items() if not c["low_quality_source"] and failed(pid, "master")]
    if retry:
        _run_jobs(tracker, retry)
        collect(retry)
        all_jobs += retry
        entries = [
            {"pid": j["pid"], "name": ctx[j["pid"]]["item"].get("resolved_name", j["pid"]), "reference": ctx[j["pid"]]["original"], "frame": ctx[j["pid"]]["frame"], "generated": {"master": j["out_path"]} if not j["result"].get("error") else {}}
            for j in retry
        ]
        if fidelity_on:
            store_verdicts(_check_fidelity(tracker, entries, "control del máster de estudio (ronda 2, solo reintentos)"), 2)

    for pid, c in ctx.items():
        if c["low_quality_source"]:
            continue
        if not failed(pid, "master"):
            c["base"] = cv2.imread(str(c["paths"]["master"]))
            c["master_ok"] = True
        else:
            c["base"] = c["original"]

    # 2. Vistas desde el máster (o desde el recorte real si el máster no pasó)
    kinds = [k for k in VIEW_KINDS if k == "front" or flows.get(view_options[k])]
    jobs = [make_job(pid, k) for pid, c in ctx.items() if not c.get("low_quality_source") for k in kinds]
    _run_jobs(tracker, jobs)
    collect(jobs)
    all_jobs += jobs
    view_pids = [pid for pid, c in ctx.items() if not c.get("low_quality_source")]
    entries = [
        {"pid": pid, "name": ctx[pid]["item"].get("resolved_name", pid), "reference": ctx[pid]["base"], "generated": {k: ctx[pid]["paths"][k] for k in kinds if ctx[pid]["paths"].get(k)}}
        for pid in view_pids
    ]
    if fidelity_on:
        store_verdicts(_check_fidelity(tracker, entries, "control de las vistas (ronda 1)"), 1)
    retry = [make_job(pid, k, ctx[pid]["fidelity"].get(k, {}).get("issues") or "generation failed") for pid in view_pids for k in kinds if failed(pid, k)]
    if retry:
        _run_jobs(tracker, retry)
        collect(retry)
        all_jobs += retry
        by_pid: Dict[str, Dict[str, Path]] = {}
        for j in retry:
            if not j["result"].get("error"):
                by_pid.setdefault(j["pid"], {})[j["kind"]] = j["out_path"]
        entries = [{"pid": pid, "name": ctx[pid]["item"].get("resolved_name", pid), "reference": ctx[pid]["base"], "generated": gen} for pid, gen in by_pid.items()]
        if fidelity_on:
            store_verdicts(_check_fidelity(tracker, entries, "control de las vistas (ronda 2, solo reintentos)"), 2)

    # 3. Salida (PNG maestros con sam-3.1 en paralelo)
    from backend.services.phase3_vision_sam import run_parallel

    master_pids = [pid for pid, c in ctx.items() if c.get("master_ok")]
    master_ok = run_parallel(master_png_job, [(ctx[pid]["base"], ctx[pid]["item"], studio_dir / f"{pid}_master.png") for pid in master_pids])
    master_png_done = dict(zip(master_pids, master_ok))
    rel = lambda p: p.relative_to(STORAGE_DIR).as_posix()
    images_api = sum(1 for j in all_jobs if not j["result"].get("error") and not j["result"].get("from_cache"))
    counts = {"masters": 0, "front_ai": 0, "front_local": 0, "angles": 0, "angles_omitted": 0, "lifestyle": 0, "lifestyle_omitted": 0, "low_quality_source": 0}
    studio_products: List[Dict[str, Any]] = []
    for pid, c in ctx.items():
        item, fid = c["item"], c["fidelity"]
        low = bool(c.get("low_quality_source"))
        counts["low_quality_source"] += int(low)
        master_rel = c["cut_rel"]
        if c.get("master_ok"):
            counts["masters"] += 1
            png_path = studio_dir / f"{pid}_master.png"
            if master_png_done.get(pid):
                master_rel = rel(png_path)

        views: Dict[str, Optional[str]] = {}
        for kind in VIEW_KINDS:
            path = path_for(pid, kind)
            if kind not in kinds:
                # Vista apagada en el flujo activo: no se genera y se borra la de una ejecución anterior.
                if path.exists():
                    path.unlink()
                views[kind] = None
                counts["views_disabled"] = counts.get("views_disabled", 0) + 1
                continue
            ok = not low and not failed(pid, kind) and bool(c["paths"].get(kind))
            if kind == "front" and not ok:
                cv2.imwrite(str(path), _local_studio_white(cv2.imread(str(STORAGE_DIR / master_rel), cv2.IMREAD_UNCHANGED)), [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                counts["front_local"] += 1
                views[kind] = rel(path)
                continue
            if not ok:
                if path.exists():
                    path.unlink()
                views[kind] = None
                if kind == "lifestyle":
                    counts["lifestyle_omitted"] += 1
                elif kind in AI_ANGLE_KINDS:
                    counts["angles_omitted"] += 1
                continue
            views[kind] = rel(path)
            counts["front_ai" if kind == "front" else "lifestyle" if kind == "lifestyle" else "angles"] += 1

        url = lambda r: f"/storage/{r}" if r else None
        studio_info = {
            "step_4a_enhancement": {
                "requested": not low,
                "executed": bool(c.get("master_ok")),
                "rejected_by_fidelity_check": not low and not c.get("master_ok"),
                "problems_addressed": c["problems"],
                "cut_by_frame_edges": c["edges"],
                "low_quality_source": low,
                "recommendation": "La toma del video no permite ver el producto con claridad: conviene volver a grabarlo." if low else "",
                "restored_image_url": url(rel(c["paths"]["master"])) if c.get("master_ok") else None,
                "source_cutout_url": url(c["cut_rel"]),
                "result_cutout_url": url(master_rel),
                "result_cutout_path": master_rel,
            },
            "step_4b_variations": {
                "studio_white_url": url(views["front"]),
                "studio_white_path": views["front"],
                "studio_white_source": "MUSE_IMAGE" if (not low and not failed(pid, "front")) else "LOCAL_REAL_CUTOUT",
                "angle_left_url": url(views["left"]),
                "angle_right_url": url(views["right"]),
                "angle_back_url": url(views["back"]),
                "ai_inferred_views": ["left", "right", "back"],
                "lifestyle_url": url(views["lifestyle"]),
                "lifestyle_path": views["lifestyle"],
                "resolution": MUSE_IMAGE_SIZE,
            },
            "fidelity_checks": fid,
        }
        studio_products.append({**item, "studio_imagen": studio_info})

    cost = round(images_api * MUSE_IMAGE_COST_USD, 2)
    tracker.record_event(
        phase=4,
        model="Estudio Fase 4",
        operation=(
            f"{len(studio_products)} producto(s): {counts['masters']} máster(es) de estudio aprobados; frente con IA {counts['front_ai']} "
            f"(+{counts['front_local']} con el recorte real); {counts['angles']} vistas laterales/trasera aprobadas ({counts['angles_omitted']} omitidas); "
            f"{counts['lifestyle']} ambiente(s) ({counts['lifestyle_omitted']} omitidos); {counts['low_quality_source']} con toma insuficiente. "
            f"{images_api} imagen(es) generadas = US${cost:.2f}."
        ),
        status="ZERO_TOKEN_LOCAL",
        details={"images_generated_api": images_api, "cost_usd": cost, **counts},
    )

    state["grounded_products"] = studio_products
    state["studio_products"] = studio_products
    state["phase4_summary"] = {
        "total_products_processed": len(studio_products),
        "enhanced_in_4a_count": counts["masters"],
        "skipped_4a_count": len(studio_products) - counts["masters"],
        "white_backgrounds_ai": counts["front_ai"],
        "white_backgrounds_local_fallback": counts["front_local"],
        "angle_views": counts["angles"],
        "angle_views_omitted": counts["angles_omitted"],
        "lifestyle_images": counts["lifestyle"],
        "lifestyle_omitted": counts["lifestyle_omitted"],
        "low_quality_sources": counts["low_quality_source"],
        "images_generated_api": images_api,
        "cost_usd": cost,
        "catalog_variations_generated": counts["front_ai"] + counts["front_local"] + counts["angles"] + counts["lifestyle"],
    }
    state["current_phase_completed"] = max(4, int(state.get("current_phase_completed", 3)))
    state["status"] = "PHASE_4_COMPLETED"
    tracker.sync_to_session(state)
    save_session_state(state)
    return state
