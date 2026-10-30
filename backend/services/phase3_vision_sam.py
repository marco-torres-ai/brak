"""
FASE 3: Identificación visual + mejor frame (muse-spark-1.3, reemplaza a Muse Glimmer) y recorte (sam-3.1).

Flujo por sesión:
1. Ventana de cada producto: intervalo en que el vendedor lo muestra según el inventario visual de la Fase 2
   (o la ventana hablada ampliada si no hay coincidencia).
2. 3 frames candidatos: el más nítido de cada tercio de la ventana (0 tokens, OpenCV).
3. Collage 1x3 por producto (1 sola imagen) → 1 llamada a muse-spark-1.3 por lote de productos: elige el mejor
   frame, nombra y describe el objeto, puntúa la calidad, da su caja y frases cortas para SAM.
4. sam-3.1 sobre el mejor frame a resolución completa: prueba las frases de Spark y se queda con la máscara que más
   coincide con la caja de Spark (IoU); si ninguna coincide, usa la caja de Spark como prompt de caja.
5. PNG RGBA recortado a la máscara.
"""
import base64
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    FramePrompt,
    PromptBox,
    PromptObject,
    build_object_prompt,
    decode_mask_to_raster,
    image_segmentation_format,
)

from backend.core.config import (
    MUSE_PHASE3_MIN_IOU,
    MUSE_PHASE3_VERIFY_CUTOUTS,
    MUSE_SAM_MODEL,
    MUSE_SPARK_MODEL,
    MUSE_SPARK_REASONING_PHASE3,
    STORAGE_DIR,
    get_session_dir,
    load_session_state,
    save_session_state,
)
from backend.core.muse_api import chat_json, segment_image_sam
from backend.core.muse_client import get_cached_result, locate_source_video, set_cached_result
from backend.core.token_tracker import TokenTracker
from backend.services.phase2_verify import name_similarity
from backend.services.phase3_verify import verify_and_repair_cutouts

PHASE3_PROMPT_VERSION = "phase3_v2"
MIN_SAM_CONFIDENCE = 0.3
SPARK_BATCH_SIZE = 8
SAMPLE_STEP_SEC = 0.25
INVENTORY_PAD_SEC = 0.5
MAX_SAM_IMAGE_SIDE = 1536
MAX_SAM_TEXT_ATTEMPTS = 3
COLLAGE_MAX_WIDTH = 1600
COLLAGE_MAX_PANEL_HEIGHT = 560

VISION_SYSTEM_PROMPT = """Eres el módulo de visión de un pipeline que convierte videos de venta en un catálogo e-commerce.
Por cada producto recibes: lo que dijo el vendedor (frase, nombre detectado, tipo de mención) y un collage con 3 fotogramas del momento en que lo muestra, numerados 1, 2 y 3 de izquierda a derecha.
Para cada producto:
1. Identifica el objeto que se vende. En menciones DEICTIC ("esto", "este de acá") es el que el vendedor sostiene, señala o acerca a la cámara en ese momento; ignora objetos del fondo.
2. slot_evaluations: por cada fotograma (1, 2, 3) indica si el objeto se ve completo (object_visible), una nota de 0 a 10 para usarlo como foto de catálogo (nitidez, luz, encuadre, oclusión por la mano) y una nota breve.
3. best_frame_slot: el fotograma con mejor nota en el que el objeto se ve.
4. resolved_name: nombre comercial en español, específico (tipo, color, marca visible). visual_category: categoría corta de tienda en español (por ejemplo "Juguetes y coleccionables", "Hogar y decoración", "Electrónica", "Accesorios"). visual_description: 1 o 2 frases en español con lo que se ve. disambiguation_note: por qué ese es el objeto ofrecido y qué otros objetos visibles se descartaron.
5. quality_score (0 a 10) del mejor fotograma para el catálogo; needs_enhancement = true si tiene desenfoque, poca luz, reflejos fuertes u oclusión que convenga restaurar; quality_diagnosis: el problema principal o "Óptimo".
6. bbox: caja del objeto DENTRO DEL PANEL del mejor fotograma, en coordenadas normalizadas 0 a 1000 sobre la imagen completa del collage que recibes (ymin, xmin, ymax, xmax). Debe abarcar el objeto completo y no la mano que lo sostiene.
7. sam_prompts: de 2 a 4 frases nominales cortas EN INGLÉS para un segmentador que solo entiende sustantivos comunes, de la más específica a la más genérica (por ejemplo "action figure", "toy"; "sunglasses", "glasses"; "spray bottle", "bottle"). Sin marcas, sin posiciones, sin listas ni verbos.
Devuelve un elemento por producto con su product_id exacto."""

VISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "slot_evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "slot": {"type": "integer"},
                                "object_visible": {"type": "boolean"},
                                "score": {"type": "number"},
                                "note": {"type": "string"},
                            },
                            "required": ["slot", "object_visible", "score", "note"],
                            "additionalProperties": False,
                        },
                    },
                    "best_frame_slot": {"type": "integer"},
                    "resolved_name": {"type": "string"},
                    "visual_category": {"type": "string"},
                    "visual_description": {"type": "string"},
                    "disambiguation_note": {"type": "string"},
                    "quality_score": {"type": "number"},
                    "needs_enhancement": {"type": "boolean"},
                    "quality_diagnosis": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "ymin": {"type": "number"},
                            "xmin": {"type": "number"},
                            "ymax": {"type": "number"},
                            "xmax": {"type": "number"},
                        },
                        "required": ["ymin", "xmin", "ymax", "xmax"],
                        "additionalProperties": False,
                    },
                    "sam_prompts": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "product_id",
                    "slot_evaluations",
                    "best_frame_slot",
                    "resolved_name",
                    "visual_category",
                    "visual_description",
                    "disambiguation_note",
                    "quality_score",
                    "needs_enhancement",
                    "quality_diagnosis",
                    "bbox",
                    "sam_prompts",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["products"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# 1-2. Ventana y frames candidatos (0 tokens)
# ---------------------------------------------------------------------------


def resolve_showcase_window(
    product: Dict[str, Any], inventory: List[Dict[str, Any]], duration_sec: float
) -> Tuple[float, float, str]:
    """Intervalo en que se muestra el producto: el del inventario visual que mejor coincide, o la ventana ampliada."""
    spoken = product.get("spoken_timestamps") or {}
    expanded = product.get("expanded_visual_window") or {}
    w_start = float(expanded.get("start_sec", spoken.get("start_sec", 0.0)))
    w_end = float(expanded.get("end_sec", spoken.get("end_sec", w_start + 3.0)))

    names = [product.get("detected_name", ""), product.get("visual_item_name", ""), product.get("spoken_item_name", "")]
    best, best_score = None, 0.0
    for item in inventory:
        if item.get("video_id") != product.get("video_id"):
            continue
        sim = max(name_similarity(n, item.get("object_name", "")) for n in names if n) if any(names) else 0.0
        overlap = item["start_sec"] <= w_end and w_start <= item["end_sec"]
        score = sim + (0.5 if overlap else 0.0)
        if score > best_score:
            best, best_score = item, score

    if best is not None and best_score >= 0.6:
        start, end, source = best["start_sec"] - INVENTORY_PAD_SEC, best["end_sec"] + INVENTORY_PAD_SEC, "VISUAL_INVENTORY"
    else:
        start, end, source = w_start, w_end, "SPOKEN_WINDOW_EXPANDED"

    if duration_sec > 0:
        start, end = max(0.0, start), min(duration_sec, end)
    if end - start < 0.75:
        mid = (start + end) / 2
        start, end = max(0.0, mid - 0.75), mid + 0.75
        if duration_sec > 0:
            end = min(duration_sec, end)
    return round(start, 2), round(end, 2), source


def _sharpness(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    scale = 320.0 / max(1, w)
    small = cv2.resize(frame, (320, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def extract_candidate_frames(
    video_path: Path, start: float, end: float, product_id: str, frames_dir: Path, session_id: str
) -> List[Dict[str, Any]]:
    """El frame más nítido de cada tercio de la ventana, a resolución original y sin retoques."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    step = max(1, int(round(fps * SAMPLE_STEP_SEC)))
    first = int(start * fps)
    last = int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    bins: List[Optional[Tuple[float, int, np.ndarray]]] = [None, None, None]
    span = max(1, last - first)
    idx = first
    while idx <= last:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (idx - first) % step == 0:
            b = min(2, int(3 * (idx - first) / (span + 1)))
            score = _sharpness(frame)
            if bins[b] is None or score > bins[b][0]:
                bins[b] = (score, idx, frame.copy())
        idx += 1
    cap.release()

    candidates = []
    for score, f_idx, frame in [b for b in bins if b is not None]:
        ts = round(f_idx / fps, 2)
        fname = f"{product_id}_cand_f{f_idx:05d}_{int(ts * 1000)}ms.jpg"
        cv2.imwrite(str(frames_dir / fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        candidates.append(
            {
                "frame_id": f"{product_id}_f{f_idx:05d}",
                "frame_index": f_idx,
                "timestamp_sec": ts,
                "laplacian_score": round(score, 2),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "file_path": f"{session_id}/frames/{fname}",
                "url": f"/storage/{session_id}/frames/{fname}",
            }
        )
    return candidates


def build_collage(candidates: List[Dict[str, Any]], out_path: Path) -> Tuple[str, Dict[str, Any]]:
    """Collage horizontal numerado que respeta la proporción de cada frame. Devuelve (MD5, posición de cada panel)."""
    images = [cv2.imread(str(STORAGE_DIR / c["file_path"])) for c in candidates]
    aspects = [img.shape[1] / img.shape[0] for img in images]
    panel_h = int(min(COLLAGE_MAX_PANEL_HEIGHT, COLLAGE_MAX_WIDTH / sum(aspects)))
    sep = np.full((panel_h, 6, 3), 30, dtype=np.uint8)
    parts: List[np.ndarray] = []
    panels: List[Dict[str, int]] = []
    x_cursor = 0
    for slot, (img, cand) in enumerate(zip(images, candidates), start=1):
        panel = cv2.resize(img, (max(1, int(panel_h * img.shape[1] / img.shape[0])), panel_h), interpolation=cv2.INTER_AREA)
        label = f"{slot}  t={cand['timestamp_sec']}s"
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 30 + 13 * len(label)), 40), (15, 23, 42), -1)
        cv2.putText(panel, label, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (56, 189, 248), 2, cv2.LINE_AA)
        if parts:
            parts.append(sep)
            x_cursor += sep.shape[1]
        panels.append({"x0": x_cursor, "width": panel.shape[1]})
        x_cursor += panel.shape[1]
        parts.append(panel)
    collage = np.hstack(parts)
    cv2.imwrite(str(out_path), collage, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    layout = {"width": int(collage.shape[1]), "height": int(collage.shape[0]), "panels": panels}
    return hashlib.md5(out_path.read_bytes()).hexdigest(), layout


# ---------------------------------------------------------------------------
# 3. muse-spark-1.3: mejor frame, nombre, calidad, caja y frases para SAM
# ---------------------------------------------------------------------------


def _product_context(prod: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "product_id": prod["product_id"],
        "video_id": prod["video_id"],
        "mention_type": prod.get("mention_type"),
        "detected_name": prod.get("detected_name"),
        "spoken_phrase": prod.get("interpreted_phrase") or prod.get("raw_spoken_phrase"),
    }


def analyze_with_spark_vision(
    tracker: TokenTracker, batch: List[Tuple[Dict[str, Any], Path, str]]
) -> Dict[str, Dict[str, Any]]:
    """1 llamada a Spark para un lote de (producto, collage, md5). Devuelve el resultado por product_id."""
    contexts = [_product_context(p) for p, _, _ in batch]
    key_src = json.dumps(
        {
            "model": MUSE_SPARK_MODEL,
            "prompt": PHASE3_PROMPT_VERSION,
            "reasoning": MUSE_SPARK_REASONING_PHASE3,
            "items": [[c, md5] for c, (_, _, md5) in zip(contexts, batch)],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    cache_key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
    ids = ", ".join(c["product_id"] for c in contexts)

    cached = get_cached_result("phase3_spark_vision", cache_key)
    if cached:
        usage = cached["usage"]
        tracker.record_event(
            phase=3,
            model="Muse Spark (Visión)",
            operation=f"Caché MD5 HIT del análisis visual de {ids}: se reutiliza sin llamar a la API.",
            tokens_saved=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            status="CACHE_HIT_MD5",
        )
        return {r["product_id"]: r for r in cached["products"]}

    content: List[Dict[str, Any]] = []
    for ctx, (_, collage_path, _) in zip(contexts, batch):
        content.append({"type": "text", "text": "Producto:\n" + json.dumps(ctx, ensure_ascii=False)})
        b64 = base64.b64encode(collage_path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    parsed, usage, _ = chat_json(
        model=MUSE_SPARK_MODEL,
        messages=[{"role": "system", "content": VISION_SYSTEM_PROMPT}, {"role": "user", "content": content}],
        schema_name="Phase3VisionSelection",
        schema=VISION_SCHEMA,
        reasoning_effort=MUSE_SPARK_REASONING_PHASE3,
    )
    results = parsed.get("products", [])
    set_cached_result("phase3_spark_vision", cache_key, {"products": results, "usage": usage})
    tracker.record_event(
        phase=3,
        model="Muse Spark (Visión)",
        operation=(
            f"{MUSE_SPARK_MODEL}: 1 llamada con {len(batch)} collage(s) 1x3 ({ids}): mejor frame, nombre, calidad, "
            f"caja y frases para SAM (razonamiento '{MUSE_SPARK_REASONING_PHASE3}': {usage['reasoning_tokens']} tokens)."
        ),
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        status="API_CALL",
        details={"products": [c["product_id"] for c in contexts], "usage": usage},
    )
    return {r["product_id"]: r for r in results}


# ---------------------------------------------------------------------------
# 4. sam-3.1
# ---------------------------------------------------------------------------


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _sam_instances(image_jpeg: bytes, image_md5: str, prompt: str, size: Tuple[int, int]) -> Tuple[List[Dict[str, Any]], bool]:
    """Llama a SAM (con caché MD5). Devuelve (instancias con su máscara a tamaño completo, si vino de la caché)."""
    cache_key = hashlib.md5(f"{MUSE_SAM_MODEL}|{image_md5}|{prompt}".encode("utf-8")).hexdigest()
    cached = get_cached_result("sam31_api", cache_key)
    if cached is not None:
        text, from_cache = cached["text"], True
    else:
        text, _ = segment_image_sam(MUSE_SAM_MODEL, image_jpeg, prompt)
        set_cached_result("sam31_api", cache_key, {"text": text, "prompt": prompt})
        from_cache = False

    parser = image_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    w, h = size
    instances: Dict[str, Dict[str, Any]] = {}
    for rec in result.records:
        if rec.kind == "box":
            instances.setdefault(rec.object_id, {})["box"] = (rec.left, rec.top, rec.right, rec.bottom)
            instances[rec.object_id]["confidence"] = rec.confidence
        elif rec.kind == "mask":
            raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(rec.mask.height, rec.mask.width)
            l, t, r, b = (int(rec.bounds.left), int(rec.bounds.top), int(rec.bounds.right), int(rec.bounds.bottom))
            bw, bh = r - l, b - t
            if raster.shape != (bh, bw):
                raster = cv2.resize(raster, (bw, bh), interpolation=cv2.INTER_NEAREST)
            full = np.zeros((h, w), dtype=np.uint8)
            full[t:b, l:r] = raster[: h - t, : w - l]
            instances.setdefault(rec.object_id, {})["mask"] = full
            instances[rec.object_id].setdefault("confidence", rec.confidence)
    out = []
    for oid, inst in instances.items():
        if "mask" in inst and inst["mask"].any():
            ys, xs = np.nonzero(inst["mask"])
            inst.setdefault("box", (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))
            out.append({"object_id": oid, **inst})
    return out, from_cache


def _matches_spark_box(inst: Dict[str, Any], spark_box: Tuple[float, float, float, float]) -> bool:
    """La máscara coincide si su caja se solapa bastante con la de Spark o si su centro cae dentro de ella."""
    if inst["iou"] >= MUSE_PHASE3_MIN_IOU:
        return True
    x1, y1, x2, y2 = inst["box"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return spark_box[0] <= cx <= spark_box[2] and spark_box[1] <= cy <= spark_box[3] and inst["iou"] >= 0.1


def _merge_parts_inside_box(
    best: Dict[str, Any], instances: List[Dict[str, Any]], spark_box: Tuple[float, float, float, float]
) -> Dict[str, Any]:
    """Productos de varias piezas (un par de audífonos, de aretes...): une las detecciones cuyo centro cae en la caja de Spark."""
    x1, y1, x2, y2 = spark_box
    mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    parts = []
    for inst in instances:
        bx1, by1, bx2, by2 = inst["box"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if inst is best or (x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my):
            parts.append(inst)
    if len(parts) < 2:
        return best
    mask = np.zeros_like(best["mask"])
    for inst in parts:
        mask |= inst["mask"]
    ys, xs = np.nonzero(mask)
    box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    return {
        **best,
        "mask": mask,
        "box": box,
        "iou": round(_iou(box, spark_box), 3),
        "merged_parts": len(parts),
        "confidence": min(i.get("confidence") or 1.0 for i in parts),
    }


# La decodificación de máscaras del parser oficial es Python puro (1-6 s por máscara): se reparte en varios procesos.
SAM_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


def run_parallel(fn: Any, args_list: List[Any]) -> List[Any]:
    """map en un pool de procesos; si el pool no se puede usar, cae a secuencial (mismo resultado, más lento)."""
    if len(args_list) <= 1 or SAM_WORKERS <= 1:
        return [fn(a) for a in args_list]
    try:
        with ProcessPoolExecutor(max_workers=min(SAM_WORKERS, len(args_list))) as pool:
            return list(pool.map(fn, args_list))
    except Exception as exc:  # BrokenProcessPool, errores de pickling o de arranque en Windows
        print(f"[MUSE-TRACE] Pool de procesos no disponible ({exc!r}); se continúa en secuencial.", flush=True)
        return [fn(a) for a in args_list]


def segment_job(args: Tuple[np.ndarray, Optional[Tuple[float, float, float, float]], List[str]]) -> Dict[str, Any]:
    return segment_product(*args)


def segment_product(
    image: np.ndarray, spark_box_px: Optional[Tuple[float, float, float, float]], sam_prompts: List[str]
) -> Dict[str, Any]:
    """Prueba las frases de Spark y elige la máscara que coincide con su caja; si no, usa la caja como prompt."""
    h, w = image.shape[:2]
    jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tobytes()
    image_md5 = hashlib.md5(jpeg).hexdigest()
    attempts: List[Dict[str, Any]] = []
    calls_api, calls_cache = 0, 0

    seen = set()
    prompts = []
    for p in sam_prompts:
        p = " ".join(str(p).strip().split())
        if p and p.lower() not in seen:
            seen.add(p.lower())
            prompts.append(p)

    best_text: Optional[Tuple[str, Dict[str, Any]]] = None
    for prompt in prompts[:MAX_SAM_TEXT_ATTEMPTS]:
        found, from_cache = _sam_instances(jpeg, image_md5, prompt, (w, h))
        instances = [i for i in found if (i.get("confidence") or 1.0) >= MIN_SAM_CONFIDENCE]
        calls_cache += int(from_cache)
        calls_api += int(not from_cache)
        if not instances:
            attempts.append({"prompt": prompt, "matches": 0, "best_iou": None})
            continue
        if spark_box_px is not None:
            for inst in instances:
                inst["iou"] = round(_iou(inst["box"], spark_box_px), 3)
            best = max(instances, key=lambda i: (i["iou"], i.get("confidence") or 0))
        else:
            best = max(instances, key=lambda i: i.get("confidence") or 0)
            best["iou"] = None
        attempts.append({"prompt": prompt, "matches": len(instances), "best_iou": best["iou"]})
        if best_text is None or (best.get("confidence") or 0) > (best_text[1].get("confidence") or 0):
            best_text = (prompt, best)
        if best["iou"] is None or _matches_spark_box(best, spark_box_px):
            if spark_box_px is not None:
                best = _merge_parts_inside_box(best, instances, spark_box_px)
            return {"mode": "TEXT_PROMPT", "prompt": prompt, "instance": best, "attempts": attempts, "calls_api": calls_api, "calls_cache": calls_cache}

    if spark_box_px is not None:
        x1, y1, x2, y2 = spark_box_px
        pad_x, pad_y = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
        box = (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(w, int(x2 + pad_x) + 1),
            min(h, int(y2 + pad_y) + 1),
        )
        box_prompt = build_object_prompt(size=(w, h), objects=[PromptObject(id=1, prompts=[FramePrompt(frame=0, box=PromptBox(*box))])])
        instances, from_cache = _sam_instances(jpeg, image_md5, box_prompt, (w, h))
        calls_api += int(not from_cache)
        calls_cache += int(from_cache)
        attempts.append({"prompt": "BOX_PROMPT", "matches": len(instances), "best_iou": None})
        instances = [i for i in instances if (i.get("confidence") or 1.0) >= MIN_SAM_CONFIDENCE]
        if instances:
            best = max(instances, key=lambda i: i.get("confidence") or 0)
            best["iou"] = round(_iou(best["box"], spark_box_px), 3)
            return {"mode": "BOX_PROMPT", "prompt": "caja de Spark", "instance": best, "attempts": attempts, "calls_api": calls_api, "calls_cache": calls_cache}

    if best_text is not None:
        return {"mode": "TEXT_PROMPT_UNVERIFIED", "prompt": best_text[0], "instance": best_text[1], "attempts": attempts, "calls_api": calls_api, "calls_cache": calls_cache}

    return {"mode": "NO_MASK", "prompt": None, "instance": None, "attempts": attempts, "calls_api": calls_api, "calls_cache": calls_cache}


# ---------------------------------------------------------------------------
# 5. Salidas
# ---------------------------------------------------------------------------


def load_frame_for_sam(candidate: Dict[str, Any]) -> np.ndarray:
    image = cv2.imread(str(STORAGE_DIR / candidate["file_path"]))
    h0, w0 = image.shape[:2]
    if max(h0, w0) > MAX_SAM_IMAGE_SIDE:
        s = MAX_SAM_IMAGE_SIDE / max(h0, w0)
        image = cv2.resize(image, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
    return image


def cutout_rgba(
    image: np.ndarray, mask: Optional[np.ndarray], fallback_box: Optional[Tuple[int, int, int, int]]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Recorte RGBA ajustado a la máscara; sin máscara, recorte rectangular de fallback_box (o del frame entero)."""
    h, w = image.shape[:2]
    if mask is None:
        x1, y1, x2, y2 = fallback_box or (0, 0, w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
    ys, xs = np.nonzero(mask)
    pad = int(0.03 * max(xs.max() - xs.min(), ys.max() - ys.min()))
    x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x2, y2 = min(w, xs.max() + 1 + pad), min(h, ys.max() + 1 + pad)
    alpha = cv2.GaussianBlur((mask * 255).astype(np.uint8), (3, 3), 0)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    crop = rgba[y1:y2, x1:x2]
    coverage = float(mask[y1:y2, x1:x2].mean())
    info = {"width": int(x2 - x1), "height": int(y2 - y1), "coverage": round(coverage, 3), "bbox_pixels": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]}
    return crop, info


def write_cutout(image: np.ndarray, mask: Optional[np.ndarray], fallback_box: Optional[Tuple[int, int, int, int]], out_path: Path) -> Dict[str, Any]:
    crop, info = cutout_rgba(image, mask, fallback_box)
    cv2.imwrite(str(out_path), crop)
    return info


def write_preview(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    sam_box: Optional[List[int]],
    spark_box: Optional[Tuple[float, float, float, float]],
    caption: str,
    out_path: Path,
) -> None:
    preview = image.copy()
    if mask is not None:
        tint = preview.copy()
        tint[mask > 0] = (0.55 * tint[mask > 0] + np.array([129, 185, 16]) * 0.45).astype(np.uint8)
        preview = tint
    thick = max(2, image.shape[1] // 250)
    if spark_box is not None:
        x1, y1, x2, y2 = (int(v) for v in spark_box)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (248, 189, 56), thick)
    if sam_box is not None:
        cv2.rectangle(preview, (sam_box[0], sam_box[1]), (sam_box[2], sam_box[3]), (129, 185, 16), thick)
    scale = max(0.5, image.shape[1] / 900)
    cv2.rectangle(preview, (0, 0), (image.shape[1], int(36 * scale)), (15, 23, 42), -1)
    cv2.putText(preview, caption[:60], (8, int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (52, 211, 153), max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.imwrite(str(out_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _spark_box_to_px(
    bbox: Dict[str, float], layout: Dict[str, Any], slot: int, w: int, h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convierte la caja de Spark (0-1000 sobre el collage completo) a píxeles del frame elegido."""
    try:
        xmin, ymin, xmax, ymax = (max(0.0, min(1000.0, float(bbox[k]))) for k in ("xmin", "ymin", "xmax", "ymax"))
    except (KeyError, TypeError, ValueError):
        return None
    panel = layout["panels"][slot - 1]

    def to_frame_x(v: float) -> float:
        return (v / 1000 * layout["width"] - panel["x0"]) * w / panel["width"]

    x1, x2 = max(0.0, to_frame_x(xmin)), min(float(w), to_frame_x(xmax))
    y1, y2 = ymin / 1000 * h, ymax / 1000 * h
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2, y2)


async def run_phase3_vision_sam(session_id: Optional[str] = None) -> Dict[str, Any]:
    state = load_session_state(session_id)
    sid = state["session_id"]
    semantic_products = state.get("semantic_products", [])
    if not semantic_products:
        raise ValueError("No existen productos en la sesión. Ejecuta primero la Fase 2.")

    videos_by_id = {v["video_id"]: v for v in state.get("videos", [])}
    inventory = (state.get("phase2_verification") or {}).get("visual_inventory", [])
    session_dir = get_session_dir(sid)
    dirs = {name: session_dir / name for name in ("frames", "collages", "bboxes", "cutouts")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tracker = TokenTracker(state.get("token_audit"))

    # 1-2. Ventanas, candidatos y collages
    prepared: List[Dict[str, Any]] = []
    for prod in semantic_products:
        pid, vid = prod["product_id"], prod["video_id"]
        video_meta = videos_by_id.get(vid, {})
        video_path = locate_source_video({**video_meta, "video_id": vid, "session_id": sid})
        duration = float((video_meta.get("quality_assessment") or {}).get("duration_sec") or 0.0)
        if video_path is None:
            tracker.record_event(phase=3, model="OpenCV", operation=f"{pid}: no se encontró el video {vid}; producto omitido.", status="SKIPPED")
            continue
        start, end, window_source = resolve_showcase_window(prod, inventory, duration)
        candidates = extract_candidate_frames(video_path, start, end, pid, dirs["frames"], sid)
        if not candidates:
            tracker.record_event(phase=3, model="OpenCV", operation=f"{pid}: no se pudieron leer frames de {vid} entre {start}s y {end}s.", status="SKIPPED")
            continue
        collage_name = f"{pid}_collage_1x3.jpg"
        collage_md5, layout = build_collage(candidates, dirs["collages"] / collage_name)
        prepared.append(
            {"prod": prod, "candidates": candidates, "collage_name": collage_name, "collage_md5": collage_md5, "layout": layout, "window": [start, end], "window_source": window_source}
        )
    tracker.record_event(
        phase=3,
        model="OpenCV Frame Selector",
        operation=f"{len(prepared)} producto(s): 3 frames candidatos por producto (el más nítido de cada tercio de su ventana) y 1 collage 1x3 cada uno.",
        status="ZERO_TOKEN_LOCAL",
    )

    # 3. Spark visión por lotes
    vision: Dict[str, Dict[str, Any]] = {}
    n_batches = max(1, -(-len(prepared) // SPARK_BATCH_SIZE))
    per_batch = -(-len(prepared) // n_batches)
    for i in range(0, len(prepared), per_batch):
        chunk = prepared[i : i + per_batch]
        vision.update(
            analyze_with_spark_vision(tracker, [(p["prod"], dirs["collages"] / p["collage_name"], p["collage_md5"]) for p in chunk])
        )

    # 4. SAM sobre el mejor frame de cada producto (en paralelo)
    results: List[Dict[str, Any]] = []
    for item in prepared:
        prod, candidates = item["prod"], item["candidates"]
        pid = prod["product_id"]
        v = vision.get(pid)
        spark_ok = v is not None
        if not spark_ok:
            v = {
                "best_frame_slot": max(range(len(candidates)), key=lambda k: candidates[k]["laplacian_score"]) + 1,
                "slot_evaluations": [],
                "resolved_name": prod.get("detected_name", pid),
                "visual_category": "General",
                "visual_description": "",
                "disambiguation_note": "Spark no devolvió resultado para este producto; se usó el frame más nítido.",
                "quality_score": 5.0,
                "needs_enhancement": True,
                "quality_diagnosis": "Sin análisis visual",
                "bbox": None,
                "sam_prompts": [],
            }
        slot = int(v.get("best_frame_slot") or 1)
        slot = slot if 1 <= slot <= len(candidates) else 1
        image = load_frame_for_sam(candidates[slot - 1])
        h, w = image.shape[:2]
        spark_box = _spark_box_to_px(v.get("bbox") or {}, item["layout"], slot, w, h)
        results.append({"item": item, "prod": prod, "v": v, "spark_ok": spark_ok, "slot": slot, "image": image, "spark_box": spark_box})
    segs = run_parallel(segment_job, [(r["image"], r["spark_box"], r["v"].get("sam_prompts") or []) for r in results])
    for r, seg in zip(results, segs):
        r["seg"] = seg

    # 5. Verificación automática de los recortes y reparación (Spark + SAM, acotada a 2 llamadas de Spark)
    if MUSE_PHASE3_VERIFY_CUTOUTS and results:
        verify_and_repair_cutouts(tracker, results)

    # 6. Archivos y salida
    grounded_products: List[Dict[str, Any]] = []
    for r in results:
        item, prod, v, seg = r["item"], r["prod"], r["v"], r["seg"]
        candidates, slot, image, spark_box = item["candidates"], r["slot"], r["image"], r["spark_box"]
        pid = prod["product_id"]
        best = candidates[slot - 1]
        h, w = image.shape[:2]
        inst = seg["instance"]
        mask = inst["mask"] if inst else None
        fallback_box = tuple(int(x) for x in spark_box) if (mask is None and spark_box) else None

        cutout_name = f"{pid}_cutout.png"
        cut = write_cutout(image, mask, fallback_box, dirs["cutouts"] / cutout_name)
        bbox_name = f"{pid}_best_frame_bbox.jpg"
        write_preview(image, mask, cut["bbox_pixels"], spark_box, f"{v['resolved_name']} | Q:{v['quality_score']}/10", dirs["bboxes"] / bbox_name)

        mode_text = {
            "TEXT_PROMPT": f"sam-3.1 con la frase '{seg['prompt']}'"
            + (f" (coincidencia con la caja de Spark: IoU {inst.get('iou')})" if inst and inst.get("iou") is not None else ""),
            "BOX_PROMPT": f"sam-3.1 con la caja de Spark como prompt (ninguna frase coincidió; IoU {inst.get('iou') if inst else '-'})",
            "TEXT_PROMPT_UNVERIFIED": f"sam-3.1 con la frase '{seg['prompt']}' (no coincidió con la caja de Spark y la caja como prompt no dio una máscara confiable; se usa la detección por texto más segura)",
            "NO_MASK": "Sin máscara confiable: recorte rectangular del objeto",
        }[seg["mode"]]
        check = r.get("cutout_check") or {}
        tracker.record_event(
            phase=3,
            model="SAM 3.1",
            operation=(
                f"{pid} '{v['resolved_name']}': {mode_text}. Intentos: "
                + "; ".join(f"{a['prompt'][:24]} -> {a['matches']}" for a in seg["attempts"])
                + f". Recorte {cut['width']}x{cut['height']}px (cobertura {cut['coverage']:.0%})"
                + (f". Verificación: {check.get('status')}" if check else "")
                + "."
            ),
            status="API_CALL" if seg["calls_api"] else "CACHE_HIT_MD5",
            details={"sam_calls_api": seg["calls_api"], "sam_calls_cache": seg["calls_cache"], "billing": "SAM cobra por imagen procesada, no reporta tokens"},
        )

        prod["resolved_visual_name"] = v["resolved_name"]
        slot_evals = []
        for k, cand in enumerate(candidates, start=1):
            ev = next((e for e in v.get("slot_evaluations", []) if int(e.get("slot", 0)) == k), {})
            slot_evals.append(
                {
                    "slot": k,
                    "timestamp_sec": cand["timestamp_sec"],
                    "selection_score": ev.get("score", "-"),
                    "object_visible": ev.get("object_visible"),
                    "note": ev.get("note", ""),
                }
            )
        cutout_ok = check.get("status") in (None, "OK", "REPAIRED")
        grounded_products.append(
            {
                "product_id": pid,
                "video_id": prod["video_id"],
                "mention_type": prod.get("mention_type"),
                "spoken_name_initial": prod.get("detected_name"),
                "resolved_name": v["resolved_name"],
                "visual_category": v["visual_category"],
                "visual_description": v["visual_description"],
                "disambiguation_note": v["disambiguation_note"],
                "price": prod.get("price"),
                "price_status": prod.get("price_status"),
                "original_price_mentioned": prod.get("original_price_mentioned"),
                "price_corrected_live": prod.get("price_corrected_live", False),
                "currency": prod.get("currency", "USD"),
                "condition": prod.get("condition"),
                "features": prod.get("features", []),
                "raw_spoken_phrase": prod.get("raw_spoken_phrase"),
                "showcase_window": {"start_sec": item["window"][0], "end_sec": item["window"][1], "source": item["window_source"]},
                "collage_1x3": {
                    "url": f"/storage/{sid}/collages/{item['collage_name']}",
                    "collage_md5": item["collage_md5"],
                    "candidate_frames": candidates,
                    "slot_evaluations": slot_evals,
                },
                # Se conserva el nombre de clave "glimmer_selection" por compatibilidad con la UI y la Fase 4;
                # el modelo que decide es muse-spark-1.3 (Glimmer no está en la API).
                "glimmer_selection": {
                    "selector_model": MUSE_SPARK_MODEL if r["spark_ok"] else "FALLBACK_SHARPEST_FRAME",
                    "best_frame_slot": slot,
                    "best_frame_id": best["frame_id"],
                    "timestamp_sec": best["timestamp_sec"],
                    "best_frame_url": best["url"],
                    "best_frame_rel_path": best["file_path"],
                    "bbox_preview_url": f"/storage/{sid}/bboxes/{bbox_name}",
                    "bbox_normalized": [round(c / d, 3) for c, d in zip((spark_box[1], spark_box[0], spark_box[3], spark_box[2]), (h, w, h, w))] if spark_box else [],
                    "bbox_pixels": cut["bbox_pixels"],
                    "quality_score": v["quality_score"],
                    "needs_enhancement": bool(v["needs_enhancement"]) or not cutout_ok,
                    "quality_diagnosis": v["quality_diagnosis"] if cutout_ok else f"{v['quality_diagnosis']} | Recorte: {check.get('problem') or check.get('status')}",
                },
                "sam31_cutout": {
                    "cutout_png_url": f"/storage/{sid}/cutouts/{cutout_name}",
                    "cutout_rel_path": f"{sid}/cutouts/{cutout_name}",
                    "width": cut["width"],
                    "height": cut["height"],
                    "foreground_coverage_ratio": cut["coverage"],
                    "sam_mode": seg["mode"],
                    "sam_prompt_used": seg["prompt"],
                    "sam_confidence": round(float(inst["confidence"]), 3) if inst and inst.get("confidence") is not None else None,
                    "iou_with_spark_bbox": inst.get("iou") if inst else None,
                    "merged_parts": inst.get("merged_parts", 1) if inst else 0,
                    "sam_attempts": seg["attempts"],
                    "mask_refinement": mode_text,
                    "verification": check,
                },
            }
        )

    state["grounded_products"] = grounded_products
    state["current_phase_completed"] = max(int(state.get("current_phase_completed", 2)), 3)
    tracker.sync_to_session(state)
    save_session_state(state)
    return state
