"""
Verificación automática de los recortes de la Fase 3 (sin revisión humana, acotada a 2 llamadas de Spark).

1. Revisión: muse-spark-1.3 ve todos los recortes (sobre fondo a cuadros) en una hoja y dictamina cada uno:
   OK, INCOMPLETE, WRONG_OBJECT, EXTRA_CONTENT o UNRECOGNIZABLE.
2. Alternativas (0 tokens): para los que fallan, sam-3.1 vuelve a segmentar con las frases de Spark sobre el mismo
   frame (sin exigir la caja) y sobre los otros frames candidatos.
3. Elección: 1 llamada a Spark con las alternativas de todos los productos que fallaron; elige la mejor o NONE.
   NONE → recorte rectangular del objeto (marcado), nunca una máscara basura.
"""
import base64
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.core.config import MUSE_SPARK_MODEL
from backend.core.muse_api import chat_json
from backend.core.muse_client import get_cached_result, set_cached_result
from backend.core.token_tracker import TokenTracker

CUTOUT_CHECK_VERSION = "cutcheck_v2"
TILE_SIZE = 300
SHEET_COLS = 4
MAX_OPTIONS = 5
PART_SEARCH_MARGIN = 0.2
SAME_MASK_IOU = 0.9

CHECK_SYSTEM_PROMPT = """Revisas recortes automáticos de productos para un catálogo e-commerce. Cada recorte es un PNG sin fondo mostrado sobre un fondo a cuadros grises, con su product_id escrito encima. Recibes también el nombre del producto esperado para cada product_id.
Para cada recorte decide verdict:
- OK: se ve el producto esperado completo o casi completo (puede faltar un borde mínimo), sin fondo ni mano relevantes.
- INCOMPLETE: falta una parte importante del producto (cortado, tapado por la mano, solo una pieza de un par o set).
- WRONG_OBJECT: es otro objeto, una superficie o el fondo.
- EXTRA_CONTENT: incluye mano, fondo u otros objetos en una proporción importante.
- UNRECOGNIZABLE: manchas o formas sin sentido; no se reconoce el producto.
problem: explicación breve en español (cadena vacía si es OK).
missing_part_prompt: si es INCOMPLETE, frase nominal corta EN INGLÉS de la parte que falta para que un segmentador la busque en la foto original (por ejemplo "felt spikes", "lid", "earbud"); cadena vacía si no es INCOMPLETE o si lo que falta es un accesorio que solo se nombra en la etiqueta."""

CHECK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["OK", "INCOMPLETE", "WRONG_OBJECT", "EXTRA_CONTENT", "UNRECOGNIZABLE"]},
                    "problem": {"type": "string"},
                    "missing_part_prompt": {"type": "string"},
                },
                "required": ["product_id", "verdict", "problem", "missing_part_prompt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["products"],
    "additionalProperties": False,
}

CHOICE_SYSTEM_PROMPT = """Para cada producto recibes su nombre esperado y una imagen con varios recortes candidatos del mismo producto, rotulados con letras (A, B, C, D, E) y mostrados sobre un fondo a cuadros grises.
Elige la letra del recorte que mejor sirve como foto de catálogo: el producto correcto, lo más completo posible, sin mano ni fondo, nítido.
Si ninguno muestra el producto correcto de forma reconocible, responde NONE. reason: breve en español."""

CHOICE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "choice": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["product_id", "choice", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["choices"],
    "additionalProperties": False,
}


def _tile(rgba: np.ndarray, label: str) -> np.ndarray:
    """Recorte sobre fondo a cuadros, escalado a una celda cuadrada, con su rótulo arriba."""
    size, header = TILE_SIZE, 34
    yy, xx = np.mgrid[0 : size - header, 0:size]
    board = np.where(((yy // 16 + xx // 16) % 2 == 0)[..., None], 205, 160).astype(np.uint8).repeat(3, axis=2)
    if rgba.size:
        scale = min((size - header - 12) / rgba.shape[0], (size - 12) / rgba.shape[1])
        img = cv2.resize(rgba, (max(1, int(rgba.shape[1] * scale)), max(1, int(rgba.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        oy, ox = (board.shape[0] - img.shape[0]) // 2, (size - img.shape[1]) // 2
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        roi = board[oy : oy + img.shape[0], ox : ox + img.shape[1]].astype(np.float32)
        board[oy : oy + img.shape[0], ox : ox + img.shape[1]] = (img[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    head = np.full((header, size, 3), (42, 23, 15), dtype=np.uint8)
    cv2.putText(head, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 189, 56), 2, cv2.LINE_AA)
    tile = np.vstack([head, board])
    cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (90, 90, 90), 1)
    return tile


def _sheet(tiles: List[np.ndarray], cols: int = SHEET_COLS) -> np.ndarray:
    blank = np.full_like(tiles[0], 255)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i : i + cols]
        row += [blank] * (cols - len(row)) if len(tiles) > cols else []
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tobytes()


def _image_part(jpeg: bytes) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}}


def _call_spark(
    tracker: TokenTracker, namespace: str, system: str, content: List[Dict[str, Any]], schema_name: str, schema: Dict[str, Any], key_src: str, label: str
) -> Dict[str, Any]:
    cache_key = hashlib.md5(f"{MUSE_SPARK_MODEL}|{CUTOUT_CHECK_VERSION}|{key_src}".encode("utf-8")).hexdigest()
    cached = get_cached_result(namespace, cache_key)
    if cached:
        usage = cached["usage"]
        tracker.record_event(
            phase=3,
            model="Muse Spark (Verificador de recortes)",
            operation=f"Caché MD5 HIT: {label} sin llamar a la API.",
            tokens_saved=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            status="CACHE_HIT_MD5",
        )
        return cached["parsed"]
    parsed, usage, _ = chat_json(
        model=MUSE_SPARK_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        schema_name=schema_name,
        schema=schema,
        reasoning_effort="low",
    )
    set_cached_result(namespace, cache_key, {"parsed": parsed, "usage": usage})
    tracker.record_event(
        phase=3,
        model="Muse Spark (Verificador de recortes)",
        operation=f"{MUSE_SPARK_MODEL}: {label} (razonamiento 'low': {usage['reasoning_tokens']} tokens).",
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        status="API_CALL",
        details={"usage": usage},
    )
    return parsed


def _mask_of(seg: Dict[str, Any]) -> Optional[np.ndarray]:
    inst = seg.get("instance")
    return inst["mask"] if inst else None


def _mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _mask_box(mask: Optional[np.ndarray]) -> Optional[Tuple[float, float, float, float]]:
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _union_box(r: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Caja que abarca la caja de Spark y, si el recorte solo está incompleto, también la máscara actual."""
    use_mask = (r.get("cutout_check") or {}).get("verdict") == "INCOMPLETE"
    boxes = [b for b in (_mask_box(_mask_of(r["seg"])) if use_mask else None, r["spark_box"]) if b is not None]
    if not boxes:
        return None
    h, w = r["image"].shape[:2]
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    px, py = 0.04 * (x2 - x1), 0.04 * (y2 - y1)
    return (max(0.0, x1 - px), max(0.0, y1 - py), min(float(w), x2 + px), min(float(h), y2 + py))


def _seg_from(mask: np.ndarray, mode: str, prompt: str, base: Dict[str, Any], confidence: Optional[float]) -> Dict[str, Any]:
    return {**base, "mode": mode, "prompt": prompt, "instance": {"mask": mask, "box": _mask_box(mask), "confidence": confidence, "iou": None}}


def _alternatives(r: Dict[str, Any], missing_part_prompt: str) -> List[Dict[str, Any]]:
    """Otras segmentaciones del mismo producto: parte faltante unida, caja ampliada, mismo frame sin exigir la caja y otros frames."""
    from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt

    from backend.services.phase3_vision_sam import _sam_instances, load_frame_for_sam, segment_product

    prompts = r["v"].get("sam_prompts") or []
    candidates = r["item"]["candidates"]
    current = _mask_of(r["seg"])
    options: List[Dict[str, Any]] = []
    image = r["image"]
    h, w = image.shape[:2]
    jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tobytes()
    image_md5 = hashlib.md5(jpeg).hexdigest()

    # Parte faltante (pinchos, tapa, la otra pieza...) pegada a la máscara actual
    part = " ".join(missing_part_prompt.split())
    cur_box = _mask_box(current)
    if part and current is not None and cur_box is not None:
        mx, my = PART_SEARCH_MARGIN * (cur_box[2] - cur_box[0]), PART_SEARCH_MARGIN * (cur_box[3] - cur_box[1])
        near = (cur_box[0] - mx, cur_box[1] - my, cur_box[2] + mx, cur_box[3] + my)
        pieces = [
            i for i in _sam_instances(jpeg, image_md5, part, (w, h))[0]
            if (i.get("confidence") or 1.0) >= 0.3 and i["box"][0] < near[2] and near[0] < i["box"][2] and i["box"][1] < near[3] and near[1] < i["box"][3]
        ]
        if pieces:
            merged = current.copy()
            for i in pieces:
                merged |= i["mask"]
            options.append({"slot": r["slot"], "image": image, "spark_box": r["spark_box"], "seg": _seg_from(merged, "TEXT_PROMPT", f"{r['seg'].get('prompt')} + {part}", r["seg"], None), "desc": f"recorte original + parte faltante '{part}'"})

    # Caja ampliada (máscara actual ∪ caja de Spark) como prompt de caja
    box = _union_box(r)
    if box is not None:
        bi = (int(box[0]), int(box[1]), min(w, int(box[2]) + 1), min(h, int(box[3]) + 1))
        box_prompt = build_object_prompt(size=(w, h), objects=[PromptObject(id=1, prompts=[FramePrompt(frame=0, box=PromptBox(*bi))])])
        found = [i for i in _sam_instances(jpeg, image_md5, box_prompt, (w, h))[0] if (i.get("confidence") or 1.0) >= 0.3]
        if found:
            best = max(found, key=lambda i: i.get("confidence") or 0)
            if _mask_iou(best["mask"], current) < SAME_MASK_IOU:
                options.append({"slot": r["slot"], "image": image, "spark_box": r["spark_box"], "seg": _seg_from(best["mask"], "BOX_PROMPT", "caja ampliada", r["seg"], best.get("confidence")), "desc": "caja ampliada como prompt de SAM"})

    text_attempts = [a for a in r["seg"].get("attempts", []) if a.get("prompt") != "BOX_PROMPT"]
    if not prompts or (text_attempts and all(a.get("matches", 0) == 0 for a in text_attempts)):
        # SAM no reconoció ninguna de las frases en el mejor frame: repetirlas en los otros frames solo gasta llamadas.
        return options[: MAX_OPTIONS - 1]

    same = segment_product(r["image"], None, prompts)
    if same["instance"] is not None and _mask_iou(same["instance"]["mask"], current) < SAME_MASK_IOU:
        options.append({"slot": r["slot"], "image": image, "spark_box": r["spark_box"], "seg": same, "desc": f"frame {r['slot']} con la frase '{same['prompt']}'"})

    scores = {int(e.get("slot", 0)): float(e.get("score", 0) or 0) for e in r["v"].get("slot_evaluations", [])}
    for slot in sorted((k for k in range(1, len(candidates) + 1) if k != r["slot"]), key=lambda k: -scores.get(k, 0)):
        other = load_frame_for_sam(candidates[slot - 1])
        seg = segment_product(other, None, prompts)
        if seg["instance"] is not None:
            options.append({"slot": slot, "image": other, "spark_box": None, "seg": seg, "desc": f"frame {slot} con la frase '{seg['prompt']}'"})
    return options[: MAX_OPTIONS - 1]


def alternatives_job(args: Tuple[Dict[str, Any], str]) -> List[Dict[str, Any]]:
    return _alternatives(*args)


def verify_and_repair_cutouts(tracker: TokenTracker, results: List[Dict[str, Any]]) -> None:
    """Revisa los recortes con Spark y repara los defectuosos. Modifica `results` en su lugar (slot, image, seg, cutout_check)."""
    from backend.services.phase3_vision_sam import cutout_rgba, run_parallel

    def current_rgba(r: Dict[str, Any]) -> np.ndarray:
        mask = _mask_of(r["seg"])
        box = tuple(int(x) for x in r["spark_box"]) if (mask is None and r["spark_box"]) else None
        return cutout_rgba(r["image"], mask, box)[0]

    # 1. Revisión de todos los recortes en una hoja (1 imagen por cada 8 productos)
    names = {r["prod"]["product_id"]: r["v"]["resolved_name"] for r in results}
    content: List[Dict[str, Any]] = [{"type": "text", "text": "Productos esperados:\n" + json.dumps(names, ensure_ascii=False)}]
    digests = []
    for i in range(0, len(results), 2 * SHEET_COLS):
        chunk = results[i : i + 2 * SHEET_COLS]
        jpeg = _jpeg(_sheet([_tile(current_rgba(r), r["prod"]["product_id"]) for r in chunk]))
        digests.append(hashlib.md5(jpeg).hexdigest())
        content.append(_image_part(jpeg))
    parsed = _call_spark(
        tracker,
        "phase3_cutout_check",
        CHECK_SYSTEM_PROMPT,
        content,
        "Phase3CutoutCheck",
        CHECK_SCHEMA,
        json.dumps([names, digests], sort_keys=True, ensure_ascii=False),
        f"revisión de {len(results)} recorte(s) en {len(digests)} hoja(s)",
    )
    verdicts = {p["product_id"]: p for p in parsed.get("products", [])}

    failing = []
    for r in results:
        pid = r["prod"]["product_id"]
        verdict = verdicts.get(pid, {"verdict": "OK", "problem": "Sin dictamen; se asume correcto."})
        r["cutout_check"] = {
            "verdict": verdict["verdict"],
            "problem": verdict["problem"],
            "missing_part_prompt": verdict.get("missing_part_prompt", ""),
            "status": "OK" if verdict["verdict"] == "OK" else None,
        }
        if verdict["verdict"] != "OK":
            failing.append(r)
    if not failing:
        tracker.record_event(phase=3, model="Verificador de recortes", operation=f"Los {len(results)} recortes pasaron la revisión.", status="ZERO_TOKEN_LOCAL")
        return

    # 2. Alternativas con SAM (0 tokens) y 3. elección con Spark
    to_choose = []
    slim = [{k: r[k] for k in ("item", "v", "slot", "image", "spark_box", "seg", "cutout_check")} for r in failing]
    alternatives = run_parallel(alternatives_job, [(r, r["cutout_check"]["missing_part_prompt"]) for r in slim])
    for r, alts in zip(failing, alternatives):
        options = [{"slot": r["slot"], "image": r["image"], "spark_box": r["spark_box"], "seg": r["seg"], "desc": "recorte original"}]
        options += alts
        r["_options"] = options
        if len(options) > 1:
            to_choose.append(r)

    choices: Dict[str, Dict[str, Any]] = {}
    if to_choose:
        content = []
        key_parts = []
        for r in to_choose:
            pid = r["prod"]["product_id"]
            tiles = []
            for letter, opt in zip("ABCDE", r["_options"]):
                mask = _mask_of(opt["seg"])
                box = tuple(int(x) for x in opt["spark_box"]) if (mask is None and opt["spark_box"]) else None
                tiles.append(_tile(cutout_rgba(opt["image"], mask, box)[0], f"{pid}  {letter}"))
            jpeg = _jpeg(np.hstack(tiles))
            key_parts.append([pid, r["v"]["resolved_name"], hashlib.md5(jpeg).hexdigest()])
            content.append({"type": "text", "text": f"product_id: {pid}\nProducto esperado: {r['v']['resolved_name']}\nOpciones: {', '.join('ABCDE'[: len(tiles)])}"})
            content.append(_image_part(jpeg))
        parsed = _call_spark(
            tracker,
            "phase3_cutout_choice",
            CHOICE_SYSTEM_PROMPT,
            content,
            "Phase3CutoutChoice",
            CHOICE_SCHEMA,
            json.dumps(key_parts, sort_keys=True, ensure_ascii=False),
            f"elección entre recortes alternativos para {len(to_choose)} producto(s)",
        )
        choices = {c["product_id"]: c for c in parsed.get("choices", [])}

    for r in failing:
        pid = r["prod"]["product_id"]
        check = r["cutout_check"]
        options = r.pop("_options")
        choice = choices.get(pid)
        letter = (choice or {}).get("choice", "").strip().upper()
        check["alternatives_tried"] = [o["desc"] for o in options[1:]]
        check["choice_reason"] = (choice or {}).get("reason", "")
        idx = "ABCDE".find(letter[:1]) if letter and not letter.startswith("NONE") else -1
        if 0 < idx < len(options):
            opt = options[idx]
            unchanged = opt["slot"] == r["slot"] and _mask_iou(_mask_of(opt["seg"]), _mask_of(r["seg"])) > 0.97
            r.update({"slot": opt["slot"], "image": opt["image"], "spark_box": opt["spark_box"], "seg": opt["seg"]})
            # Si la "reparación" casi no cambia la máscara (lo que falta no se ve en el video), no cuenta como reparada.
            check["status"] = "KEPT_BEST_AVAILABLE" if unchanged else "REPAIRED"
            check["repaired_with"] = opt["desc"]
        elif idx == 0 or (choice is None and check["verdict"] in ("INCOMPLETE", "EXTRA_CONTENT")):
            check["status"] = "KEPT_BEST_AVAILABLE"
        else:
            # Ninguna máscara sirve: recorte rectangular honesto del objeto (máscara actual ∪ caja de Spark).
            r["spark_box"] = _union_box(r) or r["spark_box"]
            r["seg"] = {**r["seg"], "mode": "NO_MASK", "prompt": None, "instance": None}
            check["status"] = "RECTANGULAR_FALLBACK"

    tracker.record_event(
        phase=3,
        model="Verificador de recortes",
        operation="; ".join(
            f"{r['prod']['product_id']}: {r['cutout_check']['verdict']} -> {r['cutout_check']['status']}"
            + (f" ({r['cutout_check'].get('repaired_with')})" if r["cutout_check"].get("repaired_with") else "")
            for r in failing
        ),
        status="ZERO_TOKEN_LOCAL",
    )
