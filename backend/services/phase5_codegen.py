"""
FASE 5: Catálogo e-commerce final (API real).

- Textos de venta con muse-spark-1.3 (1 llamada por lote de hasta 20 productos, JSON Schema estricto): título,
  descripción corta y larga, puntos destacados, categoría de tienda y etiquetas, sin inventar datos; más el nombre y el
  lema de la tienda. ("Muse Code" no es un modelo de la API: el HTML lo arma una plantilla determinista.)
- Catálogo HTML con la paleta oscura (slate-950 / emerald / cyan): filtros, buscador, selector de vistas (las vistas
  deducidas por IA van marcadas), vista ampliada, carrito con totales por moneda y "Consultar precio".
- Carpeta autocontenida storage/<sesión>/catalog/ (HTML + images/ con rutas relativas) y ZIP para descargar.
- Resumen con números reales del TokenTracker: tokens de la API, tokens evitados por caché e imágenes cobradas.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from backend.core.config import (
    MUSE_IMAGE_COST_USD,
    MUSE_SPARK_MODEL,
    STORAGE_DIR,
    load_session_state,
    save_session_state,
)
from backend.core import prompts
from backend.core.muse_api import MuseAPIError, chat_json
from backend.core.muse_client import NOT_MENTIONED_VALUES, get_cached_result, set_cached_result
from backend.core.token_tracker import TokenTracker
from backend.services.phase2_semantic import run_phase2_semantic
from backend.services.phase3_vision_sam import run_phase3_vision_sam
from backend.services.phase4_studio import run_phase4_studio

PHASE5_PROMPT_VERSION = "phase5_v2"
COPY_BATCH = 20
CATALOG_HTML = "muse_ecommerce_catalog.html"
MAX_IMAGE_SIDE = 1024

COPY_SYSTEM_PROMPT = """You are the copywriter of an e-commerce catalog. You receive products detected in a seller's videos: the name seen in the video, a tentative category, the condition and features the seller mentioned, their literal phrase (it may be in Spanish) and a description of the video scene.
Write in clear, natural US English, in an honest and clear commercial tone:
- title: short, clear commercial name (60 characters max).
- short_description: 1 sentence (140 characters max).
- description: 2 or 3 appealing sentences based ONLY on what is seen of the product and what the seller said. The scene description may mention hands, walls, tables or furniture from the video: ignore them, they are not part of the product. Do not invent measurements, materials, brands, authenticity ("original", "official"), warranties or functions that cannot be inferred from the data.
- highlights: up to 4 brief points using only the given data (type, color, condition, features mentioned). Empty list if there are none.
- category: store category. Use a small, consistent set for the whole catalog (group similar products; 6 distinct categories max).
- tags: up to 6 lowercase search keywords.
store: name (short, catchy store name that fits the products) and tagline (1 sentence).
Return one item per product with its exact product_id."""
prompts.register(
    "p5_copy", COPY_SYSTEM_PROMPT, phase=5, model="Muse Spark", group="creative",
    title="Sales copywriter",
    description="Writes the title, description, highlights, category and tags of each product, plus the store name.",
    hint="E.g.: 'Young, friendly tone with short sentences.' or 'The store is called Brak.'",
)

COPY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "store": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "tagline": {"type": "string"}},
            "required": ["name", "tagline"],
            "additionalProperties": False,
        },
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "title": {"type": "string"},
                    "short_description": {"type": "string"},
                    "description": {"type": "string"},
                    "highlights": {"type": "array", "items": {"type": "string"}},
                    "category": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["product_id", "title", "short_description", "description", "highlights", "category", "tags"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["store", "products"],
    "additionalProperties": False,
}

VIEW_SPECS = [
    # (clave, etiqueta, campo de step_4b/4a, generada por IA sin filmar)
    ("front", "Front", "studio_white_url", False),
    ("left", "Left", "angle_left_url", True),
    ("right", "Right", "angle_right_url", True),
    ("back", "Back", "angle_back_url", True),
    ("life", "Lifestyle", "lifestyle_url", False),
]


def _compute_model_breakdown(logs: list) -> Dict[str, Dict[str, int]]:
    by_model: Dict[str, Dict[str, int]] = {}
    for entry in logs:
        stats = by_model.setdefault(
            entry.get("model", "Unknown"),
            {"calls": 0, "api_calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "saved_tokens": 0},
        )
        stats["calls"] += 1
        stats["api_calls"] += int(entry.get("status") == "API_CALL")
        stats["cache_hits"] += int(entry.get("status") == "CACHE_HIT_MD5")
        stats["input_tokens"] += int(entry.get("input_tokens", 0))
        stats["output_tokens"] += int(entry.get("output_tokens", 0))
        stats["total_tokens"] += int(entry.get("step_total_tokens", 0))
        stats["saved_tokens"] += int(entry.get("tokens_saved", 0)) if entry.get("status") == "CACHE_HIT_MD5" else 0
    return by_model


def _usage_summary(logs: list) -> Dict[str, Any]:
    """Solo números reportados por la API o por la caché (no las estimaciones locales de la Fase 1)."""
    images = sum(1 for l in logs if l.get("model") == "Muse Image" and l.get("status") == "API_CALL")
    return {
        "api_tokens_used": sum(int(l.get("step_total_tokens", 0)) for l in logs),
        "tokens_avoided_by_cache": sum(int(l.get("tokens_saved", 0)) for l in logs if l.get("status") == "CACHE_HIT_MD5"),
        "images_generated": images,
        "images_cost_usd": round(images * MUSE_IMAGE_COST_USD, 2),
        "api_calls": sum(1 for l in logs if l.get("status") == "API_CALL"),
        "cache_hits": sum(1 for l in logs if l.get("status") == "CACHE_HIT_MD5"),
    }


def _write_copy(tracker: TokenTracker, products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Textos de venta con Spark (caché MD5). Si la API falla, se usan los datos de las fases anteriores."""
    store: Dict[str, str] = {}
    copy: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(products), COPY_BATCH):
        chunk = products[i : i + COPY_BATCH]
        payload = [
            {
                "product_id": p["product_id"],
                "name_seen_in_video": p.get("resolved_name"),
                "tentative_category": p.get("visual_category"),
                "condition_said": p.get("condition") if p.get("condition") not in NOT_MENTIONED_VALUES else "",
                "features_said": p.get("features") or [],
                "seller_phrase": p.get("raw_spoken_phrase") or "",
                "video_scene_description": p.get("visual_description") or "",
            }
            for p in chunk
        ]
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        key = hashlib.md5(f"{MUSE_SPARK_MODEL}|{PHASE5_PROMPT_VERSION}|{canonical}{prompts.signature('p5_copy')}".encode("utf-8")).hexdigest()
        cached = get_cached_result("phase5_copy", key)
        if cached:
            parsed, usage = cached["parsed"], cached["usage"]
            tracker.record_event(
                phase=5,
                model="Muse Spark (Copy)",
                operation=f"MD5 cache HIT: sales copy for {len(chunk)} product(s) without calling the API.",
                tokens_saved=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                status="CACHE_HIT_MD5",
            )
        else:
            try:
                parsed, usage, _ = chat_json(
                    model=MUSE_SPARK_MODEL,
                    messages=[{"role": "system", "content": prompts.get("p5_copy")}, {"role": "user", "content": canonical}],
                    schema_name="CatalogCopy",
                    schema=COPY_SCHEMA,
                    reasoning_effort="low",
                )
            except MuseAPIError as exc:
                tracker.record_event(phase=5, model="Muse Spark (Copy)", operation=f"Copywriting failed ({str(exc)[:160]}); using the names from the previous phases.", status="API_ERROR")
                continue
            set_cached_result("phase5_copy", key, {"parsed": parsed, "usage": usage})
            tracker.record_event(
                phase=5,
                model="Muse Spark (Copy)",
                operation=(
                    f"{MUSE_SPARK_MODEL}: sales copy, categories and tags for {len(chunk)} product(s) in 1 call "
                    f"(reasoning 'low': {usage['reasoning_tokens']} tokens)."
                ),
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                status="API_CALL",
            )
        if not store:
            store = parsed.get("store") or {}
        for item in parsed.get("products", []):
            copy[item["product_id"]] = item
    return {"store": store, "products": copy}


def _export_image(src_url: Optional[str], dest: Path) -> bool:
    """Copia optimizada (máx. 1024 px) de una imagen de /storage/... a la carpeta del catálogo."""
    if not src_url or "/storage/" not in src_url:
        return False
    src = STORAGE_DIR / src_url.split("/storage/", 1)[1].split("?", 1)[0]
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE:
        s = MAX_IMAGE_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if dest.suffix == ".png":
        return bool(cv2.imwrite(str(dest), img))
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return bool(cv2.imwrite(str(dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 86]))


def _catalog_products(studio_products: List[Dict[str, Any]], copy: Dict[str, Dict[str, Any]], images_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for p in studio_products:
        pid = p["product_id"]
        st = p.get("studio_imagen") or {}
        s4a, s4b = st.get("step_4a_enhancement") or {}, st.get("step_4b_variations") or {}
        views = []
        for key, label, field, ai in VIEW_SPECS:
            name = f"{pid}_{key}.jpg"
            if _export_image(s4b.get(field), images_dir / name):
                views.append({"key": key, "label": label, "src": f"images/{name}", "ai_inferred": ai, "bg": "dark" if key == "life" else "white"})
        if _export_image(s4a.get("result_cutout_url"), images_dir / f"{pid}_png.png"):
            views.append({"key": "png", "label": "No background", "src": f"images/{pid}_png.png", "ai_inferred": False, "bg": "checker"})
        c = copy.get(pid) or {}
        price = p.get("price")
        out.append(
            {
                "id": pid,
                "video_id": p.get("video_id"),
                "title": c.get("title") or p.get("resolved_name") or pid,
                "short_description": c.get("short_description") or "",
                "description": c.get("description") or "",
                "highlights": c.get("highlights") or [],
                "category": c.get("category") or p.get("visual_category") or "General",
                "tags": c.get("tags") or [],
                "price": float(price) if price not in (None, "") else None,
                "currency": p.get("currency") or "USD",
                "condition": p.get("condition") if p.get("condition") not in NOT_MENTIONED_VALUES else "",
                "seller_phrase": p.get("raw_spoken_phrase") or "",
                "real_photo_only": bool(s4a.get("low_quality_source")),
                "views": views,
            }
        )
    return out


def _build_html(store: Dict[str, str], products: List[Dict[str, Any]], usage: Dict[str, Any], session_id: str) -> str:
    data = json.dumps(
        {"store": store, "products": products, "usage": usage, "session_id": session_id},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return CATALOG_TEMPLATE.replace("__CATALOG_DATA__", data).replace("__STORE_NAME__", _esc(store.get("name") or "Catalog"))


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_catalog_zip(session_id: str) -> Path:
    """ZIP con el HTML y sus imágenes (rutas relativas): se abre sin el servidor."""
    catalog_dir = STORAGE_DIR / session_id / "catalog"
    zip_path = catalog_dir.parent / f"catalogo_{session_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(catalog_dir / CATALOG_HTML, f"catalogo_{session_id}/index.html")
        for img in sorted((catalog_dir / "images").glob("*")):
            zf.write(img, f"catalogo_{session_id}/images/{img.name}")
    return zip_path


async def run_phase5_codegen(session_id: Optional[str] = None) -> Dict[str, Any]:
    state = load_session_state(session_id)
    if not state or not state.get("session_id"):
        raise ValueError("There is no active session. Run Phases 1 to 4 first.")
    sid = state["session_id"]
    studio_products = state.get("studio_products", [])
    if not studio_products:
        raise ValueError("There are no photo-studio products (Phase 4). Run Phase 4 before the catalog.")

    tracker = TokenTracker(existing_audit=state.get("token_audit"))
    catalog_dir = STORAGE_DIR / sid / "catalog"
    images_dir = catalog_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    written = _write_copy(tracker, studio_products)
    store = {
        "name": (written["store"].get("name") or "My Store").strip(),
        "tagline": (written["store"].get("tagline") or "").strip(),
    }
    products = _catalog_products(studio_products, written["products"], images_dir)

    tracker.sync_to_session(state)
    usage = _usage_summary(state["token_audit"]["logs"])
    html = _build_html(store, products, usage, sid)
    (catalog_dir / CATALOG_HTML).write_text(html, encoding="utf-8")

    values: Dict[str, float] = {}
    for p in products:
        if p["price"] is not None:
            values[p["currency"]] = round(values.get(p["currency"], 0.0) + p["price"], 2)
    n_images = sum(len(p["views"]) for p in products)
    tracker.record_event(
        phase=5,
        model="HTML catalog",
        operation=(
            f"Catalog '{store['name']}' built with the local template: {len(products)} product(s), {n_images} optimized image(s), "
            f"{len({p['category'] for p in products})} category(ies); value: "
            + (", ".join(f"{v:,.2f} {k}" for k, v in values.items()) or "no prices")
            + "."
        ),
        status="ZERO_TOKEN_LOCAL",
    )
    tracker.sync_to_session(state)
    audit = state["token_audit"]
    audit["by_model"] = _compute_model_breakdown(tracker.logs)
    usage = _usage_summary(tracker.logs)

    catalog_url = f"/storage/{sid}/catalog/{CATALOG_HTML}"
    state["catalog_products"] = products
    state["catalog_store"] = store
    state["catalog_build"] = {
        "catalog_html_url": catalog_url,
        "preview_endpoint": f"/api/v1/phase5/catalog-preview?session_id={sid}",
        "download_endpoint": f"/api/v1/phase5/catalog-download?session_id={sid}",
    }
    state["phase5_summary"] = {
        "status": "CATALOG_READY",
        "store_name": store["name"],
        "products_published": len(products),
        "products_without_price": sum(1 for p in products if p["price"] is None),
        "images_published": n_images,
        "ai_inferred_views": sum(1 for p in products for v in p["views"] if v["ai_inferred"]),
        "categories": sorted({p["category"] for p in products}),
        "catalog_value_by_currency": values,
        "catalog_html_url": catalog_url,
        "catalog_size_kb": round(len(html.encode("utf-8")) / 1024, 1),
        **usage,
    }
    state["current_phase_completed"] = max(int(state.get("current_phase_completed", 4)), 5)
    save_session_state(state)
    return state


async def run_full_pipeline_auto(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Fases 2 -> 3 -> 4 -> 5 encadenadas sobre la sesión activa (con caché MD5 en lo ya calculado)."""
    state = load_session_state(session_id)
    if not state or not state.get("session_id"):
        raise ValueError("There is no active session with pre-processed videos. Upload the videos in Phase 1 first.")
    sid = state["session_id"]
    await run_phase2_semantic(session_id=sid)
    await run_phase3_vision_sam(session_id=sid)
    await run_phase4_studio(session_id=sid)
    return await run_phase5_codegen(session_id=sid)


CATALOG_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>__STORE_NAME__</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { background: radial-gradient(circle at top right, #0f172a 0%, #020617 100%); color: #f8fafc; font-family: Inter, system-ui, -apple-system, sans-serif; }
  .glass { background: rgba(15, 23, 42, 0.84); backdrop-filter: blur(16px); border: 1px solid rgba(148, 163, 184, 0.16); }
  .bg-checker { background-image: linear-gradient(45deg,#e2e8f0 25%,transparent 25%),linear-gradient(-45deg,#e2e8f0 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#e2e8f0 75%),linear-gradient(-45deg,transparent 75%,#e2e8f0 75%); background-size: 16px 16px; background-position: 0 0,0 8px,8px -8px,-8px 0; background-color: #f8fafc; }
  .line-clamp-2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
</style>
</head>
<body class="min-h-screen flex flex-col">
<header class="sticky top-0 z-40 glass border-b border-slate-800/80">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-4">
    <div class="flex items-center gap-3">
      <div id="logo" class="w-11 h-11 rounded-2xl bg-gradient-to-br from-emerald-500 via-cyan-500 to-indigo-600 flex items-center justify-center font-black text-white text-lg shadow-lg shadow-emerald-500/20"></div>
      <div>
        <h1 id="storeName" class="text-lg sm:text-xl font-extrabold tracking-tight text-white"></h1>
        <p id="storeTagline" class="text-xs text-slate-400"></p>
      </div>
    </div>
    <button onclick="toggleCart(true)" class="relative px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-600 hover:from-emerald-400 hover:to-cyan-500 text-slate-950 font-extrabold text-xs shadow-lg shadow-emerald-500/20 transition flex items-center gap-2">
      <span>🛒 Cart</span><span id="cartCount" class="px-2 py-0.5 rounded-full bg-slate-950 text-emerald-300 font-mono text-[11px]">0</span>
    </button>
  </div>
</header>

<section class="max-w-7xl mx-auto w-full px-4 sm:px-6 pt-6 pb-2 space-y-4">
  <div class="flex flex-wrap items-center justify-between gap-3">
    <div id="categoryBar" class="flex flex-wrap items-center gap-2"></div>
    <div class="flex items-center gap-2 w-full sm:w-auto">
      <input id="search" type="text" placeholder="Search products..." class="w-full sm:w-64 px-3.5 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white focus:outline-none focus:border-cyan-400" />
      <select id="sort" class="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-200">
        <option value="default">Featured</option>
        <option value="price_asc">Price: low to high</option>
        <option value="price_desc">Price: high to low</option>
      </select>
    </div>
  </div>
</section>

<main class="max-w-7xl mx-auto w-full px-4 sm:px-6 py-5 flex-1">
  <div id="grid" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"></div>
  <p id="emptyMsg" class="hidden text-center text-sm text-slate-400 py-16">No products match your search.</p>
</main>

<footer class="border-t border-slate-800/80 mt-6">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 py-5 text-[11px] text-slate-500 flex flex-wrap items-center justify-between gap-2">
    <span>Catalog generated from video with Meta Muse (Voice Transcribe, Spark, SAM 3.1 and Muse Image).</span>
    <span id="usageLine" class="font-mono"></span>
  </div>
</footer>

<div id="modal" class="fixed inset-0 z-50 hidden items-center justify-center bg-black/80 p-4" onclick="if (event.target === this) closeModal()">
  <div class="glass rounded-3xl max-w-4xl w-full p-6 border border-slate-700 space-y-4 max-h-[92vh] overflow-y-auto">
    <div class="flex items-start justify-between gap-3 border-b border-slate-800 pb-3">
      <div>
        <span id="mCategory" class="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300"></span>
        <h3 id="mTitle" class="text-lg font-extrabold text-white mt-1"></h3>
      </div>
      <button onclick="closeModal()" class="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs text-slate-200">✕ Close</button>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div class="space-y-3">
        <div id="mStage" class="aspect-square rounded-2xl overflow-hidden bg-white border border-slate-700 flex items-center justify-center relative">
          <img id="mImg" src="" alt="" class="w-full h-full object-contain" />
          <span id="mViewLabel" class="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-slate-950/85 text-xs font-mono text-emerald-300"></span>
        </div>
        <div id="mThumbs" class="grid grid-cols-6 gap-2"></div>
        <p id="mAiNote" class="hidden text-[11px] text-amber-300/90">Views marked "AI" were not filmed: they are a representation generated from the real product.</p>
      </div>
      <div class="space-y-4">
        <div class="flex items-baseline justify-between gap-2">
          <span id="mPrice" class="text-3xl font-black text-emerald-400"></span>
          <span id="mCondition" class="px-2.5 py-1 rounded-full text-xs bg-slate-800 text-slate-200"></span>
        </div>
        <p id="mDesc" class="text-sm text-slate-300 leading-relaxed"></p>
        <ul id="mHighlights" class="space-y-1 text-xs text-slate-300"></ul>
        <div id="mTags" class="flex flex-wrap gap-1.5"></div>
        <div id="mPhraseBox" class="p-3 rounded-2xl bg-slate-950/90 border border-slate-800 text-xs text-slate-400">
          🎙️ What the seller said: <i id="mPhrase" class="text-slate-200"></i>
        </div>
        <button id="mAction" class="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-600 hover:from-emerald-400 hover:to-cyan-500 text-slate-950 font-black text-sm shadow-lg shadow-emerald-500/20 transition"></button>
      </div>
    </div>
  </div>
</div>

<div id="cart" class="fixed inset-0 z-50 hidden justify-end bg-black/70" onclick="if (event.target === this) toggleCart(false)">
  <div class="glass w-full max-w-md h-full p-6 flex flex-col justify-between border-l border-slate-800">
    <div class="space-y-4 overflow-y-auto">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 class="text-lg font-extrabold text-white">🛒 Your cart</h3>
        <button onclick="toggleCart(false)" class="px-3 py-1 rounded-lg bg-slate-800 text-xs text-slate-300">✕</button>
      </div>
      <div id="cartItems" class="space-y-3"></div>
    </div>
    <div class="border-t border-slate-800 pt-4 space-y-3">
      <div class="flex items-start justify-between text-base font-bold">
        <span class="text-slate-300">Total:</span>
        <span id="cartTotal" class="text-right text-xl font-black text-emerald-400">—</span>
      </div>
      <button onclick="checkout()" class="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 text-slate-950 font-black text-sm shadow-lg">Place order</button>
    </div>
  </div>
</div>

<script>
const DATA = __CATALOG_DATA__;
const PRODUCTS = DATA.products;
const SYMBOL = { USD: '$', PEN: 'S/', EUR: '€', MXN: '$', COP: '$', ARS: '$', CLP: '$', BOB: 'Bs' };
let activeCategory = 'All';
const activeView = {};
let cart = [];

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const money = (v, cur) => `${SYMBOL[cur] || ''}${Number(v).toLocaleString('en-US', { minimumFractionDigits: v % 1 ? 2 : 0, maximumFractionDigits: 2 })} ${cur}`;
const priceText = (p) => (p.price == null ? 'Ask for price' : money(p.price, p.currency));
const bgClass = (v) => (v.bg === 'dark' ? 'bg-slate-950' : v.bg === 'checker' ? 'bg-checker' : 'bg-white');

function init() {
  document.getElementById('storeName').textContent = DATA.store.name;
  document.getElementById('storeTagline').textContent = DATA.store.tagline || '';
  document.getElementById('logo').textContent = (DATA.store.name || 'M').trim().charAt(0).toUpperCase();
  const u = DATA.usage || {};
  document.getElementById('usageLine').textContent =
    `${(u.api_tokens_used || 0).toLocaleString('en-US')} API tokens • ${u.images_generated || 0} images generated (US$${(u.images_cost_usd || 0).toFixed(2)}) • ${(u.tokens_avoided_by_cache || 0).toLocaleString('en-US')} tokens avoided by cache`;
  renderCategories();
  render();
}

function renderCategories() {
  const cats = ['All', ...new Set(PRODUCTS.map((p) => p.category))];
  const bar = document.getElementById('categoryBar');
  bar.innerHTML = '';
  cats.forEach((cat) => {
    const b = document.createElement('button');
    b.className = `px-3 py-1.5 rounded-xl text-xs font-semibold transition ${activeCategory === cat ? 'bg-emerald-500 text-slate-950 font-extrabold shadow' : 'bg-slate-900 text-slate-300 border border-slate-800 hover:border-slate-700'}`;
    b.textContent = cat === 'All' ? `All (${PRODUCTS.length})` : `${cat} (${PRODUCTS.filter((p) => p.category === cat).length})`;
    b.onclick = () => { activeCategory = cat; renderCategories(); render(); };
    bar.appendChild(b);
  });
}

function render() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  const sort = document.getElementById('sort').value;
  let list = PRODUCTS.filter((p) => (activeCategory === 'All' || p.category === activeCategory) &&
    (!q || `${p.title} ${p.short_description} ${p.category} ${p.tags.join(' ')}`.toLowerCase().includes(q)));
  const pv = (p) => (p.price == null ? Infinity : p.price);
  if (sort === 'price_asc') list.sort((a, b) => pv(a) - pv(b));
  if (sort === 'price_desc') list.sort((a, b) => (b.price ?? -1) - (a.price ?? -1));
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  document.getElementById('emptyMsg').classList.toggle('hidden', list.length > 0);
  list.forEach((p) => {
    const views = p.views;
    const view = views.find((v) => v.key === activeView[p.id]) || views[0];
    const pills = views.map((v) => `<button onclick="setView('${p.id}','${v.key}')" class="px-2 py-1 rounded-lg text-[10px] font-semibold transition ${v.key === (view && view.key) ? 'bg-emerald-500 text-slate-950 font-bold' : 'bg-slate-900/90 text-slate-300 hover:bg-slate-800 border border-slate-700/80'}">${esc(v.label)}${v.ai_inferred ? ' · AI' : ''}</button>`).join('');
    const card = document.createElement('div');
    card.className = 'glass rounded-3xl overflow-hidden flex flex-col justify-between border border-slate-800/90 hover:border-emerald-500/40 transition group';
    card.innerHTML = `
      <div>
        <div class="aspect-square ${view ? bgClass(view) : 'bg-slate-900'} relative overflow-hidden flex items-center justify-center cursor-pointer" onclick="openModal('${p.id}')">
          ${view ? `<img src="${view.src}" alt="${esc(p.title)}" loading="lazy" class="w-full h-full object-contain group-hover:scale-105 transition duration-300" />` : ''}
          ${view && view.ai_inferred ? '<span class="absolute top-3 left-3 px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-400 text-slate-950">AI-generated view</span>' : ''}
          ${p.real_photo_only ? '<span class="absolute top-3 left-3 px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-950/85 text-slate-200 border border-slate-600">Real photo from the video</span>' : ''}
        </div>
        ${views.length > 1 ? `<div class="px-4 py-2.5 bg-slate-950/90 border-b border-slate-800/80 flex flex-wrap gap-1">${pills}</div>` : ''}
        <div class="p-5 space-y-2">
          <div class="text-[11px] text-cyan-300/90">${esc(p.category)}</div>
          <h3 class="text-base font-extrabold text-white leading-snug">${esc(p.title)}</h3>
          <p class="text-xs text-slate-300 line-clamp-2">${esc(p.short_description)}</p>
          <div class="pt-1 flex items-center justify-between gap-2">
            <span class="${p.price == null ? 'text-base text-amber-300' : 'text-2xl text-emerald-400'} font-black">${esc(priceText(p))}</span>
            ${p.condition ? `<span class="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-300">${esc(p.condition)}</span>` : ''}
          </div>
        </div>
      </div>
      <div class="p-4 pt-0 flex gap-2">
        <button onclick="openModal('${p.id}')" class="px-3 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-200 transition">View details</button>
        ${p.price == null
          ? `<button onclick="openModal('${p.id}')" class="flex-1 py-2.5 rounded-xl bg-amber-400/90 hover:bg-amber-300 text-slate-950 font-extrabold text-xs transition">Ask for price</button>`
          : `<button onclick="addToCart('${p.id}')" class="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-600 hover:from-emerald-400 hover:to-cyan-500 text-slate-950 font-extrabold text-xs shadow-lg shadow-emerald-500/15 transition">+ Add to cart</button>`}
      </div>`;
    grid.appendChild(card);
  });
}

window.setView = (id, key) => { activeView[id] = key; render(); };

window.openModal = (id) => {
  const p = PRODUCTS.find((x) => x.id === id);
  if (!p) return;
  document.getElementById('mCategory').textContent = p.category;
  document.getElementById('mTitle').textContent = p.title;
  document.getElementById('mPrice').textContent = priceText(p);
  document.getElementById('mPrice').className = `${p.price == null ? 'text-xl text-amber-300' : 'text-3xl text-emerald-400'} font-black`;
  const cond = document.getElementById('mCondition');
  cond.textContent = p.condition; cond.classList.toggle('hidden', !p.condition);
  document.getElementById('mDesc').textContent = p.description || p.short_description;
  document.getElementById('mHighlights').innerHTML = p.highlights.map((h) => `<li>✔ ${esc(h)}</li>`).join('');
  document.getElementById('mTags').innerHTML = p.tags.map((t) => `<span class="px-2 py-0.5 rounded-full bg-slate-800 text-[10px] text-slate-300">#${esc(t)}</span>`).join('');
  document.getElementById('mPhrase').textContent = p.seller_phrase ? `"${p.seller_phrase}"` : '';
  document.getElementById('mPhraseBox').classList.toggle('hidden', !p.seller_phrase);
  document.getElementById('mAiNote').classList.toggle('hidden', !p.views.some((v) => v.ai_inferred));
  const action = document.getElementById('mAction');
  if (p.price == null) {
    action.textContent = 'Ask the seller for the price';
    action.onclick = () => alert(`Inquiry sent for: ${p.title}`);
  } else {
    action.textContent = `+ Add to cart • ${priceText(p)}`;
    action.onclick = () => { addToCart(p.id); closeModal(); };
  }
  const setStage = (v) => {
    document.getElementById('mStage').className = `aspect-square rounded-2xl overflow-hidden ${bgClass(v)} border border-slate-700 flex items-center justify-center relative`;
    document.getElementById('mImg').src = v.src;
    document.getElementById('mViewLabel').textContent = v.ai_inferred ? `${v.label} · AI-generated view` : v.label;
  };
  if (p.views.length) setStage(p.views.find((v) => v.key === activeView[p.id]) || p.views[0]);
  const thumbs = document.getElementById('mThumbs');
  thumbs.innerHTML = '';
  p.views.forEach((v) => {
    const b = document.createElement('button');
    b.className = `aspect-square rounded-xl overflow-hidden ${bgClass(v)} border border-slate-700 hover:border-emerald-400 relative`;
    b.innerHTML = `<img src="${v.src}" alt="" class="w-full h-full object-contain" /><span class="absolute bottom-0.5 inset-x-0.5 bg-slate-950/85 text-[9px] text-slate-200 truncate rounded px-1">${esc(v.label)}${v.ai_inferred ? ' · AI' : ''}</span>`;
    b.onclick = () => setStage(v);
    thumbs.appendChild(b);
  });
  const m = document.getElementById('modal');
  m.classList.remove('hidden'); m.classList.add('flex');
};

window.closeModal = () => { const m = document.getElementById('modal'); m.classList.add('hidden'); m.classList.remove('flex'); };

window.addToCart = (id) => {
  const p = PRODUCTS.find((x) => x.id === id);
  if (!p || p.price == null) return;
  const item = cart.find((i) => i.id === id);
  if (item) item.qty += 1; else cart.push({ id, qty: 1 });
  updateCart(); toggleCart(true);
};

window.changeQty = (id, delta) => {
  const item = cart.find((i) => i.id === id);
  if (!item) return;
  item.qty += delta;
  cart = cart.filter((i) => i.qty > 0);
  updateCart();
};

function totals() {
  const byCur = {};
  cart.forEach((i) => { const p = PRODUCTS.find((x) => x.id === i.id); byCur[p.currency] = (byCur[p.currency] || 0) + p.price * i.qty; });
  return byCur;
}

function updateCart() {
  document.getElementById('cartCount').textContent = cart.reduce((a, i) => a + i.qty, 0);
  const t = totals();
  document.getElementById('cartTotal').innerHTML = Object.keys(t).length ? Object.entries(t).map(([c, v]) => esc(money(v, c))).join('<br>') : '—';
  const box = document.getElementById('cartItems');
  if (!cart.length) { box.innerHTML = '<div class="text-xs text-slate-400 py-8 text-center">Your cart is empty.</div>'; return; }
  box.innerHTML = cart.map((i) => {
    const p = PRODUCTS.find((x) => x.id === i.id);
    const thumb = p.views[0] ? `<img src="${p.views[0].src}" alt="" class="w-14 h-14 rounded-xl object-contain bg-white shrink-0" />` : '';
    return `<div class="p-3 rounded-2xl bg-slate-900 border border-slate-800 flex items-center gap-3">${thumb}
      <div class="flex-1 min-w-0"><div class="text-xs font-bold text-white truncate">${esc(p.title)}</div><div class="text-xs text-emerald-400 font-bold">${esc(money(p.price, p.currency))}</div></div>
      <div class="flex items-center gap-1"><button onclick="changeQty('${p.id}',-1)" class="w-6 h-6 rounded bg-slate-800 text-slate-200">−</button><span class="w-6 text-center text-xs">${i.qty}</span><button onclick="changeQty('${p.id}',1)" class="w-6 h-6 rounded bg-slate-800 text-slate-200">+</button></div></div>`;
  }).join('');
}

window.toggleCart = (open) => { const d = document.getElementById('cart'); d.classList.toggle('hidden', !open); d.classList.toggle('flex', open); if (open) updateCart(); };

window.checkout = () => {
  if (!cart.length) return alert('Add at least one product to the cart.');
  const lines = cart.map((i) => { const p = PRODUCTS.find((x) => x.id === i.id); return `• ${i.qty} x ${p.title} (${money(p.price * i.qty, p.currency)})`; }).join('\n');
  const total = Object.entries(totals()).map(([c, v]) => money(v, c)).join(' + ');
  alert(`Order ready:\n\n${lines}\n\nTotal: ${total}`);
};

document.getElementById('search').addEventListener('input', render);
document.getElementById('sort').addEventListener('change', render);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeModal(); toggleCart(false); } });
init();
</script>
</body>
</html>
"""
