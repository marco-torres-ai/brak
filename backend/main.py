import asyncio
from typing import Any, Dict, List, Optional
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.core import flows, progress, prompts
from backend.core.config import (
    FRONTEND_DIR,
    LAPLACIAN_BLUR_THRESHOLD,
    MAX_DURATION_SEC,
    MAX_VIDEOS,
    META_MUSE_API_KEY,
    STORAGE_DIR,
    load_session_state,
    save_session_state,
)
from backend.services.phase1_preprocess import run_phase1_preprocess
from backend.services.phase2_semantic import (
    run_phase2_semantic,
    update_semantic_products,
)
from backend.services.phase3_vision_sam import run_phase3_vision_sam
from backend.services.phase4_studio import run_phase4_studio
from backend.services.phase5_codegen import (
    CATALOG_HTML,
    build_catalog_zip,
    run_full_pipeline_auto,
    run_phase5_codegen,
)

app = FastAPI(
    title="Muse Video-to-Catalog AI",
    description="Smart video-to-e-commerce-catalog pipeline with Meta Muse + SAM 3.1 and strict token auditing.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "frontend/index.html not found"}, status_code=404)
    return FileResponse(str(index_path))


@app.get("/api/health")
async def health_check():
    masked_key = (
        f"{META_MUSE_API_KEY[:10]}...{META_MUSE_API_KEY[-6:]}"
        if META_MUSE_API_KEY and len(META_MUSE_API_KEY) > 16
        else "NOT_SET"
    )
    current_state = load_session_state()
    return {
        "status": "ok",
        "service": "Muse Video-to-Catalog AI",
        "active_phase_ready": 5,
        "meta_muse_api_key_configured": bool(META_MUSE_API_KEY),
        "meta_muse_api_key_masked": masked_key,
        "limits": {
            "max_videos": MAX_VIDEOS,
            "max_duration_sec": MAX_DURATION_SEC,
            "default_laplacian_blur_threshold": LAPLACIAN_BLUR_THRESHOLD,
        },
        "current_session_id": current_state.get("session_id"),
        "current_phase_completed": current_state.get("current_phase_completed", 0),
    }


def _run_tracked(session_id: Optional[str], phase: int, make_coro: Any, phases: Optional[List[int]] = None) -> Dict[str, Any]:
    """Ejecuta una fase en este hilo (fuera del event loop) y registra su inicio y fin en el progreso en vivo."""
    custom = {str(p): prompts.describe_phase(p) for p in (phases or [phase])}
    custom = {p: d for p, d in custom.items() if d}
    flow_used = flows.describe()
    progress.begin(session_id, phase)
    try:
        state = asyncio.run(make_coro())
    except Exception as exc:
        progress.end(error=str(exc)[:300])
        raise
    progress.end()
    if (custom or flow_used) and isinstance(state, dict) and state.get("session_id"):
        # Trazabilidad: prompts personalizados y flujo distinto del completo (solo si hubo alguno).
        if custom:
            state.setdefault("prompt_customizations", {}).update(custom)
        if flow_used:
            state.setdefault("flow_customizations", {}).update({str(p): flow_used for p in (phases or [phase])})
        save_session_state(state)
    return state


def _prompt_call(fn: Any, *args: Any) -> Any:
    """Escrituras de prompts y flujos: 409 si hay una fase corriendo, 400 si la entrada es inválida."""
    if progress.snapshot().get("running"):
        raise HTTPException(status_code=409, detail="A phase is running: wait for it to finish before changing the prompts or the flow.")
    try:
        return fn(*args)
    except prompts.PromptError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/flows")
async def get_flows():
    """Flujo activo, flujos guardados y opciones (pasos opcionales y parámetros) con su valor efectivo."""
    return flows.snapshot()


@app.put("/api/v1/flows/options")
async def put_flow_options(payload: Dict[str, Any] = Body(...)):
    _prompt_call(flows.update_options, payload.get("options") or {})
    return flows.snapshot()


@app.delete("/api/v1/flows/options")
async def reset_flow_options():
    _prompt_call(flows.reset_options)
    return flows.snapshot()


@app.post("/api/v1/flows")
async def create_flow(payload: Dict[str, Any] = Body(...)):
    _prompt_call(flows.create_flow, str(payload.get("name") or ""), payload.get("copy_from"), payload.get("options"))
    return flows.snapshot()


@app.put("/api/v1/flows/active")
async def set_flow(payload: Dict[str, Any] = Body(...)):
    _prompt_call(flows.set_active, payload.get("flow"))
    return flows.snapshot()


@app.put("/api/v1/flows/{slug}")
async def rename_flow(slug: str, payload: Dict[str, Any] = Body(...)):
    _prompt_call(flows.rename_flow, slug, str(payload.get("name") or ""))
    return flows.snapshot()


@app.delete("/api/v1/flows/{slug}")
async def delete_flow(slug: str):
    _prompt_call(flows.delete_flow, slug)
    return flows.snapshot()


@app.get("/api/v1/prompts")
async def get_prompts():
    """Agentes configurables con su prompt original, el personalizado y el efectivo, más los perfiles."""
    return prompts.snapshot()


@app.put("/api/v1/prompts/{key}")
async def put_prompt(key: str, payload: Dict[str, Any] = Body(...)):
    full = payload.get("full_text")
    _prompt_call(prompts.update_agent, key, str(payload.get("extra") or ""), str(full) if full is not None else None)
    return prompts.snapshot()


@app.delete("/api/v1/prompts/{key}")
async def delete_prompt(key: str):
    _prompt_call(prompts.reset_agent, key)
    return prompts.snapshot()


@app.post("/api/v1/prompts/profiles")
async def create_prompt_profile(payload: Dict[str, Any] = Body(...)):
    _prompt_call(prompts.create_profile, str(payload.get("name") or ""), payload.get("copy_from"))
    return prompts.snapshot()


@app.put("/api/v1/prompts/profiles/active")
async def set_prompt_profile(payload: Dict[str, Any] = Body(...)):
    _prompt_call(prompts.set_active, payload.get("profile"))
    return prompts.snapshot()


@app.put("/api/v1/prompts/profiles/{slug}")
async def rename_prompt_profile(slug: str, payload: Dict[str, Any] = Body(...)):
    _prompt_call(prompts.rename_profile, slug, str(payload.get("name") or ""))
    return prompts.snapshot()


@app.delete("/api/v1/prompts/profiles/{slug}")
async def delete_prompt_profile(slug: str):
    _prompt_call(prompts.delete_profile, slug)
    return prompts.snapshot()


@app.get("/api/v1/prompts/profiles/{slug}/export")
async def export_prompt_profile(slug: str):
    try:
        data = prompts.export_profile(slug)
    except prompts.PromptError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="brak_prompts_{slug}.json"'})


@app.post("/api/v1/prompts/profiles/import")
async def import_prompt_profile(payload: Dict[str, Any] = Body(...)):
    _prompt_call(prompts.import_profile, payload)
    return prompts.snapshot()


@app.get("/api/v1/progress")
async def get_progress(session_id: Optional[str] = None):
    """Qué está haciendo la fase en curso: último evento, avisos de espera de Meta, tokens e imágenes en vivo."""
    return progress.snapshot(session_id)


@app.get("/api/v1/session/current")
async def get_current_session():
    return load_session_state()


@app.post("/api/v1/phase1/preprocess")
async def api_phase1_preprocess(
    files: List[UploadFile] = File(..., description="1 to 3 video files (max. 60 s each)"),
    blur_threshold: float = Form(LAPLACIAN_BLUR_THRESHOLD),
    sample_fps: float = Form(2.0),
):
    try:
        state = await run_phase1_preprocess(
            files=files,
            blur_threshold=blur_threshold,
            sample_fps=sample_fps,
        )
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Phase 1: {str(exc)}",
        )


@app.post("/api/v1/phase2/semantic")
async def api_phase2_semantic(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        state = await asyncio.to_thread(_run_tracked, session_id, 2, lambda: run_phase2_semantic(session_id=session_id))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Phase 2: {str(exc)}",
        )


@app.put("/api/v1/phase2/products")
async def api_phase2_update_products(
    payload: Dict[str, Any] = Body(...),
):
    try:
        products = payload.get("products", [])
        session_id = payload.get("session_id")
        state = await update_semantic_products(
            updated_products=products,
            session_id=session_id,
        )
        return state
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating Phase 2 products: {str(exc)}",
        )


@app.post("/api/v1/phase3/vision-sam")
async def api_phase3_vision_sam(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        state = await asyncio.to_thread(_run_tracked, session_id, 3, lambda: run_phase3_vision_sam(session_id=session_id))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Phase 3 (Muse Glimmer + SAM 3.1): {str(exc)}",
        )


@app.post("/api/v1/phase4/studio")
async def api_phase4_studio(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        # La Fase 4 tarda minutos (muse-image): se ejecuta en otro hilo para no congelar el servidor mientras tanto.
        state = await asyncio.to_thread(_run_tracked, session_id, 4, lambda: run_phase4_studio(session_id=session_id))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Phase 4 (Muse Image): {str(exc)}",
        )


@app.post("/api/v1/phase5/codegen")
async def api_phase5_codegen(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        state = await asyncio.to_thread(_run_tracked, session_id, 5, lambda: run_phase5_codegen(session_id=session_id))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Phase 5 (catalog): {str(exc)}",
        )


@app.post("/api/v1/pipeline/run-all")
async def api_run_full_pipeline_auto(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        # Fases 2 a 5 seguidas (~10 min con videos nuevos): en otro hilo para no congelar el servidor.
        state = await asyncio.to_thread(_run_tracked, session_id, 2, lambda: run_full_pipeline_auto(session_id=session_id), [2, 3, 4, 5])
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error running the full automatic flow (Phases 2->5): {str(exc)}",
        )


@app.get("/api/v1/phase5/catalog-preview", include_in_schema=False)
async def api_phase5_catalog_preview(session_id: Optional[str] = None):
    state = load_session_state(session_id)
    sid = state.get("session_id")
    if not sid or not (STORAGE_DIR / sid / "catalog" / CATALOG_HTML).exists():
        raise HTTPException(status_code=404, detail="The catalog has not been generated yet. Run Phase 5 first.")
    # Se sirve desde su carpeta para que las rutas relativas de images/ funcionen.
    return RedirectResponse(url=f"/storage/{sid}/catalog/{CATALOG_HTML}")


@app.get("/api/v1/phase5/catalog-download", include_in_schema=False)
async def api_phase5_catalog_download(session_id: Optional[str] = None):
    state = load_session_state(session_id)
    sid = state.get("session_id")
    if not sid or not (STORAGE_DIR / sid / "catalog" / CATALOG_HTML).exists():
        raise HTTPException(status_code=404, detail="The catalog has not been generated yet. Run Phase 5 first.")
    zip_path = await asyncio.to_thread(build_catalog_zip, sid)
    return FileResponse(str(zip_path), media_type="application/zip", filename=zip_path.name)
