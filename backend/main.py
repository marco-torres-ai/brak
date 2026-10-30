import asyncio
from typing import Any, Dict, List, Optional
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.core.config import (
    FRONTEND_DIR,
    LAPLACIAN_BLUR_THRESHOLD,
    MAX_DURATION_SEC,
    MAX_VIDEOS,
    META_MUSE_API_KEY,
    STORAGE_DIR,
    load_session_state,
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
    description="Pipeline inteligente de conversión de video a catálogo e-commerce con Meta Muse + SAM 3.1 y auditoría estricta de tokens.",
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


@app.get("/api/v1/session/current")
async def get_current_session():
    return load_session_state()


@app.post("/api/v1/phase1/preprocess")
async def api_phase1_preprocess(
    files: List[UploadFile] = File(..., description="De 1 a 3 archivos de video (máx. 60s cada uno)"),
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
            detail=f"Error procesando Fase 1: {str(exc)}",
        )


@app.post("/api/v1/phase2/semantic")
async def api_phase2_semantic(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        state = await asyncio.to_thread(lambda: asyncio.run(run_phase2_semantic(session_id=session_id)))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando Fase 2: {str(exc)}",
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
            detail=f"Error actualizando productos de Fase 2: {str(exc)}",
        )


@app.post("/api/v1/phase3/vision-sam")
async def api_phase3_vision_sam(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        state = await asyncio.to_thread(lambda: asyncio.run(run_phase3_vision_sam(session_id=session_id)))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando Fase 3 (Muse Glimmer + SAM 3.1): {str(exc)}",
        )


@app.post("/api/v1/phase4/studio")
async def api_phase4_studio(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        # La Fase 4 tarda minutos (muse-image): se ejecuta en otro hilo para no congelar el servidor mientras tanto.
        state = await asyncio.to_thread(lambda: asyncio.run(run_phase4_studio(session_id=session_id)))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando Fase 4 (Muse Image): {str(exc)}",
        )


@app.post("/api/v1/phase5/codegen")
async def api_phase5_codegen(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        state = await asyncio.to_thread(lambda: asyncio.run(run_phase5_codegen(session_id=session_id)))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando Fase 5 (catálogo): {str(exc)}",
        )


@app.post("/api/v1/pipeline/run-all")
async def api_run_full_pipeline_auto(
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    try:
        session_id = payload.get("session_id") if payload else None
        # Fases 2 a 5 seguidas (~10 min con videos nuevos): en otro hilo para no congelar el servidor.
        state = await asyncio.to_thread(lambda: asyncio.run(run_full_pipeline_auto(session_id=session_id)))
        return state
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error ejecutando el Flujo Completo Automático (Fases 2->5): {str(exc)}",
        )


@app.get("/api/v1/phase5/catalog-preview", include_in_schema=False)
async def api_phase5_catalog_preview(session_id: Optional[str] = None):
    state = load_session_state(session_id)
    sid = state.get("session_id")
    if not sid or not (STORAGE_DIR / sid / "catalog" / CATALOG_HTML).exists():
        raise HTTPException(status_code=404, detail="El catálogo aún no ha sido generado. Ejecuta primero la Fase 5.")
    # Se sirve desde su carpeta para que las rutas relativas de images/ funcionen.
    return RedirectResponse(url=f"/storage/{sid}/catalog/{CATALOG_HTML}")


@app.get("/api/v1/phase5/catalog-download", include_in_schema=False)
async def api_phase5_catalog_download(session_id: Optional[str] = None):
    state = load_session_state(session_id)
    sid = state.get("session_id")
    if not sid or not (STORAGE_DIR / sid / "catalog" / CATALOG_HTML).exists():
        raise HTTPException(status_code=404, detail="El catálogo aún no ha sido generado. Ejecuta primero la Fase 5.")
    zip_path = await asyncio.to_thread(build_catalog_zip, sid)
    return FileResponse(str(zip_path), media_type="application/zip", filename=zip_path.name)
