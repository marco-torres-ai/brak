from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import MUSE_PHASE2_VERIFY, STORAGE_DIR, load_session_state, save_session_state
from backend.core.muse_client import MuseClient
from backend.core.token_tracker import TokenTracker
from backend.services.phase2_verify import run_phase2_verification


async def run_phase2_semantic(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    FASE 2: Identificación de Audio e Intenciones (Muse Voice Transcribe + Muse Spark).
    1. Transcribe el audio 16kHz de cada video apto usando caché MD5 (0 tokens si se repite).
    2. Ejecuta Muse Spark en 1 sola llamada consolidada (Batch JSON Schema) para extraer
       productos EXPLICIT y DEICTIC, precios (incluyendo correcciones en vivo), estado/características
       y ventana temporal expandida (±1.5s), filtrando charla irrelevante.
    """
    state = load_session_state(session_id)
    videos = state.get("videos", [])
    ready_videos = [v for v in videos if v.get("status") == "READY"]

    if not ready_videos:
        raise ValueError(
            "No hay videos procesados en estado READY. Ejecuta primero la Fase 1."
        )

    tracker = TokenTracker(state.get("token_audit"))
    client = MuseClient(tracker)

    transcriptions: List[Dict[str, Any]] = []
    videos_by_id: Dict[str, Dict[str, Any]] = {}

    for video in ready_videos:
        vid_id = video["video_id"]
        videos_by_id[vid_id] = video
        audio_meta = video.get("audio", {})
        rel_path = audio_meta.get("relative_path")
        if not rel_path:
            raise ValueError(f"El video {vid_id} no posee archivo de audio 16kHz de Fase 1.")

        wav_path: Path = STORAGE_DIR / rel_path
        if not wav_path.exists():
            raise ValueError(f"No se encontró el archivo de audio en disco: {wav_path}")

        audio_md5 = audio_meta.get("audio_md5", "")
        duration_sec = float(
            video.get("quality_assessment", {}).get("duration_sec", 15.0)
        )

        trans_result = client.transcribe_video_audio(
            video_id=vid_id,
            filename=video.get("filename", vid_id),
            wav_path=wav_path,
            audio_md5=audio_md5,
            duration_sec=duration_sec,
        )
        video["transcription"] = trans_result
        transcriptions.append(trans_result)

    spark_batch = client.analyze_semantics_with_spark(
        transcriptions=transcriptions,
        videos_by_id=videos_by_id,
    )
    products = spark_batch.get("products", [])

    verification_report: Optional[Dict[str, Any]] = None
    if MUSE_PHASE2_VERIFY:
        products, verification_report = run_phase2_verification(
            tracker=tracker,
            products=products,
            visual_inventory=spark_batch.get("visual_inventory", []),
            transcriptions=transcriptions,
            videos_by_id=videos_by_id,
        )
        MuseClient._apply_segments_to_transcriptions(
            transcriptions,
            {"products": products, "discarded_chatter": spark_batch.get("discarded_chatter", [])},
        )

    state["semantic_products"] = products
    state["phase2_verification"] = verification_report
    state["semantic_summary"] = {
        "batch_md5": spark_batch.get("batch_md5"),
        "cache_hit": spark_batch.get("cache_hit", False),
        "total_products_detected": len(products),
        "discarded_chatter": spark_batch.get("discarded_chatter", []),
    }
    state["current_phase_completed"] = max(int(state.get("current_phase_completed", 1)), 2)

    tracker.sync_to_session(state)
    save_session_state(state)
    return state


async def update_semantic_products(
    updated_products: List[Dict[str, Any]],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Permite al usuario verificar y ajustar precios, nombres o notas de estado en la UI
    antes de disparar la Fase 3 (Visión). No consume tokens.
    """
    state = load_session_state(session_id)
    tracker = TokenTracker(state.get("token_audit"))

    existing_by_id = {
        p["product_id"]: p for p in state.get("semantic_products", [])
    }

    merged_list: List[Dict[str, Any]] = []
    for item in updated_products:
        pid = item.get("product_id")
        base = existing_by_id.get(pid, {})
        merged = {**base, **item}
        if merged.get("price") in (None, ""):
            merged["price"] = None
            merged["price_status"] = "CONSULTAR_PRECIO"
        else:
            merged["price"] = float(merged["price"])
            if base.get("price_status") == "CONSULTAR_PRECIO" or float(base.get("price") or -1) != merged["price"]:
                merged["price_status"] = "USER_SET"
            else:
                merged["price_status"] = base.get("price_status", "SPOKEN")
        merged["user_verified"] = True
        merged_list.append(merged)

    state["semantic_products"] = merged_list
    tracker.record_event(
        phase=2,
        model="Human-in-the-Loop State Editor",
        operation=f"Usuario verificó/actualizó manualmente {len(merged_list)} productos en session_state.json antes de Fase 3.",
        input_tokens=0,
        image_tokens=0,
        output_tokens=0,
        tokens_saved=0,
        status="ZERO_TOKEN_LOCAL",
    )
    tracker.sync_to_session(state)
    save_session_state(state)
    return state

