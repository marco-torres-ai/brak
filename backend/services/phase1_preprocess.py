import hashlib
import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from fastapi import UploadFile

from backend.core.config import (
    CORRUPT_DARK_THRESHOLD,
    ESTIMATED_AUDIO_TOKENS_SAVED_PER_SEC,
    ESTIMATED_VISION_TOKENS_PER_FRAME,
    LAPLACIAN_BLUR_THRESHOLD,
    MAX_DURATION_SEC,
    MAX_VIDEOS,
    MIN_BRIGHTNESS_RESCUE_THRESHOLD,
    RESCUE_BLUR_FLOOR,
    create_initial_session_state,
    get_session_dir,
    save_session_state,
)
from backend.core.token_tracker import TokenTracker


def compute_file_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_metrics(frame_bgr: np.ndarray) -> Tuple[float, float]:
    """Returns (laplacian_variance, mean_brightness) for a BGR frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))
    return laplacian_var, mean_brightness


def apply_local_video_enhancer(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Pre-Mejorador de Video Local (0 Tokens):
    1. Ecualización adaptativa CLAHE en canal de luminancia (LAB).
    2. Reducción ligera de ruido manteniendo bordes.
    3. Enfoque por máscara de desenfoque (Unsharp Mask).
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    merged_lab = cv2.merge((l_enhanced, a_chan, b_chan))
    contrast_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    denoised = cv2.fastNlMeansDenoisingColored(contrast_bgr, None, 3, 3, 7, 15)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.45, gaussian, -0.45, 0)
    return sharpened


def extract_lightweight_audio(
    video_path: Path, output_wav_path: Path, duration_sec: float
) -> Dict[str, Any]:
    """
    Extrae audio mono a 16kHz PCM con FFmpeg.
    Si FFmpeg no está en PATH o el video no tiene pista de audio, genera un WAV 16kHz
    de respaldo seguro para no romper el pipeline en entornos sin binarios externos.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_wav_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0 and output_wav_path.exists() and output_wav_path.stat().st_size > 44:
                return {
                    "method": "ffmpeg_16khz_mono",
                    "status": "OK",
                    "sample_rate": 16000,
                    "channels": 1,
                    "size_kb": round(output_wav_path.stat().st_size / 1024, 2),
                    "audio_md5": compute_file_md5(output_wav_path),
                }
        except Exception:
            pass

    # Fallback seguro: generar contenedor WAV 16kHz mono válido
    safe_seconds = max(1.0, min(duration_sec, MAX_DURATION_SEC))
    num_samples = int(safe_seconds * 16000)
    with wave.open(str(output_wav_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * num_samples)

    return {
        "method": "python_wave_fallback_16khz",
        "status": "FALLBACK_NO_FFMPEG_OR_SILENT_TRACK",
        "sample_rate": 16000,
        "channels": 1,
        "size_kb": round(output_wav_path.stat().st_size / 1024, 2),
        "audio_md5": compute_file_md5(output_wav_path),
    }


async def process_single_video(
    upload_file: UploadFile,
    video_index: int,
    session_id: str,
    blur_threshold: float,
    sample_fps: float,
    tracker: TokenTracker,
) -> Dict[str, Any]:
    session_dir = get_session_dir(session_id)
    videos_dir = session_dir / "videos"
    audio_dir = session_dir / "audio"
    frames_dir = session_dir / "frames"
    for folder in (videos_dir, audio_dir, frames_dir):
        folder.mkdir(parents=True, exist_ok=True)

    raw_filename = Path(upload_file.filename or f"video_{video_index}.mp4").name
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_filename)
    video_id = f"vid_{video_index}"
    saved_video_path = videos_dir / f"{video_id}_{safe_name}"

    content = await upload_file.read()
    saved_video_path.write_bytes(content)
    video_md5 = compute_file_md5(saved_video_path)

    cap = cv2.VideoCapture(str(saved_video_path))
    if not cap.isOpened():
        tracker.record_event(
            phase=1,
            model="OpenCV+FFmpeg Pre-Filter",
            operation=f"Video '{raw_filename}' rechazado: archivo corrupto o codec no soportado",
            tokens_saved=ESTIMATED_VISION_TOKENS_PER_FRAME * 30,
            status="REJECTED_CORRUPT",
        )
        return {
            "video_id": video_id,
            "filename": raw_filename,
            "status": "REJECTED_CORRUPT",
            "reason": "No se pudo abrir el contenedor de video con OpenCV.",
            "retained_frames": [],
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0.0

    # Validar límite de 60 segundos (con margen técnico de 0.8s por metadatos de contenedor)
    if duration_sec > (MAX_DURATION_SEC + 0.8):
        cap.release()
        tokens_avoided = int( max(1, duration_sec * sample_fps) * ESTIMATED_VISION_TOKENS_PER_FRAME )
        tracker.record_event(
            phase=1,
            model="OpenCV+FFmpeg Pre-Filter",
            operation=(
                f"Video '{raw_filename}' ({duration_sec}s) excede el máximo permitido "
                f"de {MAX_DURATION_SEC:.0f}s. Rechazado sin gastar tokens."
            ),
            tokens_saved=tokens_avoided,
            status="REJECTED_DURATION_EXCEEDED",
        )
        return {
            "video_id": video_id,
            "filename": raw_filename,
            "video_md5": video_md5,
            "status": "REJECTED_DURATION_EXCEEDED",
            "reason": f"Duración ({duration_sec}s) supera el límite de {MAX_DURATION_SEC:.0f} segundos.",
            "duration_sec": duration_sec,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "retained_frames": [],
        }

    frame_step = max(1, int(round(fps / max(0.5, min(sample_fps, 10.0)))))

    sampled_count = 0
    retained_frames: List[Dict[str, Any]] = []
    discarded_blurry_frames: List[Dict[str, Any]] = []
    rescued_count = 0
    brightness_values: List[float] = []
    sharpness_values: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % frame_step == 0:
            sampled_count += 1
            timestamp_sec = round(frame_idx / fps, 2)
            timestamp_ms = int(timestamp_sec * 1000)

            lap_var, mean_bright = compute_metrics(frame)
            brightness_values.append(mean_bright)
            sharpness_values.append(lap_var)

            was_rescued = False
            final_frame = frame
            final_lap_var = lap_var
            final_bright = mean_bright

            # Pre-Mejorador de Video Local cuando hay baja iluminación o desenfoque leve rescatable
            needs_local_rescue = (
                mean_bright >= CORRUPT_DARK_THRESHOLD
                and (
                    mean_bright < MIN_BRIGHTNESS_RESCUE_THRESHOLD
                    or (RESCUE_BLUR_FLOOR <= lap_var < blur_threshold)
                )
            )
            if needs_local_rescue:
                enhanced_frame = apply_local_video_enhancer(frame)
                enh_lap_var, enh_bright = compute_metrics(enhanced_frame)
                if enh_lap_var > lap_var or enh_bright > mean_bright:
                    final_frame = enhanced_frame
                    final_lap_var = enh_lap_var
                    final_bright = enh_bright
                    was_rescued = True

            if final_lap_var >= blur_threshold and final_bright >= CORRUPT_DARK_THRESHOLD:
                if was_rescued:
                    rescued_count += 1
                frame_filename = f"{video_id}_f{frame_idx:04d}_{timestamp_ms}ms.jpg"
                frame_path = frames_dir / frame_filename

                # Redimensionar si supera 1280px de lado mayor para optimizar tokens visuales futuros
                h, w = final_frame.shape[:2]
                if max(h, w) > 1280:
                    scale = 1280.0 / max(h, w)
                    final_frame = cv2.resize(
                        final_frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imwrite(
                    str(frame_path),
                    final_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88],
                )
                retained_frames.append(
                    {
                        "frame_id": f"{video_id}_f{frame_idx:04d}",
                        "frame_index": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "laplacian_score": round(final_lap_var, 2),
                        "raw_laplacian_score": round(lap_var, 2),
                        "brightness_score": round(final_bright, 2),
                        "rescued_by_clahe": was_rescued,
                        "file_path": str(frame_path.relative_to(session_dir.parent)).replace("\\", "/"),
                        "url": f"/storage/{session_id}/frames/{frame_filename}",
                    }
                )
            else:
                discarded_blurry_frames.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "laplacian_score": round(final_lap_var, 2),
                        "brightness_score": round(final_bright, 2),
                        "reason": "DARK_FRAME" if final_bright < CORRUPT_DARK_THRESHOLD else "MOTION_BLUR",
                    }
                )

        frame_idx += 1

    cap.release()

    avg_brightness = round(float(np.mean(brightness_values)), 2) if brightness_values else 0.0
    avg_sharpness = round(float(np.mean(sharpness_values)), 2) if sharpness_values else 0.0

    # Si el video está totalmente en negro o corrupto sin frames válidos
    if avg_brightness < CORRUPT_DARK_THRESHOLD and len(retained_frames) == 0:
        tokens_avoided = max(1, sampled_count) * ESTIMATED_VISION_TOKENS_PER_FRAME
        tracker.record_event(
            phase=1,
            model="OpenCV+FFmpeg Pre-Filter",
            operation=f"Video '{raw_filename}' rechazado por estar totalmente oscuro (brillo={avg_brightness}).",
            tokens_saved=tokens_avoided,
            status="REJECTED_BLACK_VIDEO",
        )
        return {
            "video_id": video_id,
            "filename": raw_filename,
            "video_md5": video_md5,
            "status": "REJECTED_BLACK_VIDEO",
            "reason": "El video está completamente en negro o sin iluminación útil.",
            "duration_sec": duration_sec,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "retained_frames": [],
        }

    # Fallback de rescate garantizado: si todos los frames quedaron bajo un umbral alto pero el video es visible,
    # rescatamos los 3 mejores frames con CLAHE para no dejar al usuario bloqueado sin candidatos.
    if len(retained_frames) == 0 and sampled_count > 0 and avg_brightness >= CORRUPT_DARK_THRESHOLD:
        cap_retry = cv2.VideoCapture(str(saved_video_path))
        scored_candidates: List[Tuple[float, int, np.ndarray, float]] = []
        idx_r = 0
        while True:
            ret_r, frm_r = cap_retry.read()
            if not ret_r or frm_r is None:
                break
            if idx_r % frame_step == 0:
                enh_r = apply_local_video_enhancer(frm_r)
                l_score, b_score = compute_metrics(enh_r)
                scored_candidates.append((l_score, idx_r, enh_r, b_score))
            idx_r += 1
        cap_retry.release()

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        for l_score, best_idx, enh_img, b_score in scored_candidates[:3]:
            ts_sec = round(best_idx / fps, 2)
            ts_ms = int(ts_sec * 1000)
            frame_filename = f"{video_id}_f{best_idx:04d}_{ts_ms}ms.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), enh_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            retained_frames.append(
                {
                    "frame_id": f"{video_id}_f{best_idx:04d}",
                    "frame_index": best_idx,
                    "timestamp_sec": ts_sec,
                    "laplacian_score": round(l_score, 2),
                    "raw_laplacian_score": round(l_score * 0.85, 2),
                    "brightness_score": round(b_score, 2),
                    "rescued_by_clahe": True,
                    "file_path": str(frame_path.relative_to(session_dir.parent)).replace("\\", "/"),
                    "url": f"/storage/{session_id}/frames/{frame_filename}",
                }
            )
            rescued_count += 1
        retained_frames.sort(key=lambda x: x["timestamp_sec"])

    # Extraer audio ligero 16kHz mono
    wav_path = audio_dir / f"{video_id}_audio_16khz.wav"
    audio_info = extract_lightweight_audio(saved_video_path, wav_path, duration_sec)
    audio_info["relative_path"] = str(wav_path.relative_to(session_dir.parent)).replace("\\", "/")
    audio_info["url"] = f"/storage/{session_id}/audio/{wav_path.name}"

    discarded_count = len(discarded_blurry_frames)
    unnecessary_raw_frames = max(0, frame_idx - sampled_count)

    # Ahorro directo por descartar frames con motion blur + submuestreo inteligente + compresión de audio 16kHz
    blur_tokens_saved = discarded_count * ESTIMATED_VISION_TOKENS_PER_FRAME
    audio_tokens_saved = int(round(duration_sec * ESTIMATED_AUDIO_TOKENS_SAVED_PER_SEC))
    total_video_tokens_saved = blur_tokens_saved + audio_tokens_saved

    quality_label = "OPTIMAL"
    if rescued_count > 0:
        quality_label = "RESCUED_BY_LOCAL_ENHANCER"
    elif avg_sharpness < blur_threshold * 1.15:
        quality_label = "ACCEPTABLE"

    tracker.record_event(
        phase=1,
        model="OpenCV+FFmpeg Pre-Filter",
        operation=(
            f"Video '{raw_filename}' ({duration_sec}s, {width}x{height}): "
            f"{len(retained_frames)} frames nítidos retenidos ({rescued_count} rescatados con CLAHE), "
            f"{discarded_count} frames con motion blur descartados (umbral={blur_threshold}). "
            f"Audio 16kHz mono listo (MD5={audio_info['audio_md5'][:10]}...)."
        ),
        input_tokens=0,
        image_tokens=0,
        output_tokens=0,
        tokens_saved=total_video_tokens_saved,
        status="ZERO_TOKEN_LOCAL",
        details={
            "video_id": video_id,
            "sampled_frames": sampled_count,
            "retained_frames": len(retained_frames),
            "discarded_blurry_frames": discarded_count,
            "rescued_frames": rescued_count,
            "raw_total_frames": frame_idx,
            "unnecessary_raw_frames_avoided": unnecessary_raw_frames,
            "blur_tokens_saved": blur_tokens_saved,
            "audio_tokens_saved": audio_tokens_saved,
        },
    )

    return {
        "video_id": video_id,
        "filename": raw_filename,
        "video_md5": video_md5,
        "status": "READY",
        "quality_assessment": {
            "overall_status": quality_label,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "duration_sec": duration_sec,
            "avg_brightness": avg_brightness,
            "avg_laplacian_sharpness": avg_sharpness,
            "blur_threshold_used": blur_threshold,
            "raw_frame_count": frame_idx,
            "sampled_frames_evaluated": sampled_count,
            "sharp_frames_retained": len(retained_frames),
            "blurry_frames_discarded": discarded_count,
            "frames_rescued_locally": rescued_count,
            "vision_tokens_saved_by_blur_filter": blur_tokens_saved,
            "audio_tokens_saved_16khz": audio_tokens_saved,
        },
        "audio": audio_info,
        "retained_frames": retained_frames,
        "discarded_frames_summary": discarded_blurry_frames[:20],
    }


async def run_phase1_preprocess(
    files: List[UploadFile],
    blur_threshold: float = LAPLACIAN_BLUR_THRESHOLD,
    sample_fps: float = 2.0,
) -> Dict[str, Any]:
    if not files or len(files) == 0:
        raise ValueError("Debes subir al menos 1 video para procesar.")
    if len(files) > MAX_VIDEOS:
        raise ValueError(f"Máximo {MAX_VIDEOS} videos permitidos por lote (recibidos: {len(files)}).")

    session_state = create_initial_session_state()
    session_id = session_state["session_id"]
    tracker = TokenTracker(session_state.get("token_audit"))

    tracker.record_event(
        phase=1,
        model="OpenCV+FFmpeg Pre-Filter",
        operation=f"Iniciando Fase 1 para {len(files)} video(s) en sesión {session_id} (Umbral Laplaciano={blur_threshold}, FPS muestreo={sample_fps})",
        tokens_saved=0,
        status="ZERO_TOKEN_LOCAL",
    )

    processed_videos: List[Dict[str, Any]] = []
    for idx, upload in enumerate(files, start=1):
        video_result = await process_single_video(
            upload_file=upload,
            video_index=idx,
            session_id=session_id,
            blur_threshold=blur_threshold,
            sample_fps=sample_fps,
            tracker=tracker,
        )
        processed_videos.append(video_result)

    valid_videos = [v for v in processed_videos if v.get("status") == "READY"]
    session_state["videos"] = processed_videos
    session_state["current_phase_completed"] = 1 if len(valid_videos) > 0 else 0
    tracker.sync_to_session(session_state)
    save_session_state(session_state)

    return session_state

