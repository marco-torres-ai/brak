import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

META_MUSE_API_KEY: str = os.getenv("META_MUSE_API_KEY", "")
MUSE_API_BASE_URL: str = os.getenv("MUSE_API_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
MUSE_VOICE_MODEL: str = os.getenv("MUSE_VOICE_MODEL", "muse-voice-transcribe-1.0")
MUSE_SPARK_MODEL: str = os.getenv("MUSE_SPARK_MODEL", "muse-spark-1.3")
# Razonamiento de Spark para la extracción de productos (Fase 2): none|minimal|low|medium|high|xhigh|max
MUSE_SPARK_REASONING_PHASE2: str = os.getenv("MUSE_SPARK_REASONING_PHASE2", "low")
# Enviar el video a Spark en la Fase 2 para corregir nombres mal transcritos (cuesta ~1.5k tokens por video de 15 s)
MUSE_SPARK_PHASE2_VIDEO: bool = os.getenv("MUSE_SPARK_PHASE2_VIDEO", "1").lower() not in ("0", "false", "no")
MUSE_SPARK_PHASE2_VIDEO_FPS: float = float(os.getenv("MUSE_SPARK_PHASE2_VIDEO_FPS", "1"))
# Verificación automática de la Fase 2 (cruce audio/video + 2da pasada dirigida solo si hay anomalías)
MUSE_PHASE2_VERIFY: bool = os.getenv("MUSE_PHASE2_VERIFY", "1").lower() not in ("0", "false", "no")
MUSE_PHASE2_VERIFY_MIN_CONFIDENCE: float = float(os.getenv("MUSE_PHASE2_VERIFY_MIN_CONFIDENCE", "0.7"))
MUSE_PHASE2_VERIFY_MAX_WINDOWS: int = int(os.getenv("MUSE_PHASE2_VERIFY_MAX_WINDOWS", "6"))
# Objeto mostrado para vender sin precio dicho: CONSULTAR_PRECIO (entra al catálogo) o EXCLUDE (solo informe)
MUSE_MISSING_PRICE_POLICY: str = os.getenv("MUSE_MISSING_PRICE_POLICY", "CONSULTAR_PRECIO").upper()
# Fase 3: muse-spark-1.3 (visión, reemplaza a Glimmer) elige el mejor frame y ubica el objeto; sam-3.1 lo recorta.
MUSE_SAM_MODEL: str = os.getenv("MUSE_SAM_MODEL", "sam-3.1")
MUSE_SPARK_REASONING_PHASE3: str = os.getenv("MUSE_SPARK_REASONING_PHASE3", "low")
MUSE_PHASE3_MIN_IOU: float = float(os.getenv("MUSE_PHASE3_MIN_IOU", "0.3"))
MUSE_PHASE3_VERIFY_CUTOUTS: bool = os.getenv("MUSE_PHASE3_VERIFY_CUTOUTS", "1").lower() not in ("0", "false", "no")
# Fase 4: muse-image-1.0 (US$0.01 por imagen generada; no se cobra por token).
MUSE_IMAGE_MODEL: str = os.getenv("MUSE_IMAGE_MODEL", "muse-image-1.0")
MUSE_IMAGE_SIZE: str = os.getenv("MUSE_IMAGE_SIZE", "1024x1024")
MUSE_IMAGE_COST_USD: float = float(os.getenv("MUSE_IMAGE_COST_USD", "0.01"))
MUSE_IMAGE_PARALLEL: int = int(os.getenv("MUSE_IMAGE_PARALLEL", "6"))
MUSE_PHASE4_LIFESTYLE: bool = os.getenv("MUSE_PHASE4_LIFESTYLE", "1").lower() not in ("0", "false", "no")
# Idiomas para sesgar el reconocimiento de voz (vacío = detección automática)
MUSE_VOICE_LANGUAGES: list = [
    lang.strip() for lang in os.getenv("MUSE_VOICE_LANGUAGES", "Spanish,English").split(",") if lang.strip()
]
MAX_VIDEOS: int = int(os.getenv("MAX_VIDEOS", "3"))
MAX_DURATION_SEC: float = float(os.getenv("MAX_DURATION_SEC", "60.0"))
LAPLACIAN_BLUR_THRESHOLD: float = float(os.getenv("LAPLACIAN_BLUR_THRESHOLD", "80.0"))

# Image & video quality thresholds
MIN_BRIGHTNESS_RESCUE_THRESHOLD: float = 75.0
CORRUPT_DARK_THRESHOLD: float = 8.0
RESCUE_BLUR_FLOOR: float = 35.0

# Vision token estimation constants (standard multimodal tile cost per frame)
ESTIMATED_VISION_TOKENS_PER_FRAME: int = 258
ESTIMATED_AUDIO_TOKENS_SAVED_PER_SEC: int = 12

STORAGE_DIR: Path = BASE_DIR / os.getenv("STORAGE_DIR", "storage")
FRONTEND_DIR: Path = BASE_DIR / "frontend"
ROOT_SESSION_STATE_PATH: Path = BASE_DIR / "session_state.json"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def create_initial_session_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if not session_id:
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return {
        "session_id": session_id,
        "created_at": now_str,
        "current_phase_completed": 0,
        "videos": [],
        "semantic_products": [],
        "grounded_products": [],
        "studio_products": [],
        "catalog_output": None,
        "token_audit": {
            "total_input_tokens": 0,
            "total_image_tokens": 0,
            "total_output_tokens": 0,
            "grand_total_tokens": 0,
            "estimated_tokens_saved": 0,
            "logs": [],
        },
    }


def get_session_dir(session_id: str) -> Path:
    session_dir = STORAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_session_state(state: Dict[str, Any]) -> Path:
    session_id = state.get("session_id", "sess_default")
    session_dir = get_session_dir(session_id)
    session_file = session_dir / "session_state.json"

    serialized = json.dumps(state, indent=4, ensure_ascii=False)
    session_file.write_text(serialized, encoding="utf-8")
    ROOT_SESSION_STATE_PATH.write_text(serialized, encoding="utf-8")
    return session_file


def load_session_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    if session_id:
        candidate = STORAGE_DIR / session_id / "session_state.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))

    if ROOT_SESSION_STATE_PATH.exists():
        try:
            return json.loads(ROOT_SESSION_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return create_initial_session_state(session_id)

