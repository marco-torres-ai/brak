# AGENT_CONTEXT.md — Memoria Central de Agentes (Muse Video-to-Catalog)

## 1. Identidad y Objetivo del Proyecto
Aplicación web de Hackathon para procesar de **1 a 3 videos cortos (máximo 60s cada uno)** donde un usuario graba un recorrido mostrando productos. El sistema procesa audio y video utilizando la suite oficial de **Meta Muse + SAM 3.1**, optimizando al extremo el uso de tokens y mostrando trazas detalladas en tiempo real (`TokenTracker`), para entregar un **Catálogo E-Commerce Interactivo**.

### Catálogo de Modelos a Utilizar
1. **Pre-filtro / Rescate de Video (OpenCV + FFmpeg):** Valida viabilidad del video, rescata iluminación/nitidez si es necesario, extrae audio mono 16kHz y descarta frames borrosos. (Gasta 0 tokens).
2. **Muse Voice Transcribe:** Extrae texto y timestamps exactos del audio 16kHz.
3. **Muse Spark:** Cerebro semántico. Filtra charla irrelevante, detecta intenciones de venta, extrae precios, estado/características y clasifica menciones como `EXPLICIT` o `DEICTIC` ("esto", "el de acá").
4. **Muse Glimmer:** Modelo multimodal de visión. Recibe un collage `1x3` de los mejores frames del intervalo temporal, identifica qué objeto es "esto", evalúa calidad (`quality_score`), decide si `needs_enhancement` es verdadero/falso, elige el `best_frame` y entrega coordenadas `[ymin, xmin, ymax, xmax]`.
5. **SAM 3.1:** Segmentación quirúrgica (*Object Multiplex*). Toma el `best_frame` y las coordenadas de Glimmer para extraer el recorte PNG transparente del producto.
6. **Muse Image:** Estudio fotográfico virtual. Aplica restauración condicional (solo si `needs_enhancement == true`) y genera variaciones de catálogo (fondo blanco e-commerce + fondo lifestyle).
7. **Muse Code:** Sintetiza y exporta el código del catálogo web interactivo (HTML/Tailwind/JS autocontenido).

---

## 2. Reglas Estrictas de Optimización de Tokens (Desarrollo y Runtime)
- **Regla de Agentes en Antigravity (Desarrollo):**
  - PROHIBIDO escanear recursivamente todo el directorio. Leer siempre este `AGENT_CONTEXT.md` y el `PLAN.md` como únicas guías.
  - PROHIBIDO avanzar a la Fase $N+1$ sin que el usuario haya ejecutado las pruebas y confirmado `TESTED_OK`.
  - PROHIBIDO modificar código de fases marcadas como `TESTED_OK`.
- **Regla de Tokens en Muse API (Runtime):**
  - **Cero bucles de conversación entre agentes:** El pipeline es un grafo secuencial determinista controlado por estado en disco (`session_state.json`).
  - **Caché por Hash MD5:** Si se re-ejecuta un audio o video idéntico en las pruebas, la API no se vuelve a llamar (0 tokens).
  - **Técnica de Collage `1x3`:** Los frames candidatos de un producto se unen en un único lienzo horizontal numerado `[Frame 1 | Frame 2 | Frame 3]` para gastar 1 sola imagen en `Muse Glimmer`.
  - **Auditoría Obligatoria:** Toda invocación a los modelos pasa por `backend/core/token_tracker.py` y se registra en el estado de sesión y en la UI.

---

## 3. Configuración Global y Entorno
- **Variable de Entorno (`.env`):** `META_MUSE_API_KEY` (la clave vive SOLO en `.env`; nunca escribirla en documentación ni en código).
- **API real:** Meta Model API — `https://api.meta.ai/v1`, cabecera `Authorization: Bearer <META_MUSE_API_KEY>`, compatible con el SDK de OpenAI. Docs: https://dev.meta.ai/docs/models.md. Cliente HTTP: `backend/core/muse_api.py`.
- **Modelos disponibles en la API (verificado con `GET /v1/models` el 2026-10-29):** `muse-voice-transcribe-1.0` (`POST /asr/transcribe`), `muse-spark-1.3` / `1.2` / `1.1` (`POST /chat/completions`, acepta imágenes), `muse-image-1.0`, `sam-3.1`, `muse-wonder-1.0`.
- **NO están en la API:** `Muse Glimmer` (open-weight, solo auto-hospedado) → se reemplaza por `muse-spark-1.3` con imagen. `Muse Code` es un producto de Meta, no un modelo → el código del catálogo lo genera `muse-spark-1.3`.
- **Modelos configurables en `.env`:** `MUSE_VOICE_MODEL`, `MUSE_SPARK_MODEL`, `MUSE_SAM_MODEL`, `MUSE_VOICE_LANGUAGES` (por defecto `Spanish`), `MUSE_SPARK_REASONING_PHASE2` / `MUSE_SPARK_REASONING_PHASE3` (por defecto `low`), `MUSE_PHASE3_MIN_IOU` (0.3), `MUSE_PHASE3_VERIFY_CUTOUTS` (1), `MUSE_IMAGE_MODEL`, `MUSE_IMAGE_SIZE` (1024x1024), `MUSE_IMAGE_PARALLEL` (4), `MUSE_PHASE4_LIFESTYLE` (1).
- **Muse Image en la API:** `POST /images/edits` (JSON, `images[].image_url` con data URL; sin máscara ni transparencia, fondo siempre opaco). Cobra **US$0.01 por imagen**, no por token (`usage` es solo referencial). ~15–25 s por imagen. Nombres de marcas o personajes (Winnie the Pooh, Iron Man, Funko) disparan `content_policy_violation`: se reintenta sin nombrar el producto.
- **SAM 3.1 en la API:** `POST /responses` con `input_text` (frase nominal corta en inglés, o prompt de caja de `build_object_prompt`) + `input_image` (data URL). La salida son tokens `<0f>id<|box;...|><|mask;...|>`; se decodifican con el parser oficial `meta-sam-parser` (en `requirements.txt`). No reporta tokens (cobra por imagen procesada). Las frases genéricas funcionan mejor ("bottle" sí, "spray bottle" no).
- **Límites Operativos:** `MAX_VIDEOS=3`, `MAX_DURATION_SEC=60`, `LAPLACIAN_BLUR_THRESHOLD=80.0`
- **Directorio de Almacenamiento Local:** `storage/<session_id>/`

---

## 4. Mapa de Directorios del Proyecto
```text
muse-catalog-hackathon/
├── AGENT_CONTEXT.md            # Memoria de estado y contratos (ESTE ARCHIVO)
├── PLAN.md                     # Hoja de ruta verificada por fases
├── CHANGELOG_AND_TOKENS.md     # Bitácora de cambios y auditoría de tokens
├── .env                        # Variables secretas y configuración
├── storage/                    # Almacenamiento local de sesiones, frames y recortes
├── backend/
│   ├── main.py                 # API FastAPI y enrutamiento por fases
│   ├── requirements.txt
│   ├── core/
│   │   ├── config.py           # Carga de .env y constantes
│   │   ├── muse_api.py         # Cliente HTTP real de la Meta Model API (reintentos, JSON Schema, usage)
│   │   ├── muse_client.py      # Lógica por fase sobre muse_api.py con Caché MD5 y TokenTracker
│   │   └── token_tracker.py    # Middleware de trazas y cálculo de ahorro de tokens
│   └── services/
│       ├── phase1_preprocess.py # Validador de video, rescate local, audio 16kHz y OpenCV
│       ├── phase2_semantic.py   # Muse Voice Transcribe + Muse Spark (JSON Schema)
│       ├── phase2_verify.py     # Verificación automática audio↔video (anomalías + 2da pasada dirigida)
│       ├── phase3_vision_sam.py # Ventana del producto -> 3 frames -> Collage 1x3 -> muse-spark-1.3 (visión) -> sam-3.1 -> PNG
│       ├── phase3_verify.py     # Verificación automática de recortes (Spark revisa, SAM genera alternativas, Spark elige)
│       ├── phase4_studio.py     # muse-image-1.0: restauración 4A condicional + fondo blanco y ambiente (4B) + control de fidelidad con Spark
│       └── phase5_codegen.py    # Textos de venta con muse-spark-1.3 + plantilla del catálogo + ZIP descargable
└── frontend/
    └── index.html              # SPA Dashboard con monitor de trazas en vivo y visor de catálogo
```

---

## 5. Estado de Ejecución por Fases

> **Auditoría del 2026-10-29:** hasta la v1.0.0 ninguna fase llamaba a la API de Muse. Las Fases 2–4 tenían nombres, recuadros, tiempos y calidades escritos a mano para los 2 videos de `sess_20261029_013005`, los tokens eran constantes inventadas y las 30 imágenes de estudio de 1024px las generó el agente de desarrollo por fuera del pipeline. Los `TESTED_OK` anteriores de las Fases 2–4 validaban esa simulación, no el sistema real. Se están reimplementando una por una con la API real.

- **FASE 1 (Calidad de Video, Pre-procesador Local y `TokenTracker`):** `TESTED_OK` — es local (OpenCV + FFmpeg) y genérica; no cambia.
- **FASE 2 (`Muse Voice Transcribe` + `Muse Spark`):** `TESTED_OK` (v2.2.1, confirmado por el usuario el 2026-10-29: prompt sin ejemplos de los videos de prueba; prueba ciega con `sess_20261029_013005` → 7/7). La v2.1.0 fue `TESTED_OK`; el usuario pidió además que los agentes prevean objetos no detectados → se añadió la **verificación automática** `backend/services/phase2_verify.py`: inventario visual + cruce local de anomalías (0 tokens) + 2da pasada dirigida (re-transcripción del tramo + 1 llamada a Spark con clips) que agrega o corrige productos sola si la confianza es ≥ 0.7. Informe en `session_state.json` → `phase2_verification`. Objetos sin precio dicho: "Consultar precio" (`MUSE_MISSING_PRICE_POLICY`).
  - `muse-voice-transcribe-1.0` en modo `ENDPOINTING` (turnos con `startMs`/`endMs`) + `muse-spark-1.3` con JSON Schema estricto y `reasoning_effort=low`, en 1 sola llamada por lote.
  - La ventana de cada producto se calcula ubicando la cita literal que devuelve Spark dentro del turno (`timing_source = QUOTE_ALIGNED_IN_TURN`; si no la encuentra, reparte el turno en partes iguales: `TURN_EVEN_SPLIT`) y luego se amplía ±1.5 s.
  - v2.1.0: Spark también recibe cada video (copia ffmpeg 480p en `storage/<sesión>/video_proxies/`, `video_url` a 1 fps) para corregir nombres mal transcritos u omitidos (`visual_item_name`, `name_source`). El audio directo (`input_audio`) no llega al modelo; el video sí. Desactivable con `MUSE_SPARK_PHASE2_VIDEO=0`.
  - Resultado en `sess_20261029_013005`: **7 productos** (5 `EXPLICIT`, 2 `DEICTIC`). El 7º son los **lentes a $5** que antes se interpretaban como corrección de precio del esqueleto. Costo real: 3,425 tokens (in=948, out=2,477, de los cuales 1,616 son de razonamiento). Re-ejecución: 0 tokens (`CACHE_HIT_MD5`).
- **FASE 3 (Identificación visual + `SAM 3.1`):** `TESTED_OK` (v3.0.0, confirmado por el usuario el 2026-10-29; API real, sin datos escritos a mano).
  - Ventana del producto = intervalo del inventario visual de la Fase 2 que mejor coincide (o la ventana hablada ampliada). 3 candidatos: el frame más nítido de cada tercio, a resolución original.
  - 1 llamada a `muse-spark-1.3` por lote de hasta 8 collages 1x3: mejor frame, nombre comercial, categoría, calidad, `needs_enhancement`, caja y frases para SAM. **Spark da la caja en coordenadas del collage completo**, no del panel: el código la convierte con la posición de cada panel.
  - `sam-3.1` sobre el mejor frame: prueba las frases de Spark y acepta la máscara que coincide con la caja (IoU ≥ 0.3 o centro dentro); une piezas de un mismo producto (par de audífonos); si nada coincide, usa la caja como prompt; descarta máscaras con confianza < 0.3.
  - Verificación automática de recortes (`phase3_verify.py`): Spark revisa todos los recortes en una hoja (OK / INCOMPLETE / WRONG_OBJECT / EXTRA_CONTENT / UNRECOGNIZABLE); para los que fallan, SAM genera alternativas (parte faltante unida, caja ampliada, mismo frame sin caja, otros frames) y Spark elige o responde NONE → recorte rectangular marcado. Estados: `OK`, `REPAIRED`, `KEPT_BEST_AVAILABLE`, `RECTANGULAR_FALLBACK`; los no OK/REPAIRED pasan a la mejora 4A.
  - Se conserva la clave `glimmer_selection` en `grounded_products` por compatibilidad con la UI y la Fase 4 (el modelo es Spark).
- **FASE 4 (`muse-image-1.0`):** `TESTED_OK` (v4.1.1, confirmado por el usuario el 2026-10-29 con videos nuevos: 15 productos, 87 imágenes, ~8 min, US$0.87). Pedido del usuario: mejorar todas las imágenes (muchas quedaban cortadas) y generar rotaciones frente/izquierda/derecha/atrás.
  - **Máster de estudio para todos los productos:** recorte sobre blanco + frame original → producto completo, nítido, sobre blanco. Solo completa lo cortado por el borde del encuadre (detectado midiendo si la caja toca el borde del frame) o tapado por la mano; nunca extiende el producto. El máster no recibe las notas libres del verificador de la Fase 3 (hacían inventar partes: piernas a un busto).
  - **Vistas desde el máster:** frente (fondo blanco), izquierda, derecha y atrás (**deducidas por IA**, marcadas así en la UI) y ambiente. PNG maestro transparente = `sam-3.1` sobre el máster.
  - **Control con `muse-spark-1.3`** (máx. 4 llamadas): máster vs recorte + FRAME original (distingue dedo tapando vs fin real del producto); vistas vs máster (mismo producto, nada agregado, ángulo real y no repetido, fondo blanco sin manos). Rechazadas → 1 reintento con los problemas → si siguen mal: frente = recorte real sin IA; el resto se omite. Si falla la llamada de control, las imágenes quedan "sin verificar" (no se cae la fase).
  - **Regla fija:** si la Fase 3 juzgó la toma insuficiente (recorte rectangular o calidad < 4), no se genera nada: foto real + recomendación de volver a grabar.
  - Costo: ~6 imágenes por producto ≈ US$0.06 (+ reintentos) y ~15k tokens de control por lote de 7. ~5–7 min para 7 productos; corre en otro hilo para no congelar el servidor. Re-ejecución: US$0 (caché).
  - Claves: `studio_imagen.step_4a_enhancement` (máster, PNG maestro, bordes cortados, `low_quality_source`) y `step_4b_variations` (`studio_white_url`, `angle_left_url`, `angle_right_url`, `angle_back_url`, `lifestyle_url`), `fidelity_checks` por vista.
- **FASE 5 (Catálogo final):** `READY_FOR_USER_TEST` (v5.0.0, API real).
  - `muse-spark-1.3` escribe en 1 llamada (lotes de 20) título, descripción corta y larga, destacados, categoría de tienda (conjunto coherente, máx. 6) y etiquetas, más nombre y lema de la tienda. Prohibido inventar medidas, materiales, autenticidad o garantías, y copiar la escena del video. "Muse Code" no existe en la API: el HTML lo arma una plantilla determinista (paleta slate-950 / emerald / cyan).
  - Tienda: filtros por categoría, buscador, orden por precio, selector de vistas (izquierda/derecha/atrás marcadas "IA"), ficha ampliada, carrito con totales por moneda, "Consultar precio" para productos sin precio, "Foto real del video" para tomas insuficientes.
  - Carpeta `storage/<sesión>/catalog/` con `muse_ecommerce_catalog.html` + `images/` (rutas relativas, máx. 1024 px). `/api/v1/phase5/catalog-preview` redirige a esa carpeta; `/catalog-download` entrega un ZIP que se abre sin el servidor (Tailwind por CDN: necesita internet para los estilos).
  - Resumen con números reales: tokens de la API, tokens evitados por caché (solo `CACHE_HIT_MD5`), imágenes cobradas y costo; desglose por modelo con llamadas API y aciertos de caché.
