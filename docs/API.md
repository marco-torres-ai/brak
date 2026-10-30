# Referencia de la API

El backend de Brak Shop es una API **HTTP + JSON** (FastAPI). Cualquier frontend (web, móvil, otro servidor) puede consumirla.

- **Base URL local:** `http://localhost:8000`
- **CORS:** abierto (`*`). Para publicarlo en internet, limítalo a tu dominio y agrega autenticación.
- **Formato:** cada fase recibe `{"session_id": "..."}` y devuelve el **estado completo de la sesión** cuando termina.
- **Una sesión a la vez:** envía siempre `session_id`; si se omite, se usa la última sesión procesada.
- **Documentación interactiva de FastAPI:** `http://localhost:8000/docs`

---

## Flujo típico

```bash
# 1) Subir videos → crea la sesión
curl -F "files=@video1.mp4" -F "files=@video2.mp4" http://localhost:8000/api/v1/phase1/preprocess
# → { "session_id": "sess_20261029_215517", "videos": [...], ... }

# 2..5) Encadenar las fases con el mismo session_id
SID=sess_20261029_215517
for p in phase2/semantic phase3/vision-sam phase4/studio phase5/codegen; do
  curl -s -X POST "http://localhost:8000/api/v1/$p" \
       -H "Content-Type: application/json" -d "{\"session_id\":\"$SID\"}" > /dev/null
done

# 6) Leer el catálogo final
curl -s http://localhost:8000/api/v1/session/current | jq '.catalog_products'
```

O en un solo paso, tras la Fase 1: `POST /api/v1/pipeline/run-all` con `{"session_id": "..."}`.

En JavaScript:

```js
const api = (path, body) =>
  fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    .then(async (r) => { const d = await r.json(); if (!r.ok) throw new Error(d.detail); return d; });

const form = new FormData();
files.forEach((f) => form.append('files', f));
const s1 = await (await fetch('/api/v1/phase1/preprocess', { method: 'POST', body: form })).json();

let state;
for (const p of ['phase2/semantic', 'phase3/vision-sam', 'phase4/studio', 'phase5/codegen']) {
  state = await api(`/api/v1/${p}`, { session_id: s1.session_id });   // puede tardar minutos
}
console.log(state.catalog_products);
```

---

## Endpoints

### `POST /api/v1/phase1/preprocess`

Sube de 1 a 3 videos y crea una sesión nueva. **Local, 0 tokens.**

| Campo (multipart) | Tipo | Por defecto |
|---|---|---|
| `files` | archivo (repetible, 1–3) | obligatorio |
| `blur_threshold` | número | `80.0` (`LAPLACIAN_BLUR_THRESHOLD`) |
| `sample_fps` | número | `2.0` |

Respuesta (extracto):

```json
{
  "session_id": "sess_20261029_215517",
  "current_phase_completed": 1,
  "videos": [
    { "video_id": "vid_1", "filename": "WhatsApp Video.mp4", "status": "READY",
      "quality_assessment": { "duration_sec": 15.36, "resolution": "576x1024", "sharp_frames_retained": 14 },
      "audio": { "status": "OK", "url": "/storage/sess_.../audio/vid_1_audio_16khz.wav" } }
  ]
}
```

`status` puede ser `READY`, `REJECTED_DURATION_EXCEEDED`, `REJECTED_BLACK_VIDEO` o `REJECTED_CORRUPT` (con su `reason`).

### `POST /api/v1/phase2/semantic`

Muse Voice Transcribe + Muse Spark + verificador audio ↔ video. ~1 min.

Agrega `semantic_products[]`:

```json
{ "product_id": "prod_2", "video_id": "vid_1", "mention_type": "DEICTIC",
  "detected_name": "Green cactus in a blue wave-pattern pot",
  "raw_spoken_phrase": "este de acá lo vendo a cinco", "price": 5.0, "currency": "USD",
  "price_status": "SPOKEN", "condition": "Not mentioned",
  "spoken_timestamps": { "start_sec": 6.57, "end_sec": 9.39, "timing_source": "QUOTE_ALIGNED_IN_TURN" } }
```

Y `phase2_verification` con `visual_inventory`, `anomalies`, `decisions` y `status` (`NO_ANOMALIES`, `VERIFIED_NO_CHANGES` o `VERIFIED_WITH_CHANGES`).

### `PUT /api/v1/phase2/products`

Edición manual de los productos de la Fase 2: `{"session_id": "...", "products": [...]}`. Un precio vacío o `null` queda como "Consultar precio".

### `POST /api/v1/phase3/vision-sam`

Muse Spark visión + SAM 3.1 + verificador de recortes. ~1–3 min.

Agrega `grounded_products[]` con:
- `glimmer_selection`: el nombre de la clave se conserva por compatibilidad; el modelo es Spark. Incluye `best_frame_url`, `quality_score`, `needs_enhancement`, `bbox_pixels` y `bbox_normalized`.
- `sam31_cutout`: `cutout_png_url`, `sam_mode`, `sam_prompt_used`, `sam_confidence`, `merged_parts` y `verification.status`, que puede ser `OK`, `REPAIRED`, `KEPT_BEST_AVAILABLE` o `RECTANGULAR_FALLBACK`.

### `POST /api/v1/phase4/studio`

Muse Image + control de fidelidad. ~30 s por producto. US$0.01 por imagen.

Agrega `studio_products[]`, donde cada producto lleva `studio_imagen`:

```json
{
  "step_4a_enhancement": { "executed": true, "cut_by_frame_edges": ["left"], "low_quality_source": false,
                           "restored_image_url": "/storage/.../studio/prod_1_master_studio.png",
                           "result_cutout_url": "/storage/.../studio/prod_1_master.png" },
  "step_4b_variations": { "studio_white_url": "...", "angle_left_url": "...", "angle_right_url": "...",
                          "angle_back_url": "...", "lifestyle_url": "...", "ai_inferred_views": ["left", "right", "back"] },
  "fidelity_checks": { "front": { "faithful": true, "issues": "", "round": 1 } }
}
```

Una vista ausente (`null`) es una vista que el control de fidelidad rechazó dos veces.

### `POST /api/v1/phase5/codegen`

Textos de venta con Muse Spark + catálogo. ~20 s.

Agrega `catalog_store` (`name`, `tagline`) y **`catalog_products[]`**, el contrato pensado para frontends:

```json
{
  "id": "prod_1", "video_id": "vid_1",
  "title": "Red and yellow chibi Iron Man figure",
  "short_description": "...", "description": "...", "highlights": ["..."],
  "category": "Toys & Collectibles", "tags": ["figure", "collectible"],
  "price": 2.0, "currency": "USD", "condition": "", "seller_phrase": "este juguete lo vendo a dos dólares",
  "real_photo_only": false,
  "views": [
    { "key": "front", "label": "Frente", "src": "images/prod_1_front.jpg", "ai_inferred": false, "bg": "white" },
    { "key": "left",  "label": "Izquierda", "src": "images/prod_1_left.jpg", "ai_inferred": true, "bg": "white" }
  ]
}
```

También agrega `phase5_summary`, con tokens de la API, tokens evitados por caché, imágenes generadas y su costo.

### `POST /api/v1/pipeline/run-all`

Ejecuta las Fases 2 → 5 seguidas (requiere la Fase 1 hecha). Devuelve el estado final.

### `GET /api/v1/progress?session_id=`

Qué está haciendo la fase en curso. Pensado para consultarlo cada 2–3 s mientras esperas la respuesta de una fase.

```json
{
  "session_id": "sess_20261029_215517", "phase": 3, "running": true,
  "elapsed_sec": 41.2, "seconds_since_update": 3.1,
  "message": "Muse Spark (Visión): muse-spark-1.3: 1 llamada con 3 collage(s) 1x3 ...",
  "warning": null,
  "tokens_total": 26208, "tokens_saved_since_start": 0, "images_since_start": 0, "api_waits": 0,
  "events": [ { "t": 1793321850.1, "phase": 3, "model": "SAM 3.1", "status": "API_CALL", "text": "..." } ]
}
```

`warning` trae texto cuando Meta tarda o hay un reintento. Por ejemplo: *"Muse Spark (0.0 MB): Meta no respondió en 90 s; se reintenta automáticamente (intento 2 de 4, esperando hasta 180 s)."*

### `GET /api/v1/session/current`

Estado completo de la última sesión procesada (incluye `token_audit.logs`).

### `GET /api/v1/phase5/catalog-preview?session_id=`

Redirige a `/storage/<sesión>/catalog/muse_ecommerce_catalog.html`.

### `GET /api/v1/phase5/catalog-download?session_id=`

ZIP con `index.html` y `images/`. Se abre con doble clic, sin servidor (necesita internet para los estilos).

### `GET /api/health`

Estado del servidor, límites de entrada y sesión actual. Incluye `meta_muse_api_key_configured` y una versión **parcialmente enmascarada** de la clave (primeros 10 y últimos 6 caracteres): no expongas este endpoint en internet sin protegerlo.

---

## Prompts de los agentes (v0.2)

Los 13 prompts de los agentes se pueden personalizar sin tocar código. Hay dos formas:
- **Instrucciones adicionales (`extra`):** se suman al final del prompt original.
- **Modo experto (`full_text`):** reemplaza el texto completo.

Los cambios se guardan en **perfiles** dentro de `storage/prompts/`, fuera de git.

- **Sin perfil activo** ("Default"), el pipeline usa exactamente los prompts y las claves de caché de siempre.
- **Con un prompt personalizado**, la clave de caché de ese agente cambia. Solo ese agente vuelve a llamar a la API; el resto sigue saliendo de la caché.
- **Formato de salida:** los JSON Schema estrictos no se editan, así que el formato nunca se rompe.
- **Trazabilidad:** la sesión guarda `prompt_customizations`, con qué perfil y qué agentes personalizados se usaron en cada fase. Solo aparece si hubo alguno.

| Clave | Fase · Modelo | Grupo | Marcadores |
|---|---|---|---|
| `p2_extract` | 2 · Muse Spark | creative | — |
| `p2_verify` | 2 · Muse Spark | advanced | — |
| `p3_vision`, `p3_cut_check`, `p3_cut_choice` | 3 · Muse Spark | advanced | — |
| `p4_master` | 4 · Muse Image | creative | `{problems}` `{desc}` obligatorios |
| `p4_view_front` / `_left` / `_right` / `_back` / `_lifestyle` | 4 · Muse Image | creative | `{desc}` obligatorio, `{category}` opcional |
| `p4_fidelity` | 4 · Muse Spark | advanced | — |
| `p5_copy` | 5 · Muse Spark | creative | — |

### `GET /api/v1/prompts`
Devuelve `active_profile` (slug o `null`), `active_profile_name`, `profiles[]` (`slug`, `name`, `custom_agents`) y `agents[]`. Cada agente trae `key`, `phase`, `model`, `group`, `title`, `description`, `kind` (`system`/`image`), `hint`, `placeholders`, `default_text`, `extra`, `full_text`, `effective_text` e `is_custom`.

### `PUT /api/v1/prompts/{key}`
Recibe `{"extra": "Tono juvenil, frases cortas.", "full_text": null}` y guarda en el perfil activo. Si el activo es el original, crea y activa "Mi perfil". Responde **400** si falta un marcador obligatorio, aparece uno desconocido o las llaves están desbalanceadas.

### `DELETE /api/v1/prompts/{key}`
Restaura el prompt original del agente en el perfil activo.

### Perfiles
- `POST /api/v1/prompts/profiles` recibe `{"name": "Lujo", "copy_from": "mi-perfil" | null}`. Crea el perfil y lo activa.
- `PUT /api/v1/prompts/profiles/active` recibe `{"profile": "lujo" | null}`. Con `null` se vuelve al original.
- `PUT /api/v1/prompts/profiles/{slug}` recibe `{"name": "Lujo boutique"}` y solo cambia el nombre (el slug y los prompts no cambian).
- `DELETE /api/v1/prompts/profiles/{slug}`
- `GET /api/v1/prompts/profiles/{slug}/export` devuelve un JSON descargable.
- `POST /api/v1/prompts/profiles/import` recibe ese mismo JSON. Crea el perfil y lo activa.

Exportar e Importar existen solo en la API: la pantalla de Brak Shop no los muestra. Los slugs `active` e `import` están reservados; un perfil llamado "Active" recibe el slug `active-2`.

Todas las escrituras devuelven el mismo cuerpo que `GET /api/v1/prompts`. Responden **409** si hay una fase corriendo.

---

## Flujos (v0.3)

Un flujo decide **qué pasos opcionales corren y con qué parámetros**. El orden de las fases es fijo, porque cada una necesita el resultado de la anterior. Los flujos se guardan en `storage/flows/`, fuera de git.

- **Sin flujo activo** ("Completo de Meta Muse"), cada opción vale lo mismo que su variable del `.env`, así que el pipeline queda idéntico.
- Cuando un flujo difiere del completo, la sesión guarda `flow_customizations` con el flujo usado y lo que cambió.

| Opción | Bolita | Tipo | Por defecto (`.env`) |
|---|---|---|---|
| `extract.video` | Detectar productos | bool | `MUSE_SPARK_PHASE2_VIDEO` |
| `verify2.enabled` | Verificar audio ↔ video | bool | `MUSE_PHASE2_VERIFY` |
| `verify3.enabled` | Verificar recortes | bool | `MUSE_PHASE3_VERIFY_CUTOUTS` |
| `images.parallel` | Foto máster | int 1–8 | `MUSE_IMAGE_PARALLEL` |
| `view.left` / `view.right` / `view.back` | Vistas IA | bool | `true` |
| `view.lifestyle` | Ambiente | bool | `MUSE_PHASE4_LIFESTYLE` |
| `fidelity.enabled` | Control de fidelidad | bool | `true` |

La vista frontal y la máster siempre se generan. Una vista apagada sale como `null` en `step_4b_variations` y no aparece en el catálogo.

- `GET /api/v1/flows` devuelve `active_flow`, `active_flow_name`, `flows[]` y `options[]` (`key`, `node`, `type`, `label`, `description`, `default`, `value`, `min`, `max`).
- `PUT /api/v1/flows/options` recibe `{"options": {"view.lifestyle": false}}` y guarda en el flujo activo. Si el activo es el completo, crea y activa "Mi flujo". Un tipo o rango inválido responde **400**.
- `DELETE /api/v1/flows/options` devuelve todas las opciones del flujo activo a su valor por defecto.
- `POST /api/v1/flows` recibe `{"name": "Rápido", "copy_from": null, "options": {...}}`. Crea el flujo y lo activa.
- `PUT /api/v1/flows/active` recibe `{"flow": "rapido" | null}`.
- `PUT /api/v1/flows/{slug}` recibe `{"name": "..."}` y solo cambia el nombre.
- `DELETE /api/v1/flows/{slug}`

Las escrituras responden **409** si hay una fase corriendo. Los slugs `active` y `options` están reservados.

---

## Construir las URLs de las imágenes

| Origen | Formato de la ruta | URL completa |
|---|---|---|
| Fases 3 y 4 (`*_url`) | `/storage/<sesión>/...` | `BASE + ruta` |
| Fase 5 (`catalog_products[].views[].src`) | `images/<archivo>` | `BASE + /storage/<sesión>/catalog/ + src` |

```js
const BASE = 'http://localhost:8000';
const viewUrl = (sid, v) => `${BASE}/storage/${sid}/catalog/${v.src}`;
const frameUrl = (p) => BASE + p.glimmer_selection.best_frame_url;
```

## Errores

- **`400`:** entrada inválida o fase anterior incompleta. El campo `detail` explica el problema, por ejemplo *"Ejecuta primero la Fase 2."*
- **`500`:** fallo procesando la fase. `detail` incluye el error de la API de Meta si lo hubo.
- **Las fases se pueden repetir:** gracias a la caché, repetir una fase que falló no vuelve a pagar lo que ya se hizo.
