# PLAN.md — Plan Maestro Verificado (Muse Video-to-Catalog AI)

## 1. Propósito del Sistema
Aplicación web que recibe de **1 a 3 videos (máximo 60 segundos cada uno)** grabados por un usuario recorriendo un espacio y mostrando objetos para la venta. El sistema procesa el flujo de video y audio mediante los **6 modelos de Meta Muse + SAM 3.1**, controlando y registrando el consumo de tokens en trazas de log en tiempo real, para entregar un **Catálogo E-Commerce Interactivo**.

---

## 2. Cobertura Total de Casos de Uso (Audio y Video)
1. **Control y Rescate de Calidad del Video (Entrada del Flujo):**
   - Al subir el video, se evalúa su viabilidad técnica (resolución, iluminación y nitidez).
   - Si el video es apto, se extraen los frames nítidos. Si un tramo del video tiene baja iluminación o desenfoque leve que impida ver los objetos, se aplica un **Pre-Mejorador de Video Local** (ecualización adaptativa CLAHE + reducción de ruido + enfoque) para rescatar los fotogramas antes de enviarlos a los modelos. Si el video está corrupto o totalmente en negro, se rechaza sin gastar tokens.
2. **Mención Deíctica / Sin Nombre (*"Quiero vender esto a 3 dólares"*):**
   - El usuario señala o enfoca el producto sin decir qué es. El flujo cruza el segundo exacto del audio con el video para que el modelo visual identifique el objeto y lo enfoque.
3. **Mención Explícita (*"Este lapicero lo quiero vender a 2 dólares"*):**
   - El usuario nombra el objeto y su precio. El sistema confirma visualmente el objeto y lo vincula.
4. **Compensación de Desfase Humano ($\pm 1.5s$):**
   - Como el usuario puede mostrar el objeto un segundo antes o después de hablar, cada marca de tiempo `[inicio, fin]` se expande en `[-1.5s, +1.5s]` para no perder la toma del producto.
5. **Recorrido Espacial y Desambiguación de Objetos:**
   - Un video puede contener múltiples productos en distintos lugares. Si en un mismo plano aparecen varios objetos, el modelo prioriza el objeto que está siendo sostenido, señalado por la mano o centrado en primer plano.
6. **Charla Irrelevante vs. Estado y Características:**
   - Comentarios ajenos a la venta se filtran y descartan.
   - Descripciones del estado (*"casi nuevo"*, *"tiene un detalle aquí"*) o características principales (*"es inalámbrico"*, *"talla M"*) se capturan y se asocian a la ficha del producto, además de resolver correcciones de precio en vivo (*"a 10... mejor a 8 dólares"* -> se queda con $8).

---

## 3. Flujo de los 6 Modelos y Optimización de Tokens

| Paso | Componente / Modelo | Función Exacta en el Flujo | Regla de Ahorro de Tokens |
| :--- | :--- | :--- | :--- |
| **0** | **Validador y Rescate de Video (OpenCV + FFmpeg)** | Verifica si la calidad del video es apta, mejora fotogramas oscuros/flojos localmente, extrae audio mono 16kHz y descarta frames con *motion blur*. | **0 Tokens.** Evita enviar frames inservibles o audio pesado a la API. |
| **1** | **Muse Voice Transcribe** | Transcribe el audio de los hasta 3 videos entregando texto y *timestamps* exactos (`[start, end]`). | Caché local por Hash MD5 del archivo de audio (re-ejecutar pruebas con el mismo video gasta **0 tokens**). |
| **2** | **Muse Spark** | Analiza la transcripción completa: filtra charla basura, detecta cada producto (`EXPLICIT` o `DEICTIC`), extrae precio, estado/características y ventana temporal. | **1 sola llamada consolidada (Batch JSON Schema)** por lote de videos. Cero bucles conversacionales entre agentes. |
| **3** | **Muse Glimmer** | Recibe los frames candidatos de cada producto en un collage `1x3`. Identifica qué es *"esto"*, elige el **Mejor Frame** (donde se vea perfecto y sin la mano tapándolo), evalúa la calidad de esa imagen capturada y da sus coordenadas (`bbox`). | **Collage `1x3`:** Envía 1 sola imagen compuesta por producto en lugar de múltiples llamadas individuales (ahorra ~66% de tokens de visión). |
| **4** | **SAM 3.1** | Recibe el **Mejor Frame** y las coordenadas (`bbox`) para detectar la silueta exacta y recortar el objeto sin fondo (PNG transparente). | **Ejecución Puntual:** Se aplica solo sobre el Mejor Frame de cada producto, no sobre todos los fotogramas del video. |
| **5** | **Muse Image** | Opera en 2 sub-etapas sobre el objeto capturado:<br>• **5A (Mejora Condicional):** Si Glimmer detectó que la calidad de la imagen recortada es baja, primero pasa por mejora/restauración. Si está bien, pasa directo.<br>• **5B (Variaciones):** Genera las variaciones del producto (fondo estudio blanco + fondo lifestyle) para el catálogo. | El paso **5A (Mejora)** se omite automáticamente cuando `quality_score >= 7.0`, ahorrando llamadas innecesarias. |
| **6** | **Muse Code** | Toma todos los objetos procesados, sus variaciones de imágenes, nombres, precios y características, y genera el código del Catálogo Web final. | **1 sola llamada al final del flujo** usando una estructura de datos limpia y validada. |

---

## 4. Directrices Estrictas para los Agentes de Antigravity
1. **Economía de Tokens del Propio Agente (Desarrollo):**
   - No listar ni leer carpetas completas repetidamente. Usar `AGENT_CONTEXT.md` como mapa único.
   - No crear sub-agentes ni tareas redundantes; escribir o modificar únicamente los archivos de la fase solicitada.
2. **Persistencia de Estado (`session_state.json`):**
   - Cada fase guarda su salida en `storage/<session_id>/session_state.json` y permite que el usuario inspeccione (y opcionalmente edite datos mal interpretados) antes de disparar la siguiente fase.
3. **Avance Bloqueado por Pruebas del Usuario:**
   - El agente implementa **una sola fase**, actualiza la documentación y se detiene. Solo el usuario autoriza el paso a la siguiente fase tras probarla.
4. **Trazabilidad en Vivo (`TokenTracker`):**
   - Todo consumo de tokens (`input`, `image`, `output`, `saved`) debe quedar registrado en consola, en `session_state.json` y visible en el panel de logs de la interfaz web.

---

## 5. Plan de Ejecución y Pruebas por Fases

### FASE 1: Calidad de Video, Pre-procesador Local y Monitor de Trazas (`TokenTracker`)
- **Desarrollo:**
  - Estructura FastAPI + Frontend Web + Configuración de `.env` (`META_MUSE_API_KEY`).
  - Módulo `TokenTracker` con panel visual de logs en vivo.
  - Servicio `phase1_preprocess.py`: Valida hasta 3 videos (máx. 60s), verifica e informa la calidad del video, aplica rescate local de iluminación/nitidez si es necesario, extrae audio WAV 16kHz y selecciona los frames nítidos.
- **Prueba del Usuario:**
  - Subir de 1 a 3 videos, revisar el diagnóstico de calidad de video, ver los frames limpios extraídos y comprobar que la traza marque `0 tokens` gastados y los tokens ahorrados.

### FASE 2: Identificación de Audio e Intenciones (`Muse Voice Transcribe` + `Muse Spark`)
- **Desarrollo:**
  - Cliente `muse_client.py` con caché MD5 y conexión directa a `TokenTracker`.
  - Servicio `phase2_semantic.py`: Transcribe el audio con timestamps y usa `Muse Spark` para listar cada producto (`EXPLICIT` o `DEICTIC`), su precio, sus características/estado y su rango de segundos (`±1.5s`), ignorando charla irrelevante.
  - Interfaz en la web que muestra los productos detectados en el audio y permite verificar/ajustar precios o notas antes de pasar a visión.
- **Prueba del Usuario:**
  - Probar con videos diciendo *"esto cuesta 3 dólares"*, *"este lapicero a 2 dólares"* y comentarios de estado, validando el JSON extraído y el log de tokens consumidos.

### FASE 3: Detección de Objetos, Mejor Frame y Recorte (`Muse Glimmer` + `SAM 3.1`)
- **Desarrollo:**
  - Servicio `phase3_vision_sam.py`: Crea el collage `1x3` de los frames de cada rango temporal y lo envía a `Muse Glimmer` para enfocar e identificar el objeto (*"esto"*), elegir el **Mejor Frame** (donde se vea perfecto) y evaluar la calidad de la imagen (`quality_score` + `needs_enhancement`).
  - Pasa el Mejor Frame y sus coordenadas a `SAM 3.1` para recortar el objeto sin fondo.
- **Prueba del Usuario:**
  - Verificar en la web el nombre real asignado a los objetos implícitos, el Mejor Frame seleccionado, el diagnóstico de calidad de la imagen y el recorte limpio de `SAM 3.1` con su traza de tokens.

### FASE 4: Mejora Condicional de Imagen y Variaciones de Catálogo (`Muse Image`)
- **Desarrollo:**
  - Servicio `phase4_studio.py`:
    - **Sub-paso 4A:** Si la imagen recortada tiene `needs_enhancement == true`, pasa por el modelo de mejora de `Muse Image`; si la calidad ya es óptima, salta directo al 4B.
    - **Sub-paso 4B:** Genera las variaciones de imagen del producto con `Muse Image` para nutrir el catálogo.
- **Prueba del Usuario:**
  - Verificar que las imágenes con baja calidad se mejoren, que las buenas pasen directo (ahorrando tokens en el log) y que se generen las variaciones de catálogo para cada producto.

### FASE 5: Creación del Catálogo Final (`Muse Code`) y Cierre
- **Desarrollo:**
  - Servicio `phase5_codegen.py`: Envía todos los productos (imágenes variadas, nombres, precios, estado/características) a `Muse Code` para autogenerar el Catálogo Web interactivo y exportable.
  - Integración de botón "Flujo Completo Automático" y resumen final de auditoría de tokens.
- **Prueba del Usuario:**
  - Ejecutar el flujo completo con hasta 3 videos, probar el catálogo web generado y validar el consumo total de tokens en las trazas.