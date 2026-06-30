# Optimización del preprocesamiento offline de video

## Flujo efectivo

1. OpenCV decodifica todos los frames y calcula una firma para cortes de escena.
2. MediaPipe detecta landmarks en resolución original y, por defecto, también a escala 1,5×.
3. NMS elimina detecciones duplicadas antes de medir calidad o extraer embeddings.
4. Cada cara usable se alinea con los cinco puntos de MediaPipe al template ArcFace 112×112.
5. El mismo ArcFace de DeepFace procesa original y flip. Ambos entran juntos en batches de doce caras y se combinan como antes.
6. Los embeddings se comparan con centroides de famosos y se asignan a tracks en orden de frame/escena.
7. Cada track usa agregado robusto, eliminación de outliers, umbral, margen top‑1/top‑2, soporte individual y votos temporales.
8. Solo los tracks desconocidos se refinan; sus embeddings alternativos se procesan por lotes y quedan guardados.
9. Se rellenan microcortes únicamente en tracks reconocidos y se genera el JSON con landmarks completos.
10. Un NPZ separado conserva bbox, 478 landmarks, calidad, embedding TTA y embedding alternativo. Cambiar umbrales o tracking vuelve a ejecutar solo las decisiones.

## Decisión offline bidireccional

- La identidad del track se acepta con al menos 65% de votos temporales consistentes, margen top-1/top-2, soporte de muestras individuales y baja proporción de competidores. Un agregado especialmente fuerte puede rescatar un track desde 45% si conserva soporte, inliers y no acumula rivales.
- Los embeddings alejados del agregado robusto se contabilizan como outliers; un track internamente inconsistente se rechaza como desconocido.
- Una pasada probabilística hacia adelante y hacia atrás usa todo el track para modular la confianza. Solo veta el nombre cuando un segmento sostenido está dominado en al menos 80% por otra identidad conocida; blur y ambigüedad reducen el porcentaje sin romper el track.
- La confianza mostrada se calcula por frame con similitud, margen, calidad y probabilidad temporal. Los frames borrosos o interpolados conservan contexto, pero reducen el porcentaje.
- El JSON conserva evidencia global y local para auditar por qué se aceptó o rechazó cada rostro.
- Los caches de decisiones anteriores se invalidan y las features costosas quedan en un NPZ reutilizable.

## Perfil reproducible

Video: `cache/videos/Video TP Vision.mp4`, 1920×1080, 30 FPS, 118,8 s. Fragmento base: 120 frames consecutivos, una cara usable por frame. No se omitieron frames ni se redujo la doble detección.

| Etapa | Base CPU, s | CPU batch‑8, s |
|---|---:|---:|
| Decodificación | 0,378 | 0,318 |
| MediaPipe primario | 1,305 | 1,268 |
| MediaPipe 1,5× | 1,663 | 1,501 |
| NMS | 0,002 | 0,002 |
| Alineamiento | 0,041 | 0,040 |
| ArcFace normal | 16,504 | — |
| ArcFace flip | 15,681 | — |
| ArcFace TTA batch | — | 5,883 |
| Comparación famosos | 0,031 | 0,019 |
| Tracking | 0,004 | 0,002 |
| Segunda pasada | 0,000 | 0,000 |
| JSON | 0,034 | 0,050 |
| **Wall análisis** | **36,837** | **10,159** |

Resultados adicionales:

- Pipeline final bidireccional, multiescala siempre activa y batch‑12: 6,014 s paralelos contra 8,907 s secuenciales para 120 frames; 62 valores de confianza distintos y records idénticos entre ambos modos.
- CPU batch‑8 + doble buffer: 7,708 s, 15,57 FPS y records idénticos a la ruta secuencial.
- Pasada completa CPU batch‑8 secuencial: 331,783 s, 10,74 FPS, 16,72 caras/s, 5.548 detecciones y cero errores de embedding. Uso medio: 5,45 núcleos lógicos equivalentes. JSON: 1,040 s.
- Recalcular decisiones desde features con otro umbral: 8,854 s; JSON: 1,050 s.
- La base del fragmento extrapola 18,4 min a dos minutos con una sola cara. El video completo tiene 1,56 caras/frame, consistente con los cerca de 30 min observados antes del batching.

En la base, ArcFace representa 87,4% del wall y MediaPipe 8,1%. Con batch secuencial representan 57,9% y 27,3%. En paralelo se solapan y sus porcentajes individuales ya no son aditivos.

Se midieron 5,45 núcleos lógicos de CPU equivalentes durante la pasada completa.

Comandos:

```powershell
python benchmarks/profile_offline_video.py "cache/videos/Video TP Vision.mp4" --frames 120 --batch-size 1
python benchmarks/profile_offline_video.py "cache/videos/Video TP Vision.mp4" --frames 120 --batch-size 8 --parallel
python benchmarks/validate_arcface_batch.py
python benchmarks/validate_pipeline_determinism.py
python benchmarks/audit_multiscale.py "cache/videos/Video TP Vision.mp4"
python benchmarks/benchmark_full_video.py "cache/videos/Video TP Vision.mp4"
```

## Precisión y decisiones de diseño

- Batch CPU: 11 imágenes, cuatro personas y tres desconocidos; coseno mínimo 0,99999994, diferencia máxima 2,53e‑7, diferencia media 3,98e‑8 y top‑5 idéntico en todos los casos.
- Las tres muestras locales desconocidas fueron rechazadas por umbral o margen; sus similitudes top‑1 quedaron entre 0,3323 y 0,3777 y ningún resultado ambiguo se aceptó como famoso.
- Paralelismo: records completos idénticos entre secuencial y doble buffer.
- La multiescala adaptativa fue rechazada como default: en los 3.564 frames conservó 96,16% de las detecciones y perdió 213, de las cuales hasta 196 eran usables. Por eso la aplicación conserva ambas escalas en todos los frames.
- NMS ya precedía al embedding y se conserva así. No hay nombres, timestamps ni resultados hardcodeados.
