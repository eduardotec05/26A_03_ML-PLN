# Análisis de Sentimientos en Reseñas — Documento de Diseño

**Fecha:** 2026-05-12  
**Estado:** Aprobado

---

## Objetivo

Construir un sistema de clasificación de sentimientos para reseñas en español que identifique si cada mensaje es **positivo**, **negativo** o **neutral**. El sistema se entrena en Google Colab y se sirve desde una aplicación web con FastAPI.

---

## Alcance

- Clasificación de texto en español en tres clases: positivo, negativo, neutral.
- Entrenamiento del modelo en Google Colab con scikit-learn + NLTK.
- Exportación del modelo entrenado como artefacto `.pkl`.
- Aplicación web FastAPI con dos modos: análisis individual y análisis batch por archivo.

---

## Arquitectura

### Fase 1 — Entrenamiento en Google Colab

**Archivo:** `sentiment_colab.ipynb`

| Celda | Contenido |
|-------|-----------|
| 1 | Instalación: `scikit-learn`, `nltk`. Descarga de `stopwords` en español. |
| 2 | Dataset manual: ~30 reseñas etiquetadas (≥10 por clase). Formato: lista de dicts `{"texto": str, "etiqueta": str}`. |
| 3 | Función `preprocessor(text)`: minúsculas, eliminación de puntuación y stopwords en español, tokenización. |
| 4 | Pipeline `TfidfVectorizer(preprocessor=preprocessor) → MultinomialNB`. División 80/20. Entrenamiento. |
| 5 | Evaluación: `accuracy_score`, `classification_report` (precision, recall, F1 por clase), matriz de confusión con `matplotlib`. |
| 6 | Serialización con `pickle.dump()` del pipeline completo. Descarga con `files.download('sentiment_model.pkl')`. |

**Artefacto exportado:** `sentiment_model.pkl` — pipeline completo (vectorizador + clasificador en un solo objeto).

---

### Fase 2 — Aplicación Web FastAPI

**Estructura de archivos:**

```
web/
├── main.py                  ← FastAPI app, carga del modelo, rutas
├── models/
│   └── sentiment_model.pkl  ← artefacto exportado desde Colab
├── templates/
│   └── index.html           ← interfaz única con dos modos (Jinja2)
└── static/
    └── style.css            ← estilos básicos
```

**Dependencias web:** `fastapi`, `uvicorn`, `jinja2`, `python-multipart`, `pandas`.

---

## Endpoints

| Método | Ruta | Entrada | Salida |
|--------|------|---------|--------|
| `GET` | `/` | — | Página HTML principal |
| `POST` | `/predict` | JSON `{"texto": "..."}` | JSON `{"etiqueta": "positivo", "confianza": 0.87}` |
| `POST` | `/predict/batch` | Archivo `.csv` (multipart/form-data, columna `texto`) | JSON `{"resultados": [{"texto": str, "etiqueta": str, "confianza": float}], "resumen": {"positivo": int, "negativo": int, "neutral": int}}` |

---

## Interfaz de Usuario

Una sola página (`index.html`) con dos secciones accesibles por pestañas o botones de modo:

### Modo Individual
- Textarea para ingresar una reseña.
- Botón "Analizar".
- Resultado: etiqueta con color (verde = positivo, rojo = negativo, gris = neutral) + porcentaje de confianza.

### Modo Batch
- Botón para subir archivo `.csv` (columna requerida: `texto`).
- Tabla de resultados: `| Reseña | Sentimiento | Confianza |` con colores por etiqueta.
- Resumen global:
  - Barras de porcentaje por clase (HTML/CSS puro o Chart.js).
  - Ejemplo: Positivo 65% · Negativo 20% · Neutral 15%.
- Gráfica de pastel con Chart.js (renderizada en el navegador).

---

## Flujo de datos completo

```
[Colab]
  reviews list → preprocessor → TF-IDF fit → NaiveBayes fit → pipeline.pkl
                                                                     ↓
[Web]
  usuario sube .pkl → main.py carga modelo al iniciar
  modo individual: textarea → POST /predict → etiqueta + confianza
  modo batch: .csv upload → POST /predict/batch → tabla + resumen + gráfica
```

---

## Dataset 

