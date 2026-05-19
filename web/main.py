"""
Archivo principal de la API. Aquí se levanta el servidor con FastAPI
y se conectan las rutas para que el modelo pueda hacer sus predicciones.
"""

import pickle
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import io

from preprocessor import preprocess

MODEL_PATH = Path("models/sentiment_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    pipeline = None


app = FastAPI(title="Análisis de Sentimientos")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class TextInput(BaseModel):
    texto: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
async def predict(data: TextInput):
    """
    Ruta para predecir un solo texto.
    Recibe el texto, lo limpia y se lo pasa al modelo para ver si es
    positivo, negativo o neutral. Devuelve la etiqueta y qué tan seguro está.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Sube sentiment_model.pkl a models/")
    processed = preprocess(data.texto) # Se limpian los datos a analizar.
    proba = pipeline.predict_proba([processed])[0] # Se le da el texto al modelo para que clasifique.
    classes = pipeline.classes_
    idx = int(proba.argmax())
    return {"etiqueta": classes[idx], "confianza": round(float(proba[idx]), 4)} 


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Esta ruta es para cuando subes un CSV.
    Lee el archivo, busca la columna 'texto', procesa todo
    y te regresa los resultados de cada uno más un resumen con los porcentajes.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Sube sentiment_model.pkl a models/")
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo CSV inválido")
    if "texto" not in df.columns:
        raise HTTPException(status_code=400, detail="El CSV debe tener una columna llamada 'texto'")

    textos = df["texto"].fillna("").tolist()
    processed = [preprocess(t) for t in textos]
    probas = pipeline.predict_proba(processed) # Se clasifican los textos.
    classes = list(pipeline.classes_)

    counts = {"positivo": 0, "negativo": 0, "neutral": 0}
    resultados = []
    # Guarda los resultados en un diccionario
    for texto, proba in zip(textos, probas):
        idx = int(proba.argmax())
        etiqueta = classes[idx]
        counts[etiqueta] += 1
        resultados.append({"texto": texto, "etiqueta": etiqueta, "confianza": round(float(proba[idx]), 4)})
    # Calcula los porcentajes
    total = len(textos)
    if total > 0:
        ordered = sorted(counts.keys())
        resumen = {}
        remaining = 100
        for i, k in enumerate(ordered):
            if i == len(ordered) - 1:
                resumen[k] = remaining
            else:
                val = round(counts[k] * 100 / total)
                resumen[k] = val
                remaining -= val
    else:
        resumen = {k: 0 for k in counts}

    return {"resultados": resultados, "resumen": resumen}
