import pickle
from pathlib import Path
import sys
import os

# Añadir el directorio padre (web) al path para importar preprocessor
sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocessor import preprocess

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_model.pkl"

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def test_samples(pipeline):
    test_cases = [
        # Positivos
        ("La cámara toma fotos espectaculares, me encanta el zoom.", "positivo"),
        ("Es el mejor dispositivo que he tenido en años, 10/10.", "positivo"),
        # Negativos
        ("El servicio al cliente es una basura, nunca contestan.", "negativo"),
        ("Se calienta demasiado y se apaga solo, no lo compren.", "negativo"),
        # Neutrales
        ("El producto es tal como se describe, nada fuera de lo común.", "neutral"),
        ("Llegó a tiempo y funciona bien para su precio.", "neutral"),
        # Negaciones (Tricky)
        ("No es una mala compra por este precio.", "positivo"), # Generalmente difícil para modelos simples
        ("No es tan bueno como esperaba.", "negativo"),
        # Sarcasmo
        ("¡Qué maravilla! Se reinició tres veces en una hora.", "negativo"),
        # Fuera de dominio (Ropa)
        ("La tela de esta camisa es muy suave y fresca.", "positivo"),
    ]

    print(f"{'Texto':<60} | {'Esperado':<10} | {'Predicho':<10} | {'Confianza':<10}")
    print("-" * 100)
    
    hits = 0
    for text, expected in test_cases:
        proc = preprocess(text)
        proba = pipeline.predict_proba([proc])[0]
        classes = pipeline.classes_
        idx = proba.argmax()
        pred = classes[idx]
        conf = proba[idx]
        
        status = "✓" if pred == expected else "✗"
        print(f"{text[:57]+'...':<60} | {expected:<10} | {pred:<10} | {conf:.2f} {status}")
        if pred == expected:
            hits += 1
            
    print("-" * 100)
    print(f"Total: {hits}/{len(test_cases)} ({hits/len(test_cases):.1%})")

if __name__ == "__main__":
    model = load_model()
    if model:
        test_samples(model)
