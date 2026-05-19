"""
Script para revisar qué palabras considera importantes el modelo
es decir si algo es positivo, negativo o neutral.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_features():
    """
    Carga el modelo y saca el top 20 de palabras clave por categoría.
    Usa las probabilidades del Naive Bayes para ver cuáles pesan más.
    """
    model_path = Path(__file__).resolve().parent.parent / "models" / "sentiment_model.pkl"
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    vec = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    feature_names = vec.get_feature_names_out()
    classes = clf.classes_
    
    # Para MultinomialNB, feature_log_prob_ contiene el log de P(x_i | y)
    for i, label in enumerate(classes):
        top_indices = np.argsort(clf.feature_log_prob_[i])[-20:][::-1]
        top_features = [feature_names[j] for j in top_indices]
        print(f"\nTop 20 palabras para {label.upper()}:")
        print(", ".join(top_features))

if __name__ == "__main__":
    analyze_features()