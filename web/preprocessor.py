import re
import unicodedata
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

_STOP_WORDS = set(stopwords.words("spanish")) - {
    "no", "ni", "nunca", "jamás", "tampoco", "sin", "nada", "nadie",
    "ningún", "ninguno", "ninguna",
}

_PALABRAS_NEG = {"no", "ni", "nunca", "jamas", "tampoco", "sin", "nada"}


def _normalizar(text: str) -> str:
    """Elimina tildes: cámara → camara, así se unifican variantes ortográficas."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _manejar_negaciones(tokens: list) -> list:
    """Prefija con NEG_ las 3 palabras siguientes a una negación.
    'no funciona bien' → ['no', 'NEG_funciona', 'NEG_bien']
    """
    resultado, neg_count = [], 0
    for tok in tokens:
        if tok in _PALABRAS_NEG:
            resultado.append(tok)
            neg_count = 3
        elif neg_count > 0:
            resultado.append("NEG_" + tok)
            neg_count -= 1
        else:
            resultado.append(tok)
    return resultado


def preprocess(text: str) -> str:
    """
    Esta es la función principal que junta todo. Agarra el texto, lo pasa a minúsculas,
    le quita puntuación, borra palabras de relleno (stopwords) y hace el manejo de negaciones.
    Regresa el string limpio y listo para usar.
    """
    text = _normalizar(text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    tokens = _manejar_negaciones(tokens)
    return " ".join(tokens)
