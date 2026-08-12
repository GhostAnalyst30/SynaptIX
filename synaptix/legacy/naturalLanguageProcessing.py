"""Clase legacy de NLP (API v0.x, mantenida por compatibilidad)."""

import os
from typing import Literal, Union

import numpy as np
import pandas as pd


class NaturalLanguageProcessing:
    """
    Clase general para Procesamiento del Lenguaje Natural (API legacy).

    Nota: la mayoría de los métodos de esta clase son experimentales.
    Para código nuevo se recomiendan los submódulos modernos de synaptix.
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray, list],
                 backend: Literal['pandas'] = 'pandas'):

        if isinstance(data, str) and os.path.exists(data):
            data = NaturalLanguageProcessing.from_file(data).data

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame({'var': data})
            else:
                data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])

        if isinstance(data, list):
            data = pd.DataFrame({'var': data})

        self.data = data
        self.backend = backend
        self._numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def from_file(path: str):
        """
        Carga automática de archivos.
        Soporta CSV, Excel, TXT, JSON, Parquet, Feather, TSV.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif ext in [".txt", ".tsv"]:
            df = pd.read_table(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".feather":
            df = pd.read_feather(path)
        else:
            raise ValueError(f"Formato no soportado: {ext}")

        return NaturalLanguageProcessing(df)

    # =========================
    # Limpieza y tokens
    # =========================
    def clean_text(self, text: str):
        pass

    def tokenize(self, text: str):
        pass

    def remove_stopwords(self, tokens: list):
        pass

    def stem(self, tokens: list):
        pass

    def lemmatize(self, tokens: list):
        pass

    # =========================
    # Vectorización
    # =========================
    def bag_of_words(self, documents: list):
        pass

    def tfidf(self, documents: list):
        pass

    def ngrams(self, documents: list, n: int = 2):
        pass

    # =========================
    # Similaridad semántica
    # =========================
    def cosine_similarity(self, vec1, vec2):
        pass

    def semantic_search(self, query: str, documents: list):
        pass

    # =========================
    # NLP aplicado
    # =========================
    def intent_classification(self, text: str):
        pass

    def sentiment_analysis(self, text: str):
        pass

    def keyword_extraction(self, text: str, top_k: int = 5):
        pass

    def help(self):
        """Muestra la ayuda de la clase NaturalLanguageProcessing."""
        print(
            "NaturalLanguageProcessing (legacy)\n"
            "Métodos: clean_text, tokenize, remove_stopwords, stem, lemmatize,\n"
            "bag_of_words, tfidf, ngrams, cosine_similarity, semantic_search,\n"
            "intent_classification, sentiment_analysis, keyword_extraction."
        )


# Alias por compatibilidad con versiones anteriores (typo original)
NaturalLanguajeProcessing = NaturalLanguageProcessing
