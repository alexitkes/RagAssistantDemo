"""
Эмбеддинги текста через OpenAI-совместимый API (ProxyAPI).

Используется endpoint POST /embeddings. Запросы отправляются батчами,
чтобы не превышать лимиты прокси. Векторы нормализуются до единичной
длины (косинусное сходство).
"""
from __future__ import annotations

import math
from typing import Iterable


def _l2_normalize(vector: list[float]) -> list[float]:
    """Нормализует вектор до единичной длины (для косинусного сходства)."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class Embedder:
    """Построитель эмбеддингов через ProxyAPI (OpenAI-совместимый /embeddings)."""

    BATCH_SIZE = 64

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Не установлен пакет openai. Выполните: pip install openai") from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        """Векторизует список текстов (чанков) для записи в хранилище."""
        texts = list(texts)
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start : start + self.BATCH_SIZE]
            embeddings.extend(self._embed(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Векторизует один запрос пользователя."""
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        # OpenAI возвращает результаты в порядке запроса; сортируем по index для надёжности
        ordered = sorted(response.data, key=lambda item: item.index)
        return [_l2_normalize(item.embedding) for item in ordered]
