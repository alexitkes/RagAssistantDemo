"""
Точка входа RAG-пакета: класс RagAssistant.

Инкапсулирует весь цикл ответа на вопрос пользователя:
  1. поиск ответа в кэше (SQLite) по SHA1 нормализованного вопроса;
  2. при промахе — поиск релевантных чанков в векторном хранилище;
  3. передача вопроса, найденных чанков и истории диалога в LLM;
  4. сохранение ответа в кэш и в историю диалога (SQLite).

ask_question возвращает только текст ответа; факт обращения к кэшу
выводится через logging.
"""
from __future__ import annotations

import logging

from config import Config

from .cache import Cache, question_key
from .embeddings import Embedder
from .llm import ask
from .storage import VectorStorage, create_storage

logger = logging.getLogger(__name__)


class RagAssistant:
    """Точка входа: выполняет всю работу по формированию ответа на вопрос."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._embedder = Embedder(
            base_url=config.proxyapi_url,
            api_key=config.proxyapi_key,
            model=config.proxyapi_embedding_model,
        )
        self._storage: VectorStorage = create_storage(config)
        self._cache = Cache(config.cache_db_path)
        # История диалога восстанавливается из SQLite (последние history_size пар)
        self._history: list[tuple[str, str]] = self._cache.get_history(config.history_size)

    def ask_question(self, question: str) -> str:
        """Возвращает текст ответа на вопрос, используя кэш, векторную БД и LLM."""
        key = question_key(question)

        # 1. Сначала ищем ответ в кэше
        cached = self._cache.get(key)
        if cached is not None:
            self._push_history(question, cached)
            logger.info("Ответ взят из кэша")
            return cached

        # 2. Ищем релевантные чанки в векторном хранилище
        query_embedding = self._embedder.embed_query(question)
        chunks = self._storage.search(query_embedding, top_k=self.config.top_k)
        if not chunks:
            logger.warning("По вопросу не найдено релевантных фрагментов в базе знаний")
            return "По вашему вопросу не найдено релевантных фрагментов в базе знаний."

        print(f"Найдено фрагментов: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            preview = " ".join(chunk.text.split())[:130]
            print(
                f"  {i}. [{chunk.source}#{chunk.index}] "
                f"(сходство ~{1 - chunk.distance:.3f}): {preview}..."
            )

        # 3. Передаём вопрос, найденные чанки и историю диалога в LLM
        answer = ask(self.config, question, chunks, self._history)
        logger.info("Ответ сформирован через LLM")

        # 4. Сохраняем ответ в кэш и в историю диалога
        self._cache.set(key, question, answer)
        self._push_history(question, answer)
        return answer

    def answer_with_contexts(self, question: str) -> tuple[str, list[str]]:
        """
        Возвращает (ответ, тексты найденных чанков) без использования кэша.

        Нужен для оценки качества RAG (например, метриками Ragas): позволяет
        получить и сам ответ, и контекст, на основе которого он сформирован.
        """
        query_embedding = self._embedder.embed_query(question)
        chunks = self._storage.search(query_embedding, top_k=self.config.top_k)
        contexts = [chunk.text for chunk in chunks]
        if not chunks:
            return "По вашему вопросу не найдено релевантных фрагментов в базе знаний.", contexts
        answer = ask(self.config, question, chunks, [])
        return answer, contexts

    def _push_history(self, question: str, answer: str) -> None:
        """Добавляет пару «вопрос-ответ» в историю (в память и в SQLite)."""
        self._history.append((question, answer))
        self._history = self._history[-self.config.history_size :]
        self._cache.add_history(question, answer)

    def close(self) -> None:
        """Закрывает соединения с векторным хранилищем и кэшем."""
        self._storage.close()
        self._cache.close()
