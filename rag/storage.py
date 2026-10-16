"""
Векторные хранилища для RAG: ChromaDB и локальный Weaviate.

Выбор хранилища определяется настройкой VECTOR_DB в .env.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from config import Config


@dataclass(frozen=True)
class Chunk:
    """Чанк (абзац документа) для записи в векторное хранилище."""

    text: str
    source: str            # имя файла, из которого взят чанк
    index: int             # порядковый номер абзаца внутри файла

    @property
    def id(self) -> str:
        return f"{self.source}##{self.index}"


@dataclass(frozen=True)
class RetrievedChunk:
    """Чанк, найденный при поиске по сходству."""

    text: str
    source: str
    index: int
    distance: float


class VectorStorage(ABC):
    """Общий интерфейс векторного хранилища."""

    @abstractmethod
    def clear(self) -> None:
        """Очищает хранилище перед повторной загрузкой базы знаний."""

    @abstractmethod
    def add_chunks(self, chunks: Iterable[Chunk], embeddings: list[list[float]]) -> None:
        """Записывает чанки вместе с их эмбеддингами в хранилище."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Возвращает top_k наиболее похожих на запрос чанков."""

    def close(self) -> None:
        """Закрывает соединение с хранилищем (переопределяется при необходимости)."""


class ChromaStorage(VectorStorage):
    """Локальная векторная БД ChromaDB (persistent-каталог на диске)."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        import chromadb  # ленивый импорт

        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: Iterable[Chunk], embeddings: list[list[float]]) -> None:
        chunks = list(chunks)
        self._collection.add(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source": chunk.source, "index": chunk.index} for chunk in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        found: list[RetrievedChunk] = []
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            found.append(
                RetrievedChunk(
                    text=text,
                    source=metadata["source"],
                    index=int(metadata["index"]),
                    distance=float(distance),
                )
            )
        return found


class WeaviateStorage(VectorStorage):
    """Векторная БД Weaviate, развёрнутая локально (self-hosted, не облако)."""

    def __init__(self, url: str, collection_name: str) -> None:
        import weaviate  # ленивый импорт

        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or "localhost"
        port = parsed.port or 8080

        self._collection_name = collection_name
        self._client = weaviate.connect_to_local(host=host, port=port)
        if not self._client.is_ready():
            self._client.close()
            raise ConnectionError(
                f"Локальный Weaviate по адресу {url} недоступен. Запустите его, например, командой:\n"
                "  docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.28"
            )

    def _ensure_collection(self) -> None:
        """
        Создаёт коллекцию (класс) с нужной схемой, если её ещё нет.

        Векторы передаём вручную, поэтому внешний модуль-векторизатор
        не требуется: подходит локальный Weaviate без моделей.
        """
        from weaviate.classes.config import DataType, Property

        if self._client.collections.exists(self._collection_name):
            return
        self._client.collections.create(
            name=self._collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

    def clear(self) -> None:
        if self._client.collections.exists(self._collection_name):
            self._client.collections.delete(self._collection_name)

    def add_chunks(self, chunks: Iterable[Chunk], embeddings: list[list[float]]) -> None:
        self._ensure_collection()
        collection = self._client.collections.get(self._collection_name)
        with collection.batch.fixed_size(batch_size=100) as batch:
            for chunk, vector in zip(chunks, embeddings):
                batch.add_object(
                    properties={
                        "text": chunk.text,
                        "source": chunk.source,
                        "chunk_index": int(chunk.index),
                    },
                    vector=list(vector),
                )

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        try:
            from weaviate.classes.query import MetadataQuery
        except ImportError:  # совместимость с новыми версиями клиента
            from weaviate.collections.queries.metadata import MetadataQuery

        self._ensure_collection()
        collection = self._client.collections.get(self._collection_name)
        response = collection.query.near_vector(
            near_vector=list(query_embedding),
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        found: list[RetrievedChunk] = []
        for obj in response.objects:
            props = obj.properties
            found.append(
                RetrievedChunk(
                    text=props["text"],
                    source=props["source"],
                    index=int(props["chunk_index"]),
                    distance=float(obj.metadata.distance or 0.0),
                )
            )
        return found

    def close(self) -> None:
        self._client.close()


def create_storage(config: Config) -> VectorStorage:
    """Фабрика: создаёт хранилище в соответствии с настройкой VECTOR_DB."""
    if config.vector_db == "chromadb":
        return ChromaStorage(config.chroma_persist_dir, config.chroma_collection)
    if config.vector_db == "weaviate":
        return WeaviateStorage(config.weaviate_url, config.weaviate_collection)
    raise ValueError(f"Неизвестная векторная БД: {config.vector_db}")
