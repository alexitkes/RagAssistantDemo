#!/usr/bin/env python3
"""
Сценарий загрузки базы знаний в RAG-хранилище.

Читает файлы *.txt и *.html из каталога data (настраивается в .env),
делит их на чанки-абзацы (по разделителю из двух символов новой строки),
строит эмбеддинги и записывает всё в векторную БД, выбранную в .env
(ChromaDB или локальный Weaviate).

Пример запуска:
    python ingest.py
"""
from __future__ import annotations

from config import Config, ConfigError
from rag.chunker import Document, chunk_by_paragraphs, load_documents
from rag.embeddings import Embedder
from rag.storage import Chunk, VectorStorage, create_storage


def build_chunks(documents: list[Document]) -> list[Chunk]:
    """Превращает документы в чанки: один абзац = один чанк."""
    chunks: list[Chunk] = []
    for document in documents:
        for index, paragraph in enumerate(chunk_by_paragraphs(document.text)):
            chunks.append(Chunk(text=paragraph, source=document.source, index=index))
    return chunks


def main() -> None:
    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        raise SystemExit(1)

    data_dir = config.data_dir
    try:
        documents = load_documents(data_dir)
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}")
        raise SystemExit(1)

    if not documents:
        print(f"В каталоге {data_dir} нет файлов *.txt или *.html. Добавьте файлы базы знаний.")
        raise SystemExit(1)

    chunks = build_chunks(documents)
    if not chunks:
        print("Не удалось выделить ни одного абзаца-чанка. Проверьте содержимое файлов базы знаний.")
        raise SystemExit(1)

    print(f"Найдено документов: {len(documents)}, чанков (абзацев): {len(chunks)}")
    print("Построение эмбеддингов через ProxyAPI...")
    embedder = Embedder(
        base_url=config.proxyapi_url,
        api_key=config.proxyapi_key,
        model=config.proxyapi_embedding_model,
    )

    print("Построение эмбеддингов и запись в векторное хранилище...")
    embeddings = embedder.embed_documents(chunk.text for chunk in chunks)

    storage: VectorStorage = create_storage(config)
    try:
        storage.clear()
        storage.add_chunks(chunks, embeddings)
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка записи в векторное хранилище {config.vector_db}: {exc}")
        raise SystemExit(1)
    finally:
        storage.close()

    print(
        f"Готово. {len(chunks)} чанков загружены в {config.vector_db}.\n"
        f"Если файлы в {data_dir} изменятся, повторно выполните: python ingest.py"
    )


if __name__ == "__main__":
    main()
