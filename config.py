"""
Конфигурация RAG-ассистента.

Все настройки приложения хранятся в файле .env (шаблон — .env.example)
и читаются модулем config.py. Класс Config — единственный источник
настроек для остальных модулей.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта: каталог, где лежит config.py
BASE_DIR = Path(__file__).resolve().parent

# Загружаем переменные окружения из .env
load_dotenv(BASE_DIR / ".env")

SUPPORTED_VECTOR_DB = {"chromadb", "weaviate"}
SUPPORTED_LLM_PROVIDERS = {"gigachat", "proxyapi"}


class ConfigError(RuntimeError):
    """Ошибка конфигурации приложения."""


def _env(name: str, default: str | None = None) -> str | None:
    """Возвращает значение переменной окружения или default, если оно пустое."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on", "да"}


@dataclass(frozen=True)
class Config:
    """Полный набор настроек приложения, полученный из .env."""

    # --- Векторная база данных ---
    vector_db: str                # "chromadb" | "weaviate"
    chroma_persist_dir: Path      # каталог с локальными данными ChromaDB
    chroma_collection: str        # имя коллекции ChromaDB
    weaviate_url: str             # URL локального Weaviate, например http://localhost:8080
    weaviate_collection: str      # имя класса (коллекции) Weaviate

    # --- LLM ---
    llm_provider: str             # "gigachat" | "proxyapi"
    gigachat_credentials: str     # client secret приложения GigaChat
    gigachat_scope: str           # GIGACHAT_API_PERS или GIGACHAT_API_B2B
    gigachat_model: str           # модель, например GigaChat-Max или GigaChat-2
    gigachat_verify_ssl: bool     # отключить проверку SSL-сертификатов при необходимости
    proxyapi_url: str             # OpenAI-совместимый base_url (для ProxyAPI)
    proxyapi_key: str             # API-ключ ProxyAPI (sk-...)
    proxyapi_model: str           # модель, например gpt-4o-mini
    proxyapi_embedding_model: str # модель эмбеддингов, например text-embedding-3-small
    temperature: float            # температура модели LLM (0.0–2.0)

    # --- Общие настройки RAG ---
    data_dir: Path                # каталог с файлами базы знаний (*.txt, *.html)
    cache_db_path: Path           # файл SQLite-кэша ответов на вопросы
    top_k: int                    # сколько релевантных чанков передавать в LLM
    history_size: int             # сколько последних пар «вопрос-ответ» передавать в LLM

    @classmethod
    def from_env(cls) -> Config:
        vector_db = (_env("VECTOR_DB", "chromadb") or "").lower()
        if vector_db not in SUPPORTED_VECTOR_DB:
            raise ConfigError(
                f"VECTOR_DB должен быть одним из: {sorted(SUPPORTED_VECTOR_DB)}, получено: {vector_db!r}"
            )

        llm_provider = (_env("LLM_PROVIDER", "gigachat") or "").lower()
        if llm_provider not in SUPPORTED_LLM_PROVIDERS:
            raise ConfigError(
                f"LLM_PROVIDER должен быть одним из: {sorted(SUPPORTED_LLM_PROVIDERS)}, получено: {llm_provider!r}"
            )

        gigachat_credentials = _env("GIGACHAT_CREDENTIALS", "") or ""
        if llm_provider == "gigachat" and not gigachat_credentials:
            raise ConfigError(
                "Для LLM_PROVIDER=gigachat нужно задать GIGACHAT_CREDENTIALS "
                "(client secret приложения с developers.sber.ru) в файле .env"
            )

        proxyapi_key = _env("PROXYAPI_KEY", "") or ""
        if not proxyapi_key:
            raise ConfigError(
                "Для построения эмбеддингов через ProxyAPI нужно задать PROXYAPI_KEY в файле .env"
            )

        try:
            top_k = int(_env("TOP_K", "5") or 5)
        except ValueError as exc:
            raise ConfigError("TOP_K должен быть целым числом") from exc
        if top_k < 1:
            raise ConfigError("TOP_K должен быть положительным числом")

        try:
            history_size = int(_env("HISTORY_SIZE", "10") or 10)
        except ValueError as exc:
            raise ConfigError("HISTORY_SIZE должен быть целым числом") from exc
        if history_size < 1:
            raise ConfigError("HISTORY_SIZE должен быть положительным числом")

        try:
            temperature = float(_env("TEMPERATURE", "0.2") or 0.2)
        except ValueError as exc:
            raise ConfigError("TEMPERATURE должен быть числом") from exc
        if not 0.0 <= temperature <= 2.0:
            raise ConfigError("TEMPERATURE должен быть в диапазоне [0.0, 2.0]")

        return cls(
            vector_db=vector_db,
            chroma_persist_dir=Path(
                _env("CHROMA_PERSIST_DIR", "./chroma_data") or "./chroma_data"
            ).resolve(),
            chroma_collection=_env("CHROMA_COLLECTION", "knowledge_base") or "knowledge_base",
            weaviate_url=_env("WEAVIATE_URL", "http://localhost:8080") or "http://localhost:8080",
            weaviate_collection=_env("WEAVIATE_COLLECTION", "Paragraph") or "Paragraph",
            llm_provider=llm_provider,
            gigachat_credentials=gigachat_credentials,
            gigachat_scope=_env("GIGACHAT_SCOPE", "GIGACHAT_API_PERS") or "GIGACHAT_API_PERS",
            gigachat_model=_env("GIGACHAT_MODEL", "GigaChat-Max") or "GigaChat-Max",
            gigachat_verify_ssl=_env_bool("GIGACHAT_VERIFY_SSL", False),
            proxyapi_url=(
                _env("PROXYAPI_URL", "https://api.proxyapi.ru/openai/v1")
                or "https://api.proxyapi.ru/openai/v1"
            ),
            proxyapi_key=proxyapi_key,
            proxyapi_model=_env("PROXYAPI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            proxyapi_embedding_model=(
                _env("PROXYAPI_EMBEDDING_MODEL", "text-embedding-3-small")
                or "text-embedding-3-small"
            ),
            data_dir=Path(_env("DATA_DIR", "./data") or "./data").resolve(),
            cache_db_path=Path(
                _env("CACHE_DB_PATH", "./cache.sqlite3") or "./cache.sqlite3"
            ).resolve(),
            top_k=top_k,
            history_size=history_size,
            temperature=temperature,
        )
