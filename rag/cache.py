"""
Кэширование ответов и истории диалога в базе данных SQLite.

Ключом записи в кэше ответов служит SHA1-хэш нормализованного текста вопроса:
  * все буквы приводятся к нижнему регистру;
  * все последовательности пробелов заменяются одним пробелом;
  * от полученной строки берётся SHA1-хэш (hex-строка).

Если задан идентификатор пользователя (uid), он также участвует в формировании
ключа, поэтому для каждого пользователя используется отдельный кэш ответов.
Пустой uid даёт «общий» кэш (поведение без учёта пользователя).

История диалога хранится в отдельной таблице dialogue_history и сохраняется
между запусками ассистента.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Нормализует вопрос: нижний регистр + схлопывание пробелов в один."""
    return _WHITESPACE_RE.sub(" ", text.strip()).lower()


def question_key(question: str, uid: str = "") -> str:
    """Возвращает ключ кэша: SHA1-хэш нормализованного текста вопроса и uid.

    При непустом uid ключ дополнительно зависит от идентификатора пользователя,
    что даёт каждому пользователю отдельный кэш. Пустой uid сохраняет прежнее
    поведение (общий кэш).
    """
    normalized = normalize_question(question)
    if uid:
        return hashlib.sha1(f"{uid}\n{normalized}".encode("utf-8")).hexdigest()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class Cache:
    """Кэш ответов и история диалога в базе данных SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cached_queries (
                key        TEXT PRIMARY KEY,
                question   TEXT NOT NULL,
                answer     TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dialogue_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                question   TEXT NOT NULL,
                answer     TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # --- Кэш ответов ---

    def get(self, key: str) -> str | None:
        """Возвращает ответ по ключу или None, если записи нет."""
        row = self._conn.execute(
            "SELECT answer FROM cached_queries WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, question: str, answer: str) -> None:
        """Записывает (или обновляет) ответ для заданного ключа."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cached_queries (key, question, answer, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, question, answer, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    # --- История диалога ---

    def add_history(self, question: str, answer: str) -> None:
        """Добавляет пару «вопрос-ответ» в историю диалога."""
        self._conn.execute(
            """
            INSERT INTO dialogue_history (question, answer, created_at)
            VALUES (?, ?, ?)
            """,
            (question, answer, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def get_history(self, limit: int) -> list[tuple[str, str]]:
        """Возвращает последние `limit` пар «вопрос-ответ» в хронологическом порядке."""
        rows = self._conn.execute(
            "SELECT question, answer FROM dialogue_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(question, answer) for question, answer in reversed(rows)]

    def close(self) -> None:
        """Закрывает соединение с базой данных."""
        self._conn.close()
