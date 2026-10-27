"""
Ограничение частоты вопросов пользователей (rate limiting).

Момент каждого заданного вопроса сохраняется в SQLite — таблица question_log
в том же файле, что и кэш ответов (см. CACHE_DB_PATH). Благодаря этому лимиты
действуют непрерывно, в том числе между перезапусками ассистента.

Лимиты задаются в Config (.env):
    MAX_QUESTIONS_PER_MINUTE — сколько вопросов в минуту задаёт один пользователь;
    MAX_QUESTIONS_PER_DAY    — сколько вопросов в сутки задаёт один пользователь.

Проверка лимита и запись момента вопроса выполняются атомарно в одной
транзакции (BEGIN IMMEDIATE), поэтому даже при конкурентных вызовах лимит
не может быть превышен гонкой «проверил → записал».
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Записи старше суток больше никогда не влияют на лимиты — их можно удалять.
_RETENTION = timedelta(days=1)


class RateLimitError(RuntimeError):
    """Превышено допустимое число вопросов пользователя за период времени."""


class RateLimiter:
    """Проверяет лимиты числа вопросов и сохраняет моменты их задавания."""

    def __init__(self, db_path: Path, per_minute: int, per_day: int) -> None:
        """Открывает SQLite-журнал вопросов и подготавливает таблицу.

        per_minute и per_day — максимальное число вопросов одного пользователя
        за минуту и за сутки соответственно.
        """
        if per_minute < 1 or per_day < 1:
            raise ValueError("Лимиты per_minute и per_day должны быть положительными")

        self.per_minute = per_minute
        self.per_day = per_day

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS question_log (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                uid      TEXT NOT NULL,
                asked_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_log_uid_asked_at
            ON question_log (uid, asked_at)
            """
        )
        # Подчищаем устаревшие записи, оставшиеся после предыдущих запусков
        self._cleanup(datetime.now(timezone.utc))
        self._conn.commit()

    def allow(self, uid: str) -> bool:
        """True, если вопрос может быть задан: лимиты не превышены, момент зафиксирован.

        Если хотя бы один лимит исчерпан, возвращает False и ничего не записывает.
        Проверка и запись выполняются атомарно в одной транзакции.
        """
        now = datetime.now(timezone.utc)
        iso_now = now.isoformat()

        try:
            # BEGIN IMMEDIATE захватывает транзакцию сразу, чтобы проверка и запись
            # не разошлись даже при одновременных вызовах из разных потоков/процессов
            self._conn.execute("BEGIN IMMEDIATE")
            self._cleanup(now)

            minute_count = self._count_since(
                uid, (now - timedelta(minutes=1)).isoformat()
            )
            day_count = self._count_since(uid, (now - timedelta(days=1)).isoformat())

            if minute_count >= self.per_minute or day_count >= self.per_day:
                self._conn.rollback()
                return False

            self._conn.execute(
                "INSERT INTO question_log (uid, asked_at) VALUES (?, ?)",
                (uid, iso_now),
            )
            self._conn.commit()
            return True
        except Exception:
            self._conn.rollback()
            raise

    def _count_since(self, uid: str, since_iso: str) -> int:
        """Число вопросов пользователя с момента since_iso (включительно)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM question_log WHERE uid = ? AND asked_at >= ?",
            (uid, since_iso),
        ).fetchone()
        return row[0] if row else 0

    def _cleanup(self, now: datetime) -> None:
        """Удаляет записи, которые уже не влияют ни на один лимит (старше суток)."""
        cutoff_iso = (now - _RETENTION).isoformat()
        self._conn.execute(
            "DELETE FROM question_log WHERE asked_at < ?", (cutoff_iso,)
        )

    def close(self) -> None:
        """Закрывает соединение с базой данных журнала."""
        self._conn.close()
