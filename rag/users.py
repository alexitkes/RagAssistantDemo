"""
Хранилище пользователей веб-ассистента в SQLite.

Класс UserStore отвечает за:
  1. создание таблицы users (если её нет) в указанном SQLite-файле;
  2. регистрацию пользователя: уникальный e-mail, отображаемое имя и пароль;
  3. вход по e-mail и паролю (проверка хэша пароля);
  4. каждому пользователю присваивается числовой id (AUTOINCREMENT).

Пароли хранятся только в виде хэшей (werkzeug.security, scrypt по умолчанию).
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

# Простая проверка формата e-mail: "что-то@домен.зона" без пробелов
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

MIN_PASSWORD_LENGTH = 8
MAX_DISPLAY_NAME_LENGTH = 64


class UserError(Exception):
    """Базовая ошибка работы с пользователями (некорректные данные и т.п.)."""


class EmailAlreadyUsedError(UserError):
    """E-mail уже зарегистрирован другим пользователем."""


@dataclass(frozen=True)
class User:
    """Зарегистрированный пользователь (без хэша пароля)."""

    id: int
    email: str
    display_name: str


def validate_email(email: str) -> str:
    """Проверяет и нормализует e-mail (приводит к нижнему регистру)."""
    email = (email or "").strip().lower()
    if not email:
        raise UserError("Укажите адрес электронной почты.")
    if len(email) > 254 or not _EMAIL_RE.match(email):
        raise UserError("Некорректный адрес электронной почты.")
    return email


def validate_display_name(name: str) -> str:
    """Проверяет и нормализует отображаемое имя."""
    name = " ".join((name or "").split())
    if not name:
        raise UserError("Укажите отображаемое имя.")
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise UserError(
            f"Отображаемое имя не должно превышать {MAX_DISPLAY_NAME_LENGTH} символов."
        )
    return name


def validate_password(password: str, confirmation: str) -> None:
    """Проверяет пароль: минимальная длина и совпадение с подтверждением."""
    if not password:
        raise UserError("Укажите пароль.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise UserError(f"Пароль должен содержать не менее {MIN_PASSWORD_LENGTH} символов.")
    if password != confirmation:
        raise UserError("Пароль и подтверждение не совпадают.")


class UserStore:
    """SQLite-хранилище зарегистрированных пользователей."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    email         TEXT NOT NULL UNIQUE,
                    display_name  TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            self._conn.commit()

    def create_user(self, email: str, display_name: str, password: str) -> User:
        """Регистрирует пользователя и возвращает его с присвоенным id.

        E-mail проверяется на корректность и уникальность, пароль — на длину.
        При попытке повторно занять e-mail возникает EmailAlreadyUsedError.
        """
        email = validate_email(email)
        display_name = validate_display_name(display_name)
        password_hash = generate_password_hash(password)

        with self._lock:
            try:
                cursor = self._conn.execute(
                    "INSERT INTO users (email, display_name, password_hash) VALUES (?, ?, ?)",
                    (email, display_name, password_hash),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise EmailAlreadyUsedError(
                    "Пользователь с таким адресом электронной почты уже зарегистрирован."
                ) from exc
            user_id = int(cursor.lastrowid)
        return User(id=user_id, email=email, display_name=display_name)

    def get_by_email(self, email: str) -> User | None:
        """Возвращает пользователя по e-mail или None, если его нет."""
        email = validate_email(email)
        with self._lock:
            row = self._conn.execute(
                "SELECT id, email, display_name FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None:
            return None
        return User(id=int(row[0]), email=str(row[1]), display_name=str(row[2]))

    def verify_login(self, email: str, password: str) -> User | None:
        """Проверяет пару «e-mail + пароль»; возвращает User при успехе, иначе None."""
        email = validate_email(email)
        if not password:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id, email, display_name, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None:
            # Фиктивная проверка хэша, чтобы время ответа не выдавало,
            # существует ли пользователь с таким e-mail
            check_password_hash("scrypt:32768:8:1$salt$hash", password)
            return None
        if not check_password_hash(str(row[3]), password):
            return None
        return User(id=int(row[0]), email=str(row[1]), display_name=str(row[2]))

    def count_users(self) -> int:
        """Возвращает количество зарегистрированных пользователей."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Закрывает соединение с базой пользователей."""
        with self._lock:
            self._conn.close()
