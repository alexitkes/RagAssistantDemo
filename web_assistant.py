#!/usr/bin/env python3
"""
RAG-ассистент: веб-приложение на Flask с регистрацией пользователей.

Скрипт:
  1. запускает локальный HTTP-сервер Flask (по умолчанию http://127.0.0.1:5000);
  2. поддерживает регистрацию (уникальный e-mail, отображаемое имя, пароль
     с подтверждением) и вход по e-mail и паролю; пользователи хранятся
     в SQLite (файл задаётся настройкой USERS_DB_PATH), каждому присваивается
     числовой id;
  3. доступ к чату с ИИ-ассистентом есть только у зарегистрированных
     пользователей: на главной странице форма для ввода вопроса; после
     отправки вопрос передаётся в RagAssistant.ask_question, а в качестве
     идентификатора пользователя передаётся строка "id_{ИД_ИЗ_БАЗЫ}".
     Это даёт каждому пользователю отдельный кэш ответов и отдельный лимит
     частоты вопросов;
  4. полученный ответ выводится на странице вместе с формой для следующего вопроса.

Конфигурация читается классом Config из config.py (переменные — в файле .env,
шаблон — .env.example).

Пример запуска:
    python web_assistant.py

Примечания:
  * разработческий сервер Flask запускается однопоточным (threaded=False):
    RagAssistant и UserStore держат по одному соединению с SQLite и общую
    историю диалога, поэтому запросы обрабатываются последовательно;
  * пароли хранятся только в виде хэшей (werkzeug.security);
  * сессии подписываются ключом FLASK_SECRET_KEY из .env (если не задан,
    ключ генерируется случайно при каждом запуске).
"""
from __future__ import annotations

import atexit
import functools
import logging
import secrets
import threading

from flask import (
    Flask,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

from config import Config, ConfigError
from rag.assistant import RagAssistant
from rag.rate_limit import RateLimitError
from rag.users import (
    EmailAlreadyUsedError,
    User,
    UserError,
    UserStore,
    validate_password,
)

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"  # HTTP-сервер слушает только локальные подключения
PORT = 5000

app = Flask(__name__)

# Ассистент и хранилище пользователей создаются лениво (при первом обращении),
# чтобы модуль можно было импортировать без настроенного .env, а ошибки
# конфигурации увидеть сразу при запуске (см. main()).
_assistant: RagAssistant | None = None
_user_store: UserStore | None = None
_services_lock = threading.Lock()

BASE_CSS = """\
:root { color-scheme: light dark; }
body {
  max-width: 760px;
  margin: 2rem auto;
  padding: 0 1rem;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  line-height: 1.55;
}
h1 { margin-bottom: 0.25rem; }
.subtitle { color: #667; margin-top: 0; }
.answer, .error, .notice {
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin: 1.5rem 0;
}
.answer { background: #eef6ff; border: 1px solid #bcd9f5; }
.error { background: #fff0f0; border: 1px solid #f0b9b9; color: #8a1f1f; }
.notice { background: #f0fff4; border: 1px solid #b9e6c3; color: #1f6a34; }
.answer h2 { margin-top: 0; }
.answer pre { white-space: pre-wrap; font-family: inherit; margin: 0.25rem 0 0; }
label { font-weight: 600; display: block; margin: 1rem 0 0.35rem; }
textarea, input[type="text"], input[type="email"], input[type="password"] {
  width: 100%;
  box-sizing: border-box;
  padding: 0.6rem 0.75rem;
  font: inherit;
  border: 1px solid #ccc;
  border-radius: 6px;
}
button {
  margin-top: 0.75rem;
  padding: 0.55rem 1.4rem;
  font: inherit;
  font-weight: 600;
  color: #fff;
  background: #1a6fb5;
  border: 0;
  border-radius: 6px;
  cursor: pointer;
}
button:hover { background: #155c96; }
button.linklike {
  margin-top: 0;
  padding: 0.15rem 0.5rem;
  background: none;
  color: #1a6fb5;
  font-weight: 600;
}
nav { margin-bottom: 1rem; color: #667; }
nav a { color: #1a6fb5; text-decoration: none; }
"""

PAGE_TEMPLATE = """\
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG-ассистент</title>
  <style>{{ base_css }}</style>
</head>
<body>
  <nav>
    Вы вошли как <strong>{{ display_name|e }}</strong>
    ({{ user_email|e }}) · <a href="{{ url_for('logout') }}">Выйти</a>
  </nav>

  <h1>RAG-ассистент</h1>
  <p class="subtitle">Задайте вопрос по базе знаний — ответ будет сформирован на основе найденных документов.</p>

  {% if error %}
  <div class="error">{{ error|e }}</div>
  {% endif %}

  {% if answer is not none %}
  <div class="answer">
    <h2>Ответ</h2>
    <p class="subtitle" style="margin-top: 0"><strong>Вопрос:</strong> {{ question|e }}</p>
    <pre>{{ answer|e }}</pre>
  </div>
  {% endif %}

  <form method="post" autocomplete="off">
    <label for="question">Ваш вопрос:</label>
    <textarea id="question" name="question" rows="5" required autofocus
              placeholder="Например: Какие требования предъявляются к ...?"></textarea>
    <button type="submit">Задать вопрос</button>
  </form>
</body>
</html>
"""

REGISTER_TEMPLATE = """\
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Регистрация — RAG-ассистент</title>
  <style>{{ base_css }}</style>
</head>
<body>
  <h1>Регистрация</h1>
  <p class="subtitle">Зарегистрируйтесь, чтобы получить доступ к чату с ассистентом.</p>

  {% if error %}
  <div class="error">{{ error|e }}</div>
  {% endif %}

  <form method="post">
    <label for="email">Электронная почта:</label>
    <input type="email" id="email" name="email" value="{{ email|e }}" required autofocus>
    <label for="display_name">Отображаемое имя:</label>
    <input type="text" id="display_name" name="display_name" value="{{ display_name|e }}" required>
    <label for="password">Пароль (не менее 8 символов):</label>
    <input type="password" id="password" name="password" required>
    <label for="password2">Подтверждение пароля:</label>
    <input type="password" id="password2" name="password2" required>
    <button type="submit">Зарегистрироваться</button>
  </form>

  <p>Уже зарегистрированы? <a href="{{ url_for('login') }}">Войти</a></p>
</body>
</html>
"""

LOGIN_TEMPLATE = """\
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Вход — RAG-ассистент</title>
  <style>{{ base_css }}</style>
</head>
<body>
  <h1>Вход</h1>
  <p class="subtitle">Войдите, чтобы задавать вопросы ассистенту.</p>

  {% if error %}
  <div class="error">{{ error|e }}</div>
  {% endif %}

  {% if registered %}
  <div class="notice">Регистрация прошла успешно. Теперь войдите, используя e-mail и пароль.</div>
  {% endif %}

  <form method="post">
    <label for="email">Электронная почта:</label>
    <input type="email" id="email" name="email" value="{{ email|e }}" required autofocus>
    <label for="password">Пароль:</label>
    <input type="password" id="password" name="password" required>
    <button type="submit">Войти</button>
  </form>

  <p>Ещё нет аккаунта? <a href="{{ url_for('register') }}">Зарегистрироваться</a></p>
</body>
</html>
"""


def get_config() -> Config:
    """Возвращает Config, где ошибки конфигурации превращаются в RuntimeError."""
    try:
        return Config.from_env()
    except ConfigError as exc:
        raise RuntimeError(f"Ошибка конфигурации: {exc}") from exc


def get_assistant() -> RagAssistant:
    """Возвращает RagAssistant, создавая его при первом обращении."""
    global _assistant
    if _assistant is None:
        with _services_lock:
            if _assistant is None:
                _assistant = RagAssistant(get_config())
    return _assistant


def get_user_store() -> UserStore:
    """Возвращает UserStore, создавая его при первом обращении."""
    global _user_store
    if _user_store is None:
        with _services_lock:
            if _user_store is None:
                _user_store = UserStore(get_config().users_db_path)
    return _user_store


def _close_services() -> None:
    """Закрывает соединения ассистента и хранилища пользователей."""
    global _assistant, _user_store
    if _assistant is not None:
        _assistant.close()
        _assistant = None
    if _user_store is not None:
        _user_store.close()
        _user_store = None


atexit.register(_close_services)


def _current_user() -> User | None:
    """Возвращает текущего вошедшего пользователя по данным сессии."""
    user_id = session.get("user_id")
    if user_id is None:
        return None
    return User(
        id=int(user_id),
        email=str(session.get("user_email", "")),
        display_name=str(session.get("user_display_name", "")),
    )


def login_required(view):
    """Декоратор: перенаправляет на страницу входа, если пользователь не вошёл."""

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if _current_user() is None:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


@app.route("/", methods=["GET", "POST"])
@login_required
def index() -> str:
    """Чат с ассистентом: форма вопроса, а после отправки — ответ и новая форма."""
    user = _current_user()
    question = ""
    answer: str | None = None
    error: str | None = None

    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        if not question:
            error = "Пожалуйста, введите вопрос."
        else:
            # Идентификатор пользователя из базы: отдельный кэш и лимиты
            # для каждого зарегистрированного пользователя
            uid = f"id_{user.id}"
            logger.info("[%s] Вопрос: '%s'", uid, question[:120])
            try:
                assistant = get_assistant()
            except RuntimeError as exc:
                logger.exception("Не удалось создать ассистента")
                error = str(exc)
            else:
                try:
                    answer = assistant.ask_question(question, uid)
                    logger.info("[%s] Ответ получен (%d симв.)", uid, len(answer))
                except RateLimitError as exc:
                    logger.warning("[%s] Превышен лимит вопросов: %s", uid, exc)
                    error = str(exc)
                except Exception:  # noqa: BLE001
                    logger.exception("[%s] Ошибка при формировании ответа", uid)
                    error = "Не удалось сформировать ответ. Попробуйте позже."

    return render_template_string(
        PAGE_TEMPLATE,
        base_css=BASE_CSS,
        display_name=user.display_name,
        user_email=user.email,
        question=question,
        answer=answer,
        error=error,
    )


@app.route("/register", methods=["GET", "POST"])
def register() -> str:
    """Регистрация: уникальный e-mail, отображаемое имя, пароль с подтверждением."""
    email = ""
    display_name = ""
    error: str | None = None

    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        display_name = (request.form.get("display_name") or "").strip()
        password = request.form.get("password") or ""
        password2 = request.form.get("password2") or ""

        try:
            validate_password(password, password2)
            user = get_user_store().create_user(email, display_name, password)
        except EmailAlreadyUsedError as exc:
            error = str(exc)
        except UserError as exc:
            error = str(exc)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка при регистрации пользователя")
            error = "Не удалось завершить регистрацию. Попробуйте позже."
        else:
            logger.info("Зарегистрирован пользователь id_%d (%s)", user.id, user.email)
            # После успешной регистрации предлагаем войти
            return redirect(url_for("login", registered=1))

    return render_template_string(
        REGISTER_TEMPLATE,
        base_css=BASE_CSS,
        email=email,
        display_name=display_name,
        error=error,
    )


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    """Вход по адресу электронной почты и паролю."""
    email = ""
    error: str | None = None
    registered = request.args.get("registered") == "1"

    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        try:
            user = get_user_store().verify_login(email, password)
        except UserError as exc:
            error = str(exc)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка при проверке учётных данных")
            error = "Не удалось выполнить вход. Попробуйте позже."
        else:
            if user is None:
                error = "Неверный адрес электронной почты или пароль."
            else:
                # Регенерируем сессию, чтобы защититься от фиксации сессии
                session.clear()
                session["user_id"] = user.id
                session["user_email"] = user.email
                session["user_display_name"] = user.display_name
                logger.info("[%s] Пользователь вошёл", f"id_{user.id}")
                next_url = request.args.get("next") or url_for("index")
                # Разрешаем редирект только на внутренние страницы
                if not next_url.startswith("/"):
                    next_url = url_for("index")
                return redirect(next_url)

    return render_template_string(
        LOGIN_TEMPLATE,
        base_css=BASE_CSS,
        email=email,
        error=error,
        registered=registered,
    )


@app.route("/logout", methods=["GET", "POST"])
def logout():
    """Выход: очищает сессию и возвращает на страницу входа."""
    session.clear()
    return redirect(url_for("login"))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Проверяем конфигурацию, создаём ассистента и хранилище пользователей
    # до запуска сервера, чтобы ошибки настроек были видны сразу,
    # а не при первом запросе.
    config = get_config()

    # Ключ подписи cookie-сессий: из .env или случайный (пока сервер запущен)
    app.secret_key = config.flask_secret_key or secrets.token_hex(32)
    if not config.flask_secret_key:
        logger.warning(
            "FLASK_SECRET_KEY не задан в .env: сессии будут сбрасываться "
            "при каждом перезапуске сервера"
        )

    store = get_user_store()
    print(f"Зарегистрированных пользователей: {store.count_users()}")

    try:
        get_assistant()
    except RuntimeError as exc:
        print(f"Ошибка: {exc}")
        raise SystemExit(1)

    print(f"Веб-ассистент запущен: http://{HOST}:{PORT}")
    print("Регистрация: /register, вход: /login")
    app.run(host=HOST, port=PORT, debug=False, threaded=False)


if __name__ == "__main__":
    main()
